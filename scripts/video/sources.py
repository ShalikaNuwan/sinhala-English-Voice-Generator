"""Give every shot a picture the channel is allowed to monetise.

    .venv/bin/python scripts/video/sources.py <shots.json>

Searches Wikimedia Commons and Openverse for the `archival` and `stock` shots, generates the
`generated` ones, downloads what it finds into `assets/` beside shots.json, and writes credits.md.
`card`, `chart` and `map` shots are left for graphics.py.

The licence gate is the point of this module. The channel is monetised and the subjects are real
criminal cases, so only public domain, CC0, PDM and CC-BY images are kept: CC-BY-SA would reach the
whole video, CC-BY-NC and CC-BY-ND forbid the use outright, and anything unrecognised is refused.
Generated likenesses of real named people are refused too, however the prompt is phrased.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings  # noqa: E402

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
# Wikimedia's robot policy refuses a User-Agent that does not say who is calling and how to reach them.
USER_AGENT = ("SinhalaEnglishVoice-Documentary/1.0 "
              "(https://github.com/ShalikaNuwan/sinhala-English-Voice-Generator; documentary research) python-httpx")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2.5-flare")
GENERATED_LICENCE = "Generated"
GENERATED_ATTRIBUTION = "Generated with OpenAI gpt-image-2.5"

SEARCHED_KINDS = ("archival", "stock")
DRAWN_KINDS = ("card", "chart", "map")
REUSE_WINDOW = 10  # shots: the same picture must not come back inside this many shots
GENERATE_WORKERS = int(os.getenv("IMAGE_WORKERS", "6"))  # pictures generated at once
RETRIES = 3
RETRY_PAUSE_S = 1.0
RETRY_MAX_S = 30.0
NEAR_CHARS = 24  # how close a "photo of" cue has to be to a name to be about that person
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}

# Terms that forbid commercial use or would spread their licence to the finished video.
_DENIED = re.compile(
    r"\b(sa|nc|nd)\b|by[\s\-]*(sa|nc|nd)|share[\s\-]*alike|non[\s\-]*commercial"
    r"|no[\s\-]*deriv|sampling|fair[\s\-]*use|all[\s\-]*rights[\s\-]*reserved|copyright"
)
# Terms that permit it, anchored so prose like "by permission of the author" is not read as CC-BY.
_ALLOWED = re.compile(
    r"^(?:cc[\s\-]*0"
    r"|cc[\s\-]*(?:by|attribution)"
    r"|(?:cc[\s\-]*)?pdm"
    r"|(?:cc[\s\-]*)?pd\b"
    r"|public[\s\-]*domain"
    r"|(?:by|attribution)(?=[\s\-]*(?:\d|international|unported|generic|deed|$))"
    r")"
)
_TAG = re.compile(r"<[^>]+>")
_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+(?:[A-Z]\.|[A-Z][a-z]+|van|von|de|del|da|di|la))+\b")
_LIKENESS = re.compile(
    r"\b(?:photo|photos|photograph|photographs|photographic|portrait|portraits|mugshot|mug shot"
    r"|headshot|face|faces|likeness|selfie|picture|pictures|depiction|rendering)\b",
    re.IGNORECASE,
)


def commercial_licence(name: str) -> bool:
    """True only for licences that allow commercial reuse: public domain, CC0, PDM, CC-BY."""
    text = re.sub(r"\s+", " ", str(name or "").strip().lower())
    if not text or _DENIED.search(text):
        return False
    return bool(_ALLOWED.match(text))


def plain_text(value: str | None, fallback: str = "Unknown") -> str:
    """Reduce an API's HTML attribution blob to one readable line."""
    text = _TAG.sub(" ", str(value or ""))
    text = unescape(text).replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip(" \t\n-,;:·|/")
    return text[:200] or fallback


def _get(url: str, params: dict | None = None, timeout: float = 30.0) -> httpx.Response:
    """One HTTP read, retried: upload.wikimedia.org rate-limits bursts and names a Retry-After."""
    for attempt in range(RETRIES):
        pause = RETRY_PAUSE_S * (attempt + 1)
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True,
                              headers={"User-Agent": USER_AGENT}) as client:
                response = client.get(url, params=params)
            after = str(response.headers.get("retry-after") or "")
            if after.isdigit():
                pause = min(float(after), RETRY_MAX_S)
            response.raise_for_status()
            return response
        except Exception:  # noqa: BLE001 - transient rate limits and resets
            if attempt == RETRIES - 1:
                raise
            time.sleep(pause)
    raise RuntimeError(url)  # unreachable


def fetch_json(url: str, params: dict) -> dict:
    """The only network read behind the searches; tests replace this."""
    return _get(url, params).json()


def fetch_bytes(url: str) -> bytes:
    """The only network read behind download(); tests replace this."""
    return _get(url, timeout=60.0).content


def search_commons(query: str, limit: int = 8) -> list[dict]:
    """Licence-clean Commons images for a query, best match first."""
    data = fetch_json(
        COMMONS_API,
        {
            "action": "query",
            "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap",
            "gsrnamespace": 6,
            "gsrlimit": limit,
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size",
            "iiurlwidth": 1920,
            "format": "json",
        },
    )
    pages = (data.get("query") or {}).get("pages") or {}
    results = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 0)):  # "index" is the search rank
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = plain_text((meta.get("LicenseShortName") or {}).get("value"), fallback="")
        if not commercial_licence(licence):
            continue
        # Prefer the 1920px thumbnail, and describe the file actually downloaded.
        url = info.get("thumburl") or info.get("url")
        if not url:
            continue
        thumb = bool(info.get("thumburl"))
        title = page.get("title") or ""
        results.append(
            {
                "title": title,
                "url": url,
                "width": int((info.get("thumbwidth") if thumb else info.get("width")) or 0),
                "height": int((info.get("thumbheight") if thumb else info.get("height")) or 0),
                "licence": licence,
                "attribution": plain_text((meta.get("Artist") or {}).get("value")),
                "source_url": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
            }
        )
    return results


def search_openverse(query: str, limit: int = 8) -> list[dict]:
    """Licence-clean Openverse images for a query; the server filters, and so do we."""
    data = fetch_json(OPENVERSE_API, {"q": query, "page_size": limit, "license": "cc0,by,pdm"})
    results = []
    for item in data.get("results") or []:
        code = str(item.get("license") or "")
        if not commercial_licence(code):
            continue
        url = item.get("url")
        if not url:
            continue
        results.append(
            {
                "title": plain_text(item.get("title"), fallback="Untitled"),
                "url": url,
                "width": int(item.get("width") or 0),
                "height": int(item.get("height") or 0),
                "licence": openverse_licence(code, item.get("license_version")),
                "attribution": plain_text(item.get("creator")),
                "source_url": item.get("foreign_landing_url") or url,
            }
        )
    return results


def openverse_licence(code: str, version: str | None) -> str:
    """Openverse's short code as a name credits.md can show: 'by' + '4.0' -> 'CC BY 4.0'."""
    label = {"cc0": "CC0", "pdm": "Public Domain Mark", "by": "CC BY"}.get(code.lower(), code.upper())
    return f"{label} {version}".strip() if version else label


def names_a_real_person(prompt: str) -> bool:
    """True when a prompt asks for the likeness of a named person, which the design forbids."""
    names = [m.span() for m in _NAME.finditer(prompt or "")]
    cues = [m.span() for m in _LIKENESS.finditer(prompt or "")]
    for name_start, name_end in names:
        for cue_start, cue_end in cues:
            gap = name_start - cue_end if cue_start < name_start else cue_start - name_end
            if gap <= NEAR_CHARS:  # the cue and the name are describing each other
                return True
    return False


def generate_image(prompt: str, destination: Path, size: str = "1536x1024") -> None:
    """Draw a shot that no archive can supply; never a real person's likeness."""
    if names_a_real_person(prompt):
        raise ValueError(f"refusing to generate a likeness of a real person: {prompt!r}")
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    result = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=size, n=1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(result.data[0].b64_json))


def download(url: str, destination: Path) -> None:
    """Fetch an image once: a non-empty file already there is left alone, so reruns are free."""
    if destination.exists() and destination.stat().st_size > 0:
        return
    data = fetch_bytes(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def suffix_for(url: str) -> str:
    """The file extension to save a found image under, defaulting to .jpg."""
    suffix = Path(httpx.URL(url).path).suffix.lower()
    return suffix if suffix in IMAGE_SUFFIXES else ".jpg"


# Words that carry no picture and only narrow a photo search to nothing.
FILLER = {
    "a", "an", "the", "at", "in", "on", "of", "with", "and", "to", "from", "by", "for",
    "over", "under", "into", "near", "during", "its", "his", "her", "their",
}


def shorten(query: str) -> str:
    """The same search with the filler dropped and only the first few picture words kept.

    Stock libraries match on keywords: a five-word phrase often finds nothing where its three
    strongest words find hundreds. Returns "" when there is nothing left to try.
    """
    words = [word for word in query.split() if word.lower() not in FILLER]
    shortened = " ".join(words[:3])
    return shortened if shortened and shortened.lower() != query.strip().lower() else ""


def search_shot(shot: dict) -> tuple[list[dict], list[str]]:
    """Candidates for one shot, preferred source first, with the reasons any search gave up."""
    query = str(shot.get("query") or shot.get("text") or "").strip()
    if not query:
        return [], ["no query"]
    order = (search_commons, search_openverse)
    if shot.get("kind") == "stock":
        order = (search_openverse, search_commons)

    failures: list[str] = []
    attempts = [query]
    briefer = shorten(query)
    if briefer:
        attempts.append(briefer)
    for attempt in attempts:
        for search in order:
            try:
                results = search(attempt)
            except Exception as exc:  # noqa: BLE001 - a dead source must not sink the shot
                failures.append(f"{search.__name__} failed: {exc}")
                continue
            if results:
                return results, failures
        if attempt == query and briefer:
            failures.append(f"nothing for {query!r}; retrying as {briefer!r}")
    return [], failures


def pick(results: list[dict], used: dict[str, int], position: int) -> dict | None:
    """The best candidate this shot has not seen lately; a repeat only when there is no choice."""
    for item in results:
        last = used.get(item["url"])
        if last is None or position - last > REUSE_WINDOW:
            return item
    if not results:
        return None
    return min(results, key=lambda item: used.get(item["url"], -1))  # longest unused


def fill_shots(shots: list[dict], assets_dir: Path, on_progress=None) -> list[dict]:
    """Fill asset, licence, attribution and source_url for every searched or generated shot.

    Returns copies of the shots in playing order. A shot that cannot be sourced keeps
    asset: None and gains a short "error", because assembly holds the previous picture instead.
    """
    filled = [dict(shot) for shot in shots]
    assets_dir.mkdir(parents=True, exist_ok=True)

    # Searches are slow and independent, so they run together; the picking stays in shot order.
    wanted = [i for i, shot in enumerate(filled) if shot.get("kind") in SEARCHED_KINDS]
    found: dict[int, tuple[list[dict], list[str]]] = {}
    if wanted:
        with ThreadPoolExecutor(max_workers=min(5, len(wanted))) as pool:
            for position, outcome in zip(wanted, pool.map(lambda i: search_shot(filled[i]), wanted)):
                found[position] = outcome

    # Generation takes about twenty seconds a picture, so a film's worth has to run in parallel too.
    # Each result is written to its own file; the loop below only records what landed.
    drawing = [i for i, shot in enumerate(filled) if shot.get("kind") == "generated"]
    made: dict[int, Exception | None] = {}
    if drawing:
        def one(i: int) -> Exception | None:
            shot = filled[i]
            index = int(shot.get("index") or i + 1)
            target = assets_dir / f"{index:04d}.png"
            if target.exists() and target.stat().st_size:
                return None  # already generated: never pay for the same picture twice
            prompt = str(shot.get("prompt") or "").strip()
            if not prompt:
                return ValueError("no prompt")
            try:
                generate_image(prompt, target)
                return None
            except Exception as exc:  # noqa: BLE001 - reported per shot, below
                return exc

        with ThreadPoolExecutor(max_workers=min(GENERATE_WORKERS, len(drawing))) as pool:
            for position, outcome in zip(drawing, pool.map(one, drawing)):
                made[position] = outcome

    used: dict[str, int] = {}
    for position, shot in enumerate(filled):
        kind = shot.get("kind")
        if kind in DRAWN_KINDS:
            continue
        shot["asset"] = shot.get("asset") or None
        index = int(shot.get("index") or position + 1)
        try:
            if kind == "generated":
                failure = made.get(position)
                if failure is not None:
                    raise failure
                name = f"{index:04d}.png"
                if not (assets_dir / name).exists():
                    raise LookupError("generation produced no file")
                shot.update({"licence": GENERATED_LICENCE, "attribution": GENERATED_ATTRIBUTION})
            elif kind in SEARCHED_KINDS:
                results, failures = found.get(position, ([], []))
                choice = pick(results, used, position)
                if choice is None:
                    raise LookupError(failures[0] if failures else "no licensed image found")
                name = f"{index:04d}{suffix_for(choice['url'])}"
                download(choice["url"], assets_dir / name)
                used[choice["url"]] = position
                shot.update(
                    {
                        "licence": choice["licence"],
                        "attribution": choice["attribution"],
                        "source_url": choice["source_url"],
                    }
                )
            else:
                continue  # an unknown kind is somebody else's stage
            shot["asset"] = f"{assets_dir.name}/{name}"  # relative to shots.json
            shot.pop("error", None)
        except Exception as exc:  # noqa: BLE001 - one bad shot must not stop the pass
            shot["asset"] = None
            shot["error"] = (str(exc).strip() or exc.__class__.__name__)[:120]
        if on_progress:
            on_progress(position + 1, len(filled), shot)
    return filled


def credits_markdown(shots: list[dict]) -> str:
    """The credits.md body: every asset used, with its author, licence and source."""
    groups = {"archival": "Archival", "stock": "Stock", "generated": "Generated"}
    lines = ["# Credits", "", "Every image in this video, with the licence it was used under.", ""]
    seen: set[str] = set()
    for kind, heading in groups.items():
        bullets = []
        for shot in shots:
            if shot.get("kind") != kind or not shot.get("asset"):
                continue
            key = str(shot.get("source_url") or shot.get("prompt") or shot["asset"])
            if key in seen:
                continue
            seen.add(key)
            where = shot.get("source_url") or f"prompt: {shot.get('prompt', '')}".strip()
            bullets.append(
                f"- `{shot['asset']}` — {shot.get('attribution') or 'Unknown'}"
                f" — {shot.get('licence') or 'Unknown licence'} — {where}"
            )
        if bullets:
            lines += [f"## {heading}", "", *bullets, ""]
    if len(lines) == 4:
        lines.append("No images have been sourced yet.")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    plan = json.loads(path.read_text(encoding="utf-8"))

    def report(done: int, total: int, shot: dict) -> None:
        state = shot.get("error") or shot.get("asset") or "drawn later"
        print(f"  {done}/{total} {shot.get('kind')}: {state}", flush=True)

    plan["shots"] = fill_shots(plan.get("shots") or [], path.parent / "assets", on_progress=report)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    (path.parent / "credits.md").write_text(credits_markdown(plan["shots"]), encoding="utf-8")
    failed = [s.get("index") for s in plan["shots"] if s.get("error")]
    print(f"sourced {sum(1 for s in plan['shots'] if s.get('asset'))} shots; {len(failed)} unsourced {failed}")
    print(f"wrote {path} and {path.parent / 'credits.md'}")


if __name__ == "__main__":
    main()

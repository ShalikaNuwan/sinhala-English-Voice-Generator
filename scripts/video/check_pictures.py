"""Look at every picture the storyboard chose, against what its beat is meant to show.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/check_pictures.py <video_dir> --free-only [--limit N]

Reads `storyboard.json` (storyboard.py) and the picture library. For each beat that runs a photograph or a
clip, a vision model sees the picture (a clip: one frame from its middle) beside the beat's intent and
narration and its scene's place and year, and answers: does it fit, is the era right, does it keep the
story's care rules (no remains, victims with dignity), and if it does not fit, what to search for instead.

A beat that fails is offered its other choices, each checked the same way, and takes the first that
passes. A beat with nothing that passes goes on the needs list: `library/needs.json` (searched by
`harvest.py --stages needs`) and `needs.md` (the gap report for the owner). Checks are cached in
`checks.json` by picture and beat, so a rerun only looks at what changed. Writes the storyboard back.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402
from storyboard import load_library, save, set_picture  # noqa: E402

FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
ALTERNATIVES = 4          # other choices tried for a beat whose picture fails
REUSE_S = 60.0            # a replacement may not already run within this long of the beat


class PictureCheck(BaseModel):
    fits: Literal["yes", "partly", "no"] = Field(description="Does the picture show what this beat is meant to show?")
    misleads: bool = Field(description=(
        "True if a viewer would take the picture as something it is not: a specific person, building, "
        "document or moment presented as the one the narration names when it is not (another house as "
        "'his house', an unrelated person who reads as someone in the story, a photograph from a later date "
        "shown as if taken at this moment), a prominent sign or brand naming a different business or place than "
        "the narration's, or a picture that contradicts what the narration says. A generic "
        "period stand-in (a street, traffic, a building of the era) for a passage with no picture of its own "
        "does NOT mislead."))
    era_ok: bool = Field(description=(
        "False if the picture belongs to the wrong time for the scene: visible anachronisms (modern cars, phones, "
        "clothes), or a photograph of a person or place from years before or after the scene shown as if it were "
        "then (a 1978 booking photograph in a 1960s scene)."))
    care_ok: bool = Field(description="False if it shows remains, a body, gore, or a victim in a degrading way.")
    setting_ok: bool = Field(default=True, description=(
        "False if the picture's time of day or season plainly contradicts the words (bright daylight under a night "
        "scene, summer under a snowstorm)."))
    problem: str = Field(default="", description="Half a sentence, only if something is wrong.")
    search: str = Field(default="", description="If fits is not yes: two to five plain words to search an archive for a better picture.")
    link_ok: bool = Field(default=True, description=(
        "Does the picture really have the link the editor declared? subject: it shows the named person, place, "
        "object or document. place_era: it shows that town or area in those years (another city does not count). "
        "action: it shows that specific object or action of the era (a shop is not a mechanic; a street is not "
        "a funeral home)."))
    detail: str = Field(default="", description=(
        "If no archive photograph is likely to show this: an object or setting that would, as an illustration "
        "(no people, no named place), e.g. 'a 1970s cash register on a counter at night'."))


SYSTEM = """You check pictures for a true-crime documentary built from real archival photographs and footage.
You see one picture and the beat it runs under: what the beat is meant to show (its intent), the narration,
and where and when the scene happens. Judge only what is visible.
- fits: yes if it shows the subject or a fair stand-in for the moment; partly if it is loosely related
  (an honest period stand-in is at least partly); no if it is unrelated.
- misleads: whether a viewer would be misled about who, where or when (see the field's description).
  This is the question that matters most; be precise.
- era_ok: false only for visible anachronisms against the scene's year.
- care_ok: false for remains, bodies, gore, or a victim shown without dignity.
- search: when fits is not yes, what an archive search should look for instead."""


def picture_bytes(video_dir: Path, beat: dict) -> bytes | None:
    """A small JPEG of the beat's picture: the photograph, or a frame from the middle of its clip."""
    asset = beat.get("asset")
    if not asset:
        return None
    path = video_dir / asset
    try:
        if beat.get("media") == "clip":
            at = (float(beat.get("clip_in") or 0) + float(beat.get("clip_out") or beat.get("clip_in") or 0)) / 2
            done = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{at:.2f}", "-i", str(path), "-frames:v", "1",
                                   "-vf", "scale=512:-2", "-f", "image2", "-c:v", "mjpeg", "-"],
                                  capture_output=True, timeout=60)
            return done.stdout or None
        image = Image.open(path).convert("RGB")
        image.thumbnail((512, 512))
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=80)
        return buf.getvalue()
    except Exception:  # noqa: BLE001 - an unreadable picture is simply not checked
        return None


def check_key(beat: dict, library_id: str) -> str:
    return hashlib.sha1(json.dumps([library_id, beat.get("intent"), beat.get("narration")[:300]],
                                   sort_keys=True).encode()).hexdigest()[:16]


def ask(client, model: str, image: bytes, beat: dict, scene: dict, rules: list[str]) -> PictureCheck:
    text = (f"Scene: {scene.get('title')} - {scene.get('setting')}, {scene.get('story_time')}. Mood: {scene.get('mood')}.\n"
            f"Beat intent: {beat.get('intent') or '-'}\nNarration: {beat['narration']}\n"
            f"Declared link: {beat.get('link') or 'place_era'} - {beat.get('link_reason') or 'not stated'}\n"
            + ("Care rules:\n" + "\n".join(f"- {r}" for r in rules[:8]) if rules else ""))
    response = client.responses.parse(model=model, input=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(image).decode(), "detail": "low"},
        ]}], text_format=PictureCheck)
    return response.output_parsed


def passes(check: dict | None) -> bool:
    """A picture stays unless it misleads, is from the wrong era, or is careless with the people in it.
    A loose stand-in stays (and goes on the wish list for a better picture)."""
    return (bool(check) and not check.get("misleads", check.get("fits") == "no") and check["era_ok"]
            and check["care_ok"] and check.get("link_ok", True) and check.get("setting_ok", True))


def reused_near(beats: list[dict], position: int, library_id: str, window_s: float = REUSE_S) -> bool:
    """Whether a picture already runs within `window_s` of the beat at `position`."""
    here = beats[position]
    return any(b.get("library_id") == library_id and abs(b["start"] - here["start"]) < window_s
               for k, b in enumerate(beats) if k != position)


def long_enough(item: dict, beat: dict) -> bool:
    """A clip can only replace a picture if it runs the whole beat and a dissolve's tail."""
    return item.get("media") != "clip" or item["length"] >= (beat["end"] - beat["start"]) + 1.0


def candidate_beat(beat: dict, item: dict) -> dict:
    """The beat as it would be with another picture, for checking that picture."""
    trial = dict(beat)
    set_picture(trial, item)
    if item.get("media") == "clip":
        trial.update({"media": "clip", "clip_in": item["in"], "clip_out": item["out"]})
    return trial


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="check only the first N beats (a trial)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    library, _ = load_library(video_dir, args.free_only)
    scenes = {s["id"]: s for s in sb["scenes"]}
    rules = sb["bible"].get("sensitivities") or []
    cache_path = video_dir / "checks.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    client = OpenAI(api_key=settings.openai_api_key)
    # Beats the ladder already derives (a reframe, a hold, an illustration) are not judged as finds of their own.
    beats = [b for b in sb["beats"] if b.get("media") in ("photo", "clip") and b.get("library_id")
             and b.get("link", "place_era") in ("subject", "place_era", "action")]
    beats = beats[:args.limit] if args.limit else beats

    def check(beat: dict, library_id: str) -> dict | None:
        key = check_key(beat, library_id)
        if key in cache:
            return cache[key]
        image = picture_bytes(video_dir, beat)
        if image is None:
            return None
        for attempt in range(3):
            try:
                result = ask(client, args.model, image, beat, scenes.get(beat.get("scene"), {}), rules).model_dump()
                result["library_id"] = library_id
                cache[key] = result
                return result
            except Exception as exc:  # noqa: BLE001 - a failed check leaves the picture as it is
                if attempt == 2:
                    print(f"  check failed for beat {beat['index']}: {str(exc)[:120]}", flush=True)
        return None

    def one(beat: dict) -> tuple[dict, dict | None, list[str]]:
        """The beat, its picture's check, and (if that fails) the other choices that pass, best first."""
        first = check(beat, beat["library_id"])
        if first is None or passes(first):
            return beat, first, []
        good = []
        options = [c for c in beat.get("choices") or [] if c != beat["library_id"] and c in library
                   and long_enough(library[c], beat)]
        for cid in options[:ALTERNATIVES]:
            if passes(check(candidate_beat(beat, library[cid]), cid)):
                good.append(cid)
        return beat, first, good

    print(f"checking {len(beats)} pictures with {args.model}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, beats))
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    swapped, needs, report = 0, {}, []
    order = sb["beats"]
    position = {b["index"]: k for k, b in enumerate(order)}
    from plan_edit import max_uses
    cap = max_uses(sb.get("duration_s") or order[-1]["end"])
    uses: dict[str, int] = {}
    for b in order:
        if b.get("library_id"):
            uses[b["library_id"]] = uses.get(b["library_id"], 0) + 1
    for beat, first, good in results:
        beat["check"] = {k: first.get(k) for k in ("fits", "misleads", "era_ok", "care_ok", "link_ok", "problem")} if first else None
        if first is None:
            continue
        if passes(first):
            if first["fits"] != "yes" and (first.get("search") or "").strip():
                beat["check"]["wish"] = first["search"].strip()      # fine as it is; a better picture would help
            continue
        # Swaps are decided in film order, so a replacement never lands within a minute of itself, and
        # never takes a picture past its share of the film: a few safe pictures must not fill every gap.
        replacement = next((c for c in good if not reused_near(order, position[beat["index"]], c)
                            and uses.get(c, 0) < cap), None)
        if replacement:
            uses[beat["library_id"]] = uses.get(beat["library_id"], 1) - 1
            uses[replacement] = uses.get(replacement, 0) + 1
            beat["check"]["replaced"] = beat["library_id"]
            item = library[replacement]
            set_picture(beat, item)
            if item.get("media") == "clip":
                beat.update({"media": "clip", "kind": "clip", "clip_id": item["clip_id"],
                             "clip_in": item["in"], "clip_out": item["out"]})
            swapped += 1
            continue
        # Nothing true to show: down the ladder (hold the picture before, or illustrate an object or setting).
        beat["link"] = "hold"
        if first.get("detail"):
            beat["detail"] = first["detail"]
        query = (first.get("search") or "").strip()
        if query:
            needs.setdefault(query.lower(), {"query": query, "beats": []})["beats"].append(beat["index"])
        m, s = divmod(int(beat["start"]), 60)
        report.append(f"| {beat['index']} | {m}:{s:02d} | {beat['library_id']} | {first['fits']}"
                      f"{'' if first['era_ok'] else ', wrong era'}{'' if first['care_ok'] else ', CARE'} | "
                      f"{(first.get('problem') or '').replace('|', '/')} | {query} |")
    save(video_dir, sb)
    (video_dir / "library" / "needs.json").write_text(json.dumps(list(needs.values()), ensure_ascii=False, indent=1),
                                                      encoding="utf-8")
    checked = [r for r in results if r[1]]
    wishes = sum(1 for b in order if (b.get("check") or {}).get("wish"))
    fit = sum(1 for _, c, _ in checked if passes(c))
    (video_dir / "needs.md").write_text(
        "# Pictures that do not fit yet\n\n"
        f"{len(checked)} pictures checked: {fit} pass (including honest stand-ins), {swapped} replaced because "
        f"they misled or were out of era, {len(report)} still need a picture ({len(needs)} searches in "
        f"library/needs.json). {wishes} stand-ins could be improved by a better picture later.\n\n"
        "| beat | time | picture | verdict | problem | search |\n|---|---|---|---|---|---|\n" + "\n".join(report) + "\n",
        encoding="utf-8")
    print(f"{len(checked)} checked: {fit} fit, {swapped} swapped for a better choice, {len(report)} still need a picture "
          f"({len(needs)} searches); wrote needs.md and library/needs.json")


if __name__ == "__main__":
    main()

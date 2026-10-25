"""Gather every usable real picture for an episode into library/ and library.json.

    .venv/bin/python scripts/video/harvest.py data/video/gacy [--stages case,entities,period,wikipedia,web]

Sources, in order of preference (Revision 2 of the documentary design):
  1. Commons case categories (Category:John Wayne Gacy and its subcategories), every licence.
  2. Commons and Openverse searches for every person, place and institution the script names.
  3. Period atmosphere: DOCUMERICA and 1970s Chicago categories on Commons.
  4. Images on the relevant English Wikipedia articles, non-free (fair use) files included.
  5. News photographs: the og:image of each page listed in library/web_seeds.json.

Duplicates go by URL and by an 8x8 average hash. Files are cached: a rerun never fetches one twice,
and keeps every caption field caption.py has already written.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sources  # noqa: E402
from sources import COMMONS_API, fetch_json, plain_text, search_openverse  # noqa: E402

from app.config import settings  # noqa: E402

# Wikimedia's policy asks for a User-Agent that says who is calling and why. sources._get reads
# this module attribute at call time, so setting it here covers search_*, fetch_json and download.
sources.USER_AGENT = (
    "SinhalaEnglishVoice-DocumentaryHarvest/1.0 "
    "(https://github.com/ShalikaNuwan/sinhala-English-Voice-Generator; documentary image library) "
    "python-httpx/0.28"
)
ENWIKI_API = "https://en.wikipedia.org/w/api.php"
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/18.0 Safari/605.1.15"
)

# Wikimedia serves thumbnails only at standard widths (https://w.wiki/GHai); others are rounded up
# or refused, and original files are rate-limited hard. 1920 is the standard step for 1080p.
THUMB_WIDTH = 1920
WIKIMEDIA_SPACING_S = 1.0  # between requests to upload.wikimedia.org, across all workers
SIZE_FLOOR = 480
HASH_DISTANCE = 5
WORKERS = 5
RASTER = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif", ".webp"}
PREFIX = {"commons": "c", "openverse": "o", "wikipedia": "w", "web": "n"}
RIGHTS_RANK = {"free": 0, "original": 1, "fair_use": 2, "generated": 3}

CASE_ROOT = "Category:John Wayne Gacy"
PERIOD_CATEGORIES = [  # (category, depth)
    ("Category:DOCUMERICA photographs of Chicago", 0),
    ("Category:DOCUMERICA photographs of Illinois", 0),
    ("Category:Chicago in the 1970s", 1),
    ("Category:Black and white photographs of Chicago in the 1970s", 0),
]
PERIOD_SKIP = re.compile(r"sport|baseball|football|hockey|basketball|cubs|white sox|bears|bulls", re.I)
PERIOD_MAX = 120

STARTER_ENTITIES = [
    ("John Wayne Gacy", "person"), ("Robert Piest", "person"), ("Nisson Pharmacy", "business"),
    ("Des Plaines Illinois", "place"), ("8213 West Summerdale Avenue", "place"),
    ("Norwood Park Township", "place"), ("Cook County Criminal Courts Building", "institution"),
    ("Cook County Jail", "institution"), ("Menard Correctional Center", "institution"),
    ("Stateville Correctional Center", "institution"), ("Des Plaines River", "place"),
    ("Waterloo Iowa", "place"), ("Anamosa State Penitentiary", "institution"),
    ("PDM Contractors", "business"), ("Pogo the Clown", "object"), ("Jaycees", "institution"),
    ("Sam Amirante", "person"), ("Terry Sullivan", "person"), ("William Kunkle", "person"),
    ("Joseph Kozenczak", "person"), ("Jeffrey Rignall", "person"), ("Michael Rossi", "person"),
    ("David Cram", "person"), ("Gregory Godzik", "person"), ("John Butkovich", "person"),
    ("Michael Bonnin", "person"), ("John Mowery", "person"), ("James Mazzara", "person"),
    ("Rosalynn Carter", "person"),
]

# Articles accepted as they are; any other title must link to the Gacy article to count.
WIKI_ARTICLES = [
    "John Wayne Gacy", "Des Plaines, Illinois", "Norwood Park Township, Cook County, Illinois",
    "Cook County Jail", "Menard Correctional Center", "Stateville Correctional Center",
    "Des Plaines River", "Waterloo, Iowa", "Anamosa State Penitentiary", "Sam Amirante",
    "William Kunkle", "Rosalynn Carter", "United States Junior Chamber", "Robert Piest",
    "Cook County Sheriff's Office", "Norwood Park, Chicago", "Chicago Police Department",
]
WIKI_PER_ARTICLE = 10
WIKI_SKIP = re.compile(
    r"logo|icon|flag of|seal of|symbol|commons-|wiki|edit-|question_book|ambox|padlock|portal"
    r"|disambig|crystal|nuvola|signature|coat of arms|locator|location map|\bmap\b.*\.svg",
    re.I,
)

_lock = threading.Lock()


# --- perceptual hash and dedupe (pure) -------------------------------------------------------

def average_hash(image: Image.Image) -> int:
    """64-bit average hash: 8x8 greyscale, one bit per pixel brighter than the mean."""
    small = image.convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    pixels = list(small.tobytes())
    mean = sum(pixels) / len(pixels)
    bits = 0
    for value in pixels:
        bits = (bits << 1) | (1 if value > mean else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def _better(a: dict, b: dict) -> bool:
    """True when a should be kept over b: the freer copy, then the larger file, then the earlier."""
    rank_a = RIGHTS_RANK.get(a.get("rights"), 9)
    rank_b = RIGHTS_RANK.get(b.get("rights"), 9)
    if rank_a != rank_b:
        return rank_a < rank_b
    return int(a.get("width") or 0) * int(a.get("height") or 0) > int(b.get("width") or 0) * int(b.get("height") or 0)


def dedupe(items: list[dict], threshold: int = HASH_DISTANCE) -> tuple[list[dict], list[tuple[str, str]]]:
    """Drop repeats by URL, then by perceptual hash (Hamming <= threshold).

    Of two copies of one picture the larger file is kept; a free copy is kept over a fair-use one
    whatever the sizes, so a picture is never made claimable when a free version exists. Returns
    the kept items in input order and (dropped id, kept id) pairs.
    """
    dropped: list[tuple[str, str]] = []
    by_url: dict[str, dict] = {}
    unique: list[dict] = []
    for item in items:
        url = item.get("url")
        if url and url in by_url:
            dropped.append((item["id"], by_url[url]["id"]))
            continue
        if url:
            by_url[url] = item
        unique.append(item)

    kept: list[dict] = []
    for item in unique:
        if item.get("hash") is None:
            kept.append(item)
            continue
        twin = next((k for k in kept if k.get("hash") is not None
                     and hamming(k["hash"], item["hash"]) <= threshold), None)
        if twin is None:
            kept.append(item)
        elif _better(item, twin):
            kept[kept.index(twin)] = item
            dropped.append((twin["id"], item["id"]))
        else:
            dropped.append((item["id"], twin["id"]))
    order = {id(item): n for n, item in enumerate(unique)}
    kept.sort(key=lambda item: order[id(item)])
    return kept, dropped


def big_enough(item: dict, floor: int = SIZE_FLOOR) -> bool:
    """The short side must reach the floor, except for the irreplaceable case-category files."""
    if item.get("case"):
        return True
    return min(int(item.get("width") or 0), int(item.get("height") or 0)) >= floor


# --- page image extraction (pure) ------------------------------------------------------------

class _PageImages(HTMLParser):
    META_KEYS = ("og:image:secure_url", "og:image", "og:image:url", "twitter:image", "twitter:image:src")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, str] = {}
        self.depth = 0  # inside <article> or <main>
        self.images: list[tuple[int, str]] = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or "").strip().lower()
            if key in self.META_KEYS and a.get("content") and key not in self.meta:
                self.meta[key] = a["content"].strip()
        elif tag in ("article", "main"):
            self.depth += 1
        elif tag == "img" and self.depth:
            src = a.get("src") or a.get("data-src") or a.get("data-original") or ""
            if not src or src.startswith("data:"):
                return
            try:
                area = int(a.get("width") or 0) * int(a.get("height") or 0)
            except ValueError:
                area = 0
            self.images.append((area, src.strip()))

    def handle_endtag(self, tag):
        if tag in ("article", "main") and self.depth:
            self.depth -= 1


# Site-wide share pictures (logos, live-stream cards, placeholders) say nothing about the story.
GENERIC_IMAGE = re.compile(
    r"logo|favicon|placeholder|default[-_]?(?:image|share|og|social)|og[-_]default|fallback"
    r"|live[-_]thumbnail|news[-_]stream|sprite|blank\.(?:gif|png)",
    re.I,
)


def page_image(html: str, base_url: str) -> str | None:
    """The page's lead picture: og:image, then twitter:image, then the largest article <img>."""
    if not html:
        return None
    parser = _PageImages()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 - broken markup still yields what was parsed
        pass
    for key in ("og:image:secure_url", "og:image", "og:image:url", "twitter:image", "twitter:image:src"):
        if parser.meta.get(key) and not GENERIC_IMAGE.search(parser.meta[key]):
            return urljoin(base_url, parser.meta[key])
    images = [pair for pair in parser.images if not GENERIC_IMAGE.search(pair[1])]
    if images:
        area, src = max(images, key=lambda pair: pair[0])
        return urljoin(base_url, src)
    return None


# --- Commons and Wikipedia (pure where it can be) ----------------------------------------------

def walk_category(root: str, fetch=None, max_depth: int = 3, skip=None) -> list[str]:
    """Every file title under a category and its subcategories, breadth first, each once."""
    fetch = fetch or fetch_json
    files: list[str] = []
    seen_files: set[str] = set()
    seen_cats = {root}
    queue = [(root, 0)]
    while queue:
        category, depth = queue.pop(0)
        cont = None
        while True:
            params = {"action": "query", "list": "categorymembers", "cmtitle": category,
                      "cmtype": "file|subcat", "cmlimit": 500, "format": "json"}
            if cont:
                params["cmcontinue"] = cont
            data = fetch(COMMONS_API, params)
            for member in (data.get("query") or {}).get("categorymembers") or []:
                title = member.get("title") or ""
                if member.get("ns") == 6 and title not in seen_files:
                    seen_files.add(title)
                    files.append(title)
                elif member.get("ns") == 14 and depth < max_depth and title not in seen_cats:
                    if skip and skip.search(title):
                        continue
                    seen_cats.add(title)
                    queue.append((title, depth + 1))
            cont = (data.get("continue") or {}).get("cmcontinue")
            if not cont:
                break
    return files


def rights_for(source: str, licence: str, non_free: bool = False) -> str:
    """free for Commons and Openverse, fair_use for news and non-free Wikipedia files."""
    if source == "generated":
        return "generated"
    if source == "web" or non_free:
        return "fair_use"
    return "free"


def _title_text(title: str) -> str:
    """'File:John Wayne Gacy 1978 mugshot.jpg' -> 'John Wayne Gacy 1978 mugshot'."""
    name = re.sub(r"^File:", "", title)
    return re.sub(r"\.[A-Za-z0-9]{2,4}$", "", name).replace("_", " ").strip()


def _file_page(host: str, title: str) -> str:
    return f"https://{host}/wiki/" + title.replace(" ", "_")


def _rendition(info: dict) -> tuple[str | None, int, int]:
    """The large rendition to download and its real size."""
    width, height = int(info.get("width") or 0), int(info.get("height") or 0)
    if info.get("thumburl"):
        thumb_w, thumb_h = int(info.get("thumbwidth") or 0), int(info.get("thumbheight") or 0)
        if width and thumb_w > width:  # an unscaled original: the API echoes the asked-for width
            return info["thumburl"], width, height
        return info["thumburl"], thumb_w, thumb_h
    return info.get("url"), width, height


def _is_raster(title: str) -> bool:
    return Path(title.lower()).suffix in RASTER


def commons_item(page: dict, found_by: str, case: bool = False) -> dict | None:
    """A Commons imageinfo page as a library candidate; every licence is accepted."""
    title = page.get("title") or ""
    if not _is_raster(title):
        return None
    info = (page.get("imageinfo") or [{}])[0]
    url, width, height = _rendition(info)
    if not url:
        return None
    meta = info.get("extmetadata") or {}
    licence = plain_text((meta.get("LicenseShortName") or {}).get("value"), fallback="Unknown licence")
    return {
        "source": "commons", "title": _title_text(title), "url": url,
        "source_url": _file_page("commons.wikimedia.org", title),
        "licence": licence,
        "attribution": plain_text((meta.get("Artist") or {}).get("value")),
        "rights": rights_for("commons", licence),
        "width": width, "height": height, "case": case, "found_by": found_by,
        "description": plain_text((meta.get("ImageDescription") or {}).get("value"), fallback="")[:240],
    }


def wikipedia_item(page: dict, found_by: str) -> dict | None:
    """An en.wikipedia imageinfo page: local non-free files are fair use, shared ones are Commons'."""
    title = page.get("title") or ""
    if not _is_raster(title):
        return None
    info = (page.get("imageinfo") or [{}])[0]
    url, width, height = _rendition(info)
    if not url:
        return None
    meta = info.get("extmetadata") or {}
    non_free = str((meta.get("NonFree") or {}).get("value") or "").lower() in ("true", "1", "yes")
    shared = page.get("imagerepository") == "shared"
    if non_free:
        licence = "Copyrighted"
    else:
        licence = plain_text((meta.get("LicenseShortName") or {}).get("value"), fallback="Unknown licence")
    host = "commons.wikimedia.org" if shared else "en.wikipedia.org"
    return {
        "source": "wikipedia", "title": _title_text(title), "url": url,
        "source_url": _file_page(host, title),
        "licence": licence,
        "attribution": plain_text((meta.get("Artist") or {}).get("value")),
        "rights": rights_for("wikipedia", licence, non_free=non_free),
        "width": width, "height": height, "case": False, "found_by": found_by,
    }


def outlet(page_url: str) -> str:
    host = urlparse(page_url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def web_item(seed: dict, image_url: str, title: str = "") -> dict:
    return {
        "source": "web", "title": (title or seed.get("subject") or "").strip()[:200],
        "url": image_url, "source_url": seed["page_url"],
        "licence": "Copyrighted", "attribution": outlet(seed["page_url"]),
        "rights": rights_for("web", "Copyrighted"),
        "width": 0, "height": 0, "case": False, "found_by": f"news:{seed.get('subject', '')}",
    }


def openverse_item(found: dict, found_by: str) -> dict:
    return {
        "source": "openverse", "title": found["title"], "url": found["url"],
        "source_url": found["source_url"], "licence": found["licence"],
        "attribution": found["attribution"], "rights": rights_for("openverse", found["licence"]),
        "width": found["width"], "height": found["height"], "case": False, "found_by": found_by,
    }


_STOP = {"the", "of", "and", "a", "an", "in", "at", "on", "illinois", "iowa", "chicago", "county",
         "west", "east", "north", "south", "avenue", "street", "center", "centre", "file", "jpg"}


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOP]


# People famous enough in their own right that a picture under their name is very likely them.
# Anyone else (victims, detectives, lawyers) shares a name with strangers, so a hit must also
# mention Gacy: a namesake's photograph shown as a victim would be a serious error.
PUBLIC_FIGURES = {"john wayne gacy", "rosalynn carter", "jimmy carter", "tom dart", "robert ressler",
                  "walter jacobson", "dean corll"}


def search_query(entity: dict) -> str:
    """The phrase to search for an entity.

    A place or institution is searched by its own name (the model tends to add "John Wayne Gacy",
    which narrows a jail or a river to nothing); a private person by name plus "Gacy", since only
    a picture tied to the case can be trusted to show them; an object keeps the model's phrase.
    """
    name = re.sub(r"^(?:dr|detective)\.?\s+", "", entity["name"].strip(), flags=re.I)
    kind = entity.get("kind")
    if kind == "person":
        return name if name.lower() in PUBLIC_FIGURES else f"{name} Gacy"
    if kind == "object":
        return entity.get("query") or name
    query = entity.get("query") or name
    query = re.sub(r"\b(?:john wayne gacy|gacy|arrest|\d{4})\b", " ", query, flags=re.I)
    return re.sub(r"\s+", " ", query).strip() or name


def relevant(name: str, title: str, person: bool) -> bool:
    """A search hit counts when its title carries most of the name, and the surname for a person."""
    wanted = _tokens(name)
    have = set(re.findall(r"[a-z0-9]+", unquote(title).lower()))
    if not wanted:
        return False
    if person and name.lower().strip() not in PUBLIC_FIGURES and "gacy" not in have:
        return False
    if person and wanted[-1] not in have:
        return False
    hits = sum(1 for t in wanted if t in have)
    return hits >= max(1, math.ceil(0.6 * len(wanted)))


def next_id(prefix: str, taken: set[str]) -> str:
    numbers = [int(t[1:]) for t in taken if t.startswith(prefix) and t[1:].isdigit()]
    return f"{prefix}{(max(numbers) if numbers else 0) + 1:04d}"


# --- network stages ---------------------------------------------------------------------------

def _log(*parts) -> None:
    with _lock:
        print(*parts, flush=True)


def imageinfo(api: str, titles: list[str]) -> list[dict]:
    """imageinfo for many file titles, 50 a request, in the order given."""
    pages: dict[str, dict] = {}
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = fetch_json(api, {
            "action": "query", "titles": "|".join(batch), "prop": "imageinfo",
            "iiprop": "url|size|extmetadata", "iiurlwidth": THUMB_WIDTH, "format": "json",
            "iiextmetadatafilter": "LicenseShortName|Artist|NonFree|ImageDescription|DateTimeOriginal",
        })
        query = data.get("query") or {}
        renamed = {n["to"]: n["from"] for n in query.get("normalized") or []}
        for page in (query.get("pages") or {}).values():
            pages[renamed.get(page.get("title"), page.get("title"))] = page
            pages[page.get("title")] = page
    return [pages[t] for t in titles if t in pages and pages[t].get("imageinfo")]


def stage_case() -> list[dict]:
    titles = walk_category(CASE_ROOT, max_depth=3)
    _log(f"case: {len(titles)} files under {CASE_ROOT}")
    return [i for i in (commons_item(p, f"case:{CASE_ROOT}", case=True)
                        for p in imageinfo(COMMONS_API, titles)) if i]


def commons_search(query: str, limit: int = 10) -> list[dict]:
    """Commons search accepting every licence (sources.search_commons drops CC BY-SA)."""
    data = fetch_json(COMMONS_API, {
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:bitmap",
        "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo",
        "iiprop": "url|size|extmetadata", "iiurlwidth": THUMB_WIDTH, "format": "json",
        "iiextmetadatafilter": "LicenseShortName|Artist|ImageDescription",
    })
    pages = (data.get("query") or {}).get("pages") or {}
    return sorted(pages.values(), key=lambda p: p.get("index", 0))


def extract_entities(script: str, cache: Path) -> list[dict]:
    """Every named person, place, institution, business and object, from the text model; cached."""
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    from openai import OpenAI
    from pydantic import BaseModel

    class Entity(BaseModel):
        name: str
        kind: str  # person | place | institution | business | object
        note: str
        query: str

    class Entities(BaseModel):
        entities: list[Entity]

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.parse(
        model=settings.text_model,
        input=[
            {"role": "system", "content": (
                "You index a true-crime documentary script for a picture researcher. List every named "
                "person, place, institution, business and object in the script, once each. kind is one "
                "of person, place, institution, business, object. note is one line on what it is in "
                "this story. query is the best short search phrase to find a real photograph of it in "
                "an image archive (full name plus city or state where that disambiguates; for a person "
                "just the full name). Use the spelling the historical record uses.")},
            {"role": "user", "content": script},
        ],
        text_format=Entities,
    )
    found = [e.model_dump() for e in response.output_parsed.entities]
    cache.write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    return found


# Names too broad to search for a picture of this story, or that are not about it at all.
ENTITY_SKIP = {
    "chicago", "illinois", "iowa", "california", "nebraska", "las vegas", "cook county", "youtube",
    "democratic party", "republican party", "fbi", "john wayne", "alexander hamilton",
    "united states secret service", "defense diaries", "greyhound", "motorola", "oldsmobile",
    "plymouth", "kfc",
}


def searchable(entity: dict) -> bool:
    """False for broad or bare names: a lone first name, a state, a body number."""
    name = entity["name"].strip()
    key = re.sub(r"\W+", " ", name.lower()).strip()
    if key in ENTITY_SKIP or re.match(r"body \d+$", key):
        return False
    if entity.get("kind") == "person" and len(name.split()) < 2:
        return False
    return True


def merge_entities(found: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for entity in found:
        merged.setdefault(re.sub(r"\W+", " ", entity["name"].lower()).strip(), entity)
    for name, kind in STARTER_ENTITIES:
        key = re.sub(r"\W+", " ", name.lower()).strip()
        if key not in merged:
            merged[key] = {"name": name, "kind": kind, "note": "starter list", "query": name}
    return [entity for entity in merged.values() if searchable(entity)]


OPENVERSE_GATE_S = 3.5  # anonymous Openverse allows 20 searches a minute and 200 a day


class SearchCache:
    """Search results kept on disk, so a rerun spends none of Openverse's 200-a-day allowance."""

    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.lock = threading.Lock()

    def get(self, key: str, search):
        with self.lock:
            if key in self.data:
                return self.data[key]
        value = search()  # failures raise and are not cached
        with self.lock:
            self.data[key] = value
            write_atomic(self.path, self.data)
        return value


def stage_entities(video_dir: Path, script_path: Path, per_source: int = 3) -> list[dict]:
    entities = merge_entities(extract_entities(script_path.read_text(encoding="utf-8"),
                                               video_dir / "library" / "entities.json"))
    _log(f"entities: {len(entities)}")
    cache = SearchCache(video_dir / "library" / "search_cache.json")
    openverse_gate = HostGate(OPENVERSE_GATE_S)

    def openverse(query: str) -> list[dict]:
        openverse_gate.wait()
        return search_openverse(query, limit=12)

    def one(entity: dict) -> list[dict]:
        name, person = entity["name"], entity.get("kind") == "person"
        query = search_query(entity)
        out: list[dict] = []
        try:
            pages = cache.get(f"commons:{query}", lambda: commons_search(query))
            hits = [commons_item(p, f"entity:{name}") for p in pages]
            hits = [h for h in hits if h and relevant(name, h["title"] + " " + h.get("description", ""), person)]
            out += hits[:per_source]
        except Exception as exc:  # noqa: BLE001
            _log(f"  commons search failed for {name}: {str(exc)[:100]}")
        if person and name.lower() not in PUBLIC_FIGURES:
            _log(f"  {name}: {len(out)}")
            return out  # Openverse titles almost never tie a private person to the case
        try:
            found = cache.get(f"openverse:{query}", lambda: openverse(query))
            hits = [openverse_item(h, f"entity:{name}") for h in found]
            out += [h for h in hits if relevant(name, h["title"], person)][:per_source]
        except Exception as exc:  # noqa: BLE001
            _log(f"  openverse search failed for {name}: {str(exc)[:100]}")
        _log(f"  {name}: {len(out)}")
        return out

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one, entities))
    return [h for hits in results for h in hits]


def pick_period(candidates: list[dict], cache: Path, limit: int = PERIOD_MAX) -> list[dict]:
    """The text model chooses the period pictures most useful under this story; cached."""
    if cache.exists():
        wanted = set(json.loads(cache.read_text(encoding="utf-8")))
        return [c for c in candidates if c["source_url"] in wanted]
    from openai import OpenAI
    from pydantic import BaseModel

    class Picks(BaseModel):
        numbers: list[int]

    listing = "\n".join(f"{n}. {c['title'][:140]} | {c.get('description', '')[:160]}"
                        for n, c in enumerate(candidates))
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.responses.parse(
        model=settings.text_model,
        input=[
            {"role": "system", "content": (
                f"Pick up to {limit} photographs to use as period atmosphere in a documentary about "
                "the John Wayne Gacy murders (Chicago and its north-west suburbs, 1970s, ending with "
                "his December 1978 arrest and 1980 trial). Prefer real 1970s photographs of: streets "
                "and traffic, houses and suburbs, cars, police, courts and public buildings, winter "
                "and snow, offices, bars and taverns, shops, night scenes, ordinary people in public. "
                "Avoid sport, celebrities, parades and festivals, art works, and anything clearly "
                "outside the 1970s or Illinois. Return the numbers.")},
            {"role": "user", "content": listing},
        ],
        text_format=Picks,
    )
    chosen = [candidates[n] for n in response.output_parsed.numbers if 0 <= n < len(candidates)][:limit]
    cache.write_text(json.dumps([c["source_url"] for c in chosen], indent=1), encoding="utf-8")
    return chosen


def stage_period(video_dir: Path) -> list[dict]:
    titles: list[str] = []
    for category, depth in PERIOD_CATEGORIES:
        for title in walk_category(category, max_depth=depth, skip=PERIOD_SKIP):
            if title not in titles:
                titles.append(title)
    pool = [i for i in (commons_item(p, "period") for p in imageinfo(COMMONS_API, titles)) if i]
    pool = [i for i in pool if big_enough(i)]
    _log(f"period: {len(titles)} files, {len(pool)} large enough")
    chosen = pick_period(pool, video_dir / "library" / "period_pick.json")
    for item in chosen:
        item["found_by"] = "period"
    _log(f"period: picked {len(chosen)}")
    return chosen


def _article_images(title: str) -> list[str]:
    data = fetch_json(ENWIKI_API, {"action": "parse", "page": title, "prop": "images",
                                   "redirects": 1, "format": "json"})
    names = (data.get("parse") or {}).get("images") or []
    return ["File:" + n.replace("_", " ") for n in names]


def _existing_articles(titles: list[str], require_link: str | None) -> list[str]:
    out: list[str] = []
    for start in range(0, len(titles), 40):
        params = {"action": "query", "titles": "|".join(titles[start:start + 40]), "redirects": 1,
                  "prop": "pageprops", "ppprop": "disambiguation", "format": "json"}
        if require_link:
            params.update({"prop": "pageprops|links", "pltitles": require_link, "pllimit": "max"})
        pages = ((fetch_json(ENWIKI_API, params).get("query") or {}).get("pages") or {}).values()
        for page in pages:
            if "missing" in page or "disambiguation" in (page.get("pageprops") or {}):
                continue
            if require_link and not page.get("links") and page.get("title") != require_link:
                continue
            out.append(page["title"])
    return out


def stage_wikipedia(video_dir: Path) -> list[dict]:
    articles = _existing_articles(WIKI_ARTICLES, None)
    entity_file = video_dir / "library" / "entities.json"
    if entity_file.exists():
        extra = [e["name"] for e in json.loads(entity_file.read_text(encoding="utf-8"))
                 if e["name"] not in WIKI_ARTICLES]
        articles += [a for a in _existing_articles(extra, "John Wayne Gacy") if a not in articles]
    _log(f"wikipedia: {len(articles)} articles: {articles}")
    out: list[dict] = []
    for article in articles:
        try:
            files = [f for f in _article_images(article) if _is_raster(f) and not WIKI_SKIP.search(f)]
        except Exception as exc:  # noqa: BLE001
            _log(f"  {article}: {exc}")
            continue
        if article != "John Wayne Gacy":
            files = files[:WIKI_PER_ARTICLE]
        items = [wikipedia_item(p, f"wikipedia:{article}") for p in imageinfo(ENWIKI_API, files)]
        items = [i for i in items if i]
        _log(f"  {article}: {len(items)} images")
        out += items
    return out


def _browser_get(url: str, timeout: float = 12.0, referer: str | None = None) -> httpx.Response:
    headers = {"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9",
               "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,image/*,*/*;q=0.8"}
    if referer:
        headers["Referer"] = referer
    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
    response.raise_for_status()
    return response


def _page_title(html: str) -> str:
    match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.I) \
        or re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    return plain_text(match.group(1), fallback="") if match else ""


def stage_web(video_dir: Path, blocked: list[str]) -> list[dict]:
    seeds_path = video_dir / "library" / "web_seeds.json"
    if not seeds_path.exists():
        _log("web: no web_seeds.json")
        return []
    seeds = json.loads(seeds_path.read_text(encoding="utf-8"))
    # A second seed file from the research pass: the same shape, and an entry may name the exact
    # picture (`image_url`) instead of leaving it to the page's headline image.
    extra_path = video_dir / "library" / "web_seeds_extra.json"
    if extra_path.exists():
        seeds += json.loads(extra_path.read_text(encoding="utf-8"))
    cache_path = video_dir / "library" / "web_pages.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    def one(seed: dict) -> dict | None:
        page = seed["page_url"]
        if seed.get("image_url"):
            return web_item(seed, seed["image_url"], seed.get("title", ""))
        if page in cache:
            hit = cache[page]
            return web_item(seed, hit["image"], hit.get("title", "")) if hit.get("image") else None
        try:
            response = _browser_get(page)
            html = response.text
            image = page_image(html, str(response.url))
            title = _page_title(html)
        except Exception as exc:  # noqa: BLE001 - refusals and paywalls are skipped, never bypassed
            blocked.append(f"{outlet(page)}: {str(exc)[:90]}")
            return None
        with _lock:
            cache[page] = {"image": image, "title": title}
        return web_item(seed, image, title) if image else None

    with ThreadPoolExecutor(max_workers=5) as pool:
        found = [i for i in pool.map(one, seeds) if i]
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    _log(f"web: {len(found)} of {len(seeds)} pages gave a picture")
    return found


# --- download, measure, dedupe, write ----------------------------------------------------------

class HostGate:
    """Spaces requests to one host across all workers, and pauses them all on a Retry-After."""

    def __init__(self, spacing: float):
        self.lock = threading.Lock()
        self.spacing = spacing
        self.next_at = 0.0
        self.blocked_until = 0.0

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.monotonic()
                slot = max(now, self.next_at, self.blocked_until)
                self.next_at = slot + self.spacing
            time.sleep(max(0.0, slot - now))
            with self.lock:  # a back_off that landed while this worker slept still holds it
                if time.monotonic() >= self.blocked_until:
                    return

    def back_off(self, seconds: float) -> None:
        with self.lock:
            self.blocked_until = max(self.blocked_until, time.monotonic() + seconds)


WIKIMEDIA_GATE = HostGate(WIKIMEDIA_SPACING_S)


def _fetch_wikimedia(url: str, destination: Path, attempts: int = 5) -> None:
    """upload.wikimedia.org, honouring its Retry-After in full (sources caps it at 30 s)."""
    for attempt in range(attempts):
        WIKIMEDIA_GATE.wait()
        with httpx.Client(timeout=60.0, follow_redirects=True,
                          headers={"User-Agent": sources.USER_AGENT}) as client:
            response = client.get(url)
        if response.status_code in (429, 503) and attempt < attempts - 1:
            after = str(response.headers.get("retry-after") or "")
            pause = float(after) if after.isdigit() else 30.0 * (attempt + 1)
            _log(f"  upload.wikimedia.org said {response.status_code}; pausing {pause:.0f}s")
            WIKIMEDIA_GATE.back_off(pause)
            continue
        response.raise_for_status()
        destination.write_bytes(response.content)
        return


def _fetch_image(item: dict, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        return
    if item["source"] == "web":
        data = _browser_get(item["url"], timeout=20.0, referer=item["source_url"]).content
        destination.write_bytes(data)
        return
    if urlparse(item["url"]).netloc == "upload.wikimedia.org":
        _fetch_wikimedia(item["url"], destination)
        return
    sources.download(item["url"], destination)


def materialise(item: dict, lib_dir: Path) -> dict:
    """Download (once), normalise to JPEG/PNG, and record the real size and hash."""
    stem = lib_dir / item["id"]
    existing = [p for p in lib_dir.glob(item["id"] + ".*") if p.suffix in (".jpg", ".png")]
    if existing and existing[0].stat().st_size and item.get("hash") is not None:
        return item
    raw = stem.with_suffix(".download")
    target = existing[0] if existing else raw
    if not existing:
        _fetch_image(item, raw)
    with Image.open(target) as image:
        image.load()
        fmt = image.format
        if fmt in ("JPEG", "PNG"):
            final = stem.with_suffix(".jpg" if fmt == "JPEG" else ".png")
            if target != final:
                target.replace(final)
        else:
            final = stem.with_suffix(".jpg")
            frame = image.convert("RGB")
            frame.save(final, "JPEG", quality=92)
            target.unlink(missing_ok=True)
        item["width"], item["height"] = image.size
        item["hash"] = average_hash(image)
    item["file"] = f"library/{final.name}"
    return item


HARVEST_FIELDS = ("id", "file", "source", "title", "source_url", "licence", "attribution",
                  "rights", "width", "height", "found_by")
CAPTION_FIELDS = ("caption", "subjects", "places", "era", "colour", "kind", "graphic", "quality")


def library_entry(item: dict, previous: dict | None) -> dict:
    entry = {key: item.get(key) for key in HARVEST_FIELDS}
    for key in CAPTION_FIELDS:
        entry[key] = (previous or {}).get(key)
    if entry["graphic"] is None:
        entry["graphic"] = None
    return entry


def write_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--script", default="data/server_pull/gacy-2026-10-13/gacy_english_transcript.txt")
    parser.add_argument("--stages", default="case,entities,period,wikipedia,web")
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    lib_dir = video_dir / "library"
    lib_dir.mkdir(parents=True, exist_ok=True)
    stages = set(args.stages.split(","))
    blocked: list[str] = []

    # The cache remembers every URL ever seen: its id, its measured size and hash, and why it
    # was dropped, so a rerun neither renumbers nor refetches anything.
    cache_path = lib_dir / "harvest_cache.json"
    cache: dict[str, dict] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

    candidates: list[dict] = []
    if "case" in stages:
        candidates += stage_case()
    if "entities" in stages:
        candidates += stage_entities(video_dir, Path(args.script))
    if "period" in stages:
        candidates += stage_period(video_dir)
    if "wikipedia" in stages:
        candidates += stage_wikipedia(video_dir)
    if "web" in stages:
        candidates += stage_web(video_dir, blocked)
    _log(f"candidates: {len(candidates)}")

    # Merge with cache, give ids, drop by URL and by reported size before downloading.
    taken = {rec["id"] for rec in cache.values()}
    work: list[dict] = []
    seen_urls: set[str] = set()
    for cand in candidates:
        if cand["url"] in seen_urls:
            continue
        seen_urls.add(cand["url"])
        rec = cache.get(cand["url"])
        if rec is None:
            rec = {"id": next_id(PREFIX[cand["source"]], taken)}
            taken.add(rec["id"])
            cache[cand["url"]] = rec
        item = dict(cand, id=rec["id"])
        for key in ("width", "height", "hash", "file"):
            if rec.get(key) is not None:
                item[key] = rec[key]
        if rec.get("failed"):
            continue
        if item["width"] and item["height"] and not big_enough(item):
            rec["dropped"] = "too small"
            continue
        work.append(item)

    def one(item: dict) -> dict | None:
        rec = cache[item["url"]]
        try:
            materialise(item, lib_dir)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            with _lock:
                rec["failed"] = str(exc)[:120]
                blocked.append(f"{urlparse(item['url']).netloc}: {str(exc)[:90]}")
            return None
        with _lock:
            rec.update({k: item[k] for k in ("width", "height", "hash", "file")})
        return item

    _log(f"downloading/measuring {len(work)}")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ready = [i for i in pool.map(one, work) if i]
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    small = [i for i in ready if not big_enough(i)]
    for item in small:
        cache[item["url"]]["dropped"] = "too small"
    ready = [i for i in ready if big_enough(i)]
    kept, dropped = dedupe(ready)
    by_id = {i["id"]: i for i in ready}
    for gone, keeper in dropped:
        cache[by_id[gone]["url"]]["dropped"] = f"duplicate of {keeper}"
    for item in kept:
        cache[item["url"]].pop("dropped", None)
    cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")

    library_path = video_dir / "library.json"
    previous = {}
    if library_path.exists():
        previous = {i["id"]: i for i in json.loads(library_path.read_text(encoding="utf-8")).get("items", [])}
    entries = [library_entry(item, previous.get(item["id"])) for item in kept]
    entries.sort(key=lambda e: ("cwon".index(e["id"][0]) if e["id"][0] in "cwon" else 9, e["id"]))
    with _lock:  # caption.py may be writing too; it re-reads before every write
        write_atomic(library_path, {"items": entries})

    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["source"]] = counts.get(entry["source"], 0) + 1
    _log(f"library: {len(entries)} items {counts}; {len(small)} under {SIZE_FLOOR}px, "
         f"{len(dropped)} duplicates")
    if blocked:
        (lib_dir / "blocked.txt").write_text("\n".join(blocked) + "\n", encoding="utf-8")
        _log(f"{len(blocked)} refusals/failures written to library/blocked.txt")


if __name__ == "__main__":
    main()

"""Find short real video clips to cut between the photographs, and describe their usable seconds.

    .venv/bin/python scripts/video/footage.py search   data/video/gacy
    .venv/bin/python scripts/video/footage.py build    data/video/gacy [--budget-gb 1] [--workers 3]
    .venv/bin/python scripts/video/footage.py generate data/video/gacy "snow past a streetlight" --seconds 4

search: queries the Internet Archive and Wikimedia Commons (and the US National Archives catalogue,
when its API answers without a key) for period footage this story needs, keeps only films whose
licence allows commercial reuse, and ranks them for relevance into clips/candidates.json.

build: for each ranked film, downloads the smallest derivative of at least 480p, finds its scene
cuts with ffmpeg, samples one frame per scene, asks a vision model which scenes show the period
setting cleanly (no titles, no modern elements, no remains, no violence, no close-up faces), and
shapes the kept scenes into 4-15 s segments. The best of those (spread across the film) are checked
again from three stills each, and at most MAX_SEGMENTS_PER_FILM survive. Only the kept seconds are
stored: they are re-encoded,
silent, one small file per kept segment (clips/vNNNN.mp4, with a second of margin either side), and
the full film is deleted. Each item therefore holds one segment; its `in`/`out` are seconds into that
file, and `source_in`/`source_out` say where it came from in the original film.

The catalogue is clips.json ({"items": [...]}, shaped as the design spec's Revision 3 shows). It
is rewritten atomically after every film, and clips/state.json remembers every film already
fetched, judged, rejected or stored, so a rerun never downloads or captions anything twice.

generate: one atmospheric clip from sora-2, never a person's likeness or face.
recheck: the three-still check for clips stored before it existed. borders: crop black borders out
of clips stored before cut_clip did it. drop: remove clips by id after a human look. prune: keep the best --per-film
clips of each film, delete the others, renumber the ids.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sources import USER_AGENT, commercial_licence, fetch_json, names_a_real_person, plain_text  # noqa: E402

from app.config import settings  # noqa: E402

FFMPEG = os.getenv("VIDEO_FFMPEG", "/opt/miniconda3/envs/video/bin/ffmpeg")
FFPROBE = os.getenv("VIDEO_FFPROBE", "/opt/miniconda3/envs/video/bin/ffprobe")
ARCHIVE_SEARCH = "https://archive.org/advancedsearch.php"
ARCHIVE_METADATA = "https://archive.org/metadata/{}"
ARCHIVE_DOWNLOAD = "https://archive.org/download/{}/{}"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
NARA_SEARCH = "https://catalog.archives.gov/api/v2/records/search"

MIN_SEGMENT_S = 4.0
MAX_SEGMENT_S = 15.0
CUT_GAP_S = 0.25  # cuts closer together than this are one cut (a flash or a dissolve)
MARGIN_S = 1.0  # context kept either side of a segment in its own file
INSET_S = 0.2  # trimmed off each edge of a kept segment so no frame of a neighbouring shot leaks in
SCENE_THRESHOLD = 0.3
MAX_SEGMENTS_PER_FILM = 10
VERIFY_POOL = 18  # best segments re-checked from three stills each, before the final choice
MIN_HEIGHT = 480
MAX_SOURCE_BYTES = 220_000_000  # one film; anything bigger is skipped rather than fetched
DEFAULT_BUDGET_BYTES = 1_000_000_000
DISK_RESERVE_BYTES = 600_000_000  # never let a download take the disk below this
FRAMES_PER_REQUEST = 10
LATEST_YEAR = 1995  # footage dated after this cannot show the period

VIDEO_SECONDS = (4, 8, 12)  # what the videos API offers
GENERATED_LICENCE = "Generated"

# Collections whose public-domain marks can be trusted. Anyone can upload to opensource_movies and
# tick "public domain"; studio features turn up there, so it and its like are refused.
TRUSTED_COLLECTIONS = {
    "prelinger", "prelingerhomemovies", "prelinger_library", "ephemera", "avgeeks", "fedflix",
    "usgovfilms", "newsandpublicaffairs", "us_national_archives", "nasa", "classic_tv_commercials",
}
PRELINGER = {"prelinger", "prelingerhomemovies"}
VIDEO_SUFFIXES = (".mp4", ".m4v", ".mov", ".mpeg", ".mpg", ".ogv", ".webm", ".avi", ".mkv")

_FACE = re.compile(
    r"\b(?:face|faces|facial|portrait|portraits|headshot|head shot|mugshot|mug shot|selfie|likeness"
    r"|close[\s\-]?up of (?:a |an |the )?(?:man|woman|person|boy|girl|teen|teenager|child|detective"
    r"|officer|killer|victim|suspect|people))\b",
    re.IGNORECASE,
)


# ---------- licences ----------

def _collections(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value.lower()]
    return [str(c).lower() for c in value]


def trusted_collection(collections) -> bool:
    return any(c in TRUSTED_COLLECTIONS for c in _collections(collections))


def archive_licence(licenseurl: str | None, collections) -> str | None:
    """The licence name for an archive.org item, or None when it does not allow commercial reuse.

    Prelinger films carry no licence URL and are public domain. Elsewhere the licence URL decides,
    and only inside collections whose marks are trustworthy.
    """
    names = _collections(collections)
    if not trusted_collection(names):
        return None
    url = str(licenseurl or "").strip().lower()
    if not url:
        return "Public domain (Prelinger Archives)" if PRELINGER & set(names) else None
    if "creativecommons.org" not in url:
        return None
    path = re.sub(r"^https?://(?:www\.)?creativecommons\.org/", "", url).strip("/")
    version = re.search(r"/(\d+(?:\.\d+)?)", "/" + path)
    ver = f" {version.group(1)}" if version else ""
    if path.startswith("licenses/publicdomain"):
        name = "Public domain"
    elif path.startswith("publicdomain/mark"):
        name = f"Public Domain Mark{ver}"
    elif path.startswith("publicdomain/zero"):
        name = f"CC0{ver}"
    elif re.match(r"licenses/by/", path + "/"):
        name = f"CC BY{ver}"
    else:
        return None  # by-sa, by-nc, by-nd and the rest
    return name if commercial_licence(name) else None


def commons_licence(name: str | None) -> str | None:
    """A Commons LicenseShortName that allows commercial reuse, else None."""
    text = plain_text(name, fallback="")
    return text if text and commercial_licence(text) else None


def year_of(text: str | None) -> int | None:
    match = re.search(r"\b(18[89]\d|19\d\d|20\d\d)\b", str(text or ""))
    return int(match.group(1)) if match else None


# ---------- choosing what to download ----------

def _int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def pick_archive_file(files: list[dict], min_height: int = MIN_HEIGHT,
                      max_bytes: int = MAX_SOURCE_BYTES) -> dict | None:
    """The smallest video file of at least min_height within max_bytes; else the best SD one."""
    videos = [
        f for f in files
        if str(f.get("name", "")).lower().endswith(VIDEO_SUFFIXES)
        and _int(f.get("height")) > 0 and 0 < _int(f.get("size")) <= max_bytes
    ]
    tall = [f for f in videos if _int(f.get("height")) >= min_height]
    if tall:
        return min(tall, key=lambda f: _int(f.get("size")))
    sd = [f for f in videos if _int(f.get("width")) >= 640]  # 640x360 widescreen SD
    if sd:
        return max(sd, key=lambda f: (_int(f.get("height")), -_int(f.get("size"))))
    return None


def pick_commons_derivative(derivatives: list[dict], min_height: int = MIN_HEIGHT) -> dict | None:
    """The smallest transcode of at least min_height; the original only when nothing else is."""
    sized = [d for d in derivatives if _int(d.get("height")) > 0]
    tall = [d for d in sized if _int(d.get("height")) >= min_height]
    transcodes = [d for d in tall if d.get("transcodekey")]
    pool = transcodes or tall
    if not pool:
        return None
    return min(pool, key=lambda d: (_int(d.get("height")), _int(d.get("bandwidth"))))


# ---------- scenes and segments ----------

_PTS = re.compile(r"pts_time:\s*([0-9.]+)")
_SCD = re.compile(r"lavfi\.scd\.time:\s*([0-9.]+)")


def parse_scene_cuts(log: str) -> list[float]:
    """Cut times from ffmpeg's showinfo (after select=gt(scene,..)) or scdet output, sorted, unique."""
    found = [float(m) for m in _PTS.findall(log or "")] + [float(m) for m in _SCD.findall(log or "")]
    return sorted(set(found))


def scenes_from_cuts(cuts: list[float], duration: float, min_gap: float = CUT_GAP_S) -> list[tuple[float, float]]:
    """The film as consecutive (start, end) scenes, one per cut; cuts too close together are one."""
    points = [0.0]
    for cut in sorted(cuts):
        if min_gap <= cut <= duration - min_gap and cut - points[-1] >= min_gap:
            points.append(round(cut, 3))
    points.append(round(duration, 3))
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def split_long(spans: list[tuple[float, float]], max_s: float = MAX_SEGMENT_S) -> list[tuple[float, float]]:
    """Scenes longer than max_s cut into equal pieces of at most max_s, so each gets its own look."""
    out: list[tuple[float, float]] = []
    for start, end in spans:
        length = end - start
        parts = max(1, math.ceil(length / max_s - 1e-9))
        step = length / parts
        for n in range(parts):
            a = start + n * step
            b = end if n == parts - 1 else start + (n + 1) * step
            out.append((round(a, 3), round(b, 3)))
    return out


def _combine(group: list[dict]) -> dict:
    captions: list[str] = []
    for p in group:
        text = str(p.get("caption") or "").strip()
        if text and text not in captions:
            captions.append(text)
    eras = Counter(str(p.get("era") or "") for p in group)
    era = max(eras, key=lambda e: (eras[e], -[str(p.get("era") or "") for p in group].index(e)))
    merged = {
        "in": group[0]["in"], "out": group[-1]["out"], "caption": " ".join(captions), "era": era,
        "quality": min(int(p.get("quality") or 1) for p in group),
        "graphic": any(bool(p.get("graphic")) for p in group),
    }
    return merged


def merge_short(pieces: list[dict], min_s: float = MIN_SEGMENT_S, max_s: float = MAX_SEGMENT_S) -> list[dict]:
    """Kept pieces shaped into segments of min_s to max_s seconds.

    Adjacent pieces (one ends where the next starts) are joined while either is shorter than min_s
    and the total stays within max_s. A short piece with no such neighbour is dropped.
    """
    segments: list[dict] = []
    group: list[dict] = []

    def close() -> None:
        if group and group[-1]["out"] - group[0]["in"] >= min_s - 1e-9:
            segments.append(_combine(group))

    for p in sorted(pieces, key=lambda p: p["in"]):
        if group:
            length = group[-1]["out"] - group[0]["in"]
            own = p["out"] - p["in"]
            contiguous = abs(p["in"] - group[-1]["out"]) < 1e-3
            if contiguous and (length < min_s or own < min_s) and length + own <= max_s + 1e-9:
                group.append(p)
                continue
            close()
        group = [p]
    close()
    return segments


def best_segments(segments: list[dict], limit: int) -> list[dict]:
    """At most limit segments, best quality first but spread across the film; returned in film order.

    A long film yields dozens of usable scenes that look alike; a few of the best, far apart, serve
    the edit better and keep the library small.
    """
    if len(segments) <= limit:
        return sorted(segments, key=lambda s: s["in"])
    span = max(s["out"] for s in segments) - min(s["in"] for s in segments)
    gap = span / (limit * 2)
    ranked = sorted(segments, key=lambda s: (-int(s.get("quality") or 0), s["in"]))
    chosen: list[dict] = []
    for s in ranked:
        if len(chosen) < limit and all(abs(s["in"] - c["in"]) >= gap for c in chosen):
            chosen.append(s)
    for s in ranked:  # a short film: fill up with what is left, best first
        if len(chosen) < limit and s not in chosen:
            chosen.append(s)
    return sorted(chosen, key=lambda s: s["in"])


def acceptable(j: dict) -> bool:
    """A judged scene that may be used: clean period footage, nothing unsuitable in frame."""
    return bool(
        j.get("usable")
        and str(j.get("caption") or "").strip()
        and not j.get("graphic")
        and not j.get("violence")
        and not j.get("on_screen_text")
        and not j.get("modern_elements")
        and not j.get("close_up_face")
        and int(j.get("quality") or 0) >= 2
    )


# ---------- the catalogue and its state ----------

def write_atomic(path: Path, data) -> None:
    """Write JSON beside the target and swap it in, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    try:
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _read(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", key)[:120]


class Library:
    """clips.json plus clips/state.json, re-read on every change so parallel work never loses an edit."""

    def __init__(self, root: Path, budget_bytes: int = DEFAULT_BUDGET_BYTES):
        self.root = Path(root)
        self.clips_dir = self.root / "clips"
        self.catalogue_path = self.root / "clips.json"
        self.state_path = self.clips_dir / "state.json"
        self.budget_bytes = budget_bytes
        self.lock = threading.RLock()

    # state
    def _state(self) -> dict:
        state = _read(self.state_path, {})
        state.setdefault("ids", {})
        state.setdefault("sources", {})
        state.setdefault("downloaded_bytes", 0)
        return state

    def source_state(self, key: str) -> dict | None:
        return self._state()["sources"].get(key)

    def mark(self, key: str, **fields) -> None:
        with self.lock:
            state = self._state()
            entry = state["sources"].setdefault(key, {})
            entry.update(fields)
            write_atomic(self.state_path, state)

    def reserve_id(self, key: str) -> str:
        with self.lock:
            state = self._state()
            if key in state["ids"]:
                return state["ids"][key]
            used = set(state["ids"].values()) | {i.get("id") for i in self.items()}
            n = 1 + max([int(i[1:]) for i in used if re.fullmatch(r"v\d{4}", str(i))] or [0])
            state["ids"][key] = f"v{n:04d}"
            write_atomic(self.state_path, state)
            return state["ids"][key]

    def downloaded_bytes(self) -> int:
        return int(self._state()["downloaded_bytes"])

    def can_download(self, size: int) -> bool:
        return self.downloaded_bytes() + int(size) <= self.budget_bytes

    def count_download(self, size: int) -> None:
        with self.lock:
            state = self._state()
            state["downloaded_bytes"] = int(state["downloaded_bytes"]) + int(size)
            write_atomic(self.state_path, state)

    def work_dir(self, key: str) -> Path:
        return self.clips_dir / "work" / _slug(key)

    # catalogue
    def items(self) -> list[dict]:
        return _read(self.catalogue_path, {"items": []}).get("items", [])

    def add_item(self, item: dict) -> None:
        with self.lock:
            data = _read(self.catalogue_path, {"items": []})
            items = [i for i in data.get("items", []) if i.get("id") != item["id"]]
            items.append(item)
            items.sort(key=lambda i: i["id"])
            data["items"] = items
            write_atomic(self.catalogue_path, data)


# ---------- one film, end to end ----------

DONE = {"stored", "rejected"}


def process_source(candidate: dict, lib: Library, steps,
                   log: Callable[[str], None] = lambda s: None) -> list[dict] | None:
    """Fetch, judge and store one film; returns the new catalogue items, or None when nothing new was stored.

    Each kept segment is cut, with MARGIN_S of context either side, to its own small file and gets its
    own item and id; then the full film is deleted. Every expensive step leaves a mark, so a rerun
    picks up where a crash left off: the download is kept until every segment is stored, the
    judgements are saved before anything else happens, and a segment already in clips.json is not
    cut again.
    """
    key = candidate["key"]
    previous = lib.source_state(key) or {}
    if previous.get("status") in DONE:
        return None
    work = lib.work_dir(key)
    judged_path = work / "judged.json"
    source = work / f"source{candidate.get('suffix') or '.mp4'}"
    judged = _read(judged_path, None)

    if judged is None:
        if not (source.exists() and source.stat().st_size > 0):
            size = int(candidate.get("size") or 0)
            with lib.lock:
                if not lib.can_download(size):
                    log(f"skip {key}: over the download budget")
                    return None
                lib.count_download(size)
            work.mkdir(parents=True, exist_ok=True)
            lib.mark(key, status="downloading", title=candidate.get("title"))
            part = source.with_name(source.name + ".part")
            part.unlink(missing_ok=True)  # a download cut short earlier is never resumed blind
            try:
                got = steps.download(candidate, part)
            except Exception:
                part.unlink(missing_ok=True)
                lib.count_download(-size)  # nothing kept, so nothing counted against the budget
                raise
            os.replace(part, source)
            lib.mark(key, status="downloaded", bytes=int(got))
        info = steps.probe(source)
        cuts = steps.detect(source, info["duration_s"])
        spans = split_long(scenes_from_cuts(cuts, info["duration_s"]))
        frames_dir = work / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frames = steps.frames(source, spans, frames_dir)
        verdicts = steps.judge(candidate, frames, spans)
        judged = {"info": info, "cuts": cuts,
                  "pieces": [{"in": a, "out": b, **v} for (a, b), v in zip(spans, verdicts)]}
        write_atomic(judged_path, judged)
        shutil.rmtree(frames_dir, ignore_errors=True)
        lib.mark(key, status="judged", scenes=len(spans))

    kept = [
        {k: p[k] for k in ("in", "out", "caption", "era", "quality", "graphic")}
        for p in judged["pieces"] if acceptable(p)
    ]
    segments = []
    for seg in merge_short(kept):
        seg["in"], seg["out"] = round(seg["in"] + INSET_S, 3), round(seg["out"] - INSET_S, 3)
        if seg["out"] - seg["in"] >= MIN_SEGMENT_S - 1e-9:
            segments.append(seg)
    if segments and hasattr(steps, "verify"):
        verified = judged.get("verified")
        if verified is None:
            pool = best_segments(segments, VERIFY_POOL)
            verdicts = steps.verify(candidate, source, pool, work / "strips")
            verified = [{**seg, "caption": v.get("caption") or seg["caption"],
                         "quality": min(int(seg["quality"]), int(v.get("quality") or 0)), "era": v.get("era") or seg["era"]}
                        for seg, v in zip(pool, verdicts) if acceptable(v)]
            judged["verified"] = verified
            write_atomic(judged_path, judged)
            shutil.rmtree(work / "strips", ignore_errors=True)
            lib.mark(key, status="verified", verified=f"{len(verified)}/{len(pool)}")
        segments = verified
    segments = best_segments(segments, MAX_SEGMENTS_PER_FILM)
    if not segments:
        lib.mark(key, status="rejected", reason="no usable scenes")
        shutil.rmtree(work, ignore_errors=True)
        log(f"rejected {key}: nothing usable in {len(judged['pieces'])} scenes")
        return None

    duration = float(judged["info"]["duration_s"])
    stored = {item["id"] for item in lib.items()}
    lib.mark(key, status="cutting")
    new_items: list[dict] = []
    ids: list[str] = []
    for n, seg in enumerate(segments):
        clip_id = lib.reserve_id(f"{key}#{n}")
        ids.append(clip_id)
        if clip_id in stored:
            continue
        start = round(max(0.0, seg["in"] - MARGIN_S), 3)
        end = round(min(duration, seg["out"] + MARGIN_S), 3)
        destination = lib.clips_dir / f"{clip_id}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        info = steps.cut(source, start, end, destination)
        length = float(info["duration_s"])
        item = {
            "id": clip_id, "file": f"clips/{clip_id}.mp4", "source": candidate["source"],
            "title": candidate.get("title") or "", "source_url": candidate.get("source_url") or "",
            "licence": candidate.get("licence") or "", "attribution": candidate.get("attribution") or "",
            "rights": "free",
            "duration_s": round(length, 3), "width": int(info["width"]),
            "height": int(info["height"]), "fps": round(float(info["fps"]), 3),
            "segments": [{
                "in": round(seg["in"] - start, 3), "out": round(min(seg["out"] - start, length), 3),
                "caption": seg["caption"], "era": seg["era"], "quality": int(seg["quality"]),
                "graphic": bool(seg["graphic"]), "source_in": seg["in"], "source_out": seg["out"],
                **({"checked": True} if "verified" in judged else {}),
            }],
        }
        lib.add_item(item)
        new_items.append(item)
    lib.mark(key, status="stored", clip_ids=ids, segments=len(ids), source_duration_s=duration)
    shutil.rmtree(work, ignore_errors=True)
    log(f"stored {ids[0]}..{ids[-1]} <- {key}: {len(ids)} segments")
    return new_items


# ---------- the real steps: HTTP, ffmpeg, the vision model ----------

def _run(args: list[str], timeout: float = 1800) -> subprocess.CompletedProcess:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{Path(args[0]).name} failed: {result.stderr.strip()[-400:]}")
    return result


def probe_video(path: Path) -> dict:
    out = _run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height,avg_frame_rate,r_frame_rate,duration:format=duration", "-of", "json",
                str(path)]).stdout
    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]

    def rate(text: str) -> float:
        num, _, den = str(text or "0/1").partition("/")
        return float(num) / float(den or 1) if float(den or 1) else 0.0

    fps = rate(stream.get("avg_frame_rate")) or rate(stream.get("r_frame_rate"))
    duration = float((data.get("format") or {}).get("duration") or stream.get("duration") or 0)
    return {"duration_s": round(duration, 3), "width": int(stream.get("width") or 0),
            "height": int(stream.get("height") or 0), "fps": round(fps, 3)}


class RealSteps:
    def __init__(self, client=None, model: str | None = None):
        self.client = client
        self.model = model or settings.text_model

    def download(self, candidate: dict, destination: Path) -> int:
        for attempt in range(4):
            try:
                return self._download_once(candidate, destination)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429 and shutil.which("curl"):
                    # upload.wikimedia.org answers python clients with 429 while curl, with the same
                    # User-Agent, is served at once; the edge fingerprints the client, not the rate.
                    return self._download_with_curl(candidate, destination)
                if exc.response.status_code not in (500, 502, 503, 504) or attempt == 3:
                    raise
                after = str(exc.response.headers.get("retry-after") or "")
                time.sleep(min(float(after), 120.0) if after.isdigit() else 30.0 * (attempt + 1))
        raise RuntimeError("unreachable")

    def _download_with_curl(self, candidate: dict, destination: Path) -> int:
        _run(["curl", "-fsSL", "--retry", "3", "--max-filesize", str(int(MAX_SOURCE_BYTES * 1.2)),
              "-A", USER_AGENT, "-o", str(destination), candidate["download_url"]], timeout=3600)
        time.sleep(1.0)
        return destination.stat().st_size

    def _download_once(self, candidate: dict, destination: Path) -> int:
        total = 0
        with httpx.Client(timeout=httpx.Timeout(60.0, read=120.0), follow_redirects=True,
                          headers={"User-Agent": USER_AGENT}) as client:
            with client.stream("GET", candidate["download_url"]) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
                        total += len(chunk)
                        if total > MAX_SOURCE_BYTES * 1.2:
                            raise RuntimeError("download larger than its listing")
        time.sleep(1.0)  # be polite to the archive between films
        return total

    def probe(self, path: Path) -> dict:
        return probe_video(path)

    def detect(self, path: Path, duration: float) -> list[float]:
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-an", "-sn",
             "-vf", f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"scene detection failed: {result.stderr.strip()[-300:]}")
        return parse_scene_cuts(result.stderr)

    def frames(self, path: Path, spans: list[tuple[float, float]], folder: Path) -> list[Path]:
        def one(n: int) -> Path:
            a, b = spans[n]
            target = folder / f"f{n:04d}.jpg"
            if not target.exists():
                _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{(a + b) / 2:.3f}", "-i", str(path),
                      "-frames:v", "1", "-vf", "scale=512:-2", "-q:v", "4", "-y", str(target)], timeout=120)
            return target

        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(one, range(len(spans))))

    def judge(self, candidate: dict, frames: list[Path], spans: list[tuple[float, float]]) -> list[dict]:
        verdicts: list[dict] = []
        for start in range(0, len(frames), FRAMES_PER_REQUEST):
            batch = list(range(start, min(start + FRAMES_PER_REQUEST, len(frames))))
            verdicts.extend(judge_batch(self.client, self.model, candidate, [(n, frames[n], spans[n]) for n in batch]))
        return verdicts

    def cut(self, source: Path, start: float, end: float, destination: Path) -> dict:
        return cut_clip(source, start, end, destination)

    def verify(self, candidate: dict, source: Path, segments: list[dict], folder: Path) -> list[dict]:
        """Judge each chosen segment again from three stills (start, middle, end) side by side.

        Scene detection misses cuts in grey, grainy film, so one frame per scene can hide a title
        card or a close-up in the same stretch; the strip catches most of those.
        """
        folder.mkdir(parents=True, exist_ok=True)
        strips = [segment_strip(source, s["in"], s["out"], folder / f"s{n:04d}.jpg") for n, s in enumerate(segments)]
        verdicts: list[dict] = []
        for start in range(0, len(strips), FRAMES_PER_REQUEST):
            batch = list(range(start, min(start + FRAMES_PER_REQUEST, len(strips))))
            verdicts.extend(judge_batch(
                self.client, self.model, candidate,
                [(n, strips[n], (segments[n]["in"], segments[n]["out"])) for n in batch],
                detail="high", label="segment", note=VERIFY_NOTE))
        return verdicts


JUDGE_SYSTEM = (
    "You pick b-roll for a true-crime documentary set in 1960s-1990s America: Chicago and its suburbs "
    "(Des Plaines, Norwood Park) mostly in winter, 1960s Waterloo, Iowa, a police investigation, a house "
    "search, a 1980 trial, Illinois prisons, a 1994 execution. Wanted: city and suburban streets, night "
    "roads, snow and winter driving, police cars, officers at work, police stations, offices, courtrooms, "
    "prisons, houses, construction work, shops, small towns, crowds seen at a distance. You are shown "
    "numbered frames, one per scene of an old film. For each frame decide:\n"
    "usable: true only if it is clean live-action footage that could stand in for that period setting "
    "(not animation, diagrams, maps, charts, a blank or black frame, a test pattern, or an unreadable blur);\n"
    "caption: one factual sentence saying what is visible, with period clues (e.g. 'A police car drives "
    "along a snowy residential street, 1970s.'). Name a city only if the film's title or description "
    "establishes it; never guess who a person is;\n"
    "era: the decade the scene appears to be from, e.g. '1960s';\n"
    "quality: 1-5, how good it would look full-screen in a 1080p documentary (focus, exposure, damage);\n"
    "graphic: human remains, a body or a corpse visible; violence: a fight, a weapon in use, injury, blood, "
    "a crash with victims; on_screen_text: any title, credit, caption, subtitle, logo, watermark, timecode "
    "or film leader numbers over the picture (ordinary signs that are part of the scene do not count); "
    "modern_elements: anything that could not be from before 1995; close_up_face: one person's face fills "
    "a large part of the frame and could be recognised."
)


def judge_batch(client, model: str, candidate: dict, frames: list[tuple[int, Path, tuple[float, float]]],
                detail: str = "low", note: str = "", label: str = "scene") -> list[dict]:
    """One vision request for up to FRAMES_PER_REQUEST scene frames; verdicts in the frames' order."""
    from pydantic import BaseModel, Field

    class Verdict(BaseModel):
        frame: int
        usable: bool
        caption: str
        era: str
        quality: int = Field(ge=1, le=5)
        graphic: bool
        violence: bool
        on_screen_text: bool
        modern_elements: bool
        close_up_face: bool

    class Verdicts(BaseModel):
        frames: list[Verdict]

    context = (f"Film: {candidate.get('title')}\nYear: {candidate.get('year') or 'unknown'}\n"
               f"Description: {str(candidate.get('description') or '')[:700]}")
    content: list[dict] = [{"type": "input_text", "text": context + (f"\n\n{note}" if note else "")}]
    for n, path, (a, b) in frames:
        content.append({"type": "input_text", "text": f"Frame {n} ({label} {a:.1f}-{b:.1f} s):"})
        content.append({"type": "input_image", "image_url": "data:image/jpeg;base64,"
                        + base64.b64encode(Path(path).read_bytes()).decode(), "detail": detail})
    wanted = [n for n, _, _ in frames]
    for attempt in range(3):
        try:
            response = client.responses.parse(
                model=model,
                input=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": content}],
                text_format=Verdicts,
            )
            parsed = response.output_parsed
            by_frame = {v.frame: v.model_dump() for v in (parsed.frames if parsed else [])}
            missing = [n for n in wanted if n not in by_frame]
            if missing and attempt < 2:
                continue
            return [by_frame.get(n) or {"usable": False, "caption": "", "quality": 1} for n in wanted]
        except Exception:  # noqa: BLE001 - retried, then raised
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("unreachable")


VERIFY_NOTE = (
    "Each image below is one candidate segment shown as three stills side by side: its start, middle "
    "and end. Judge the segment as a whole: it is usable only if all three stills pass every rule, and "
    "on_screen_text, close_up_face, graphic, violence or modern_elements is true if any still shows it. "
    "The caption describes what the whole segment shows."
)


def segment_strip(source: Path, start: float, end: float, destination: Path, height: int = 300) -> Path:
    """Three stills from a segment (start, middle, end) side by side in one JPEG."""
    from PIL import Image

    stills = []
    for k, at in enumerate((start + 0.3, (start + end) / 2, max(start, end - 0.3))):
        still = destination.with_name(f"{destination.stem}_{k}.jpg")
        _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(source), "-frames:v", "1",
              "-vf", f"scale=-2:{height},setsar=1", "-q:v", "3", "-y", str(still)], timeout=120)
        stills.append(still)
    images = [Image.open(s).convert("RGB") for s in stills]
    strip = Image.new("RGB", (sum(i.width for i in images) + 8 * (len(images) - 1), height), "white")
    x = 0
    for image in images:
        strip.paste(image, (x, 0))
        x += image.width + 8
    strip.save(destination, "JPEG", quality=88)
    for s in stills:
        s.unlink(missing_ok=True)
    return destination


_CROP = re.compile(r"crop=(\d+):(\d+):(\d+):(\d+)")
CROP_LIMIT = 40  # luma; telecine bars are rarely pure black
MIN_BORDER_PX = 4


def parse_crop(log: str) -> tuple[int, int, int, int] | None:
    """The last crop=w:h:x:y estimate in ffmpeg cropdetect output (with reset=0, the whole clip's)."""
    found = _CROP.findall(log or "")
    return tuple(int(v) for v in found[-1]) if found else None


def border_crop(width: int, height: int, crop) -> tuple[int, int, int, int] | None:
    """The crop to apply, or None when the borders are noise or the 'border' is really a dark scene."""
    if not crop:
        return None
    w, h, x, y = crop
    left, right, top, bottom = x, width - w - x, y, height - h - y
    if w * h < 0.5 * width * height:
        # Too much to be borders alone: a dark part of the picture joined in. Keep only an axis whose
        # two borders match, as telecine pillarbox or letterbox bars do.
        bars_x = left > MIN_BORDER_PX and right > MIN_BORDER_PX and abs(left - right) <= 6
        bars_y = top > MIN_BORDER_PX and bottom > MIN_BORDER_PX and abs(top - bottom) <= 6
        w, x = (w, x) if bars_x else (width, 0)
        h, y = (h, y) if bars_y else (height, 0)
        if not (bars_x or bars_y) or w * h < 0.5 * width * height:
            return None
        left, right, top, bottom = x, width - w - x, y, height - h - y
    if max(left, right, top, bottom) <= MIN_BORDER_PX:
        return None
    return (w - w % 2, h - h % 2, x, y)


def detect_crop(source: Path, start: float, end: float) -> tuple[int, int, int, int] | None:
    info = probe_video(source)
    result = subprocess.run(
        [FFMPEG, "-hide_banner", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{end - start:.3f}", "-an",
         "-vf", f"cropdetect=limit={CROP_LIMIT}:round=2:reset=0", "-f", "null", "-"],
        capture_output=True, text=True, timeout=600)
    return border_crop(info["width"], info["height"], parse_crop(result.stderr))


def cut_clip(source: Path, start: float, end: float, destination: Path) -> dict:
    """Re-encode source[start:end], silent, H.264, square pixels, black borders cropped; its probe."""
    tmp = destination.with_name(destination.stem + ".tmp.mp4")
    crop = detect_crop(source, start, end)
    cropping = "crop={}:{}:{}:{},".format(*crop) if crop else ""
    try:
        _run([FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.3f}", "-i", str(source),
              "-t", f"{end - start:.3f}", "-map", "0:v:0", "-an", "-sn", "-dn",
              "-vf", f"{cropping}scale=trunc(iw*sar/2)*2:trunc(ih/2)*2,setsar=1",
              "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", "-y", str(tmp)])
        info = probe_video(tmp)
        if info["duration_s"] <= 0:
            raise RuntimeError("cut produced an empty file")
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return info


# ---------- searching ----------

ARCHIVE_QUERIES = [
    "chicago", "chicago street", "chicago traffic", "chicago night", "snow", "winter", "winter driving",
    "blizzard", "snowstorm", "police", "police car", "police patrol", "police station", "detective",
    "crime laboratory", "fingerprint", "court", "courtroom", "trial", "judge jury", "prison", "jail",
    "penitentiary", "inmate", "suburb", "suburban", "housing", "construction", "contractor", "night driving",
    "city at night", "street scenes", "downtown", "traffic", "iowa", "small town", "main street",
    "drugstore", "pharmacy", "newspaper", "television news", "missing", "runaway", "highway", "illinois",
]

COMMONS_QUERIES = [
    "Chicago 1970s", "Chicago 1960s", "Chicago scenes", "National Archives Chicago", "National Archives police",
    "National Archives prison", "National Archives snow", "National Archives winter", "National Archives court",
    "1970s police", "1970s street", "1970s traffic", "prison 1970s", "Iowa 1960s", "snow 1970s",
    "police car 1970s", "courthouse", "penitentiary", "Illinois",
]

ARCHIVE_FILTER = (
    "mediatype:movies AND year:[1955 TO 1994] AND collection:("
    + " OR ".join(sorted(TRUSTED_COLLECTIONS)) + ")"
)


def search_archive(query: str, rows: int = 40) -> list[dict]:
    data = fetch_json(ARCHIVE_SEARCH, {
        "q": f"({query}) AND {ARCHIVE_FILTER}",
        "fl[]": ["identifier", "title", "licenseurl", "year", "collection", "description", "downloads"],
        "rows": rows, "sort[]": "downloads desc", "output": "json",
    })
    return (data.get("response") or {}).get("docs") or []


def archive_candidate(identifier: str) -> dict | None:
    """A downloadable, licence-clean candidate for one archive.org item, or None."""
    data = fetch_json(ARCHIVE_METADATA.format(identifier), {})
    meta = data.get("metadata") or {}
    licence = archive_licence(meta.get("licenseurl"), meta.get("collection"))
    if not licence:
        return None
    choice = pick_archive_file(data.get("files") or [])
    if not choice:
        return None
    creator = meta.get("creator") or meta.get("sponsor") or ""
    if isinstance(creator, list):
        creator = "; ".join(creator)
    collections = _collections(meta.get("collection"))
    if PRELINGER & set(collections):
        attribution = "Prelinger Archives" + (f" ({plain_text(creator, '')})" if creator else "")
    else:
        attribution = plain_text(creator, fallback="Internet Archive")
    title = plain_text(meta.get("title"), fallback=identifier)
    return {
        "key": f"archive.org:{identifier}", "source": "archive.org", "identifier": identifier,
        "title": title, "year": year_of(meta.get("year") or meta.get("date")),
        "description": plain_text(meta.get("description"), fallback="")[:1200],
        "collections": collections, "licence": licence, "attribution": attribution,
        "source_url": f"https://archive.org/details/{identifier}",
        "download_url": ARCHIVE_DOWNLOAD.format(identifier, choice["name"]),
        "size": _int(choice.get("size")), "width": _int(choice.get("width")), "height": _int(choice.get("height")),
        "length_s": float(choice.get("length") or 0) if re.fullmatch(r"[0-9.]+", str(choice.get("length") or "")) else None,
        "suffix": Path(choice["name"]).suffix.lower() or ".mp4",
    }


def search_commons_video(query: str, limit: int = 20) -> list[dict]:
    data = fetch_json(COMMONS_API, {
        "action": "query", "generator": "search", "gsrsearch": f"{query} filetype:video", "gsrnamespace": 6,
        "gsrlimit": limit, "prop": "videoinfo", "viprop": "url|size|extmetadata|derivatives", "format": "json",
    })
    out = []
    for page in sorted(((data.get("query") or {}).get("pages") or {}).values(), key=lambda p: p.get("index", 0)):
        info = (page.get("videoinfo") or [{}])[0]
        meta = info.get("extmetadata") or {}
        licence = commons_licence((meta.get("LicenseShortName") or {}).get("value"))
        if not licence:
            continue
        dated = plain_text((meta.get("DateTimeOriginal") or {}).get("value"), fallback="")
        year = year_of(dated) or year_of(page.get("title"))
        if not year or year > LATEST_YEAR:
            continue
        choice = pick_commons_derivative(info.get("derivatives") or [])
        if not choice:
            continue
        duration = float(info.get("duration") or 0)
        bandwidth = _int(choice.get("bandwidth"))
        size = int(duration * bandwidth / 8) if bandwidth and duration else _int(info.get("size"))
        if size > MAX_SOURCE_BYTES:
            continue
        title = page.get("title") or ""
        out.append({
            "key": f"commons:{title}", "source": "commons", "identifier": title, "title": title.removeprefix("File:"),
            "year": year, "description": plain_text((meta.get("ImageDescription") or {}).get("value"), "")[:1200],
            "licence": licence, "attribution": plain_text((meta.get("Artist") or {}).get("value")),
            "source_url": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
            "download_url": choice["src"], "size": size, "width": _int(choice.get("width")),
            "height": _int(choice.get("height")), "length_s": duration or None,
            "suffix": Path(httpx.URL(choice["src"]).path).suffix.lower() or ".webm",
        })
    return out


def nara_available() -> tuple[bool, str]:
    """Whether the National Archives catalogue API answers JSON without a key."""
    try:
        with httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
            response = client.get(NARA_SEARCH, params={"q": "chicago police", "rows": 1})
        if "json" not in response.headers.get("content-type", ""):
            return False, f"HTTP {response.status_code}, {response.headers.get('content-type')} (needs an API key)"
        response.json()
        return True, "ok"
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:160]


RANK_SYSTEM = (
    "Rate public-domain films as sources of b-roll for a true-crime documentary about the John Wayne Gacy "
    "case: 1970s Chicago and its north-west suburbs (Des Plaines, Norwood Park) in winter; 1960s Waterloo, "
    "Iowa; police investigation, surveillance and stations; a house search; construction work; a 1980 "
    "courtroom trial; Illinois prisons; an execution in 1994. score 0-5: 5 = likely to contain clean "
    "live-action shots of exactly those settings in the right era; 3 = some useful generic period street, "
    "car, police, court, prison or winter shots; 0-1 = cartoons, lectures, studio interiors, wrong country, "
    "pre-1955, sport, music, or nothing relevant. Judge from title, year and description only."
)


def rank_candidates(client, model: str, candidates: list[dict]) -> list[dict]:
    from pydantic import BaseModel, Field

    class Score(BaseModel):
        key: str
        score: int = Field(ge=0, le=5)
        reason: str

    class Scores(BaseModel):
        scores: list[Score]

    for start in range(0, len(candidates), 25):
        batch = candidates[start:start + 25]
        text = "\n\n".join(
            f"key: {c['key']}\ntitle: {c['title']}\nyear: {c.get('year')}\nlength: {c.get('length_s')}\n"
            f"description: {str(c.get('description') or '')[:400]}" for c in batch)
        response = client.responses.parse(
            model=model, input=[{"role": "system", "content": RANK_SYSTEM}, {"role": "user", "content": text}],
            text_format=Scores)
        scores = {s.key: s for s in (response.output_parsed.scores if response.output_parsed else [])}
        for c in batch:
            s = scores.get(c["key"])
            c["score"], c["score_reason"] = (s.score, s.reason) if s else (0, "unscored")
    return sorted(candidates, key=lambda c: (-c.get("score", 0), c.get("size") or 0))


def gather(log=print) -> tuple[list[dict], list[str]]:
    """Every licence-clean candidate from every source, de-duplicated, with what failed."""
    failures: list[str] = []
    ids: dict[str, dict] = {}
    for query in ARCHIVE_QUERIES:
        try:
            for doc in search_archive(query):
                ids.setdefault(doc["identifier"], doc)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"archive.org search {query!r}: {exc}")
    log(f"archive.org: {len(ids)} items found; reading metadata")

    def meta(identifier: str):
        try:
            return archive_candidate(identifier)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"archive.org metadata {identifier}: {str(exc)[:100]}")
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        found = [c for c in pool.map(meta, list(ids)) if c]
    log(f"archive.org: {len(found)} licence-clean, downloadable")
    commons: dict[str, dict] = {}
    for query in COMMONS_QUERIES:
        try:
            for c in search_commons_video(query):
                commons.setdefault(c["key"], c)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"commons search {query!r}: {exc}")
    log(f"commons: {len(commons)} licence-clean, pre-{LATEST_YEAR + 1} videos")
    ok, why = nara_available()
    if not ok:
        failures.append(f"nara: catalogue API unusable without a key: {why}")
    return found + list(commons.values()), failures


# ---------- generated clips ----------

def refusal(prompt: str) -> str | None:
    """Why a generation prompt is refused, or None: no real person's likeness, no faces at all."""
    if names_a_real_person(prompt):
        return "names a real person"
    if _FACE.search(prompt or ""):
        return "asks for a face or portrait"
    return None


def generate_clip(prompt: str, seconds: int, destination: Path, model: str = "sora-2", *, client=None,
                  size: str = "1280x720", poll_s: float = 10.0, timeout_s: float = 1200.0,
                  sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> dict:
    """Generate one atmospheric clip with the videos API: create the job, poll it, download the mp4.

    Returns what the API reported (id, status, seconds, size, model, timestamps) plus elapsed_s.
    """
    reason = refusal(prompt)
    if reason:
        raise ValueError(f"refusing to generate {prompt!r}: {reason}")
    if int(seconds) not in VIDEO_SECONDS:
        raise ValueError(f"seconds must be one of {VIDEO_SECONDS}, not {seconds}")
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=settings.openai_api_key)
    started = clock()
    video = client.videos.create(model=model, prompt=prompt, seconds=str(int(seconds)), size=size)
    while video.status not in ("completed", "failed"):
        if clock() - started > timeout_s:
            raise TimeoutError(f"video {video.id} still {video.status} after {timeout_s:.0f} s")
        sleep(poll_s)
        video = client.videos.retrieve(video.id)
    if video.status == "failed":
        error = getattr(video, "error", None)
        detail = getattr(error, "message", None) or str(error or "unknown error")
        raise RuntimeError(f"video {video.id} failed: {detail}")
    content = client.videos.download_content(video.id, variant="video")
    data = content.content if hasattr(content, "content") else content.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, destination)
    report = {
        "id": video.id, "status": video.status, "model": getattr(video, "model", model),
        "seconds": getattr(video, "seconds", str(seconds)), "size": getattr(video, "size", size),
        "created_at": getattr(video, "created_at", None), "completed_at": getattr(video, "completed_at", None),
        "expires_at": getattr(video, "expires_at", None), "elapsed_s": round(clock() - started, 1),
        "bytes": len(data),
    }
    try:
        report["raw"] = video.model_dump()
    except Exception:  # noqa: BLE001 - fakes and old SDKs
        pass
    return report


# ---------- command line ----------

def cmd_search(args) -> None:
    from openai import OpenAI

    root = Path(args.video_dir)
    candidates, failures = gather()
    client = OpenAI(api_key=settings.openai_api_key)
    ranked = rank_candidates(client, args.model, candidates)
    write_atomic(root / "clips" / "candidates.json", {"candidates": ranked, "failures": failures})
    print(f"{len(ranked)} candidates; {sum(1 for c in ranked if c['score'] >= args.min_score)} "
          f"scored {args.min_score}+; {len(failures)} failures -> {root / 'clips' / 'candidates.json'}")
    for f in failures[:20]:
        print("  FAILED", f)


def cmd_build(args) -> None:
    from openai import OpenAI

    root = Path(args.video_dir)
    lib = Library(root, budget_bytes=int(args.budget_gb * 1e9))
    data = _read(root / "clips" / "candidates.json", None)
    if data is None:
        sys.exit("run `search` first")
    chosen = [c for c in data["candidates"] if c.get("score", 0) >= args.min_score]
    if args.only:
        wanted = set(args.only.split(","))
        chosen = [c for c in data["candidates"] if c["key"] in wanted or c.get("identifier") in wanted]
    skip = set(filter(None, args.skip.split(",")))
    chosen = [c for c in chosen if c["key"] not in skip and c.get("identifier") not in skip][: args.limit]
    steps = RealSteps(OpenAI(api_key=settings.openai_api_key), args.model)
    lock = threading.Lock()
    stats = {"stored": 0, "failed": 0}

    def segments_so_far() -> int:
        return sum(len(i.get("segments") or []) for i in lib.items())

    def one(c: dict) -> None:
        if segments_so_far() >= args.target:
            return
        free = shutil.disk_usage(root).free
        if free < int(c.get("size") or 0) + DISK_RESERVE_BYTES and not (lib.work_dir(c["key"]) / "judged.json").exists():
            print(f"  skip {c['key']}: only {free / 1e9:.2f} GB free", flush=True)
            return
        try:
            item = process_source(c, lib, steps, log=lambda s: print("  " + s, flush=True))
            with lock:
                stats["stored"] += bool(item)
        except Exception as exc:  # noqa: BLE001 - one bad film must not stop the build
            with lock:
                stats["failed"] += 1
            lib.mark(c["key"], last_error=str(exc)[:200])
            print(f"  FAILED {c['key']}: {str(exc)[:200]}", flush=True)

    print(f"building from {len(chosen)} candidates; {lib.downloaded_bytes() / 1e9:.2f} GB already downloaded", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, chosen))
    items = lib.items()
    print(f"stored {stats['stored']} new, failed {stats['failed']}; library {len(items)} clips, "
          f"{sum(len(i['segments']) for i in items)} segments; downloaded {lib.downloaded_bytes() / 1e9:.2f} GB")


def cmd_generate(args) -> None:
    root = Path(args.video_dir)
    lib = Library(root)
    key = f"generated:{args.prompt}"
    if (lib.source_state(key) or {}).get("status") == "stored":
        sys.exit(f"already generated as {lib.source_state(key)['clip_id']}: never pay for the same clip twice")
    pending = lib.clips_dir / "work" / f"generated-{abs(hash(args.prompt)) % 10**8}.mp4"
    try:
        report = generate_clip(args.prompt, args.seconds, pending, model=args.video_model)
    except Exception as exc:  # noqa: BLE001 - recorded so the report can say what the API returned
        lib.mark(key, status="failed", last_error=f"{exc.__class__.__name__}: {str(exc)[:300]}")
        raise
    clip_id = lib.reserve_id(key)  # only a clip that exists takes an id
    destination = lib.clips_dir / f"{clip_id}.mp4"
    os.replace(pending, destination)
    info = probe_video(destination)
    item = {
        "id": clip_id, "file": f"clips/{clip_id}.mp4", "source": "generated", "title": args.prompt,
        "source_url": "", "licence": GENERATED_LICENCE, "attribution": f"Generated with OpenAI {args.video_model}",
        "rights": "generated", "duration_s": info["duration_s"], "width": info["width"],
        "height": info["height"], "fps": info["fps"],
        "segments": [{"in": 0.0, "out": info["duration_s"], "caption": args.caption or args.prompt,
                      "era": args.era, "quality": args.quality, "graphic": False}],
    }
    lib.add_item(item)
    lib.mark(key, status="stored", clip_id=clip_id, generation=report)
    print(json.dumps({k: v for k, v in report.items() if k != "raw"}, indent=1))
    print(f"wrote {destination} ({info})")


def cmd_recheck(args) -> None:
    """Run the three-still check on clips already stored (from their own files); drop the ones that fail."""
    from openai import OpenAI

    root = Path(args.video_dir)
    lib = Library(root)
    steps = RealSteps(OpenAI(api_key=settings.openai_api_key), args.model)
    candidates = {c["key"]: c for c in (_read(root / "clips" / "candidates.json", {}) or {}).get("candidates", [])}
    state = lib._state()
    owner = {i: k for k, e in state["sources"].items() for i in e.get("clip_ids") or []}
    items = [i for i in lib.items() if i.get("rights") == "free" and not i["segments"][0].get("checked")]
    dropped = 0
    for key in sorted({owner.get(i["id"], "") for i in items}):
        group = [i for i in items if owner.get(i["id"], "") == key]
        candidate = candidates.get(key) or {"title": group[0]["title"]}
        for item in group:
            seg = item["segments"][0]
            verdict = steps.verify(candidate, root / item["file"], [seg], lib.clips_dir / "work" / "strips")[0]
            if acceptable(verdict):
                seg.update(caption=verdict["caption"], era=verdict.get("era") or seg["era"],
                           quality=min(int(seg["quality"]), int(verdict["quality"])), checked=True)
                lib.add_item(item)
                print(f"  ok {item['id']}: {seg['caption'][:100]}", flush=True)
            else:
                flags = [f for f in ("graphic", "violence", "on_screen_text", "modern_elements", "close_up_face")
                         if verdict.get(f)] or ["unusable" if not verdict.get("usable") else f"quality {verdict.get('quality')}"]
                with lib.lock:
                    data = _read(lib.catalogue_path, {"items": []})
                    data["items"] = [i for i in data["items"] if i["id"] != item["id"]]
                    write_atomic(lib.catalogue_path, data)
                (root / item["file"]).unlink(missing_ok=True)
                dropped += 1
                print(f"  DROPPED {item['id']} ({', '.join(flags)}): {verdict.get('caption', '')[:100]}", flush=True)
    shutil.rmtree(lib.clips_dir / "work" / "strips", ignore_errors=True)
    print(f"rechecked {len(items)}, dropped {dropped}")


def cmd_borders(args) -> None:
    """Crop black borders out of clips stored before cut_clip did it (re-encodes only those with borders)."""
    root = Path(args.video_dir)
    lib = Library(root)
    fixed = 0
    for item in lib.items():
        path = root / item["file"]
        if not detect_crop(path, 0.0, float(item["duration_s"])):
            continue
        info = cut_clip(path, 0.0, float(item["duration_s"]) + 1.0, path)
        lib.add_item({**item, "width": info["width"], "height": info["height"]})
        fixed += 1
        print(f"  cropped {item['id']}: {item['width']}x{item['height']} -> {info['width']}x{info['height']}", flush=True)
    print(f"cropped {fixed} clips")


def cmd_drop(args) -> None:
    """Remove clips by id after a human look, with the reason kept in clips/state.json."""
    root = Path(args.video_dir)
    lib = Library(root)
    wanted = set(args.ids.split(","))
    with lib.lock:
        data = _read(lib.catalogue_path, {"items": []})
        gone = [i for i in data["items"] if i["id"] in wanted]
        data["items"] = [i for i in data["items"] if i["id"] not in wanted]
        write_atomic(lib.catalogue_path, data)
        state = lib._state()
        state.setdefault("dropped", []).extend(
            {"id": i["id"], "title": i["title"], "caption": i["segments"][0]["caption"], "reason": args.reason}
            for i in gone)
        for entry in state["sources"].values():
            if entry.get("clip_ids"):
                entry["clip_ids"] = [c for c in entry["clip_ids"] if c not in wanted]
        write_atomic(lib.state_path, state)
    for i in gone:
        (root / i["file"]).unlink(missing_ok=True)
    print(f"dropped {len(gone)}: {', '.join(i['id'] for i in gone)}")


def cmd_prune(args) -> None:
    """Cut every stored film down to its best --per-film segments, delete the rest, renumber the ids."""
    root = Path(args.video_dir)
    lib = Library(root)
    with lib.lock:
        state = lib._state()
        items = {i["id"]: i for i in lib.items()}
        keep: list[tuple[str, dict]] = []  # (film key, item), film order kept
        for key, entry in state["sources"].items():
            ids = [i for i in entry.get("clip_ids") or [] if i in items]
            if not ids:
                continue
            segs = [{**items[i]["segments"][0], "in": items[i]["segments"][0]["source_in"], "_id": i} for i in ids]
            for s in best_segments(segs, args.per_film):
                keep.append((key, items[s["_id"]]))
        kept_ids = {item["id"] for _, item in keep}
        orphans = [i for i in items if i not in kept_ids and not any(i in (e.get("clip_ids") or [])
                                                                      for e in state["sources"].values())]
        for i in orphans:  # generated clips and anything not from a film stay as they are
            keep.append((None, items[i]))
        for i in items:
            if i not in kept_ids and i not in orphans:
                (root / items[i]["file"]).unlink(missing_ok=True)
        keep.sort(key=lambda pair: pair[1]["id"])
        staged = []
        for n, (key, item) in enumerate(keep, start=1):
            new_id = f"v{n:04d}"
            tmp = lib.clips_dir / f"renumber-{new_id}.mp4"
            os.replace(root / item["file"], tmp)
            staged.append((key, item, new_id, tmp))
        new_items, new_ids = [], {}
        per_key: dict[str, list[str]] = {}
        for key, item, new_id, tmp in staged:
            os.replace(tmp, lib.clips_dir / f"{new_id}.mp4")
            new_items.append({**item, "id": new_id, "file": f"clips/{new_id}.mp4"})
            if key:
                per_key.setdefault(key, []).append(new_id)
                new_ids[f"{key}#{len(per_key[key]) - 1}"] = new_id
        for key, entry in state["sources"].items():
            if entry.get("clip_ids") is not None:
                entry["clip_ids"] = per_key.get(key, [])
                entry["segments"] = len(entry["clip_ids"])
        state["ids"] = new_ids
        write_atomic(root / "clips.json", {"items": new_items})
        write_atomic(lib.state_path, state)
    print(f"kept {len(new_items)} of {len(items)} clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("search")
    s.add_argument("video_dir")
    s.add_argument("--model", default=settings.text_model)
    s.add_argument("--min-score", type=int, default=3)
    b = sub.add_parser("build")
    b.add_argument("video_dir")
    b.add_argument("--model", default=settings.text_model)
    b.add_argument("--budget-gb", type=float, default=1.0)
    b.add_argument("--min-score", type=int, default=3)
    b.add_argument("--target", type=int, default=120, help="stop starting new films at this many segments")
    b.add_argument("--workers", type=int, default=3)
    b.add_argument("--limit", type=int, default=60)
    b.add_argument("--only", default="")
    b.add_argument("--skip", default="")
    g = sub.add_parser("generate")
    g.add_argument("video_dir")
    g.add_argument("prompt")
    g.add_argument("--seconds", type=int, default=4)
    g.add_argument("--video-model", default="sora-2")
    g.add_argument("--caption", default="")
    g.add_argument("--era", default="1970s")
    g.add_argument("--quality", type=int, default=3)
    rc = sub.add_parser("recheck")
    rc.add_argument("video_dir")
    rc.add_argument("--model", default=settings.text_model)
    bo = sub.add_parser("borders")
    bo.add_argument("video_dir")
    dr = sub.add_parser("drop")
    dr.add_argument("video_dir")
    dr.add_argument("ids", help="comma-separated clip ids")
    dr.add_argument("--reason", required=True)
    pr = sub.add_parser("prune")
    pr.add_argument("video_dir")
    pr.add_argument("--per-film", type=int, default=MAX_SEGMENTS_PER_FILM)
    args = parser.parse_args()
    {"search": cmd_search, "build": cmd_build, "generate": cmd_generate, "prune": cmd_prune,
     "recheck": cmd_recheck, "borders": cmd_borders, "drop": cmd_drop}[args.command](args)


if __name__ == "__main__":
    main()

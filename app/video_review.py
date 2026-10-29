"""Storyboard review: a film's shots, what each one shows, the owner's marks, and replacing a picture.

Pure file work on one film folder (data/video/<film>): storyboard.json and shots.json (the engine's own
files, changed only through the engine's own functions), library.json and clips.json (the pictures), and
review.json (the owner's marks), review/regen.json (the owner's reviews and their regenerations, made by
app/video_regen.py). The HTTP layer is app/video_api.py; renders run in app/video_jobs.py.

Every shot is put in one category:
- actual: a real picture of the person, place or thing named (link "subject", or a profile/exhibit
  built on a real photograph);
- stand-in: a real archival photograph or footage that is not of this moment (link place_era/action,
  archival film, or a period/setting picture);
- illustration: drawn by the image model;
- graphic: made for the film (name and quote cards, key-fact labels, maps, boards, charts);
- missing: no picture at all.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
import time
import uuid
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SCRIPTS = ROOT / "scripts" / "video"
VIDEO_FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"

CATEGORIES = ("actual", "stand-in", "illustration", "graphic", "missing")
GRAPHIC_KINDS = {"card", "keyfact", "map", "board", "chart"}
DEVICE_KINDS = {"profile", "exhibit", "highlight"}
# Found by the case's own searches: pictures of the case itself. Wikipedia and named-entity pictures are real
# photos of a place or thing, often from another year or of another of its kind, so on their own they stand in.
ACTUAL_SOURCES = {"case", "news"}
KIND_WORDS = {"generated": "illustration", "card": "name card", "keyfact": "key-fact label", "map": "map",
              "board": "evidence board", "chart": "chart", "profile": "profile card",
              "exhibit": "document close-up", "highlight": "document highlight"}
MAX_OPTIONS = 12
REGEN_KEYS = ("review", "state", "step", "explanation", "check", "clip", "error", "at")
INTERRUPTED = "interrupted: regenerate again"
# This server process. A regeneration queued or running under another one (before a restart) has no worker.
WORKER = uuid.uuid4().hex
# The storyboard is read, changed and written back whole: one writer at a time (requests and the regen worker).
LOCK = threading.RLock()


def locked(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        with LOCK:
            return fn(*args, **kwargs)
    return run


class FilmError(Exception):
    """A request about a film that cannot be met (unknown film or shot, unusable picture, refused drawing)."""


def engine():
    """Make scripts/video importable and return the storyboard module."""
    for path in (str(ROOT), str(VIDEO_SCRIPTS)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import storyboard
    return storyboard


def film_dir(root: Path, film: str) -> Path:
    if not film or film.startswith(".") or "/" in film or "\\" in film:
        raise FilmError("no such film")
    d = Path(root) / film
    if not (d / "storyboard.json").exists():
        raise FilmError("no such film")
    return d


def load_storyboard(d: Path) -> dict:
    return json.loads((d / "storyboard.json").read_text(encoding="utf-8"))


def load_library(d: Path, free_only: bool = False) -> dict[str, dict]:
    library, _ = engine().load_library(d, free_only)
    return library


def find_beat(sb: dict, n: int) -> dict:
    for b in sb["beats"]:
        if b["index"] == n:
            return b
    raise FilmError(f"no shot {n}")


def picture_category(item: dict | None) -> str:
    if item is None:
        return "missing"
    found = (item.get("found_by") or "").split(":")[0]
    return "actual" if found in ACTUAL_SOURCES else "stand-in"


def found_words(item: dict) -> str:
    """Where the picture was found, for the owner: "wikipedia: Volkswagen Beetle", "period"."""
    return (item.get("found_by") or ("archive footage" if item.get("media") == "clip" else "")).replace(":", ": ", 1)


def category(beat: dict, library: dict[str, dict]) -> str:
    kind = beat.get("kind")
    if kind == "generated" or beat.get("rights") == "generated":
        return "illustration"
    if kind in GRAPHIC_KINDS:
        return "graphic"
    if kind in DEVICE_KINDS:
        return "actual" if (beat.get("subject_id") or beat.get("library_id")) else "graphic"
    item = library.get(beat.get("library_id") or "")
    if beat.get("media") == "clip":
        return picture_category(item) if item else "stand-in"
    if item is None:
        if beat.get("media") == "motion":
            return "graphic"
        return "stand-in" if beat.get("asset") else "missing"
    if beat.get("owner"):
        return picture_category(item)            # the owner's pick: what the picture itself is
    link = beat.get("link")
    if link == "subject":
        return "actual"
    if link in ("place_era", "action"):
        return "stand-in"
    return picture_category(item)                # reframe, hold: what the picture itself is


def kind_words(beat: dict) -> str:
    if beat.get("kind") in KIND_WORDS:
        return KIND_WORDS[beat["kind"]]
    if beat.get("media") == "clip":
        return "archival footage"
    if beat.get("crop"):
        return "closer framing of a photo"
    return "archival photo"


def marks_path(d: Path) -> Path:
    return d / "review.json"


def load_marks(d: Path) -> dict:
    p = marks_path(d)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def current_mark(marks: dict, beat: dict) -> dict | None:
    m = marks.get(str(beat["index"]))
    if not m or m.get("asset") != beat.get("asset"):
        return None                              # the mark was about a picture the shot no longer shows
    return {"status": m["status"], "note": m.get("note", "")}


@locked
def set_mark(d: Path, n: int, status: str | None, note: str = "") -> dict | None:
    if status not in ("ok", "not_ok", None):
        raise FilmError("status must be ok, not_ok or empty")
    beat = find_beat(load_storyboard(d), n)
    marks = load_marks(d)
    if status is None:
        marks.pop(str(n), None)
    else:
        marks[str(n)] = {"status": status, "note": note, "asset": beat.get("asset"), "at": time.time()}
    marks_path(d).write_text(json.dumps(marks, ensure_ascii=False, indent=1), encoding="utf-8")
    return current_mark(marks, beat)


def regen_path(d: Path) -> Path:
    return d / "review" / "regen.json"


def load_regen(d: Path) -> dict:
    p = regen_path(d)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def regen_view(entry: dict | None) -> dict | None:
    """A shot's review and its regeneration, for the page; one left queued or running by an earlier server
    process has nobody working on it."""
    if not entry:
        return None
    out = {"review": "", "state": "failed", "step": "", "explanation": "", "check": None, "clip": None,
           "error": "", "at": 0.0} | {k: entry[k] for k in REGEN_KEYS if k in entry}
    if out["state"] in ("queued", "running") and entry.get("worker") != WORKER:
        out.update(state="failed", step="", error=INTERRUPTED)
    return out


def regenerating(regen: dict | None) -> bool:
    return bool(regen) and regen["state"] in ("queued", "running")


def animatic_time(d: Path) -> float | None:
    p = d / "animatic.mp4"
    return p.stat().st_mtime if p.exists() else None


def built_from(d: Path) -> float | None:
    """When the review cut read the storyboard: the start of the rebuild that wrote it, else its file time.
    A picture chosen while a rebuild runs is not in that cut, though the file is written after the choice."""
    made = animatic_time(d)
    job_path = d / "review" / "job.json"
    if made is None or not job_path.exists():
        return made
    job = json.loads(job_path.read_text(encoding="utf-8"))
    started = job.get("started_at")
    return started if job.get("kind") == "rebuild" and started and started <= made else made


def final_film(d: Path) -> str | None:
    for name in (f"{d.name}_1080p.mp4", "cut.mp4"):
        if (d / name).exists():
            return name
    # A film rendered with its own --out name (gacy_documentary_v2_1080p.mp4); samples are not the film.
    named = sorted((p for p in d.glob("*_1080p.mp4") if not p.name.startswith("sample")),
                   key=lambda p: p.stat().st_mtime)
    return named[-1].name if named else None


def shot_view(beat: dict, library: dict[str, dict], marks: dict, rebuilt_at: float | None,
              regen: dict | None = None) -> dict:
    item = library.get(beat.get("library_id") or "")
    cat = category(beat, library)
    owner = beat.get("owner")
    check = beat.get("check") or None
    return {
        "n": beat["index"], "start": beat["start"], "end": beat["end"], "category": cat, "kind": kind_words(beat),
        "narration": beat.get("narration", ""), "asset": beat.get("asset"),
        "picture": None if item is None else {
            "id": item["id"], "file": item.get("file"), "media": item.get("media") or "photo",
            "caption": item.get("caption", ""), "source": item.get("source"), "source_url": item.get("source_url"),
            "licence": item.get("licence"), "found": found_words(item)},
        "illustration": beat.get("detail") if cat == "illustration" else None,
        "reason": beat.get("link_reason", ""),
        "check": None if not check else {"fits": check.get("fits"), "misleads": check.get("misleads"),
                                         "problem": check.get("problem", "")},
        "owner": owner, "mark": current_mark(marks, beat), "quote": bool(beat.get("quote_ids")),
        "key_fact": beat.get("key_fact"), "regen": regen_view((regen or {}).get(str(beat["index"]))),
        "frame_current": bool(rebuilt_at) and (not owner or owner.get("at", 0) < rebuilt_at),
    }


def film_view(d: Path) -> dict:
    sb = load_storyboard(d)
    library = load_library(d)
    marks = load_marks(d)
    regen = load_regen(d)
    rebuilt_at = built_from(d)
    by_scene: dict = {}
    for b in sb["beats"]:
        by_scene.setdefault(b.get("scene"), []).append(shot_view(b, library, marks, rebuilt_at, regen))
    scenes = [{"id": s["id"], "title": s.get("title", ""), "story_time": s.get("story_time", ""),
               "look": s.get("look", ""), "start": s.get("start"), "end": s.get("end"),
               "shots": by_scene.get(s["id"], [])} for s in sb.get("scenes", [])]
    shots = [s for sc in scenes for s in sc["shots"]]
    summary = {c: sum(1 for s in shots if s["category"] == c) for c in CATEGORIES}
    summary.update(shots=len(shots), reviewed=sum(1 for s in shots if s["mark"]),
                   not_ok=sum(1 for s in shots if s["mark"] and s["mark"]["status"] == "not_ok"),
                   approved=sum(1 for s in shots if s["mark"] and s["mark"]["status"] == "ok"),
                   regenerating=sum(1 for s in shots if regenerating(s["regen"])))
    return {"film": d.name, "minutes": round(sb.get("duration_s", 0) / 60, 1),
            "animatic": rebuilt_at is not None, "animatic_version": int(animatic_time(d) or 0),
            "final": final_film(d), "summary": summary, "scenes": scenes,
            "render_command": f".venv/bin/python scripts/video/documentary.py render data/video/{d.name}"}


@locked
def approve_unreviewed(d: Path) -> int:
    """Approve every shot the owner has neither marked nor written a review for; returns how many."""
    sb = load_storyboard(d)
    marks = load_marks(d)
    regen = load_regen(d)
    now = time.time()
    approved = 0
    for b in sb["beats"]:
        key = str(b["index"])
        if current_mark(marks, b) or ((regen.get(key) or {}).get("review") or "").strip():
            continue
        marks[key] = {"status": "ok", "note": "", "asset": b.get("asset"), "at": now}
        approved += 1
    if approved:
        marks_path(d).write_text(json.dumps(marks, ensure_ascii=False, indent=1), encoding="utf-8")
    return approved


def list_films(root: Path) -> list[dict]:
    root = Path(root)
    if not root.exists():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if (p / "storyboard.json").exists()):
        sb = load_storyboard(d)
        marks = load_marks(d)
        mk = [current_mark(marks, b) for b in sb["beats"]]
        out.append({"film": d.name, "shots": len(sb["beats"]), "minutes": round(sb.get("duration_s", 0) / 60, 1),
                    "reviewed": sum(1 for m in mk if m), "not_ok": sum(1 for m in mk if m and m["status"] == "not_ok"),
                    "animatic": (d / "animatic.mp4").exists(), "final": final_film(d)})
    return out


def illustration_allowed(sb: dict, beat: dict) -> tuple[bool, str]:
    scene = next((s for s in sb.get("scenes", []) if s["id"] == beat.get("scene")), {})
    if beat.get("kind") in GRAPHIC_KINDS or scene.get("scenario") == "roll_call":
        return False, "cards, labels and the victims' register are made by the engine, not illustrated"
    if beat.get("quote_ids"):
        return False, "a quote card is shown over this shot"
    return True, ""


def alternatives(d: Path, n: int) -> dict:
    storyboard = engine()
    sb = load_storyboard(d)
    beat = find_beat(sb, n)
    allowed, why = illustration_allowed(sb, beat)
    if not allowed:
        return {"shot": n, "options": [], "illustration": {"allowed": False, "why": why}}
    library = load_library(d, free_only=True)
    scene_choices = [c for b in sb["beats"] if b.get("scene") == beat.get("scene") for c in b.get("choices") or []]
    duration = beat["end"] - beat["start"]
    options = []
    for c in dict.fromkeys((beat.get("choices") or []) + scene_choices):
        item = library.get(c)
        if not item or c == beat.get("library_id") or not storyboard.usable(item):
            continue
        if item.get("media") == "clip" and item.get("length", 0) < duration + 1.0:
            continue
        options.append({"id": c, "media": item.get("media") or "photo", "file": item.get("file"),
                        "caption": item.get("caption", ""), "licence": item.get("licence"),
                        "source": item.get("source"), "found": found_words(item),
                        "category": picture_category(item)})
        if len(options) >= MAX_OPTIONS:
            break
    return {"shot": n, "options": options, "illustration": {"allowed": True, "why": ""}}


def choose(d: Path, n: int, choice: str, describe=None, generate=None) -> dict:
    storyboard = engine()
    sb = load_storyboard(d)
    beat = find_beat(sb, n)
    allowed, why = illustration_allowed(sb, beat)
    if not allowed:
        raise FilmError(why)
    if choice == "illustration":
        import ladder
        describe = describe or ladder.describe_detail
        generate = generate or ladder.generate_detail
        scene = next((s for s in sb.get("scenes", []) if s["id"] == beat.get("scene")), {})
        sentences_path = d / "sentences.json"
        sentences = json.loads(sentences_path.read_text(encoding="utf-8")) if sentences_path.exists() else []
        words = " ".join(s["text"] for s in sentences
                         if s["end"] > beat["start"] and s["start"] < beat["end"]) or beat.get("narration", "")
        detail = describe(words, scene)
        if not ladder.safe_detail(detail, sb.get("bible") or {}):
            raise FilmError(f"the illustration rules refused this drawing: {detail}")
        path = generate(detail, scene.get("story_time") or "", d)

        def change(beat: dict) -> None:
            for key in ("clip_id", "clip_in", "clip_out"):
                beat.pop(key, None)
            # The fields ladder.illustrate_holds gives a drawn beat (its transition is left to the edit).
            beat.update({"asset": str(Path(path).relative_to(d)), "media": "photo", "kind": "generated",
                         "library_id": None, "colour": "colour", "rights": "generated", "detail": detail,
                         "licence": "Generated for this film (illustration)", "attribution": "Illustration",
                         "link": "generated", "effects": [], "crop": None})
    else:
        library = load_library(d, free_only=True)
        item = library.get(choice)
        if not item or not storyboard.usable(item):
            raise FilmError("that picture is not available")
        if item.get("media") == "clip" and item.get("length", 0) < beat["end"] - beat["start"] + 1.0:
            raise FilmError("that clip is shorter than the shot")

        def change(beat: dict) -> None:
            storyboard.set_picture(beat, item)
            beat.update({"crop": None, "detail": ""})
            if item.get("media") == "clip":
                beat.update({"media": "clip", "kind": "clip", "clip_id": item["clip_id"],
                             "clip_in": item["in"], "clip_out": item["out"]})
            else:
                beat["kind"] = "archival"
    # Read again under the lock: a drawing takes a while, and other shots may have changed meanwhile.
    with LOCK:
        sb = load_storyboard(d)
        beat = find_beat(sb, n)
        change(beat)
        beat["owner"] = {"choice": choice, "at": time.time()}
        storyboard.save(d, sb)
    return shot_view(beat, load_library(d), load_marks(d), built_from(d), load_regen(d))


def _grab(source: Path, at: float, out: Path) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        done = subprocess.run([VIDEO_FFMPEG, "-v", "error", "-y", "-ss", f"{max(0.0, at):.3f}", "-i", str(source),
                               "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", str(out)],
                              capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out if done.returncode == 0 and out.exists() else None


def frame(d: Path, n: int) -> Path | None:
    """The shot's frame in the review cut (cached); None without a review cut or when it is out of date."""
    beat = find_beat(load_storyboard(d), n)
    rebuilt_at = built_from(d)
    owner = beat.get("owner")
    if rebuilt_at is None or (owner and owner.get("at", 0) >= rebuilt_at):
        return None
    key = hashlib.sha1(f"{rebuilt_at}|{beat['start']}|{beat['end']}|{beat.get('asset')}|{beat.get('crop')}"
                       .encode()).hexdigest()[:10]
    out = d / "review" / "frames" / f"{n}_{key}.jpg"
    return out if out.exists() else _grab(d / "animatic.mp4", (beat["start"] + beat["end"]) / 2, out)


def thumb(d: Path, item_id: str) -> Path | None:
    """A still for a library picture or a clip (a frame a second into the clip)."""
    item = load_library(d).get(item_id)
    if not item or not item.get("file"):
        return None
    if item.get("media") != "clip":
        p = d / item["file"]
        return p if p.exists() else None
    safe = item_id.replace("#", "_").replace("/", "_")
    out = d / "review" / "thumbs" / f"{safe}.jpg"
    return out if out.exists() else _grab(d / item["file"], item.get("in", 0.0) + min(1.0, item.get("length", 2) / 2), out)

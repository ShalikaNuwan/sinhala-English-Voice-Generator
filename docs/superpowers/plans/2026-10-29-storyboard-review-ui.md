# Storyboard Review UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Video tab in the local app where the owner reviews a film's storyboard shot by shot (what each shot shows, and whether it is an actual picture, a stand-in, an illustration or a graphic), marks shots OK / not OK, swaps a shot's picture or asks for a new illustration, rebuilds the 480p review cut, and starts the final render.

**Architecture:** Three new backend modules in `app/`: `video_review.py` (pure file work on `data/video/<film>`: films, shot categories, marks, alternatives, choosing a picture, frame grabs), `video_jobs.py` (one background `build.py` process per film, state on disk) and `video_api.py` (a FastAPI router under `/api/video`, mounted in `app/main.py`). The storyboard engine (`scripts/video/storyboard.py`, `ladder.py`) learns to leave shots marked `owner` alone. The front end adds a tab and a new `app/static/video.js`.

**Tech Stack:** Python 3.12, FastAPI 0.141 / Starlette 1.6 (range requests built in), vanilla JS, ffmpeg 9 at `/opt/miniconda3/envs/video/bin/ffmpeg`, pytest.

Spec: `docs/superpowers/specs/2026-10-29-storyboard-review-ui-design.md`.

**Owner rule: never commit or push.** Every task ends with the tests passing and the work left uncommitted; the owner reviews the diff and approves commits separately.

**Run tests with:** `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q <files>` from the repo root `/Users/shalikanuwan/Documents/Projects/Sinhala-English-Voice`.

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/video/storyboard.py` (modify) | picture-moving rules skip shots with `owner` |
| `scripts/video/ladder.py` (modify) | ladder rules skip shots with `owner` |
| `app/video_review.py` (create) | films, categories, film view, marks, alternatives, choose, frames, thumbs |
| `app/video_jobs.py` (create) | background rebuild/render processes, status from their logs |
| `app/video_api.py` (create) | HTTP endpoints under `/api/video` |
| `app/main.py` (modify) | include the router |
| `app/static/index.html` (modify) | Voice / Video tabs and the Video tab's markup |
| `app/static/video.js` (create) | the Video tab's behaviour |
| `app/static/style.css` (modify) | tabs, badges, shot rows, alternatives |
| `tests/test_video_owner_lock.py` (create) | engine rules respect `owner` |
| `tests/test_video_review.py` (create) | domain module |
| `tests/test_video_jobs.py` (create) | job runner |
| `tests/test_video_api.py` (create) | endpoints |

---

### Task 0: Branch

- [ ] **Step 1:** From the repo root, create the working branch (no upstream):

```bash
git checkout -b storyboard-review
git branch --show-current
```
Expected: `storyboard-review`

---

### Task 1: The engine leaves the owner's choices alone

**Files:**
- Modify: `scripts/video/storyboard.py` (functions `respect_links`, `guard_dates`, `guard_victims`, `guard_mourning`, `speaker_pictures`, `merge_repeats`)
- Modify: `scripts/video/ladder.py` (functions `vary_subjects`, `split_long_holds`, `break_loops`, `illustrate_holds`, `apply_ladder`)
- Test: `tests/test_video_owner_lock.py`

A shot the owner chose carries `owner: {"choice": "<library id>" | "illustration", "at": <epoch seconds>}`. Rules that would move its picture must skip it. `replace_excluded` (remains/bodies) must NOT skip it.

- [ ] **Step 1: Write the failing tests** in `tests/test_video_owner_lock.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import ladder  # noqa: E402
import storyboard  # noqa: E402

OWNER = {"choice": "x", "at": 1.0}


def item(i, caption="A street.", **kw):
    return {"id": i, "file": f"library/{i}.jpg", "caption": caption, **kw}


def beat(i, start, end, pid, **kw):
    return {"index": i, "scene": 1, "start": start, "end": end, "media": "photo", "library_id": pid,
            "asset": f"library/{pid}.jpg", "narration": f"words {i}", "choices": [], **kw}


def test_respect_links_puts_a_beat_back_on_its_own_choice_but_not_the_owners():
    library = {"a": item("a"), "x": item("x")}
    planned = beat(1, 0, 5, "x", link="subject", choices=["a"])
    chosen = beat(2, 5, 10, "x", link="subject", choices=["a"], owner=OWNER)
    storyboard.respect_links([planned, chosen], library)
    assert planned["library_id"] == "a"                  # the rule works
    assert chosen["library_id"] == "x"                   # and leaves the owner's pick


def test_guard_dates_leaves_the_owners_pick():
    library = {"late": item("late", "A street in 1990.", era="1990"), "ok": item("ok", "A street in 1975.")}
    scenes = {1: {"id": 1, "story_time": "1975"}}
    planned = beat(1, 0, 5, "late", choices=["ok"])
    chosen = beat(2, 5, 10, "late", choices=["ok"], owner=OWNER)
    storyboard.guard_dates([planned, chosen], library, scenes)
    assert planned["library_id"] == "ok" and chosen["library_id"] == "late"


def test_merge_repeats_never_joins_an_owners_pick():
    scenes = [{"id": 1, "beats": []}]
    entries = [beat(1, 0, 5, "x"), beat(2, 5, 10, "x", owner=OWNER)]
    storyboard.merge_repeats(entries, scenes)
    assert len(entries) == 2


def test_excluded_pictures_are_replaced_even_when_the_owner_chose_them():
    library = {"bad": item("bad", excluded=True), "ok": item("ok")}
    chosen = beat(1, 0, 5, "bad", choices=["ok"], owner=OWNER)
    storyboard.replace_excluded([chosen], library)
    assert chosen["library_id"] == "ok"


def test_illustrate_holds_leaves_an_owners_long_stand_in(tmp_path):
    made = []

    def generate(detail, era, video_dir):
        out = video_dir / "assets" / "generated" / f"d{len(made)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        made.append(detail)
        return out

    entries = [beat(1, 0, 30, "aerial", link="place_era", owner=OWNER)]
    ladder.illustrate_holds(entries, tmp_path, {"people": [], "places": []}, {1: {"id": 1}},
                            describe=lambda w, s: "a kitchen table", generate=generate, library={})
    assert made == [] and len(entries) == 1 and entries[0]["library_id"] == "aerial"
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_owner_lock.py`
Expected: 4 failures (respect_links, guard_dates, merge_repeats, illustrate_holds); `test_excluded_pictures_are_replaced_even_when_the_owner_chose_them` passes already.

- [ ] **Step 3: Implement.** In `scripts/video/storyboard.py`:

`respect_links`: as the first statement inside `for e in entries:` add
```python
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
```
Do the same (first statement in the per-beat loop) in `guard_dates`, `guard_victims` (its `for e in entries:` loop), `guard_mourning` (its `for e in entries:` loop) and `speaker_pictures` (its `for position, e in enumerate(entries):` loop). Do NOT change `replace_excluded`.

`merge_repeats`: extend the `same = (...)` condition with `and not e.get("owner") and not prev.get("owner")`:
```python
        same = (prev is not None and e.get("library_id") and e.get("library_id") == prev.get("library_id")
                and e.get("media") == prev.get("media") == "photo" and e.get("scene") == prev.get("scene")
                and e["end"] - prev["start"] <= MERGE_MAX_S and not e.get("owner") and not prev.get("owner"))
```

In `scripts/video/ladder.py`:
- `vary_subjects`, inside `for k in range(1, len(entries)):` after `e = entries[k]` (or as the first statement that reads the beat), add `if e.get("owner"): continue`.
- `split_long_holds`, in `for e in entries:`, add `or e.get("owner")` to the first `if (...)` condition that sends a beat straight to `out.append(e)`.
- `break_loops`, in `for k in range(3, len(entries)):`, after `e = entries[k]` add `if e.get("owner"): continue`.
- `illustrate_holds`, as the first statement inside `for e in entries:` add:
```python
        if e.get("owner"):
            out.append(e)                 # the owner chose this picture in the review page
            continue
```
- `apply_ladder`, as the first statement inside `for at, e in enumerate(entries):` add:
```python
        if e.get("owner"):
            keep.append(e)
            continue
```

- [ ] **Step 4: Run the new tests and the engine's existing tests**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_owner_lock.py tests/test_video_storyboard.py tests/test_video_ladder.py`
Expected: all pass.

---

### Task 2: The review domain module

**Files:**
- Create: `app/video_review.py`
- Test: `tests/test_video_review.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_video_review.py`:

```python
import json
import time
from pathlib import Path

import pytest

from app import video_review as review


def make_film(root: Path, name="demo") -> Path:
    d = root / name
    (d / "library").mkdir(parents=True)
    (d / "assets" / "generated").mkdir(parents=True)
    items = [
        {"id": "c1", "file": "library/c1.jpg", "caption": "Rex Heuermann in 2023.", "found_by": "case:Category:Rex",
         "rights": "free", "licence": "Public domain", "source": "commons"},
        {"id": "c2", "file": "library/c2.jpg", "caption": "A Long Island street, 1970s.", "found_by": "period",
         "rights": "free", "licence": "Public domain", "source": "commons"},
        {"id": "c3", "file": "library/c3.jpg", "caption": "Oak Beach, 2010.", "found_by": "need:oak beach",
         "rights": "free", "licence": "CC0", "source": "openverse"},
        {"id": "bad", "file": "library/bad.jpg", "caption": "Remains.", "found_by": "case", "rights": "free",
         "licence": "Public domain", "graphic": True},
    ]
    for i in items:
        (d / i["file"]).write_bytes(b"jpg")
    (d / "library.json").write_text(json.dumps({"items": items}))
    (d / "clips.json").write_text(json.dumps({"items": [{"id": "v1", "file": "clips/v1.mp4", "source": "archive",
        "licence": "Public domain", "rights": "free", "segments": [
            {"in": 0.0, "out": 3.0, "caption": "Police car, 1970s.", "quality": 4},
            {"in": 0.0, "out": 20.0, "caption": "Traffic, 1970s.", "quality": 4}]}]}))
    (d / "assets" / "generated" / "detail_a.png").write_bytes(b"png")
    beats = [
        {"index": 1, "scene": 1, "start": 0.0, "end": 5.0, "kind": "archival", "media": "photo", "link": "subject",
         "library_id": "c1", "asset": "library/c1.jpg", "narration": "Rex Heuermann was arrested.",
         "choices": ["c1", "c2", "c3", "bad", "v1#0", "v1#1"], "link_reason": "Shows Rex.",
         "check": {"fits": "yes", "misleads": False, "problem": ""}},
        {"index": 2, "scene": 1, "start": 5.0, "end": 10.0, "kind": "period", "media": "photo", "link": "place_era",
         "library_id": "c2", "asset": "library/c2.jpg", "narration": "Long Island at night.", "choices": ["c2"]},
        {"index": 3, "scene": 1, "start": 10.0, "end": 14.0, "kind": "generated", "media": "photo", "link": "generated",
         "library_id": None, "asset": "assets/generated/detail_a.png", "detail": "a cash register at dusk",
         "narration": "She worked the register.", "choices": []},
        {"index": 4, "scene": 1, "start": 14.0, "end": 17.0, "kind": "keyfact", "media": "photo", "link": "reframe",
         "library_id": None, "asset": "assets/cards/keyfact_a.png", "key_fact": "blue jacket",
         "narration": "He wore a blue jacket.", "choices": []},
        {"index": 5, "scene": 1, "start": 17.0, "end": 20.0, "kind": "archival", "media": "photo", "link": "reframe",
         "library_id": "c3", "asset": "library/c3.jpg", "crop": [0.1, 0.1, 0.6, 0.6], "narration": "Oak Beach.",
         "choices": ["c3"]},
        {"index": 6, "scene": 1, "start": 20.0, "end": 28.0, "kind": "clip", "media": "clip", "link": "action",
         "library_id": "v1#1", "asset": "clips/v1.mp4", "clip_id": "v1", "clip_in": 0.0, "clip_out": 20.0,
         "narration": "Cars passed.", "choices": ["v1#1"]},
    ]
    sb = {"duration_s": 28.0, "bible": {"people": [], "places": []},
          "scenes": [{"id": 1, "title": "The arrest", "story_time": "2023", "look": "investigation", "start": 0.0,
                      "end": 28.0, "beats": [1, 2, 3, 4, 5, 6]}],
          "quotes": [], "beats": beats}
    (d / "storyboard.json").write_text(json.dumps(sb))
    (d / "shots.json").write_text(json.dumps({"audio": "", "width": 1920, "height": 1080, "fps": 30, "shots": beats}))
    return d


def test_categories_follow_what_each_shot_shows(tmp_path):
    d = make_film(tmp_path)
    view = review.film_view(d)
    cats = {s["n"]: s["category"] for sc in view["scenes"] for s in sc["shots"]}
    assert cats == {1: "actual", 2: "stand-in", 3: "illustration", 4: "graphic", 5: "stand-in", 6: "stand-in"}
    assert view["summary"]["actual"] == 1 and view["summary"]["stand-in"] == 3
    shot3 = view["scenes"][0]["shots"][2]
    assert shot3["illustration"] == "a cash register at dusk" and shot3["kind"] == "illustration"


def test_list_films_finds_folders_with_a_storyboard(tmp_path):
    make_film(tmp_path, "one")
    (tmp_path / "not_a_film").mkdir()
    films = review.list_films(tmp_path)
    assert [f["film"] for f in films] == ["one"] and films[0]["shots"] == 6


def test_film_dir_refuses_unknown_and_escaping_names(tmp_path):
    make_film(tmp_path)
    assert review.film_dir(tmp_path, "demo") == tmp_path / "demo"
    for bad in ("nope", "../demo", "", ".hidden"):
        with pytest.raises(review.FilmError):
            review.film_dir(tmp_path, bad)


def test_marks_are_kept_and_dropped_when_the_picture_changes(tmp_path):
    d = make_film(tmp_path)
    review.set_mark(d, 2, "not_ok", "wrong town")
    shot = review.film_view(d)["scenes"][0]["shots"][1]
    assert shot["mark"] == {"status": "not_ok", "note": "wrong town"}
    sb = json.loads((d / "storyboard.json").read_text())
    sb["beats"][1]["asset"] = "library/c3.jpg"
    (d / "storyboard.json").write_text(json.dumps(sb))
    assert review.film_view(d)["scenes"][0]["shots"][1]["mark"] is None
    review.set_mark(d, 2, None, "")
    assert "2" not in json.loads((d / "review.json").read_text())


def test_alternatives_are_the_shots_usable_candidates_and_long_enough_clips(tmp_path):
    d = make_film(tmp_path)
    alt = review.alternatives(d, 1)
    ids = [o["id"] for o in alt["options"]]
    assert ids == ["c2", "c3", "v1#1"]          # not itself, not the graphic one, not the 3 s clip for a 5 s shot
    assert alt["options"][0]["category"] == "stand-in" and alt["illustration"]["allowed"] is True
    graphic = review.alternatives(d, 4)
    assert graphic["options"] == [] and graphic["illustration"]["allowed"] is False


def test_choosing_another_picture_updates_both_files_and_marks_it_the_owners(tmp_path):
    d = make_film(tmp_path)
    shot = review.choose(d, 1, "c2")
    assert shot["picture"]["id"] == "c2" and shot["owner"]["choice"] == "c2"
    sb = json.loads((d / "storyboard.json").read_text())
    shots = json.loads((d / "shots.json").read_text())["shots"]
    assert sb["beats"][0]["library_id"] == "c2" == shots[0]["library_id"]
    assert sb["beats"][0]["asset"] == "library/c2.jpg" and sb["beats"][0]["owner"]["choice"] == "c2"
    clip = review.choose(d, 2, "v1#1")
    assert clip["picture"]["media"] == "clip"
    beat = json.loads((d / "storyboard.json").read_text())["beats"][1]
    assert beat["media"] == "clip" and beat["clip_in"] == 0.0 and beat["clip_out"] == 20.0
    with pytest.raises(review.FilmError):
        review.choose(d, 1, "bad")


def test_a_new_illustration_is_drawn_and_an_unsafe_one_refused(tmp_path):
    d = make_film(tmp_path)

    def generate(detail, era, video_dir):
        out = video_dir / "assets" / "generated" / "detail_new.png"
        out.write_bytes(b"png")
        return out

    shot = review.choose(d, 2, "illustration", describe=lambda words, scene: "an empty road at night",
                         generate=generate)
    assert shot["category"] == "illustration" and shot["illustration"] == "an empty road at night"
    with pytest.raises(review.FilmError):
        review.choose(d, 1, "illustration", describe=lambda words, scene: "a man waiting by a car",
                      generate=generate)


def test_frames_are_current_only_after_a_rebuild(tmp_path):
    d = make_film(tmp_path)
    (d / "animatic.mp4").write_bytes(b"mp4")
    view = review.film_view(d)
    assert view["animatic"] and all(s["frame_current"] for s in view["scenes"][0]["shots"])
    review.choose(d, 1, "c2")
    time.sleep(0.01)
    shots = review.film_view(d)["scenes"][0]["shots"]
    assert shots[0]["frame_current"] is False and shots[1]["frame_current"] is True
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_review.py`
Expected: errors, `ModuleNotFoundError: No module named 'app.video_review'`.

- [ ] **Step 3: Implement** `app/video_review.py`:

```python
"""Storyboard review: a film's shots, what each one shows, the owner's marks, and replacing a picture.

Pure file work on one film folder (data/video/<film>): storyboard.json and shots.json (the engine's own
files, changed only through the engine's own functions), library.json and clips.json (the pictures), and
review.json (the owner's marks). The HTTP layer is app/video_api.py; renders run in app/video_jobs.py.

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
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIDEO_SCRIPTS = ROOT / "scripts" / "video"
VIDEO_FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"

CATEGORIES = ("actual", "stand-in", "illustration", "graphic", "missing")
GRAPHIC_KINDS = {"card", "keyfact", "map", "board", "chart"}
DEVICE_KINDS = {"profile", "exhibit", "highlight"}
STAND_IN_SOURCES = {"period", "need", "archive"}
KIND_WORDS = {"generated": "illustration", "card": "name card", "keyfact": "key-fact label", "map": "map",
              "board": "evidence board", "chart": "chart", "profile": "profile card",
              "exhibit": "document close-up", "highlight": "document highlight"}
MAX_OPTIONS = 12


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
    if item.get("media") == "clip":
        return "stand-in"
    found = (item.get("found_by") or "").split(":")[0]
    return "stand-in" if found in STAND_IN_SOURCES else "actual"


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
        return "actual" if item and (item.get("found_by") or "").startswith("case") else "stand-in"
    if item is None:
        if beat.get("media") == "motion":
            return "graphic"
        return "stand-in" if beat.get("asset") else "missing"
    if beat.get("owner"):
        return picture_category(item)
    link = beat.get("link")
    if link == "subject":
        return "actual"
    if link in ("place_era", "action"):
        return "stand-in"
    return picture_category(item)            # reframe, hold: what the picture itself is


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


def animatic_time(d: Path) -> float | None:
    p = d / "animatic.mp4"
    return p.stat().st_mtime if p.exists() else None


def final_film(d: Path) -> str | None:
    for name in (f"{d.name}_1080p.mp4", "cut.mp4"):
        if (d / name).exists():
            return name
    return None


def shot_view(beat: dict, library: dict[str, dict], marks: dict, rebuilt_at: float | None) -> dict:
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
            "licence": item.get("licence")},
        "illustration": beat.get("detail") if cat == "illustration" else None,
        "reason": beat.get("link_reason", ""),
        "check": None if not check else {"fits": check.get("fits"), "misleads": check.get("misleads"),
                                         "problem": check.get("problem", "")},
        "owner": owner, "mark": current_mark(marks, beat), "quote": bool(beat.get("quote_ids")),
        "key_fact": beat.get("key_fact"),
        "frame_current": bool(rebuilt_at) and (not owner or owner.get("at", 0) < rebuilt_at),
    }


def film_view(d: Path) -> dict:
    sb = load_storyboard(d)
    library = load_library(d)
    marks = load_marks(d)
    rebuilt_at = animatic_time(d)
    by_scene: dict = {}
    for b in sb["beats"]:
        by_scene.setdefault(b.get("scene"), []).append(shot_view(b, library, marks, rebuilt_at))
    scenes = [{"id": s["id"], "title": s.get("title", ""), "story_time": s.get("story_time", ""),
               "look": s.get("look", ""), "start": s.get("start"), "end": s.get("end"),
               "shots": by_scene.get(s["id"], [])} for s in sb.get("scenes", [])]
    shots = [s for sc in scenes for s in sc["shots"]]
    summary = {c: sum(1 for s in shots if s["category"] == c) for c in CATEGORIES}
    summary.update(shots=len(shots), reviewed=sum(1 for s in shots if s["mark"]),
                   not_ok=sum(1 for s in shots if s["mark"] and s["mark"]["status"] == "not_ok"))
    return {"film": d.name, "minutes": round(sb.get("duration_s", 0) / 60, 1),
            "animatic": rebuilt_at is not None, "animatic_version": int(rebuilt_at or 0),
            "final": final_film(d), "summary": summary, "scenes": scenes,
            "render_command": f".venv/bin/python scripts/video/documentary.py render data/video/{d.name}"}


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
                        "source": item.get("source"), "category": picture_category(item)})
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
        for key in ("clip_id", "clip_in", "clip_out"):
            beat.pop(key, None)
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
        storyboard.set_picture(beat, item)
        beat.update({"crop": None, "detail": ""})
        if item.get("media") == "clip":
            beat.update({"media": "clip", "kind": "clip", "clip_id": item["clip_id"],
                         "clip_in": item["in"], "clip_out": item["out"]})
        else:
            beat["kind"] = "archival"
    beat["owner"] = {"choice": choice, "at": time.time()}
    storyboard.save(d, sb)
    return shot_view(beat, load_library(d), load_marks(d), animatic_time(d))


def _grab(source: Path, at: float, out: Path) -> Path | None:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = subprocess.run([VIDEO_FFMPEG, "-v", "error", "-y", "-ss", f"{max(0.0, at):.3f}", "-i", str(source),
                           "-frames:v", "1", "-vf", "scale=640:-2", "-q:v", "3", str(out)],
                          capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
    return out if done.returncode == 0 and out.exists() else None


def frame(d: Path, n: int) -> Path | None:
    """The shot's frame in the review cut (cached); None without a review cut or when it is out of date."""
    beat = find_beat(load_storyboard(d), n)
    rebuilt_at = animatic_time(d)
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
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_review.py`
Expected: all 8 pass. (If `test_a_new_illustration_is_drawn_and_an_unsafe_one_refused` fails because `safe_detail` lets "a man waiting by a car" through, check `ladder.safe_detail`: it must refuse the word "man".)

---

### Task 3: The job runner

**Files:**
- Create: `app/video_jobs.py`
- Test: `tests/test_video_jobs.py`

- [ ] **Step 1: Write the failing tests** in `tests/test_video_jobs.py`:

```python
import json
import time
from pathlib import Path

import pytest

from app import video_jobs


def film(tmp_path: Path) -> Path:
    d = tmp_path / "demo"
    d.mkdir()
    (d / "storyboard.json").write_text(json.dumps({"duration_s": 60.0, "beats": []}))
    return d


def wait_for(jobs, d, state, seconds=10):
    end = time.time() + seconds
    while time.time() < end:
        s = jobs.status(d)
        if s and s["state"] == state:
            return s
        time.sleep(0.1)
    raise AssertionError(f"job never reached {state}: {jobs.status(d)}")


def test_a_job_runs_reports_progress_and_finishes(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs, "commands", lambda d, kind, sound=False: [
        "python3", "-c", "import time\nfor k in range(1, 4):\n print(f'  shot {k}/3 (#{k}) 1.0s cut', flush=True)"])
    jobs = video_jobs.Jobs()
    started = jobs.start(d, "rebuild")
    assert started["kind"] == "rebuild"
    done = wait_for(jobs, d, "done")
    assert done["progress"] == 1.0


def test_a_failed_job_reports_its_last_lines_and_a_second_start_is_refused(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs, "commands", lambda d, kind, sound=False: [
        "python3", "-c", "import sys, time\nprint('  shot 1/4 (#1) 1.0s cut', flush=True)\ntime.sleep(0.6)\n"
                         "print('boom', flush=True)\nsys.exit(3)"])
    jobs = video_jobs.Jobs()
    jobs.start(d, "rebuild")
    with pytest.raises(video_jobs.JobBusy):
        jobs.start(d, "render")
    failed = wait_for(jobs, d, "failed")
    assert failed["progress"] == 0.25 and "boom" in failed["tail"][-1]


def test_status_is_empty_before_any_job(tmp_path):
    assert video_jobs.Jobs().status(film(tmp_path)) is None


def test_a_render_is_refused_without_enough_disk(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 ** 9})())
    with pytest.raises(video_jobs.NoSpace):
        video_jobs.Jobs().start(d, "render")


def test_commands_are_the_engines_own():
    d = Path("/data/video/demo")
    rebuild = video_jobs.commands(d, "rebuild")
    render = video_jobs.commands(d, "render", sound=True)
    assert rebuild[-4:] == [str(d), "--animatic", "--out", str(d / "animatic.mp4")]
    assert render[-4:] == [str(d), "--out", str(d / "demo_1080p.mp4"), "--sound"]
    assert rebuild[2].endswith("scripts/video/build.py")
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_jobs.py`
Expected: `ModuleNotFoundError: No module named 'app.video_jobs'`.

- [ ] **Step 3: Implement** `app/video_jobs.py`:

```python
"""Background renders for the storyboard review: one process per film at a time, its state on disk.

A job is `build.py` run on the film (the 480p review cut, or the 1080p final film), started detached
through /bin/sh so that it outlives a server restart. Its output goes to <film>/logs/review_<kind>.log,
which ends with "__exit__ <code>" when the process finishes; <film>/review/job.json remembers which job
ran. Status is read back from those two files: done or failed from the exit line, running while the
process is alive, stopped when it died without one. Progress is the latest "shot k/N" line of the log.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "video" / "build.py"
EXIT = "__exit__"
FULL_FILM_BYTES = 20e9          # free space a 90-minute final render needs
FULL_FILM_S = 90 * 60


class JobBusy(Exception):
    pass


class NoSpace(Exception):
    pass


def commands(d: Path, kind: str, sound: bool = False) -> list[str]:
    if kind == "rebuild":
        return [sys.executable, "-u", str(BUILD), str(d), "--animatic", "--out", str(d / "animatic.mp4")]
    if kind == "render":
        return [sys.executable, "-u", str(BUILD), str(d), "--out", str(d / f"{d.name}_1080p.mp4"),
                *(["--sound"] if sound else [])]
    raise ValueError(f"unknown job {kind}")


def load_env() -> dict:
    """The environment for a job: this process's, with the project's .env (API keys) added."""
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def check_space(d: Path) -> None:
    duration = json.loads((d / "storyboard.json").read_text(encoding="utf-8")).get("duration_s", FULL_FILM_S)
    need = max(5e9, FULL_FILM_BYTES * duration / FULL_FILM_S)
    free = shutil.disk_usage(d).free
    if free < need:
        raise NoSpace(f"the final render needs about {need / 1e9:.0f} GB free; {free / 1e9:.0f} GB is free")


class Jobs:
    def state_path(self, d: Path) -> Path:
        return d / "review" / "job.json"

    def status(self, d: Path) -> dict | None:
        p = self.state_path(d)
        if not p.exists():
            return None
        job = json.loads(p.read_text(encoding="utf-8"))
        log = Path(job["log"])
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        exits = re.findall(rf"^{EXIT} (-?\d+)\s*$", text, re.M)
        if exits:
            state = "done" if exits[-1] == "0" else "failed"
        elif alive(job["pid"]):
            state = "running"
        else:
            state = "stopped"
        shots = re.findall(r"shot (\d+)/(\d+)", text)
        progress = 1.0 if state == "done" else (int(shots[-1][0]) / int(shots[-1][1]) if shots else 0.0)
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith(EXIT)]
        tail = lines[-12:] if state in ("failed", "stopped") else lines[-1:]
        return {**job, "state": state, "progress": round(progress, 3), "tail": tail}

    def start(self, d: Path, kind: str, sound: bool = False) -> dict:
        current = self.status(d)
        if current and current["state"] == "running":
            raise JobBusy(f"a {current['kind']} is already running for this film")
        if kind == "render":
            check_space(d)
        (d / "review").mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        log = d / "logs" / f"review_{kind}.log"
        command = " ".join(shlex.quote(c) for c in commands(d, kind, sound))
        shell = f"{command} > {shlex.quote(str(log))} 2>&1; echo \"{EXIT} $?\" >> {shlex.quote(str(log))}"
        proc = subprocess.Popen(["/bin/sh", "-c", shell], cwd=ROOT, env=load_env(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        job = {"kind": kind, "pid": proc.pid, "sound": sound, "log": str(log), "command": command,
               "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.state_path(d).write_text(json.dumps(job, indent=1), encoding="utf-8")
        return self.status(d)
```

Note on `alive`: the `/bin/sh` wrapper is a child of the server; when it finishes it becomes a zombie until reaped, and `os.kill(pid, 0)` still succeeds on a zombie. That is fine: the exit line is checked first, so a finished job reads as done/failed regardless.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_jobs.py`
Expected: all 5 pass.

---

### Task 4: The HTTP endpoints

**Files:**
- Create: `app/video_api.py`
- Modify: `app/main.py` (include the router after the `app.mount(...)` lines)
- Test: `tests/test_video_api.py`

Depends on Tasks 2 and 3.

- [ ] **Step 1: Write the failing tests** in `tests/test_video_api.py`:

```python
import importlib
import json

from tests.test_video_review import make_film


def client_for(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import app.config
    import app.main
    import app.video_api
    importlib.reload(app.config)
    importlib.reload(app.video_api)
    module = importlib.reload(app.main)
    from fastapi.testclient import TestClient
    return TestClient(module.app), app.video_api


class FakeJobs:
    def __init__(self):
        self.started = []
        self.state = None

    def status(self, d):
        return self.state

    def start(self, d, kind, sound=False):
        from app.video_jobs import JobBusy
        if self.state and self.state["state"] == "running":
            raise JobBusy("a rebuild is already running for this film")
        self.started.append((d.name, kind, sound))
        self.state = {"kind": kind, "state": "running", "progress": 0.0, "tail": []}
        return self.state


def test_films_and_film_view(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    films = client.get("/api/video/films").json()["films"]
    assert [f["film"] for f in films] == ["demo"]
    view = client.get("/api/video/films/demo").json()
    assert view["summary"]["shots"] == 6 and "job" in view
    assert client.get("/api/video/films/nope").status_code == 404


def test_mark_alternatives_and_choose(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    r = client.put("/api/video/films/demo/shots/2/mark", json={"status": "not_ok", "note": "wrong town"})
    assert r.status_code == 200 and r.json()["mark"]["status"] == "not_ok"
    alt = client.get("/api/video/films/demo/shots/1/alternatives").json()
    assert [o["id"] for o in alt["options"]] == ["c2", "c3", "v1#1"]
    chosen = client.post("/api/video/films/demo/shots/1/choose", json={"choice": "c3"})
    assert chosen.status_code == 200 and chosen.json()["picture"]["id"] == "c3"
    refused = client.post("/api/video/films/demo/shots/1/choose", json={"choice": "bad"})
    assert refused.status_code == 400


def test_rebuild_render_and_busy(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    fake = FakeJobs()
    monkeypatch.setattr(api, "jobs", fake)
    assert client.post("/api/video/films/demo/rebuild").status_code == 202
    assert client.post("/api/video/films/demo/render", json={"sound": True}).status_code == 409
    assert fake.started == [("demo", "rebuild", False)]
    assert client.get("/api/video/films/demo/job").json()["state"] == "running"


def test_files_are_served_inside_the_film_only(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    assert client.get("/api/video/films/demo/file/library/c1.jpg").status_code == 200
    assert client.get("/api/video/films/demo/file/../storyboard.json").status_code in (403, 404)
    assert client.get("/api/video/films/demo/file/%2E%2E/%2E%2E/app.db").status_code in (403, 404)
    assert client.get("/api/video/films/demo/thumb/c1.jpg").status_code == 200
    assert client.get("/api/video/films/demo/frame/1.jpg").status_code == 404     # no review cut yet
```

- [ ] **Step 2: Run them to see them fail**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_api.py`
Expected: `ModuleNotFoundError: No module named 'app.video_api'`.

- [ ] **Step 3: Implement** `app/video_api.py`:

```python
"""HTTP endpoints for the storyboard review (Video tab), under /api/video."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import video_review as review
from .config import settings
from .video_jobs import JobBusy, Jobs, NoSpace

router = APIRouter(prefix="/api/video")
jobs = Jobs()


class MarkIn(BaseModel):
    status: Literal["ok", "not_ok"] | None = None
    note: str = ""


class ChoiceIn(BaseModel):
    choice: str


class RenderIn(BaseModel):
    sound: bool = False


def video_root() -> Path:
    return settings.data_dir / "video"


def film_or_404(film: str) -> Path:
    try:
        return review.film_dir(video_root(), film)
    except review.FilmError:
        raise HTTPException(404, "Film not found")


@router.get("/films")
def films() -> dict:
    return {"films": review.list_films(video_root())}


@router.get("/films/{film}")
def film_view(film: str) -> dict:
    d = film_or_404(film)
    return review.film_view(d) | {"job": jobs.status(d)}


@router.get("/films/{film}/shots/{n}/alternatives")
def alternatives(film: str, n: int) -> dict:
    try:
        return review.alternatives(film_or_404(film), n)
    except review.FilmError as exc:
        raise HTTPException(404, str(exc))


@router.put("/films/{film}/shots/{n}/mark")
def mark(film: str, n: int, payload: MarkIn) -> dict:
    try:
        return {"shot": n, "mark": review.set_mark(film_or_404(film), n, payload.status, payload.note)}
    except review.FilmError as exc:
        raise HTTPException(400, str(exc))


@router.post("/films/{film}/shots/{n}/choose")
def choose(film: str, n: int, payload: ChoiceIn) -> dict:
    try:
        return review.choose(film_or_404(film), n, payload.choice)
    except review.FilmError as exc:
        raise HTTPException(400, str(exc))


@router.post("/films/{film}/rebuild", status_code=202)
def rebuild(film: str) -> dict:
    try:
        return jobs.start(film_or_404(film), "rebuild")
    except JobBusy as exc:
        raise HTTPException(409, str(exc))


@router.post("/films/{film}/render", status_code=202)
def render(film: str, payload: RenderIn | None = None) -> dict:
    try:
        return jobs.start(film_or_404(film), "render", sound=bool(payload and payload.sound))
    except (JobBusy, NoSpace) as exc:
        raise HTTPException(409, str(exc))


@router.get("/films/{film}/job")
def job(film: str) -> dict:
    return jobs.status(film_or_404(film)) or {"state": "none"}


@router.get("/films/{film}/frame/{n}.jpg")
def frame(film: str, n: int) -> FileResponse:
    try:
        path = review.frame(film_or_404(film), n)
    except review.FilmError:
        path = None
    if not path:
        raise HTTPException(404, "No frame yet: rebuild the review cut")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/films/{film}/thumb/{item_id}.jpg")
def thumb(film: str, item_id: str) -> FileResponse:
    path = review.thumb(film_or_404(film), item_id)
    if not path:
        raise HTTPException(404, "No picture")
    return FileResponse(path)


@router.get("/films/{film}/file/{path:path}")
def film_file(film: str, path: str) -> FileResponse:
    d = film_or_404(film)
    parts = Path(path).parts
    if not parts or Path(path).is_absolute() or ".." in parts:
        raise HTTPException(403, "Outside the film")
    target = d.joinpath(*parts)
    if not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target)
```

In `app/main.py`, add the import next to the other local imports and include the router right after the two `app.mount(...)` lines:
```python
from .video_api import router as video_router
```
```python
app.include_router(video_router)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q tests/test_video_api.py tests/test_api.py`
Expected: all pass (the voice API's own tests unaffected).

---

### Task 5: The Video tab

**Files:**
- Modify: `app/static/index.html`
- Create: `app/static/video.js`
- Modify: `app/static/style.css`

Independent of Tasks 1–4 except for the API contract above (endpoints and JSON shapes from Task 2's `film_view`, `shot_view`, `alternatives`, Task 3's job status).

- [ ] **Step 1: Tabs and markup.** In `app/static/index.html`:

1. Directly after `</header>` add:
```html
  <nav class="tabs">
    <button class="tab active" data-tab="voice">Voice</button>
    <button class="tab" data-tab="video">Video</button>
  </nav>
```
2. Change the existing `<main>` opening tag to `<main id="voice-tab">`.
3. After the voice `</main>` closing tag add:
```html
  <main id="video-tab" class="hidden">
    <section class="card" id="films-card">
      <h2>Films</h2>
      <p class="muted">Films prepared with <code>documentary.py prepare</code>. Open one to review its shots.</p>
      <div id="films" class="project-list"></div>
    </section>

    <section class="card hidden" id="film-card">
      <div class="section-head">
        <div><p class="eyebrow">Storyboard review</p><h2 id="film-title"></h2></div>
        <button id="film-close" class="secondary">All films</button>
      </div>
      <video id="player" controls preload="metadata"></video>
      <p id="film-summary" class="muted"></p>
      <div class="actions" id="filters"></div>
      <div class="actions">
        <button id="rebuild">Rebuild review cut</button>
        <label class="inline"><input type="checkbox" id="render-sound"> Mix music and effects</label>
        <button id="render" class="secondary">Render final film</button>
        <code id="render-command" class="muted"></code>
      </div>
      <div id="job" class="progress-wrap hidden">
        <div class="progress"><div id="job-bar"></div></div>
        <p id="job-text" class="muted"></p>
        <pre id="job-tail" class="job-tail" hidden></pre>
      </div>
      <p id="film-error" class="error" hidden></p>
      <div id="scenes"></div>
    </section>
  </main>
```
4. After `<script src="/static/app.js"></script>` add `<script src="/static/video.js"></script>`.

- [ ] **Step 2: Create `app/static/video.js`:**

```javascript
/* Video tab: review a film's storyboard shot by shot, swap pictures, rebuild the review cut, render. */
(() => {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const enc = encodeURIComponent;
  const api = async (url, options = {}) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      let message = response.statusText;
      try { message = (await response.json()).detail || message; } catch (_) {}
      throw new Error(message);
    }
    return response.json();
  };
  const send = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body ?? {}) });
  const LABELS = { actual: "Actual", "stand-in": "Stand-in", illustration: "Illustration", graphic: "Graphic", missing: "Missing" };
  const FILTERS = [["all", "All"], ["not-actual", "Not actual only"], ["illustration", "Illustration"], ["stand-in", "Stand-in"],
                   ["graphic", "Graphic"], ["not-ok", "Marked not OK"], ["unreviewed", "Not reviewed"]];

  let film = null, view = null, filter = "all", jobTimer = null;
  const base = () => `/api/video/films/${enc(film)}`;
  const fileUrl = (path) => `${base()}/file/${path.split("/").map(enc).join("/")}`;
  const clock = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;

  function showError(message) { $("film-error").textContent = message || ""; $("film-error").hidden = !message; }

  /* ------------------------------------------------------------- tabs */
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("voice-tab").classList.toggle("hidden", tab.dataset.tab !== "voice");
    $("video-tab").classList.toggle("hidden", tab.dataset.tab !== "video");
    if (tab.dataset.tab === "video" && !film) loadFilms();
  }));

  /* ------------------------------------------------------------ films */
  async function loadFilms() {
    const { films } = await api("/api/video/films");
    $("films").innerHTML = films.length ? films.map((f) => `
      <div class="project-row">
        <strong>${esc(f.film)}</strong>
        <span class="muted">${f.shots} shots · ${f.minutes} min · reviewed ${f.reviewed}/${f.shots}${f.not_ok ? ` · ${f.not_ok} not OK` : ""}</span>
        <span class="pill">${f.final ? "final film ready" : f.animatic ? "review cut ready" : "no review cut yet"}</span>
        <button data-open="${esc(f.film)}">Open</button>
      </div>`).join("") : `<p class="muted">No films yet.</p>`;
  }
  $("films").addEventListener("click", (e) => { const name = e.target.dataset.open; if (name) openFilm(name); });
  $("film-close").addEventListener("click", () => { film = null; $("film-card").classList.add("hidden"); $("films-card").classList.remove("hidden"); loadFilms(); });

  async function openFilm(name) {
    film = name;
    $("films-card").classList.add("hidden");
    $("film-card").classList.remove("hidden");
    await refresh();
  }

  async function refresh() {
    view = await api(base());
    $("film-title").textContent = view.film;
    const src = view.final ? fileUrl(view.final) : view.animatic ? fileUrl("animatic.mp4") : "";
    const player = $("player");
    if (src && !player.src.endsWith(src + `?v=${view.animatic_version}`)) player.src = src + `?v=${view.animatic_version}`;
    player.hidden = !src;
    const s = view.summary;
    $("film-summary").textContent = `${s.shots} shots: ${s.actual} actual · ${s["stand-in"]} stand-in · ${s.illustration} illustration · ${s.graphic} graphic`
      + (s.missing ? ` · ${s.missing} missing` : "") + ` — reviewed ${s.reviewed}/${s.shots}` + (s.not_ok ? `, ${s.not_ok} not OK` : "")
      + (view.animatic ? "" : " — no review cut yet: rebuild it to see the frames");
    $("render-command").textContent = `${view.render_command} [--sound]`;
    $("filters").innerHTML = FILTERS.map(([key, label]) => `<button class="${key === filter ? "" : "secondary"}" data-filter="${key}">${label}</button>`).join("");
    renderScenes();
    showJob(view.job);
  }

  $("filters").addEventListener("click", (e) => { if (e.target.dataset.filter) { filter = e.target.dataset.filter; refresh(); } });

  /* ------------------------------------------------------------ shots */
  function visible(shot) {
    if (filter === "all") return true;
    if (filter === "not-actual") return shot.category !== "actual";
    if (filter === "not-ok") return shot.mark?.status === "not_ok";
    if (filter === "unreviewed") return !shot.mark;
    return shot.category === filter;
  }

  function sourceImage(shot) {
    if (shot.picture?.media === "clip") return `${base()}/thumb/${enc(shot.picture.id)}.jpg`;
    if (shot.picture?.file) return fileUrl(shot.picture.file);
    if (shot.asset && /\.(png|jpe?g|webp)$/i.test(shot.asset)) return fileUrl(shot.asset);
    return "";
  }

  function shotHtml(shot) {
    const frame = shot.frame_current ? `<img loading="lazy" src="${base()}/frame/${shot.n}.jpg?v=${view.animatic_version}" alt="frame">`
      : `<div class="placeholder">${view.animatic ? "Changed: rebuild the review cut to see this frame" : "No review cut yet"}</div>`;
    const src = sourceImage(shot);
    const p = shot.picture;
    const details = [
      p ? `<div><strong>Picture:</strong> ${esc(p.caption)} <span class="muted">(${esc(p.source || "")}${p.licence ? `, ${esc(p.licence)}` : ""})</span>${p.source_url ? ` <a href="${esc(p.source_url)}" target="_blank" rel="noopener">source</a>` : ""}</div>` : "",
      shot.illustration ? `<div><strong>Drawn:</strong> ${esc(shot.illustration)}</div>` : "",
      shot.key_fact ? `<div><strong>Label:</strong> ${esc(shot.key_fact.toUpperCase())}</div>` : "",
      shot.reason ? `<div class="muted"><strong>Why:</strong> ${esc(shot.reason)}</div>` : "",
      shot.check?.problem ? `<div class="warn"><strong>Check:</strong> ${esc(shot.check.problem)}</div>` : "",
      shot.owner ? `<div class="muted">Your choice</div>` : "",
      shot.quote ? `<div class="muted">A quote card is shown over this shot</div>` : "",
    ].join("");
    const mark = shot.mark?.status;
    return `
      <div class="shot ${mark === "not_ok" ? "shot-bad" : mark === "ok" ? "shot-ok" : ""}" data-n="${shot.n}">
        <div class="shot-media">${frame}${src ? `<img loading="lazy" class="source" src="${src}" alt="source picture">` : ""}</div>
        <div class="shot-body">
          <div class="shot-head">
            <button class="secondary play" data-play="${shot.start}">▶ ${clock(shot.start)}–${clock(shot.end)}</button>
            <span class="badge badge-${shot.category}">${LABELS[shot.category]}</span>
            <span class="muted">${esc(shot.kind)}</span>
          </div>
          <p class="narration">“${esc(shot.narration)}”</p>
          ${details}
          <div class="actions">
            <button class="${mark === "ok" ? "" : "secondary"}" data-mark="ok">✓ OK</button>
            <button class="${mark === "not_ok" ? "danger" : "secondary"}" data-mark="not_ok">✗ Not OK</button>
            ${mark === "not_ok" ? `<input class="note" placeholder="What is wrong?" value="${esc(shot.mark.note)}"><button class="secondary" data-alternatives>Choose another…</button>` : ""}
          </div>
          <div class="alternatives" hidden></div>
        </div>
      </div>`;
  }

  function renderScenes() {
    $("scenes").innerHTML = view.scenes.map((sc) => {
      const shots = sc.shots.filter(visible);
      if (!shots.length) return "";
      return `<div class="scene"><h3>${esc(sc.title)} <span class="muted">${esc(sc.story_time)} · ${esc(sc.look)} · ${clock(sc.start || 0)}</span></h3>${shots.map(shotHtml).join("")}</div>`;
    }).join("") || `<p class="muted">No shots match this filter.</p>`;
  }

  $("scenes").addEventListener("click", async (e) => {
    const row = e.target.closest(".shot");
    if (!row) return;
    const n = row.dataset.n;
    const t = e.target;
    try {
      showError("");
      if (t.dataset.play) { const player = $("player"); player.currentTime = Number(t.dataset.play); player.play(); return; }
      if (t.dataset.mark) {
        const current = view.scenes.flatMap((s) => s.shots).find((s) => String(s.n) === n)?.mark?.status;
        const status = current === t.dataset.mark ? null : t.dataset.mark;
        await api(`${base()}/shots/${n}/mark`, send("PUT", { status, note: row.querySelector(".note")?.value || "" }));
        await refresh();
        return;
      }
      if (t.dataset.alternatives !== undefined) { await showAlternatives(row, n); return; }
      if (t.dataset.choose) {
        t.disabled = true;
        t.textContent = t.dataset.choose === "illustration" ? "Drawing…" : "Using…";
        await api(`${base()}/shots/${n}/choose`, send("POST", { choice: t.dataset.choose }));
        await refresh();
      }
    } catch (error) { showError(error.message); await refresh().catch(() => {}); }
  });

  $("scenes").addEventListener("change", async (e) => {
    if (!e.target.classList.contains("note")) return;
    const n = e.target.closest(".shot").dataset.n;
    try { await api(`${base()}/shots/${n}/mark`, send("PUT", { status: "not_ok", note: e.target.value })); } catch (error) { showError(error.message); }
  });

  async function showAlternatives(row, n) {
    const box = row.querySelector(".alternatives");
    box.hidden = false;
    box.innerHTML = `<p class="muted">Loading…</p>`;
    const alt = await api(`${base()}/shots/${n}/alternatives`);
    const options = alt.options.map((o) => `
      <div class="option">
        <img loading="lazy" src="${o.media === "clip" ? `${base()}/thumb/${enc(o.id)}.jpg` : fileUrl(o.file)}" alt="">
        <span class="badge badge-${o.category}">${LABELS[o.category]}</span>
        <span class="muted">${o.media === "clip" ? "footage · " : ""}${esc(o.caption)}</span>
        <button data-choose="${esc(o.id)}">Use this</button>
      </div>`).join("");
    const draw = alt.illustration.allowed ? `<div class="option"><div class="placeholder">New illustration</div><span class="muted">An object or setting matching the words (never a person or a named place), labelled ILLUSTRATION.</span><button data-choose="illustration">Draw one</button></div>`
      : `<p class="muted">${esc(alt.illustration.why)}</p>`;
    box.innerHTML = (options || `<p class="muted">No other pictures were found for this shot.</p>`) + draw;
  }

  /* -------------------------------------------------------------- jobs */
  function showJob(job) {
    clearTimeout(jobTimer);
    const running = job && job.state === "running";
    $("rebuild").disabled = running;
    $("render").disabled = running;
    if (!job || job.state === "none") { $("job").classList.add("hidden"); return; }
    $("job").classList.remove("hidden");
    $("job-bar").style.width = `${Math.round((job.progress || 0) * 100)}%`;
    const what = job.kind === "render" ? "Final render" : "Review cut";
    const words = { running: "running", done: "done", failed: "failed", stopped: "stopped before finishing" }[job.state] || job.state;
    $("job-text").textContent = `${what}: ${words} (${Math.round((job.progress || 0) * 100)}%)` + (job.tail?.length && job.state === "running" ? ` — ${job.tail[job.tail.length - 1]}` : "");
    $("job-tail").hidden = !(job.state === "failed" || job.state === "stopped");
    $("job-tail").textContent = (job.tail || []).join("\n");
    if (running) jobTimer = setTimeout(async () => {
      try {
        const next = await api(`${base()}/job`);
        if (next.state !== "running") await refresh(); else showJob(next);
      } catch (error) { showError(error.message); }
    }, 3000);
  }

  $("rebuild").addEventListener("click", async () => {
    try { showError(""); showJob(await api(`${base()}/rebuild`, send("POST"))); } catch (error) { showError(error.message); }
  });
  $("render").addEventListener("click", async () => {
    try { showError(""); showJob(await api(`${base()}/render`, send("POST", { sound: $("render-sound").checked }))); } catch (error) { showError(error.message); }
  });
})();
```

- [ ] **Step 3: Styles.** Append to `app/static/style.css`:

```css
.tabs { display:flex; gap:8px; width:min(1040px, calc(100% - 32px)); margin:20px auto 0; }
.tab { background:#e7edf3; color:var(--ink); }
.tab.active { background:var(--blue); color:white; }
.hidden { display:none !important; }
#player { width:100%; max-height:420px; background:#000; border-radius:12px; margin-bottom:12px; }
.scene { margin-top:22px; }
.scene h3 { margin:0 0 6px; font-size:1.05rem; }
.shot { display:grid; grid-template-columns:320px 1fr; gap:16px; border-top:1px solid var(--line); padding:14px 0; }
.shot-ok { box-shadow:inset 4px 0 0 var(--ok); padding-left:10px; }
.shot-bad { box-shadow:inset 4px 0 0 #c0362c; padding-left:10px; }
.shot-media { display:grid; gap:6px; }
.shot-media img { width:100%; border-radius:8px; background:#0b1d33; }
.shot-media img.source { width:45%; opacity:.9; }
.placeholder { display:grid; place-items:center; min-height:150px; border-radius:8px; background:#e5ecf2; color:var(--muted); text-align:center; padding:12px; }
.shot-head { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
.play { padding:6px 10px; }
.narration { margin:8px 0; font-style:italic; }
.badge { padding:3px 9px; border-radius:999px; font-size:.78rem; font-weight:800; color:white; }
.badge-actual { background:#087443; }
.badge-stand-in { background:#a15c00; }
.badge-illustration { background:#6941c6; }
.badge-graphic { background:#5d6b78; }
.badge-missing { background:#c0362c; }
.warn { color:var(--warn); }
button.danger { background:#c0362c; }
.note { max-width:280px; }
.alternatives { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); gap:12px; margin-top:10px; }
.option { display:grid; gap:6px; align-content:start; border:1px solid var(--line); border-radius:10px; padding:8px; }
.option img { width:100%; border-radius:6px; }
.job-tail { background:#0b1d33; color:#c7d5e4; padding:10px; border-radius:8px; overflow:auto; max-height:220px; }
@media (max-width: 720px) { .shot { grid-template-columns:1fr; } }
```

If `style.css` already defines `.hidden`, keep one definition (with `display:none !important`).

- [ ] **Step 4: Check the page loads without errors.** Start the app and open the Video tab:

```bash
set -a && source .env && set +a && .venv/bin/python -m uvicorn app.main:app --port 8000
```
Open http://localhost:8000, click **Video**: the films list loads; the Voice tab still works. Stop the server afterwards.

---

### Task 6: Real check on the Bundy test film, then the owner reviews the diff

- [ ] **Step 1:** Make the test film visible to the app (a link, nothing copied):
```bash
ln -s /Users/shalikanuwan/.claude/jobs/4a7c1675/tmp/second_case/film data/video/bundy_test
```
- [ ] **Step 2:** Run the whole test suite:
```bash
.venv/bin/python -m pytest -p no:warnings -o addopts="" -q
```
Expected: all pass (previous 1081 plus the new tests).
- [ ] **Step 3:** Start the app (Task 5 Step 4 command), open Video → bundy_test. Check: the summary counts; frames show; ▶ seeks the player; mark a shot not OK, **Choose another…**, use another picture; the shot shows "Changed: rebuild…"; **Rebuild review cut** runs to done; the frame now shows the new picture.
- [ ] **Step 4:** Remove the link (`rm data/video/bundy_test`), stop the server, and show the owner `git status` and `git diff --stat`. Do not commit.

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import critics  # noqa: E402

BIBLE = {"people": [
    {"name": "John Wayne Gacy", "victim": False, "photo_ids": ["c0011"], "aliases": ["Gacy"]},
    {"name": "Robert Jerome Piest", "victim": True, "photo_ids": ["c0037"], "aliases": ["Rob"]},
    {"name": "Robert Schultz", "victim": False, "photo_ids": [], "aliases": ["Schultz"]}]}


def test_a_beat_that_names_one_person_but_shows_anothers_face_is_flagged():
    beats = [{"index": 1, "start": 0, "library_id": "c0011", "narration": "Rob told her to wait."},
             {"index": 2, "start": 9, "library_id": "c0011", "narration": "Gacy and Rob spoke outside."},
             {"index": 3, "start": 18, "library_id": "c0037", "narration": "Rob never came back."}]
    issues = critics.continuity(beats, BIBLE)
    assert [i["beat"] for i in issues] == [1] and "Robert Jerome Piest" in issues[0]["note"]


def test_tense_scenes_that_hold_too_long_and_rushed_memorials_are_flagged():
    beats = [{"index": i, "start": i * 16.0, "end": (i + 1) * 16.0} for i in range(3)]
    beats += [{"index": 10 + i, "start": 100 + i * 3.0, "end": 103 + i * 3.0} for i in range(3)]
    scenes = [{"id": 1, "start": 0, "intensity": 9, "look": "revelation", "beats": [0, 1, 2]},
              {"id": 2, "start": 100, "intensity": 7, "look": "memorial", "beats": [10, 11, 12]},
              {"id": 3, "start": 200, "intensity": 4, "look": "investigation", "beats": [0, 1]}]
    assert [(i["scene"], "memorial" in i["note"]) for i in critics.rhythm(scenes, beats)] == [(1, False), (2, True)]


def test_a_picture_back_within_a_minute_is_flagged_but_not_after():
    beats = [{"index": 1, "start": 0, "end": 10, "library_id": "a"},
             {"index": 2, "start": 40, "end": 50, "library_id": "a"},
             {"index": 3, "start": 200, "end": 210, "library_id": "a"},
             {"index": 4, "start": 215, "end": 220, "library_id": "b", "media": "motion"}]
    assert [i["beat"] for i in critics.repetition(beats)] == [2]

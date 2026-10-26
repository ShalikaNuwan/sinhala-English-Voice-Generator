from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import story_bible  # noqa: E402


def sentences(n: int, length: float = 5.0):
    return [{"index": i, "text": f"Sentence {i}.", "start": i * length, "end": (i + 1) * length} for i in range(n)]


def test_spans_cover_the_range_once_in_order():
    spans = story_bible.cover_range([(0, 3, "a"), (4, 9, "b")], 0, 9)
    assert [(a, b, o) for a, b, o in spans] == [(0, 3, "a"), (4, 9, "b")]


def test_an_overlap_is_trimmed_from_the_later_span_and_a_gap_goes_to_the_span_before():
    spans = story_bible.cover_range([(0, 5, "a"), (3, 6, "b"), (9, 12, "c")], 0, 12)
    assert [(a, b, o) for a, b, o in spans] == [(0, 5, "a"), (6, 8, "b"), (9, 12, "c")]


def test_a_late_first_span_starts_at_the_beginning_and_the_last_runs_to_the_end():
    spans = story_bible.cover_range([(2, 4, "a"), (5, 7, "b")], 0, 10)
    assert [(a, b) for a, b, _ in spans] == [(0, 4), (5, 10)]


def test_no_usable_span_still_covers_the_range():
    assert story_bible.cover_range([(20, 30, "x")], 0, 9) == [(0, 9, None)]


def make_bible(chapters, people=()):
    return story_bible.Bible(
        logline="A story.", people=list(people), places=[], timeline=[
            story_bible.Event(date="1978", what="It happened.", sentence=3),
            story_bible.Event(date="2000", what="Out of range.", sentence=999)],
        chapters=chapters, motifs=[], sensitivities=["Never show remains."])


def chapter(first, last, **kw):
    base = dict(title="T", first=first, last=last, purpose="p", tone="t", intensity=5, era="1978", look="investigation")
    base.update(kw)
    return story_bible.Chapter(**base)


def test_tidy_makes_the_chapters_whole_and_times_them():
    data = story_bible.tidy(make_bible([chapter(0, 4), chapter(7, 9, intensity=14)]), sentences(12), set())
    assert [(c["id"], c["first"], c["last"]) for c in data["chapters"]] == [(1, 0, 6), (2, 7, 11)]
    assert data["chapters"][1]["start"] == 35.0 and data["chapters"][1]["end"] == 60.0
    assert data["chapters"][1]["intensity"] == 10          # clamped to the scale


def test_tidy_keeps_only_photograph_ids_the_library_has_and_events_inside_the_film():
    person = story_bible.Person(name="John Wayne Gacy", role="the killer", importance=3, photo_ids=["c0011", "zzz"])
    data = story_bible.tidy(make_bible([chapter(0, 9)], [person]), sentences(10), {"c0011"})
    assert data["people"][0]["photo_ids"] == ["c0011"]
    assert [e["sentence"] for e in data["timeline"]] == [3]


def test_the_library_index_lists_people_and_places_with_their_photographs():
    items = [{"id": "c0011", "subjects": ["John Wayne Gacy"], "places": ["Des Plaines"]},
             {"id": "c0018", "subjects": ["John Wayne Gacy"], "places": []}]
    text = story_bible.library_index(items)
    assert "John Wayne Gacy: c0011, c0018" in text and "Des Plaines: c0011" in text


def test_the_narration_is_numbered_with_minute_marks():
    text = story_bible.numbered([{"index": 0, "text": "Think about it.", "start": 0.0},
                                 {"index": 1, "text": "Later.", "start": 125.0}])
    assert text.splitlines() == ["(0) [0:00] Think about it.", "(1) [2:05] Later."]

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import sentences  # noqa: E402


def test_the_script_splits_into_sentences_and_drops_break_tags():
    text = 'Think about it. <break time="0.4s" />\nHe said, “I know.” Then he left! Was it 1978? Yes.\n'
    assert sentences.script_sentences(text) == ["Think about it.", "He said, “I know.”", "Then he left!", "Was it 1978?", "Yes."]


def test_an_abbreviation_mid_sentence_is_not_a_sentence_end_when_a_lower_case_word_follows():
    assert sentences.script_sentences("He was born in Chicago, Ill. and grew up there.") == [
        "He was born in Chicago, Ill. and grew up there."]


def words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_sentences_take_their_first_word_and_butt_against_each_other():
    heard = words(("Think", 0.2, 0.5), ("about", 0.5, 0.8), ("it", 0.8, 1.0), ("He", 2.0, 2.2), ("left", 2.2, 2.6))
    timed = sentences.time_sentences(["Think about it.", "He left."], heard, duration=4.0)
    assert [(s["start"], s["end"]) for s in timed] == [(0.0, 2.0), (2.0, 4.0)]
    assert timed[0]["matched"] == 1.0


def test_an_unheard_sentence_borrows_the_gap_and_never_runs_backwards():
    heard = words(("One", 0.0, 0.5), ("Three", 3.0, 3.4))
    timed = sentences.time_sentences(["One.", "Two.", "Three."], heard, duration=5.0)
    starts = [s["start"] for s in timed]
    assert starts == sorted(starts) and timed[1]["matched"] == 0.0


def test_a_time_inside_a_shortened_pause_lands_proportionally_in_the_new_cut():
    old = words(("a", 0.0, 0.4), ("b", 10.0, 10.4), ("c", 20.0, 20.4))
    new = words(("a", 0.0, 0.4), ("b", 6.0, 6.4), ("c", 12.0, 12.4))   # every pause shortened
    points = sentences.time_map(old, new)
    assert sentences.map_time(5.0, points) == 3.0
    assert sentences.map_time(15.0, points) == 9.0
    assert sentences.map_time(25.0, points) == 17.0  # past the end: carried at the same rate as the tail


def test_a_middle_initial_does_not_end_a_sentence():
    assert sentences.script_sentences("Michael M. Marino, 14 years old, on October 24th. William Bundy, 19.") == [
        "Michael M. Marino, 14 years old, on October 24th.", "William Bundy, 19."]

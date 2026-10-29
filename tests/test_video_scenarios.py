from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import quotes  # noqa: E402
import scenarios  # noqa: E402

SENTENCE = {"index": 7, "text": "Darrel Julius Samson, 18 years old, on April 6th. ", "start": 10.0, "end": 14.0}


def entry(**kw):
    base = dict(kind="person", name="Darrel Julius Samson", age="18", date="April 6th", sentence=7)
    base.update(kw)
    return scenarios.RollEntry(**base)


def test_an_entry_is_kept_only_if_every_field_on_its_card_is_in_the_narration():
    assert scenarios.stated(entry(), {7: SENTENCE})
    assert not scenarios.stated(entry(age="19"), {7: SENTENCE})
    assert not scenarios.stated(entry(date="April 7th"), {7: SENTENCE})
    assert not scenarios.stated(entry(name="Darrel Samsonite"), {7: SENTENCE})
    assert not scenarios.stated(entry(sentence=8), {7: SENTENCE})


def test_names_take_the_spelling_the_bible_or_library_uses_when_it_clearly_matches():
    known = ["Michael Bonnin", "James Haakenson", "John Prestidge", "Michael Rossi", "Darrell Julius Sampson"]
    # Only the misspelled parts change; the narrator's middle names and suffix stay.
    assert scenarios.canonical("Michael Lawrence Bronin", known) == "Michael Lawrence Bonnin"
    assert scenarios.canonical("James Byron Harkinson", known) == "James Byron Haakenson"
    assert scenarios.canonical("John Steven Prestige", known) == "John Steven Prestidge"
    assert scenarios.canonical("Kenneth Ray Parker", known) == "Kenneth Ray Parker"     # nobody close: as narrated
    assert scenarios.canonical("Darrel Julius Samson", known) == "Darrell Julius Sampson"
    assert scenarios.canonical("William Huey Carroll Jr.", known + ["William Carroll"]) == "William Huey Carroll Jr."


def test_a_card_cuts_in_when_its_name_is_spoken():
    sentences = [SENTENCE]
    tokens = quotes.script_tokens(sentences)
    heard = [{"word": w, "start": 10.4 + n * 0.3, "end": 10.6 + n * 0.3} for n, w in enumerate(SENTENCE["text"].split())]
    times = quotes.token_times(tokens, heard)
    assert scenarios.spoken_at(entry(), SENTENCE, tokens, times) == 10.4


def test_cards_are_drawn_for_people_unidentified_victims_and_year_headings():
    for e in ({"kind": "person", "display": "Darrel Julius Samson", "age": "18", "date": "April 6th", "year": "1976"},
              {"kind": "unidentified", "display": "Unidentified male, body number 5", "date": "between June and August"},
              {"kind": "heading", "year": "1977", "count": "ten people"}):
        img = scenarios.card_image(e, None, 640, 360)
        assert img.size == (640, 360) and img.getbbox() is not None


def test_a_roll_call_scene_keeps_its_lead_in_and_cuts_a_card_at_each_name(tmp_path):
    scene = {"id": 5, "chapter": 2, "start": 0.0, "end": 30.0, "intensity": 6}
    old = [{"index": 1, "start": 0.0, "end": 9.0, "kind": "period", "scene": 5},
           {"index": 2, "start": 9.0, "end": 30.0, "kind": "period", "scene": 5}]
    entries = [{"kind": "heading", "year": "1976", "count": "15 people", "at": 6.0},
               {"kind": "person", "display": "Darrel Julius Samson", "age": "18", "date": "April 6th", "year": "1976", "at": 10.4},
               {"kind": "unidentified", "display": "Unidentified male, body number 5", "at": 20.0}]
    sentences = [{"index": i, "text": f"S{i}.", "start": i * 5.0, "end": (i + 1) * 5.0} for i in range(6)]
    beats = scenarios.roll_call_beats(scene, old, entries, sentences, tmp_path, {})
    assert [(b["start"], b["end"], b["kind"]) for b in beats] == [
        (0.0, 6.0, "period"), (6.0, 10.4, "card"), (10.4, 20.0, "card"), (20.0, 30.0, "card")]
    assert all(b["look"] == "memorial" and (tmp_path / b["asset"]).exists() for b in beats[1:])
    assert [b["transition_in"] for b in beats[1:]] == ["dissolve", "cut", "cut"]      # never name into name


def test_a_misheard_name_still_cuts_in_at_its_sentence():
    sentences = [SENTENCE]
    tokens = quotes.script_tokens(sentences)
    heard = [{"word": "Daryl", "start": 10.3, "end": 10.6}] + [
        {"word": w, "start": 10.7 + n * 0.3, "end": 10.9 + n * 0.3} for n, w in enumerate(SENTENCE["text"].split()[1:])]
    times = quotes.token_times(tokens, heard)
    assert scenarios.spoken_at(entry(), SENTENCE, tokens, times) == 10.7
    assert scenarios.spoken_at(entry(), SENTENCE, tokens, [None] * len(tokens)) == 10.0


def test_a_name_cut_at_an_initial_is_checked_across_the_split():
    split = {3: {"index": 3, "text": "Michael M.", "start": 0, "end": 1},
             4: {"index": 4, "text": "Marino, 14 years old, on October 24th.", "start": 1, "end": 4}}
    assert scenarios.stated(entry(name="Michael M. Marino", age="14", date="October 24th", sentence=4), split)


def test_approximately_timed_entries_are_spread_between_their_exact_neighbours():
    entries = [{"at": 0.0, "exact": True}, {"at": 3.0, "exact": False}, {"at": 3.0, "exact": False},
               {"at": 3.2, "exact": False}, {"at": 20.0, "exact": True}]
    scenarios.spread(entries, end=30.0)
    assert [e["at"] for e in entries] == [0.0, 5.0, 10.0, 15.0, 20.0]


def test_cards_write_dates_counts_and_unidentified_victims_one_consistent_way():
    assert scenarios.tidy_date("April 6th") == "April 6" and scenarios.tidy_date("between June 16th and 23rd") == "between June 16 and 23"
    assert scenarios.tidy_count("ten people") == "10 people" and scenarios.tidy_count("15 people") == "15 people"
    assert scenarios.split_unidentified("Unidentified male, body number 26") == ("Unidentified victim", "Body No. 26")


def test_a_register_shows_portraits_only_when_every_named_person_has_one(tmp_path):
    from PIL import Image
    Image.new("RGB", (400, 500), (90, 90, 90)).save(tmp_path / "p.jpg")
    library = {"p1": {"id": "p1", "file": "p.jpg"}}
    scene = {"id": 1, "chapter": 1, "start": 0.0, "end": 20.0, "intensity": 4}
    sentences = [{"index": 0, "start": 0, "end": 10, "text": "Ann Lee, 18."}, {"index": 1, "start": 10, "end": 20, "text": "Bo Ray, 19."}]

    def entries(photo_b):
        return [{"kind": "person", "display": "Ann Lee", "photo": "p1", "at": 0.0, "sentence": 0},
                {"kind": "person", "display": "Bo Ray", "photo": photo_b, "at": 10.0, "sentence": 1}]
    mixed = scenarios.roll_call_beats(scene, [], entries(None), sentences, tmp_path, library)
    whole = scenarios.roll_call_beats(scene, [], entries("p1"), sentences, tmp_path, library)
    centre = lambda beat: Image.open(tmp_path / beat["asset"]).convert("L").getpixel((int(1920 * 0.2), 540))
    assert centre(mixed[0]) < 40          # no portrait on Ann's card, as Bo has none
    assert centre(whole[0]) > 60          # both have one: Ann's portrait is shown


def test_a_heading_spanning_years_is_stated_when_each_year_is():
    sentences = {5: {"index": 5, "text": "After he started killing in 1972, Gacy had killed only three people by 1975."}}
    ok = scenarios.RollEntry(kind="heading", year="1972–1975", count="three people", sentence=5)
    wrong = scenarios.RollEntry(kind="heading", year="1972–1976", count="three people", sentence=5)
    assert scenarios.stated(ok, sentences) and not scenarios.stated(wrong, sentences)

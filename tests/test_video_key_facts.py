import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import key_facts  # noqa: E402
import storyboard  # noqa: E402

SENTENCES = [{"index": 31, "start": 121.6, "end": 124.0, "text": "That day, Rob was wearing a blue jacket."},
             {"index": 36, "start": 138.8, "end": 141.0, "text": "That jacket is important to this story."},
             {"index": 40, "start": 150.0, "end": 152.0, "text": "Rob Piest left the store."}]
BIBLE = {"people": [{"name": "Rob Piest"}], "places": [{"name": "Nisson Pharmacy"}]}


def tokens_and_times():
    from quotes import script_tokens
    tokens = script_tokens(SENTENCES)
    times = [None] * len(tokens)
    for k, t in enumerate(tokens):
        if t["sentence"] == 36 and t["norm"] == "jacket":
            times[k] = (139.3, 139.7)
    return tokens, times


def beat(i, start, end, **kw):
    return {"index": i, "scene": 1, "start": start, "end": end, "media": "photo", "asset": f"p{i}.jpg",
            "narration": f"words {i}", "kind": "archival", **kw}


def test_only_the_narrations_own_words_about_an_object_are_labels():
    by = {s["index"]: s for s in SENTENCES}
    assert key_facts.valid(key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36]), by, BIBLE)
    assert not key_facts.valid(key_facts.KeyFact(label="red jacket", stated_in=31, at=[36]), by, BIBLE)
    assert not key_facts.valid(key_facts.KeyFact(label="Rob Piest", stated_in=40, at=[40]), by, BIBLE)
    assert not key_facts.valid(key_facts.KeyFact(label="the blue jacket he wore", stated_in=31, at=[36]), by, BIBLE)
    # The label may appear only where the object is mentioned.
    assert key_facts.anchors(key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36, 40]), by) == [36]


def test_a_label_cuts_into_the_shot_when_the_word_is_said_and_the_shot_carries_on(tmp_path):
    tokens, times = tokens_and_times()
    entries = [beat(1, 130.0, 145.0), beat(2, 145.0, 150.0)]
    facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
    placed = key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    assert placed == [{"label": "blue jacket", "sentence": 36, "at": 139.3}]
    # The 2.7 s left after the label is too short to cut back to, so the label holds to the end of the shot.
    assert [(e["start"], e["end"], e["kind"]) for e in entries] == [
        (130.0, 139.3, "archival"), (139.3, 145.0, "keyfact"), (145.0, 150.0, "archival")]
    assert (tmp_path / entries[1]["asset"]).exists() and entries[1]["rights"] == "original"
    storyboard.strip_key_facts(entries)
    assert [(e["start"], e["end"]) for e in entries] == [(130.0, 145.0), (145.0, 150.0)]


def test_a_label_never_interrupts_a_quote_card_or_the_register(tmp_path):
    tokens, times = tokens_and_times()
    for extra in ({"quote_ids": [3]}, {"scenario": "roll_call"}):
        entries = [beat(1, 130.0, 145.0, **extra)]
        facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
        assert key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path) == []
        assert len(entries) == 1


def test_a_label_absorbs_a_short_piece_of_shot_rather_than_leave_a_flash(tmp_path):
    tokens, times = tokens_and_times()
    entries = [beat(1, 137.6, 145.4)]          # the word comes 1.7 s in: too little shot to show before it
    facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
    key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    assert [(e["start"], round(e["end"], 2), e["kind"]) for e in entries] == [
        (137.6, 141.3, "keyfact"), (141.3, 145.4, "archival")]

    storyboard.strip_key_facts(entries)       # the label's time goes back to the shot it was cut from
    assert [(e["start"], e["end"]) for e in entries] == [(137.6, 145.4)]


def test_placing_labels_twice_gives_the_same_cut(tmp_path):
    tokens, times = tokens_and_times()
    facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
    entries = [beat(1, 120.0, 137.6), beat(2, 137.6, 145.4)]
    key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    first = [(e["start"], e["end"], e["kind"]) for e in entries]
    storyboard.strip_key_facts(entries)
    key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    assert [(e["start"], e["end"], e["kind"]) for e in entries] == first



def test_a_label_that_took_a_whole_shot_gives_it_back(tmp_path):
    tokens, times = tokens_and_times()
    facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
    entries = [beat(1, 120.0, 138.0), beat(2, 138.0, 143.0), beat(3, 143.0, 150.0)]
    key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    assert [e["kind"] for e in entries] == ["archival", "keyfact", "archival"]   # beat 2 is all label
    storyboard.strip_key_facts(entries)
    assert [(e["start"], e["end"], e["asset"]) for e in entries] == [
        (120.0, 138.0, "p1.jpg"), (138.0, 143.0, "p2.jpg"), (143.0, 150.0, "p3.jpg")]



def test_a_label_near_the_end_of_a_shot_starts_earlier_so_it_can_be_read(tmp_path):
    tokens, times = tokens_and_times()
    entries = [beat(1, 132.0, 140.1), beat(2, 140.1, 150.0)]     # "jacket" at 139.3, 0.8 s before the cut
    facts = [key_facts.KeyFact(label="blue jacket", stated_in=31, at=[36])]
    key_facts.place(entries, facts, SENTENCES, tokens, times, BIBLE, tmp_path)
    label = next(e for e in entries if e["kind"] == "keyfact")
    assert round(label["end"] - label["start"], 2) == key_facts.KEY_FACT_S and label["end"] == 140.1

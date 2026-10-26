from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import quotes  # noqa: E402


def sents(*texts, length=4.0):
    return [{"index": i, "text": t, "start": i * length, "end": (i + 1) * length} for i, t in enumerate(texts)]


def test_a_quote_inside_one_sentence_is_found_with_the_words_before_it():
    spans = quotes.find_spans(sents("Initially, he insisted, “I didn’t do any of this.”"))
    assert [(s["id"], s["text"], s["lead"], s["first"], s["last"]) for s in spans] == [
        ("q001", "I didn’t do any of this.", "Initially, he insisted,", 0, 0)]


def test_a_quote_that_runs_over_two_sentences_is_one_span():
    spans = quotes.find_spans(sents("Rob told her, “Wait here for a little while.", "I will be right back.” Then he left."))
    assert len(spans) == 1
    assert spans[0]["text"] == "Wait here for a little while. I will be right back."
    assert (spans[0]["first"], spans[0]["last"], spans[0]["closed"]) == (0, 1, True)


def test_an_unclosed_quote_is_cut_where_its_first_sentence_ends():
    spans = quotes.find_spans(sents("He said, “Mike, why is this?", "One.", "Two.", "Three.", "Four."))
    assert spans[0]["text"] == "Mike, why is this?" and spans[0]["closed"] is False


def test_straight_quotes_pair_up_and_several_quotes_in_a_sentence_are_separate():
    spans = quotes.find_spans(sents('The ring read "JAS" and the card said "best wishes."'))
    assert [s["text"] for s in spans] == ["JAS", "best wishes."]


def test_card_lines_break_at_words_and_come_out_about_even():
    assert quotes.lines_of("I didn't do any of this, and I don't know anything about it.", width=24) == [
        "I didn't do any of", "this, and I don't know", "anything about it."]


def test_no_line_is_left_holding_one_stray_word():
    lines = quotes.lines_of("Wait here for a little while. I'm just going to meet someone about a job.", width=40)
    assert len(lines) == 2 and min(len(l) for l in lines) > 20


def test_pages_split_evenly_and_never_leave_one_line_alone():
    assert quotes.pages_of(4) == [[0, 1], [2, 3]]
    assert quotes.pages_of(5) == [[0, 1, 2], [3, 4]]
    assert quotes.pages_of(3) == [[0, 1, 2]] and quotes.pages_of(0) == []


def words(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_a_quote_is_timed_to_its_own_words_not_its_sentence():
    sentences = sents("He walked in and said, “I came to meet him.”")
    heard = words(("He", 0.1, 0.3), ("walked", 0.3, 0.6), ("in", 0.6, 0.7), ("and", 0.7, 0.8), ("said", 0.8, 1.1),
                  ("I", 1.5, 1.6), ("came", 1.6, 1.9), ("to", 1.9, 2.0), ("meet", 2.0, 2.3), ("him", 2.3, 2.7))
    span = quotes.find_spans(sentences)[0]
    tokens = quotes.script_tokens(sentences)
    quotes.time_span(span, tokens, quotes.token_times(tokens, heard), sentences)
    assert (span["start"], span["end"], span["word_count"], span["heard"]) == (1.5, 2.7, 5, 1.0)
    assert span["lines"] == [{"text": "I came to meet him.", "at": 1.5}]


def test_a_long_quote_reveals_line_by_line_in_pages():
    text = "one two three four five six seven eight nine ten eleven twelve"
    sentences = sents(f"He said, “{text}.”")
    heard = words(("He", 0.0, 0.2), ("said", 0.2, 0.4)) + words(*[(w, 1.0 + k, 1.5 + k) for k, w in enumerate(text.split())])
    span = quotes.find_spans(sentences)[0]
    tokens = quotes.script_tokens(sentences)
    old = quotes.LINE_CHARS
    quotes.LINE_CHARS = 20
    try:
        quotes.time_span(span, tokens, quotes.token_times(tokens, heard), sentences)
    finally:
        quotes.LINE_CHARS = old
    # Each line appears when its first word is said (word k is said at 1.0 + k seconds).
    spoken = text.split()
    starts = [l["at"] for l in span["lines"]]
    first_words = [l["text"].split()[0].rstrip(".") for l in span["lines"]]
    assert len(span["lines"]) > 1 and starts == [1.0 + spoken.index(w) for w in first_words]
    assert sum(len(p) for p in span["pages"]) == len(span["lines"]) and all(len(p) <= 3 for p in span["pages"])


def call(id, kind="spoken", speaker="A", group="perpetrator", sensitive=False):
    return quotes.QuoteCall(id=id, kind=kind, speaker=speaker, speaker_group=group, sensitive=sensitive)


def span(id, start, end):
    return {"id": id, "start": start, "end": end}


def test_terms_stay_in_the_narration_written_words_become_exhibits_and_sensitive_speech_is_held_back():
    spans = [span("q1", 0, 1), span("q2", 30, 31), span("q3", 60, 61), span("q4", 90, 91)]
    quotes.decide(spans, {"q1": call("q1", kind="term"), "q2": call("q2", kind="written"),
                          "q3": call("q3", sensitive=True), "q4": call("q4", group="family")})
    assert [(s["show"], s["style"]) for s in spans] == [
        ("narration", None), ("card", "exhibit"), ("narration", "card"), ("card", "memorial")]


def test_cards_are_spaced_unless_two_people_answer_each_other():
    spans = [span("q1", 0, 2), span("q2", 4, 6), span("q3", 10, 12), span("q4", 40, 42)]
    quotes.decide(spans, {"q1": call("q1", speaker="Kozenczak", group="police"), "q2": call("q2", speaker="Gacy"),
                          "q3": call("q3", speaker="Gacy"), "q4": call("q4", speaker="Gacy")})
    assert [s["show"] for s in spans] == ["card", "card", "narration", "card"]
    assert spans[2]["held_back"] == "within 20 s of another card"

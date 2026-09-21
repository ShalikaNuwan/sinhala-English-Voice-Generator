from __future__ import annotations

from app import breaks


def phrase(start, end, text="ඇය කියනවා"):
    return {"start_ms": start, "end_ms": end, "text": text}


def test_phrases_that_run_on_form_one_run_and_a_real_pause_ends_it():
    runs = breaks.runs([phrase(0, 8000, "A"), phrase(8100, 15000, "B"), phrase(15600, 30000, "C")], 45000)

    assert runs == [
        {"start_ms": 0, "end_ms": 15000, "text": "A B", "gap_after_ms": 600},
        {"start_ms": 15600, "end_ms": 30000, "text": "C", "gap_after_ms": 15000},
    ]


def test_the_last_gap_reaches_the_next_segments_first_phrase_when_known():
    runs = breaks.runs([phrase(0, 44700)], 45000, next_phrase_start_ms=45300)

    assert runs[-1]["gap_after_ms"] == 600


def test_runs_ignore_empty_phrases_and_sort_by_time():
    runs = breaks.runs([phrase(5000, 6000, "later"), phrase(0, 1000, "first"), phrase(2000, 3000, "   ")], 7000)

    assert [r["text"] for r in runs] == ["first", "later"]


def test_break_table():
    assert breaks.break_seconds(100, "and then,") is None
    assert breaks.break_seconds(200, "and then,") == 0.2
    assert breaks.break_seconds(200, "She never came home.") == 0.4  # a sentence end is lifted to the minimum
    assert breaks.break_seconds(760, "Frightened.”") == 0.8
    assert breaks.break_seconds(2600, "Long Island…") == 2.6
    assert breaks.break_seconds(5000, "anything") == 3.0
    assert breaks.break_seconds(444, "no punctuation") == 0.4


def test_mark_renders_the_elevenlabs_tag():
    assert breaks.mark(0.8) == '<break time="0.8s" />'
    assert breaks.mark(3.0) == '<break time="3.0s" />'


def test_lines_must_cover_every_run_once_in_order():
    assert breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [2, 3]}], 3)
    assert not breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [3]}], 3)
    assert not breaks.covers_runs([{"text": "a", "runs": [2]}, {"text": "b", "runs": [1]}], 2)
    assert not breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [1, 2]}], 2)
    assert not breaks.covers_runs([], 1)


def test_attach_breaks_gives_each_line_the_silence_of_its_last_run():
    run_list = breaks.runs([phrase(0, 8000, "A"), phrase(8600, 15000, "B"), phrase(15300, 20000, "C")], 20000, 21000)
    lines = [{"text": "First line.", "runs": [1]}, {"text": "Second, then third.", "runs": [2, 3]}]

    enriched = breaks.attach_breaks(lines, run_list)

    assert enriched[0] == {"text": "First line.", "runs": [1], "start_ms": 0, "end_ms": 8000, "gap_after_ms": 600, "break_s": 0.6}
    assert enriched[1]["start_ms"] == 8600 and enriched[1]["end_ms"] == 20000
    assert enriched[1]["gap_after_ms"] == 1000 and enriched[1]["break_s"] == 1.0


def test_default_lines_share_the_span_by_length_and_use_default_breaks():
    lines = breaks.default_lines("Short one.\n\nA much longer paragraph here.", 10000, 20000)

    assert [line["text"] for line in lines] == ["Short one.", "A much longer paragraph here."]
    assert lines[0]["start_ms"] == 10000 and lines[1]["end_ms"] == 20000
    assert lines[0]["gap_after_ms"] == breaks.DEFAULT_PARAGRAPH_MS and lines[0]["break_s"] == 0.9
    assert lines[1]["gap_after_ms"] == breaks.DEFAULT_SENTENCE_MS and lines[1]["break_s"] == 0.5
    assert breaks.default_lines("   ", 0, 1000) == []


def test_render_writes_a_mark_after_every_line_that_has_one():
    text = breaks.render([{"text": "One.", "break_s": 0.6}, {"text": "Two", "break_s": None}, {"text": "Three.", "break_s": 3.0}])

    assert text == 'One. <break time="0.6s" />\nTwo\nThree. <break time="3.0s" />\n'


def test_parts_split_by_characters_and_by_marks():
    long_line = {"text": "x" * 5000, "break_s": 0.5}
    by_chars = breaks.split_parts([long_line, long_line, long_line])
    assert [len(p) for p in by_chars] == [1, 1, 1]

    marked = [{"text": "m.", "break_s": 0.4} for _ in range(85)]
    by_marks = breaks.split_parts(marked)
    assert [len(p) for p in by_marks] == [40, 40, 5]

    unmarked = [{"text": "m", "break_s": None} for _ in range(85)]
    assert len(breaks.split_parts(unmarked)) == 1

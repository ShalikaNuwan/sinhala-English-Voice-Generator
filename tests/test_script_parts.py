from __future__ import annotations

from app import breaks, script_parts


def seg(index, kind="narration", text="", lines=None, start=None, end=None, audio=None):
    return {"segment_index": index, "kind": kind, "narration_en": text, "lines": lines or [], "transcript_si": "call text",
            "start_ms": start if start is not None else index * 1000, "end_ms": end if end is not None else index * 1000 + 1000,
            "tts_audio_path": audio}


def line(text, break_s, start, end):
    return {"text": text, "runs": [1], "start_ms": start, "end_ms": end, "gap_after_ms": round((break_s or 0) * 1000), "break_s": break_s}


def test_parts_alternate_narration_runs_and_recordings_and_carry_lines():
    parts = script_parts.plan_parts([
        seg(1, lines=[line("Intro one.", 0.6, 1000, 1500), line("Intro two.", 0.4, 1600, 2000)]),
        seg(2, kind="original", audio="/a/0002_original.wav"),
        seg(3, text="After the call.\n\nAnd more.", start=3000, end=4000),
    ])

    assert [p["kind"] for p in parts] == ["narration", "recording", "narration"]
    assert [l["text"] for l in parts[0]["lines"]] == ["Intro one.", "Intro two."]
    assert parts[0]["segments"] == [1] and parts[0]["start_ms"] == 1000 and parts[0]["end_ms"] == 2000
    # A segment without lines gets default lines with default breaks.
    assert [l["text"] for l in parts[2]["lines"]] == ["After the call.", "And more."]
    assert [l["break_s"] for l in parts[2]["lines"]] == [0.9, 0.5]


def test_narration_parts_are_split_under_the_generation_limits():
    big = [line("x" * 5000, 0.5, i * 10, i * 10 + 5) for i in range(3)]
    parts = script_parts.plan_parts([seg(1, lines=big)])
    pieces = script_parts.expand(parts)

    assert [p["kind"] for p in pieces] == ["narration", "narration", "narration"]
    assert all(len(p["lines"]) == 1 for p in pieces)
    assert [p["number"] for p in pieces] == [1, 2, 3]


def test_expand_numbers_files_in_playing_order():
    parts = script_parts.plan_parts([seg(1, lines=[line("A.", 0.5, 0, 500)]), seg(2, kind="original", audio="/a/x.wav"), seg(3, lines=[line("B.", None, 2000, 2500)])])

    pieces = script_parts.expand(parts)

    assert [(p["number"], p["kind"]) for p in pieces] == [(1, "narration"), (2, "recording"), (3, "narration")]
    assert script_parts.file_name(pieces[0]) == "01_narration.txt"
    assert script_parts.file_name(pieces[1]) == "02_recording_segment002.wav"


def test_index_lines_report_marks_and_target_length():
    parts = script_parts.plan_parts([seg(1, lines=[line("A.", 0.6, 0, 8000), line("B.", None, 8600, 12000)]), seg(2, kind="original", audio="/a/x.wav", start=12000, end=20000)])
    pieces = script_parts.expand(parts)

    lines = script_parts.index_lines(pieces)

    assert lines[-2] == "- `01_narration.txt`: narration, segments 1, source 0.0-12.0 s, target 12.0 s, 28 characters, 1 break mark"
    assert lines[-1] == "- `02_recording_segment002.wav`: kept recording, segment 2, source 12.0-20.0 s, duration 8.0 s"


def test_render_of_a_piece_matches_breaks_render():
    piece = {"kind": "narration", "lines": [line("A.", 0.6, 0, 1), line("B", None, 2, 3)]}
    assert script_parts.render(piece) == breaks.render(piece["lines"])


def test_empty_narration_is_skipped_and_order_is_by_index():
    parts = script_parts.plan_parts([seg(2, text="Second."), seg(1, text="First."), seg(3, text="   ")])

    assert len(parts) == 1 and parts[0]["segments"] == [1, 2]
    assert [l["text"] for l in parts[0]["lines"]] == ["First.", "Second."]


def test_split_pieces_name_only_the_segments_they_contain():
    first = [line("x" * 5000, 0.5, 0, 5), line("y" * 5000, 0.5, 10, 15)]
    pieces = script_parts.expand(script_parts.plan_parts([seg(1, lines=first), seg(2, lines=[line("Tail.", 0.5, 20, 25)])]))

    assert [p["segments"] for p in pieces] == [[1], [1, 2]]


def test_the_programme_s_last_line_carries_no_mark():
    pieces = script_parts.expand(script_parts.plan_parts([seg(1, lines=[line("A.", 0.6, 0, 1), line("B.", 0.5, 2, 3)])]))

    assert script_parts.render(pieces[-1]) == 'A. <break time="0.6s" />\nB.\n'


def test_render_flattens_paragraph_breaks_inside_a_line():
    piece = {"kind": "narration", "lines": [line("One.\n\nTwo.", 0.5, 0, 1)]}
    assert script_parts.render(piece) == 'One. Two. <break time="0.5s" />\n'


def test_index_notes_silence_before_narration_that_resumes_late():
    parts = script_parts.plan_parts([
        seg(1, kind="original", audio="/a/x.wav", start=0, end=10000),
        seg(2, lines=[line("Back.", None, 15000, 16000)]),
    ])
    lines = script_parts.index_lines(script_parts.expand(parts))
    assert lines[-1].endswith("0 break marks, starts 5.0 s after the previous file")


def test_index_marks_a_part_the_diarizer_barely_heard():
    thin = [dict(line("A.", 0.6, 0, 8000), coverage=0.97), dict(line("B.", None, 8600, 12000), coverage=0.42)]
    lines = script_parts.index_lines(script_parts.expand(script_parts.plan_parts([seg(1, lines=thin)])))

    assert lines[-1].endswith(", coverage 42% (check breaks by ear)")  # the lowest of the part's lines


def test_index_says_nothing_about_coverage_when_the_phrases_are_full():
    full = [dict(line("A.", 0.6, 0, 8000), coverage=0.97)]
    lines = script_parts.index_lines(script_parts.expand(script_parts.plan_parts([seg(1, lines=full)])))

    assert "coverage" not in lines[-1]

from __future__ import annotations

from app import script_parts


def seg(index, kind="narration", text="", start=None, end=None, audio=None):
    return {"segment_index": index, "kind": kind, "narration_en": text, "transcript_si": "call text",
            "start_ms": start if start is not None else index * 1000, "end_ms": end if end is not None else index * 1000 + 1000,
            "tts_audio_path": audio}


def test_parts_alternate_narration_runs_and_recordings():
    parts = script_parts.plan_parts([
        seg(1, text="Intro one."), seg(2, text="Intro two."),
        seg(3, kind="original", audio="/a/0003_original.wav"),
        seg(4, kind="original", audio="/a/0004_original.wav"),
        seg(5, text="After the call."),
    ])

    assert [p["kind"] for p in parts] == ["narration", "recording", "recording", "narration"]
    assert parts[0]["text"] == "Intro one.\n\nIntro two."
    assert parts[0]["segments"] == [1, 2] and parts[0]["start_ms"] == 1000 and parts[0]["end_ms"] == 3000
    assert parts[1]["audio_path"] == "/a/0003_original.wav" and parts[1]["segment_index"] == 3
    assert parts[3]["text"] == "After the call." and parts[3]["segments"] == [5]


def test_empty_narration_is_skipped_and_order_is_by_index():
    parts = script_parts.plan_parts([seg(2, text="Second."), seg(1, text="First."), seg(3, text="   ")])

    assert len(parts) == 1 and parts[0]["text"] == "First.\n\nSecond." and parts[0]["segments"] == [1, 2]


def test_a_project_with_no_recordings_is_one_part():
    parts = script_parts.plan_parts([seg(1, text="A."), seg(2, text="B.")])

    assert [p["kind"] for p in parts] == ["narration"]


def test_index_lines_name_every_file_in_order():
    parts = script_parts.plan_parts([seg(1, text="Intro."), seg(2, kind="original", audio="/a/x.wav"), seg(3, text="Outro.")])

    lines = script_parts.index_lines(parts, ["01_narration.txt", "02_recording_segment002.wav", "03_narration.txt"])

    assert lines[-3].startswith("- `01_narration.txt`: narration, segments 1,")
    assert lines[-2].startswith("- `02_recording_segment002.wav`: kept recording, segment 2,")
    assert lines[-1].startswith("- `03_narration.txt`: narration, segments 3,")

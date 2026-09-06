from __future__ import annotations

import json
from pathlib import Path

from app import recordings

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "gilgo_probe_spans.json").read_text(encoding="utf-8"))
SAMPLE_SPANS = FIXTURE["sample"]
CHUNK2_SPANS = FIXTURE["chunk2"]
CHUNK2_MS = 37878


def span(speaker, start, end, text=""):
    return {"speaker": speaker, "start": start, "end": end, "text": text}


SINHALA = "ඇය කියනවා කවුරුහරි එනවා කියලා."
DEVANAGARI = "यकर तममा हितवा महत"


def test_script_test_separates_narrator_speech_from_english():
    assert recordings.has_foreign_script(SINHALA)
    assert recordings.has_foreign_script(DEVANAGARI)
    assert recordings.has_foreign_script("Rex Heuermann " + SINHALA)
    assert not recordings.has_foreign_script("Hello? Do you need the police?")
    assert not recordings.has_foreign_script("")


def test_reference_is_the_dominant_speakers_longest_span_trimmed_to_ten_seconds():
    assert recordings.choose_reference(SAMPLE_SPANS) == (16.43, 26.43)


def test_reference_prefers_the_speaker_with_the_most_speech():
    spans = [span("A", 0, 3, SINHALA), span("B", 3, 8, SINHALA), span("A", 8, 12, SINHALA)]

    assert recordings.choose_reference(spans) == (8.0, 12.0)


def test_reference_needs_at_least_two_seconds():
    assert recordings.choose_reference([span("A", 0, 1.5, SINHALA), span("A", 2, 3.4, SINHALA)]) is None
    assert recordings.choose_reference([]) is None


def test_the_real_911_call_chunk_yields_one_recording():
    # 13.9 s − 150 ms pad; 36.486 s + 150 ms pad
    assert recordings.original_spans(CHUNK2_SPANS, CHUNK2_MS) == [(13750, 36636)]


def test_narrator_speech_closes_a_recording():
    spans = [
        span(recordings.NARRATOR, 0, 5, SINHALA),
        span("A", 5.2, 8, "Hello?"),
        span("A", 9, 12, "Do you need the police?"),
        span(recordings.NARRATOR, 12.5, 20, SINHALA),
        span("A", 20.5, 25, "Where are you?"),
        span(recordings.NARRATOR, 25.5, 30, SINHALA),
    ]

    assert recordings.original_spans(spans, 30000) == [(5050, 12150), (20350, 25150)]


def test_a_mislabelled_english_word_inside_a_recording_stays_inside():
    spans = [
        span("A", 0, 3, "Hello?"),
        span(recordings.NARRATOR, 4, 4.7, "Okay."),
        span("A", 5, 8, "Do you need the police?"),
        span(recordings.NARRATOR, 9, 15, SINHALA),
    ]

    assert recordings.original_spans(spans, 15000) == [(0, 8150)]


def test_an_english_word_by_the_narrator_outside_a_recording_is_not_one():
    spans = [span(recordings.NARRATOR, 0, 5, SINHALA), span(recordings.NARRATOR, 5, 6, "Gilgo Beach"), span(recordings.NARRATOR, 6, 10, SINHALA)]

    assert recordings.original_spans(spans, 10000) == []


def test_a_gap_over_ten_seconds_splits_recordings():
    spans = [span("A", 0, 3, "Hello?"), span("A", 14, 17, "Hello?")]

    assert recordings.original_spans(spans, 20000) == [(0, 3150), (13850, 17150)]


def test_recordings_under_one_and_a_half_seconds_are_ignored():
    spans = [span(recordings.NARRATOR, 0, 5, SINHALA), span("A", 5.2, 6.0, "Yeah."), span(recordings.NARRATOR, 6.5, 10, SINHALA)]

    assert recordings.original_spans(spans, 10000) == []


def test_recordings_are_padded_but_clamped_to_the_chunk():
    spans = [span("A", 0.05, 2.0, "Hello?"), span("A", 2.2, 4.95, "Hello?")]

    assert recordings.original_spans(spans, 5000) == [(0, 5000)]


def test_narration_spans_are_where_the_narrator_is_heard():
    assert recordings.narration_spans(CHUNK2_SPANS) == [(0, 9250), (9500, 13800)]


def test_cut_plan_on_the_real_chunk_keeps_the_intro_and_absorbs_the_calls_tail():
    originals = recordings.original_spans(CHUNK2_SPANS, CHUNK2_MS)
    narration = recordings.narration_spans(CHUNK2_SPANS)

    plan = recordings.cut_plan(45000, 45000 + CHUNK2_MS, originals, narration)

    assert plan == [(45000, 45000 + 13750, "narration"), (45000 + 13750, 45000 + CHUNK2_MS, "original")]


def test_cut_plan_keeps_narration_on_both_sides_of_a_recording():
    plan = recordings.cut_plan(0, 45000, [(19850, 30150)], [(0, 20000), (30000, 45000)])

    assert plan == [(0, 19850, "narration"), (19850, 30150, "original"), (30150, 45000, "narration")]


def test_cut_plan_absorbs_leading_silence_into_the_recording():
    plan = recordings.cut_plan(100000, 110000, [(400, 6000)], [(6200, 10000)])

    assert plan == [(100000, 106000, "original"), (106000, 110000, "narration")]


def test_cut_plan_without_recordings_is_the_whole_chunk():
    assert recordings.cut_plan(0, 45000, [], [(0, 45000)]) == [(0, 45000, "narration")]
    assert recordings.cut_plan(0, 45000, [], []) == [(0, 45000, "narration")]


def test_cut_plan_can_be_entirely_a_recording():
    assert recordings.cut_plan(0, 20000, [(0, 20000)], []) == [(0, 20000, "original")]


def test_clip_text_collects_what_is_said_in_the_recording():
    (start, end), = recordings.original_spans(CHUNK2_SPANS, CHUNK2_MS)
    text = recordings.clip_text(CHUNK2_SPANS, start, end)

    assert text.startswith("Hi, how can I assist you? Hello? Hello?")
    assert text.endswith("Do you need the police? Where?")
    assert "Okay." in text
    assert "නිව්" not in text


def test_english_run_flags_a_recording_left_in_a_transcript():
    transcript = SINHALA + " 9-1-1, how can I assist you? Hello? Hello? Hello, you dialed into the 911 system."

    assert recordings.english_run(transcript)


def test_english_run_ignores_names_and_short_phrases_inside_narration():
    assert not recordings.english_run(SINHALA + " Rex Heuermann " + SINHALA + " Gilgo Beach Killer " + SINHALA)
    assert not recordings.english_run("")


def test_a_long_gap_followed_by_the_narrator_closes_cleanly():
    spans = [span("A", 0, 3, "Hello?"), span(recordings.NARRATOR, 14, 20, SINHALA)]

    assert recordings.original_spans(spans, 20000) == [(0, 3150)]


def test_inverted_spans_do_not_poison_a_recording():
    spans = [span("A", 8, 5, "Hello?"), span("A", 9, 12, "Do you need the police?")]

    assert recordings.original_spans(spans, 15000) == [(7850, 12150)]


def test_cut_plan_keeps_nested_originals_whole():
    plan = recordings.cut_plan(0, 30000, [(0, 20000), (5000, 10000)], [(20000, 30000)])

    assert plan == [(0, 20000, "original"), (20000, 30000, "narration")]


def test_cut_plan_with_two_separated_recordings():
    plan = recordings.cut_plan(0, 30000, [(5000, 10000), (20000, 25000)], [(0, 5000), (10000, 20000), (25000, 30000)])

    assert plan == [
        (0, 5000, "narration"), (5000, 10000, "original"), (10000, 20000, "narration"),
        (20000, 25000, "original"), (25000, 30000, "narration"),
    ]


def test_english_run_fires_on_an_all_english_transcript():
    assert recordings.english_run("Where? Tell me. Where on Long Island are you? I don't know.")


# What the pipeline actually saw on a real chunk: the narrator's Sinhala came back in Latin letters.
TRANSLITERATED = "Kedath Dhahaye Mai Palaveni Dah Pandra Hatharai Panas Ekata Niv York State Police Dispatch"


def test_a_narrator_span_counts_as_narration_even_when_written_in_latin_letters():
    spans = [
        span(recordings.NARRATOR, 0, 9.85, TRANSLITERATED),
        span("A", 9.85, 9.95, "9"),
        span(recordings.NARRATOR, 9.95, 13.85, "Ea Sehema Kalabala Vela Vageema Baya Vela Vage Tamai Kathakaranai."),
        span("A", 14.25, 15.45, "-1-1, how can I assist you?"),
        span("A", 18.05, 18.55, "Hello?"),
        span("A", 21.15, 21.50, "Hello?"),
        span("A", 24.44, 26.29, "Hello, are you dialed into the 911 system?"),
        span("A", 26.54, 27.69, "How can I assist you?"),
        span("B", 30.74, 31.14, "Okay."),
        span("A", 35.99, 36.44, "Where?"),
    ]

    assert recordings.narration_spans(spans) == [(0, 9850), (9950, 13850)]
    assert recordings.original_spans(spans, 37482) == [(14100, 36590)]
    plan = recordings.cut_plan(45000, 45000 + 37482, recordings.original_spans(spans, 37482), recordings.narration_spans(spans))
    assert plan == [(45000, 45000 + 14100, "narration"), (45000 + 14100, 45000 + 37482, "original")]


def test_a_long_narrator_labelled_latin_span_closes_a_recording():
    spans = [
        span("A", 0, 3, "Hello?"),
        span(recordings.NARRATOR, 4, 9, "Ea Sehema Kalabala Vela Vageema Baya Vela"),
        span("A", 10, 13, "Do you need the police?"),
    ]

    assert recordings.original_spans(spans, 15000) == [(0, 3150), (9850, 13150)]


def test_a_short_narrator_labelled_aside_is_still_absorbed():
    spans = [span("A", 0, 3, "Hello?"), span(recordings.NARRATOR, 4, 4.7, "Okay."), span("A", 5, 8, "Where?")]

    assert recordings.original_spans(spans, 10000) == [(0, 8150)]


def test_a_stray_fragment_between_narrator_spans_is_not_a_recording():
    spans = [
        span(recordings.NARRATOR, 0, 9.85, TRANSLITERATED),
        span("A", 9.85, 9.95, "9"),
        span(recordings.NARRATOR, 9.95, 20, TRANSLITERATED),
    ]

    assert recordings.original_spans(spans, 20000) == []

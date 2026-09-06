from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from app import pauses


SCRIPT = (
    "On May 1st, 2010, at 4:51 in the morning, a call comes in.\n\n"
    "It's from a young woman named Shannon Gilbert. She sounds agitated… and frightened — badly.\n"
    "Whose house is it?"
)


def test_script_tokens_carry_the_break_after_each_word():
    tokens = pauses.script_tokens(SCRIPT)
    by_word = dict(tokens)

    assert by_word["1st"] == "clause"
    assert by_word["2010"] == "clause"
    assert by_word["morning"] == "clause"
    assert by_word["in"] == "paragraph"  # sentence end followed by a blank line
    assert by_word["gilbert"] == "sentence"
    assert by_word["agitated"] == "beat"
    assert by_word["frightened"] == "dash"  # the lone dash attaches to the word before it
    assert by_word["badly"] == "sentence"
    assert by_word["it"] == "sentence"
    assert by_word["young"] == "none"
    assert [word for word, _ in tokens][:4] == ["on", "may", "1st", "2010"]


def test_a_beat_before_a_blank_line_stays_a_beat():
    assert dict(pauses.script_tokens("Then silence…\n\nNothing.")) == {"then": "none", "silence": "beat", "nothing": "sentence"}


def test_abbreviations_and_closers_are_classified_sensibly():
    by_word = dict(pauses.script_tokens('Mr. Smith arrived at 4 a.m. and said, “Agitated.” Then… “Why?” wait- no.'))

    assert by_word["mr"] == "none"
    assert by_word["am"] == "none"
    assert by_word["agitated"] == "sentence"
    assert by_word["then"] == "beat"
    assert by_word["why"] == "sentence"
    assert by_word["wait"] == "dash"


def test_normalise_keeps_letters_and_digits_only():
    assert pauses.normalise("“Gilbert.”") == "gilbert"
    assert pauses.normalise("4:51") == "451"
    assert pauses.normalise("—") == ""


def spoken(*items):
    return [{"word": w, "start": s, "end": e} for w, s, e in items]


def test_align_tolerates_a_transcriber_rewriting_a_token():
    tokens = pauses.script_tokens("On May 1st, 2010, a call.")
    heard = spoken(("On", 0.0, 0.2), ("May", 0.2, 0.5), ("1", 0.5, 0.8), ("2010", 1.2, 1.7), ("a", 2.2, 2.3), ("call", 2.3, 2.6))

    mapping = pauses.align(tokens, heard)

    assert mapping[0] == 0 and mapping[1] == 1 and mapping[3] == 3 and mapping[5] == 5
    assert mapping[2] == 2  # "1" sits between two matches and is filled in by position


def test_align_survives_an_extra_and_a_missing_word():
    tokens = pauses.script_tokens("She sounds agitated and frightened.")
    heard = spoken(("She", 0, 0.2), ("uh", 0.2, 0.3), ("sounds", 0.3, 0.6), ("frightened", 1.0, 1.4))

    mapping = pauses.align(tokens, heard)

    assert mapping == {0: 0, 2: 1, 3: 4}


def test_classify_uses_the_last_word_ending_before_the_pause():
    tokens = pauses.script_tokens("First part, second part. Third part.")
    heard = spoken(("First", 0.0, 0.5), ("part", 0.5, 1.0), ("second", 1.9, 2.4), ("part", 2.4, 2.9), ("Third", 4.3, 4.8), ("part", 4.8, 5.3))
    mapping = pauses.align(tokens, heard)

    classified = pauses.classify([(1.02, 1.9), (2.95, 4.3), (5.3, 5.6)], heard, mapping, tokens)

    assert classified == [(1.02, 1.9, "clause"), (2.95, 4.3, "sentence"), (5.3, 5.6, "sentence")]


def test_classify_marks_an_unmapped_word_as_unknown():
    tokens = pauses.script_tokens("Hello there.")
    heard = spoken(("Goodbye", 0.0, 0.5), ("there", 0.9, 1.3))

    classified = pauses.classify([(0.5, 0.9)], heard, pauses.align(tokens, heard), tokens)

    assert classified == [(0.5, 0.9, "unknown")]


def test_targets_fall_inside_the_scaled_band_and_are_reproducible():
    classified = [(1.0, 1.9, "clause"), (2.9, 4.3, "sentence"), (5.0, 6.8, "beat")]

    first = pauses.choose_targets(classified, "continuous", random.Random("seg-1"), 10.0)
    again = pauses.choose_targets(classified, "continuous", random.Random("seg-1"), 10.0)
    slow = pauses.choose_targets(classified, "measured", random.Random("seg-1"), 10.0)

    assert first == again
    for (_, _, target), cls in zip(first, ["clause", "sentence", "beat"]):
        low, high = pauses.BANDS[cls]
        assert round(low * 0.85) <= target <= round(high * 0.85)
    for (_, _, target), cls in zip(slow, ["clause", "sentence", "beat"]):
        low, high = pauses.BANDS[cls]
        assert round(low * 1.15) <= target <= round(high * 1.15)


def test_unpunctuated_pauses_are_capped_never_lengthened():
    plan = pauses.choose_targets([(1.0, 1.5, "none"), (2.0, 2.1, "none")], "moderate", random.Random(0), 5.0)

    assert [target for _, _, target in plan] == [pauses.UNPUNCTUATED_MAX_MS, 100]


def test_edges_are_trimmed_to_a_breath():
    plan = pauses.choose_targets([(0.0, 0.8, "none"), (4.5, 5.0, "sentence")], "moderate", random.Random(0), 5.0)

    assert [target for _, _, target in plan] == [pauses.EDGE_SILENCE_MS, pauses.EDGE_SILENCE_MS]


def test_unknown_pauses_keep_their_length():
    plan = pauses.choose_targets([(1.0, 1.8, "unknown")], "moderate", random.Random(0), 5.0)

    assert plan == [(1.0, 1.8, 800)]


def bursts(path: Path, pattern: list[tuple[float, float]]) -> Path:
    """Tone bursts separated by silences: pattern is [(tone_s, silence_after_s), ...]."""
    inputs, labels = [], []
    for index, (tone_s, gap_s) in enumerate(pattern):
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={tone_s}"]
        labels.append(f"[{len(labels)}:a]")
        if gap_s > 0:
            inputs += ["-f", "lavfi", "-t", f"{gap_s}", "-i", "anullsrc=r=24000:cl=mono"]
            labels.append(f"[{len(labels)}:a]")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]",
         "-map", "[out]", "-ar", "24000", "-ac", "1", str(path)],
        check=True,
    )
    return path


def test_detect_pauses_finds_interior_and_trailing_silence(tmp_path):
    path = bursts(tmp_path / "in.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    found = pauses.detect_pauses(path)

    assert len(found) == 3
    assert abs(found[0][0] - 1.0) < 0.05 and abs(found[0][1] - 1.9) < 0.05
    assert abs(found[1][0] - 2.9) < 0.05 and abs(found[1][1] - 4.3) < 0.05
    assert abs(found[2][0] - 5.3) < 0.05 and abs(found[2][1] - 5.8) < 0.05


def test_rebuild_sets_every_gap_to_its_target_and_leaves_speech_alone(tmp_path):
    source = bursts(tmp_path / "in.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])
    plan = [(1.0, 1.9, 250), (2.9, 4.3, 1000), (5.3, 5.8, 60)]

    pauses.rebuild(source, plan, 5.8, tmp_path / "out.wav")

    found = pauses.detect_pauses(tmp_path / "out.wav")
    gaps = [round((end - start) * 1000) for start, end in found]
    assert len(gaps) == 2  # the 60 ms tail sits below the detection floor by design
    assert abs(gaps[0] - 250) <= 30 and abs(gaps[1] - 1000) <= 30
    runs = [found[0][0], found[1][0] - found[0][1]]
    assert all(abs(run - 1.0) < 0.05 for run in runs)
    # Three 1 s runs plus 250 + 1000 + 60 ms of silence.
    assert abs(pauses._duration_s(tmp_path / "out.wav") - 4.31) < 0.03


def test_rebuild_refuses_overlapping_pauses(tmp_path):
    source = bursts(tmp_path / "in.wav", [(1.0, 0.9), (1.0, 0.5)])

    with pytest.raises(RuntimeError, match="Overlapping"):
        pauses.rebuild(source, [(1.0, 2.0, 200), (1.5, 3.0, 200)], 3.4, tmp_path / "out.wav")


def test_profile_and_issues(tmp_path):
    path = bursts(tmp_path / "in.wav", [(1.0, 0.35), (1.0, 0.5), (1.0, 2.2), (1.0, 0.05)])

    profile = pauses.pause_profile(path)

    assert profile["count"] == 3  # the 50 ms tail is below the detection floor
    assert profile["max_ms"] >= 2150
    issues = pauses.profile_issues({"profile": profile, "unpunctuated_over_cap": 1, "pace": "moderate"})
    assert any("longest pause" in issue for issue in issues)
    assert any("unpunctuated" in issue for issue in issues)
    assert pauses.profile_issues({"profile": {"max_ms": 900}, "unpunctuated_over_cap": 0, "pace": "moderate"}) == []


def test_profile_of_a_file_without_pauses(tmp_path):
    source = bursts(tmp_path / "solid.wav", [(2.0, 0.0)])

    assert pauses.pause_profile(source) == {"count": 0, "median_ms": 0, "p90_ms": 0, "max_ms": 0, "per_minute": 0.0}


def test_shape_end_to_end_with_word_timestamps(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])
    script = "First part, second part. Third part."
    heard = spoken(("First", 0.0, 0.5), ("part", 0.5, 1.0), ("second", 1.9, 2.4), ("part", 2.4, 2.9), ("Third", 4.3, 4.8), ("part", 4.8, 5.3))

    result = pauses.shape(source, script, heard, "continuous", tmp_path / "shaped.wav", seed="seg")

    assert result["method"] == "words"
    found = pauses.detect_pauses(tmp_path / "shaped.wav")
    gaps = [round((end - start) * 1000) for start, end in found]
    clause_low, clause_high = pauses.BANDS["clause"]
    sentence_low, sentence_high = pauses.BANDS["sentence"]
    assert len(gaps) == 2
    assert clause_low * 0.85 - 30 <= gaps[0] <= clause_high * 0.85 + 30
    assert sentence_low * 0.85 - 30 <= gaps[1] <= sentence_high * 0.85 + 30
    assert result["plan"][-1]["target_ms"] <= pauses.EDGE_SILENCE_MS
    assert abs(pauses._duration_s(tmp_path / "shaped.wav") - (3.0 + (gaps[0] + gaps[1] + result["plan"][-1]["target_ms"]) / 1000)) < 0.05
    assert result["classes"] == {"clause": 1, "sentence": 1, "edge": 1}
    assert result["profile"]["count"] == 2


def test_shape_refuses_when_too_few_words_align(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])
    heard = spoken(("alpha", 0.0, 0.5), ("beta", 0.5, 1.0), ("gamma", 1.9, 2.4), ("part", 2.4, 2.9))

    with pytest.raises(pauses.AlignmentError):
        pauses.shape(source, "First part, second part. Third part.", heard, "moderate", tmp_path / "shaped.wav", seed="seg")


def test_shape_falls_back_to_punctuation_order_without_timestamps(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    result = pauses.shape(source, "First part, second part. Third part.", [], "moderate", tmp_path / "shaped.wav", seed="seg")

    assert result["method"] == "order"
    assert result["classes"] == {"clause": 1, "sentence": 1, "edge": 1}


def test_order_fallback_refuses_an_extra_pause(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.9), (1.0, 0.5)])

    with pytest.raises(pauses.AlignmentError):
        pauses.shape(source, "First part, second part. Third part.", [], "moderate", tmp_path / "shaped.wav", seed="seg")


def test_shape_refuses_when_nothing_lines_up(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    with pytest.raises(pauses.AlignmentError):
        pauses.shape(source, "One sentence with no breaks at all", [], "moderate", tmp_path / "shaped.wav", seed="seg")


def test_duration_failure_is_loud(tmp_path):
    with pytest.raises(RuntimeError):
        pauses._duration_s(tmp_path / "missing.wav")


def test_a_one_word_sentence_like_no_still_breaks():
    assert dict(pauses.script_tokens("No. She never called.")) == {"no": "sentence", "she": "none", "never": "none", "called": "sentence"}
    assert dict(pauses.script_tokens("At 4 a.m. she called."))["am"] == "none"


def test_classify_survives_timestamp_drift_on_both_sides_of_a_pause():
    """Seen on a real segment: "me" stretched 300 ms into the silence and "Even" started early."""
    tokens = pauses.script_tokens("trying to kill me.\n\nEven when the operator asks.")
    heard = spoken(("trying", 0.0, 0.3), ("to", 0.3, 0.4), ("kill", 0.4, 0.7), ("me", 0.7, 1.2), ("Even", 0.85, 2.3), ("when", 2.3, 2.5))

    classified = pauses.classify([(0.9, 1.9)], heard, pauses.align(tokens, heard), tokens)

    assert classified == [(0.9, 1.9, "paragraph")]


def test_a_pause_spanned_by_its_nearest_word_is_unknown_not_attributed_earlier():
    tokens = pauses.script_tokens("First part. Second part.")
    heard = spoken(("First", 0.0, 0.25), ("part", 0.25, 0.5), ("Second", 0.6, 0.9), ("part", 0.9, 1.2))

    classified = pauses.classify([(0.30, 0.42)], heard, pauses.align(tokens, heard), tokens)

    assert classified == [(0.30, 0.42, "unknown")]

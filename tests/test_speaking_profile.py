from __future__ import annotations

import subprocess
from pathlib import Path

from app import speaking_profile


def build_speech_like_wav(path: Path) -> None:
    """Three 3s bursts of tone separated by two 1.5s silences: 12s, 25% silence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-f", "lavfi", "-t", "1.5", "-i", "anullsrc=r=44100:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-f", "lavfi", "-t", "1.5", "-i", "anullsrc=r=44100:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-filter_complex", "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[out]",
            "-map", "[out]", "-ar", "24000", "-ac", "1", str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_derives_labels_for_a_deliberate_moderate_narrator():
    # The real measurements taken from Recording 550.wav.
    labels = speaking_profile.derive(
        {"speech_ratio": 0.90, "mean_pause_ms": 1321, "loudness_range_lu": 4.0}
    )

    assert labels == {
        "pace": "moderate",
        "pause_style": "deliberate",
        "dynamics": "moderately dynamic",
    }


def test_derives_labels_for_a_fast_flat_narrator():
    labels = speaking_profile.derive(
        {"speech_ratio": 0.97, "mean_pause_ms": 300, "loudness_range_lu": 1.5}
    )

    assert labels == {
        "pace": "continuous",
        "pause_style": "brisk",
        "dynamics": "controlled",
    }


def test_derives_labels_for_a_slow_dramatic_narrator():
    labels = speaking_profile.derive(
        {"speech_ratio": 0.60, "mean_pause_ms": 2400, "loudness_range_lu": 9.0}
    )

    assert labels == {
        "pace": "measured",
        "pause_style": "dramatic",
        "dynamics": "highly dynamic",
    }


def test_derives_labels_when_the_narrator_never_pauses():
    labels = speaking_profile.derive(
        {"speech_ratio": 1.0, "mean_pause_ms": 0, "loudness_range_lu": 0.0}
    )

    assert labels["pace"] == "continuous"
    assert labels["pause_style"] == "brisk"


def test_measures_pauses_and_speech_ratio_from_real_audio(tmp_path):
    source = tmp_path / "narration.wav"
    build_speech_like_wav(source)

    measured = speaking_profile.measure(source)

    assert measured["pause_count"] == 2
    assert 1400 <= measured["mean_pause_ms"] <= 1600
    assert 1400 <= measured["longest_pause_ms"] <= 1600
    assert 0.70 <= measured["speech_ratio"] <= 0.80
    assert 11.5 <= measured["duration_s"] <= 12.5
    assert isinstance(measured["loudness_range_lu"], float)


def test_measures_continuous_audio_as_having_no_pauses(tmp_path):
    source = tmp_path / "continuous.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=200:duration=6", "-ar", "24000", "-ac", "1", str(source)],
        check=True, capture_output=True,
    )

    measured = speaking_profile.measure(source)

    assert measured["pause_count"] == 0
    assert measured["mean_pause_ms"] == 0
    assert measured["speech_ratio"] == 1.0


def test_ignores_leading_silence_before_the_narrator_starts(tmp_path):
    """Dead air before the first word is not a narration beat and must not skew the mean."""
    source = tmp_path / "late_start.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-t", "2.0", "-i", "anullsrc=r=44100:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-f", "lavfi", "-t", "1.5", "-i", "anullsrc=r=44100:cl=mono",
            "-f", "lavfi", "-i", "sine=frequency=200:duration=3",
            "-filter_complex", "[0:a][1:a][2:a][3:a]concat=n=4:v=0:a=1[out]",
            "-map", "[out]", "-ar", "24000", "-ac", "1", str(source),
        ],
        check=True, capture_output=True,
    )

    measured = speaking_profile.measure(source)

    assert measured["pause_count"] == 1
    assert 1400 <= measured["mean_pause_ms"] <= 1600
    # 6s speech and one 1.5s pause once the 2s of dead air is excluded.
    assert 0.78 <= measured["speech_ratio"] <= 0.82


def test_samples_the_middle_of_a_long_recording_skipping_the_intro():
    assert speaking_profile.sample_window(1800.0) == (5.0, 45.0)


def test_samples_a_short_recording_whole():
    assert speaking_profile.sample_window(30.0) == (0.0, 30.0)


def test_samples_a_recording_just_under_the_threshold_whole():
    start, length = speaking_profile.sample_window(49.0)
    assert (start, length) == (0.0, 49.0)

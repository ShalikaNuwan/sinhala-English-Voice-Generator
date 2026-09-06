import subprocess
from pathlib import Path

import pytest

from app.audio import AudioError
from app.audio import AudioService


def tone(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "24000", "-ac", "1", str(path)],
        check=True,
    )
    return path


def test_short_audio_is_one_segment():
    assert AudioService.plan_segments(35_000, [12_000, 25_000]) == [(0, 35_000)]


def test_prefers_silence_near_target():
    assert AudioService.plan_segments(95_000, [22_000, 39_000, 58_000, 80_000]) == [
        (0, 39_000),
        (39_000, 95_000),
    ]


def test_falls_back_to_safe_fixed_boundaries():
    segments = AudioService.plan_segments(130_000, [])
    assert segments == [(0, 45_000), (45_000, 90_000), (90_000, 130_000)]
    assert all(end > start for start, end in segments)


def test_avoids_tiny_tail_for_continuous_one_minute_audio():
    assert AudioService.plan_segments(64_740, []) == [(0, 64_740)]


def test_assembly_inserts_the_requested_silence_between_segments(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / f"{i}.wav", 1.0) for i in range(3)]
    wav = tmp_path / "out" / "final.wav"

    audio.assemble(parts, wav, tmp_path / "out" / "final.mp3", gaps_ms=[500, 1000])

    duration = audio.probe(wav).duration_ms
    assert 4350 <= duration <= 4650  # 3 x 1000 ms of tone + 1500 ms of silence


def test_assembly_without_gaps_is_unchanged(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / f"{i}.wav", 1.0) for i in range(2)]
    wav = tmp_path / "out" / "final.wav"

    audio.assemble(parts, wav, tmp_path / "out" / "final.mp3")

    assert 1900 <= audio.probe(wav).duration_ms <= 2100


def test_assembly_rejects_a_gap_list_of_the_wrong_length(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / f"{i}.wav", 1.0) for i in range(2)]

    with pytest.raises(AudioError):
        audio.assemble(parts, tmp_path / "final.wav", tmp_path / "final.mp3", gaps_ms=[500, 500])

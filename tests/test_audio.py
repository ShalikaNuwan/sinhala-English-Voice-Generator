import re
import subprocess
from pathlib import Path

import pytest

from app.audio import AudioError, AudioService


def tone(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "24000", "-ac", "1", str(path)],
        check=True,
    )
    return path


def mean_volume_db(path: Path, start_s: float, end_s: float) -> float:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}",
         "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    match = re.search(r"mean_volume:\s*(-?[0-9.]+) dB", result.stderr)
    assert match, result.stderr
    return float(match.group(1))


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
    # The silence must land between the segments, not be appended at the end.
    assert mean_volume_db(wav, 1.05, 1.45) < -60
    assert mean_volume_db(wav, 2.60, 3.40) < -60
    assert mean_volume_db(wav, 1.60, 2.40) > -30
    assert (tmp_path / "out" / "final.mp3").exists()


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


def test_assembly_rejects_an_empty_gap_list_for_multiple_segments(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / f"{i}.wav", 1.0) for i in range(3)]

    with pytest.raises(AudioError):
        audio.assemble(parts, tmp_path / "final.wav", tmp_path / "final.mp3", gaps_ms=[])


def test_assembly_rejects_negative_gaps(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / f"{i}.wav", 1.0) for i in range(2)]

    with pytest.raises(AudioError):
        audio.assemble(parts, tmp_path / "final.wav", tmp_path / "final.mp3", gaps_ms=[-500])


def test_assembly_accepts_an_empty_gap_list_for_a_single_segment(tmp_path):
    audio = AudioService()
    parts = [tone(tmp_path / "0.wav", 1.0)]
    wav = tmp_path / "out" / "final.wav"

    audio.assemble(parts, wav, tmp_path / "out" / "final.mp3", gaps_ms=[])

    assert audio.probe(wav).channels == 1
    assert 900 <= audio.probe(wav).duration_ms <= 1100


def test_assembly_joins_a_legacy_mp3_segment_with_a_wav_segment(tmp_path):
    audio = AudioService()
    wav_part = tone(tmp_path / "0.wav", 1.0)
    mp3_part = tmp_path / "1.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav_part), "-codec:a", "libmp3lame", "-b:a", "128k", str(mp3_part)],
        check=True,
    )
    wav = tmp_path / "out" / "final.wav"

    audio.assemble([wav_part, mp3_part], wav, tmp_path / "out" / "final.mp3", gaps_ms=[600])

    info = audio.probe(wav)
    assert info.channels == 1 and info.sample_rate == 24000
    assert 2450 <= info.duration_ms <= 2750
    assert mean_volume_db(wav, 1.05, 1.55) < -60


def test_extract_cuts_a_padded_piece_as_mono_24k(tmp_path):
    audio = AudioService()
    source = tone(tmp_path / "source.wav", 3.0)

    info = audio.extract(source, 1000, 2000, tmp_path / "out" / "piece.wav")

    assert info.channels == 1 and info.sample_rate == 24000
    assert 1350 <= info.duration_ms <= 1450  # 1000 ms plus 200 ms of padding on each side


def test_extract_does_not_pad_past_the_edges(tmp_path):
    audio = AudioService()
    source = tone(tmp_path / "source.wav", 3.0)

    first = audio.extract(source, 0, 1000, tmp_path / "first.wav")
    last = audio.extract(source, 2000, 3000, tmp_path / "last.wav")
    exact = audio.extract(source, 1000, 2000, tmp_path / "exact.wav", pad_ms=0)

    assert 1150 <= first.duration_ms <= 1250
    assert 1150 <= last.duration_ms <= 1250
    assert 950 <= exact.duration_ms <= 1050


@pytest.mark.parametrize("clip_ms", [1500, 2200, 3000, 5000])
def test_extract_levelled_normalises_loudness_regardless_of_length(tmp_path, clip_ms):
    audio = AudioService()
    source = tmp_path / "quiet.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
         "-af", "volume=-30dB", "-ar", "24000", "-ac", "1", str(source)],
        check=True,
    )

    info = audio.extract_levelled(source, 500, 500 + clip_ms, tmp_path / "out" / f"clip_{clip_ms}.wav")

    assert info.channels == 1 and info.sample_rate == 24000
    assert clip_ms - 100 <= info.duration_ms <= clip_ms + 100
    # Measured over the middle of the clip, clear of the 5 ms fades.
    assert mean_volume_db(tmp_path / "out" / f"clip_{clip_ms}.wav", 0.1, clip_ms / 1000 - 0.1) > -25


def test_extract_levelled_leaves_silence_alone(tmp_path):
    audio = AudioService()
    source = tmp_path / "silence.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-t", "3", "-i", "anullsrc=r=24000:cl=mono", str(source)],
        check=True,
    )

    info = audio.extract_levelled(source, 0, 2000, tmp_path / "out" / "silent.wav")

    assert 1900 <= info.duration_ms <= 2100
    assert mean_volume_db(tmp_path / "out" / "silent.wav", 0.1, 1.9) < -60

"""Describes how the source narrator speaks so the English narration can match them."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


# Pauses shorter than this are word gaps, not narration beats.
SILENCE_THRESHOLD_DB = -32
SILENCE_MINIMUM_S = 0.30
# Silence starting this close to the top of the file is dead air, not a pause.
LEADING_SILENCE_TOLERANCE_S = 0.05

# Tuning thresholds. Adjust these after listening to real output.
CONTINUOUS_SPEECH_RATIO = 0.93
MODERATE_SPEECH_RATIO = 0.85
BRISK_PAUSE_MS = 600
DELIBERATE_PAUSE_MS = 1500
CONTROLLED_RANGE_LU = 3.0
DYNAMIC_RANGE_LU = 7.0

# How much audio the tone analysis listens to, and where it starts.
SAMPLE_START_S = 5.0
SAMPLE_LENGTH_S = 45.0
SAMPLE_WHOLE_FILE_BELOW_S = 50.0


def derive(measured: dict) -> dict:
    """Turn raw acoustic measurements into labels a TTS instruction can use."""
    ratio = measured["speech_ratio"]
    if ratio >= CONTINUOUS_SPEECH_RATIO:
        pace = "continuous"
    elif ratio >= MODERATE_SPEECH_RATIO:
        pace = "moderate"
    else:
        pace = "measured"

    pause_ms = measured["mean_pause_ms"]
    if pause_ms < BRISK_PAUSE_MS:
        pause_style = "brisk"
    elif pause_ms <= DELIBERATE_PAUSE_MS:
        pause_style = "deliberate"
    else:
        pause_style = "dramatic"

    loudness_range = measured["loudness_range_lu"]
    if loudness_range < CONTROLLED_RANGE_LU:
        dynamics = "controlled"
    elif loudness_range <= DYNAMIC_RANGE_LU:
        dynamics = "moderately dynamic"
    else:
        dynamics = "highly dynamic"

    return {"pace": pace, "pause_style": pause_style, "dynamics": dynamics}


def _run(command: list[str]) -> str:
    """Run an ffmpeg/ffprobe command and return its stderr, where the analysis lands."""
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stderr + result.stdout


def _duration_s(path: Path, ffprobe: str) -> float:
    output = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return float(output) if output else 0.0


def _pause_seconds(path: Path, ffmpeg: str) -> tuple[list[float], float]:
    """Return the pauses between speech, plus where the narrator actually starts."""
    output = _run([
        ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
        "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={SILENCE_MINIMUM_S}",
        "-f", "null", "-",
    ])
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", output)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", output)]
    spans = list(zip(starts, ends))
    # Dead air before the first word is not a narration beat; it would skew every average.
    speech_starts_at = 0.0
    if spans and spans[0][0] <= LEADING_SILENCE_TOLERANCE_S:
        speech_starts_at = spans[0][1]
        spans = spans[1:]
    return [end - start for start, end in spans], speech_starts_at


def _loudness_range_lu(path: Path, ffmpeg: str) -> float:
    output = _run([ffmpeg, "-v", "info", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"])
    # ebur128 prints a running value per frame; only the trailing summary is meaningful.
    summary = output.split("Summary:")[-1]
    match = re.search(r"LRA:\s*(-?[0-9.]+) LU", summary)
    return float(match.group(1)) if match else 0.0


def measure(path: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    """Measure how the speaker in this recording uses time and volume."""
    path = Path(path)
    duration_s = _duration_s(path, ffprobe)
    pauses, speech_starts_at = _pause_seconds(path, ffmpeg)
    narration_s = duration_s - speech_starts_at
    speaking_s = narration_s - sum(pauses)
    return {
        "duration_s": round(duration_s, 2),
        "pause_count": len(pauses),
        "mean_pause_ms": round(sum(pauses) / len(pauses) * 1000) if pauses else 0,
        "longest_pause_ms": round(max(pauses) * 1000) if pauses else 0,
        "pauses_per_min": round(len(pauses) / (narration_s / 60), 1) if narration_s else 0.0,
        "speech_ratio": round(speaking_s / narration_s, 2) if narration_s else 0.0,
        "loudness_range_lu": _loudness_range_lu(path, ffmpeg),
    }


def sample_window(duration_s: float) -> tuple[float, float]:
    """Pick the excerpt to analyse: skip the intro on long files, take short ones whole."""
    if duration_s < SAMPLE_WHOLE_FILE_BELOW_S:
        return (0.0, duration_s)
    return (SAMPLE_START_S, SAMPLE_LENGTH_S)


def extract_sample(source: Path, destination: Path, duration_s: float, ffmpeg: str = "ffmpeg") -> Path:
    """Write the excerpt the tone analysis should listen to."""
    start_s, length_s = sample_window(duration_s)
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-y", "-v", "error",
            "-ss", f"{start_s:.3f}", "-t", f"{length_s:.3f}", "-i", str(source),
            "-ac", "1", "-ar", "24000", str(destination),
        ],
        check=True,
        capture_output=True,
    )
    return destination

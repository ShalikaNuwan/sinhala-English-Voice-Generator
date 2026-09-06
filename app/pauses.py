"""Shapes the silences in a voiced segment so the breaks sound like a person telling a story.

The voice model produces the speech; this module decides the pauses. Word timestamps tie each
detected pause to the script, the script's punctuation says what kind of break it is, and each kind
gets a human-like length with a little variation. Speech is never stretched or re-levelled.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import statistics
import subprocess
from pathlib import Path


# Target pause lengths in ms for an intimate English storyteller at a moderate pace.
BANDS: dict[str, tuple[int, int]] = {
    "clause": (150, 280),
    "dash": (250, 400),
    "sentence": (420, 650),
    "paragraph": (850, 1200),
    "beat": (1200, 1600),
}
# Scale applied to every band from the source narrator's derived pace.
PACE_SCALE = {"continuous": 0.85, "moderate": 1.0, "measured": 1.15}
# A break the script did not ask for is a breath, not a pause.
UNPUNCTUATED_MAX_MS = 200
# Silence kept at the start and end of a segment; the assembly gap supplies the rest.
EDGE_SILENCE_MS = 60
SILENCE_DB = -40
SILENCE_MIN_S = 0.08
# A word may end this long after a pause starts and still be the word before it.
WORD_TOLERANCE_S = 0.15
# Without word timestamps only clear pauses are matched, in order, to script breaks.
FALLBACK_MIN_PAUSE_S = 0.25
# A pause may exceed the longest band by this much before QA complains.
OVERRUN_TOLERANCE_MS = 200

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_CLOSERS = "\"'”’)]"
_BREAK_ORDER = ["clause", "dash", "sentence", "paragraph", "beat"]


class AlignmentError(RuntimeError):
    """The pauses could not be tied to the script; the raw audio should be kept."""


def normalise(word: str) -> str:
    return _NON_ALNUM.sub("", word.lower())


def _class_after(raw: str) -> str:
    stripped = raw.rstrip(_CLOSERS)
    if stripped in {"—", "–", "-"}:
        return "dash"
    if stripped.endswith("…") or stripped.endswith("..."):
        return "beat"
    if stripped.endswith(("—", "–")):
        return "dash"
    if stripped.endswith((".", "!", "?")):
        return "sentence"
    if stripped.endswith((",", ";", ":")):
        return "clause"
    return "none"


def script_tokens(script: str) -> list[tuple[str, str]]:
    """(normalised word, break class after it). A blank line turns the last break into a paragraph."""
    tokens: list[tuple[str, str]] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", (script or "").strip()) if p.strip()]
    for p_index, paragraph in enumerate(paragraphs):
        words = paragraph.split()
        for w_index, raw in enumerate(words):
            cls = _class_after(raw)
            if w_index == len(words) - 1 and p_index < len(paragraphs) - 1 and cls != "beat":
                cls = "paragraph"
            norm = normalise(raw)
            if not norm:
                if tokens and cls != "none":
                    tokens[-1] = (tokens[-1][0], cls)
                continue
            tokens.append((norm, cls))
    return tokens


def align(tokens: list[tuple[str, str]], spoken: list[dict]) -> dict[int, int]:
    """Spoken word index -> script token index for words the transcriber got right, with single
    mismatches between two matches filled in by position."""
    heard = [normalise(item.get("word") or "") for item in spoken]
    expected = [word for word, _ in tokens]
    matcher = difflib.SequenceMatcher(a=heard, b=expected, autojunk=False)
    mapping: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapping[block.a + offset] = block.b + offset
    for index in range(1, len(heard) - 1):
        before, after = mapping.get(index - 1), mapping.get(index + 1)
        if index not in mapping and before is not None and after is not None and after - before == 2:
            mapping[index] = before + 1
    return mapping


def _duration_s(path: Path, ffprobe: str = "ffprobe") -> float:
    output = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    return float(output) if output else 0.0


def detect_pauses(path: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> list[tuple[float, float]]:
    """Silences in seconds, including any at the very start or end."""
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", f"silencedetect=noise={SILENCE_DB}dB:d={SILENCE_MIN_S}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start:\s*(-?[0-9.]+)", result.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", result.stderr)]
    if len(ends) < len(starts):
        ends.append(_duration_s(path, ffprobe))
    return [(max(0.0, start), end) for start, end in zip(starts, ends) if end > start]


def classify(
    pauses: list[tuple[float, float]],
    spoken: list[dict],
    mapping: dict[int, int],
    tokens: list[tuple[str, str]],
) -> list[tuple[float, float, str]]:
    """Name each pause by the break that follows the last word spoken before it."""
    result = []
    for start, end in pauses:
        preceding = [index for index, item in enumerate(spoken) if float(item.get("end") or 0) <= start + WORD_TOLERANCE_S]
        cls = "none"
        if preceding:
            token_index = mapping.get(preceding[-1])
            if token_index is not None:
                cls = tokens[token_index][1]
        result.append((start, end, cls))
    return result


def classify_by_order(
    pauses: list[tuple[float, float]], tokens: list[tuple[str, str]], duration_s: float,
) -> list[tuple[float, float, str]] | None:
    """Without word timestamps: match clear interior pauses, in order, to the script's breaks."""
    interior = [(s, e) for s, e in pauses if s > 0.005 and e < duration_s - 0.005]
    clear = [(s, e) for s, e in interior if e - s >= FALLBACK_MIN_PAUSE_S]
    breaks = [cls for _, cls in tokens[:-1] if cls != "none"]
    if len(clear) != len(breaks):
        return None
    classes = dict(zip(clear, breaks))
    return [(s, e, classes.get((s, e), "none")) for s, e in pauses]


def choose_targets(
    classified: list[tuple[float, float, str]], pace: str, rng: random.Random, duration_s: float,
) -> list[tuple[float, float, int]]:
    """Pick a length for every pause: a jittered band value, a capped breath, or a trimmed edge."""
    scale = PACE_SCALE.get(pace, 1.0)
    plan = []
    for start, end, cls in classified:
        current_ms = round((end - start) * 1000)
        if start <= 0.005 or end >= duration_s - 0.005:
            target = min(current_ms, EDGE_SILENCE_MS)
        elif cls in BANDS:
            low, high = BANDS[cls]
            target = round(rng.uniform(low * scale, high * scale))
        else:
            target = min(current_ms, UNPUNCTUATED_MAX_MS)
        plan.append((start, end, target))
    return plan


def rebuild(path: Path, plan: list[tuple[float, float, int]], duration_s: float, destination: Path, ffmpeg: str = "ffmpeg") -> None:
    """Re-time every pause in one FFmpeg pass; speech runs are copied untouched."""
    parts: list[str] = []
    labels: list[str] = []
    cursor = 0.0

    def speech(start: float, end: float) -> None:
        if end - start <= 0.001:
            return
        label = f"[s{len(labels)}]"
        parts.append(f"[0:a]atrim=start={start:.4f}:end={end:.4f},asetpts=PTS-STARTPTS{label}")
        labels.append(label)

    def silence(ms: int) -> None:
        if ms <= 0:
            return
        label = f"[g{len(labels)}]"
        parts.append(f"aevalsrc=0:d={ms / 1000:.4f}:s=24000:c=mono{label}")
        labels.append(label)

    for start, end, target in sorted(plan):
        speech(cursor, start)
        silence(target)
        cursor = end
    speech(cursor, duration_s)
    if not labels:
        raise RuntimeError("Nothing to rebuild")
    graph = ";".join(parts + [f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", str(path), "-filter_complex", graph,
         "-map", "[out]", "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", str(destination)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "Pause rebuild failed")[-1200:])


def pause_profile(path: Path, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe") -> dict:
    """How the pauses of a file are distributed, edges excluded."""
    duration = _duration_s(path, ffprobe)
    interior = [(s, e) for s, e in detect_pauses(path, ffmpeg, ffprobe) if s > 0.005 and e < duration - 0.005]
    lengths = sorted(round((e - s) * 1000) for s, e in interior)
    if not lengths:
        return {"count": 0, "median_ms": 0, "p90_ms": 0, "max_ms": 0, "micro": 0, "per_minute": 0.0}
    return {
        "count": len(lengths),
        "median_ms": round(statistics.median(lengths)),
        "p90_ms": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))],
        "max_ms": lengths[-1],
        "micro": sum(1 for ms in lengths if ms < 300),
        "per_minute": round(len(lengths) / (duration / 60), 1) if duration else 0.0,
    }


def profile_issues(result: dict) -> list[str]:
    """QA text for a shaped segment whose pauses are still outside the natural range."""
    issues = []
    profile = result.get("profile") or {}
    scale = PACE_SCALE.get(result.get("pace", "moderate"), 1.0)
    ceiling = round(BANDS["beat"][1] * scale) + OVERRUN_TOLERANCE_MS
    if profile.get("max_ms", 0) > ceiling:
        issues.append(f"Pauses: longest pause {profile['max_ms']} ms exceeds the natural ceiling of {ceiling} ms")
    if result.get("unpunctuated_over_cap", 0):
        issues.append(f"Pauses: {result['unpunctuated_over_cap']} unpunctuated break(s) over {UNPUNCTUATED_MAX_MS} ms remain")
    return issues


def shape(
    path: Path,
    script: str,
    spoken: list[dict],
    pace: str,
    destination: Path,
    seed: str,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> dict:
    """Shape every pause of a voiced segment and return what was done and how it measures."""
    duration = _duration_s(path, ffprobe)
    found = detect_pauses(path, ffmpeg, ffprobe)
    tokens = script_tokens(script)
    if spoken:
        mapping = align(tokens, spoken)
        if spoken and len(mapping) < max(1, len(spoken) // 2):
            raise AlignmentError(f"only {len(mapping)} of {len(spoken)} spoken words matched the script")
        classified = classify(found, spoken, mapping, tokens)
        method = "words"
    else:
        classified = classify_by_order(found, tokens, duration)
        if classified is None:
            raise AlignmentError("no word timestamps and the pauses do not match the script's breaks")
        method = "order"
    plan = choose_targets(classified, pace, random.Random(seed), duration)
    rebuild(path, plan, duration, destination, ffmpeg)
    classes: dict[str, int] = {}
    over_cap = 0
    for (start, end, cls), (_, _, target) in zip(classified, plan):
        name = "edge" if start <= 0.005 or end >= duration - 0.005 else cls
        classes[name] = classes.get(name, 0) + 1
        if name == "none" and target > UNPUNCTUATED_MAX_MS:
            over_cap += 1
    return {
        "method": method,
        "pace": pace,
        "classes": classes,
        "unpunctuated_over_cap": over_cap,
        "plan": [{"start": round(s, 3), "end": round(e, 3), "target_ms": t} for s, e, t in plan],
        "profile": pause_profile(destination, ffmpeg, ffprobe),
    }

# Natural Pauses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every pause in the voiced narration a pipeline decision with human-like lengths, tied to the script's punctuation, measured and checked, instead of whatever the voice model happened to produce.

**Architecture:** A new pure-plus-FFmpeg module `app/pauses.py` detects the silences in a raw voiced segment, aligns them to the script through word timestamps from `whisper-1`, classifies each by the punctuation that precedes it, chooses a target length inside a per-class band (scaled by the narrator's pace, jittered by a seeded random generator), rebuilds the audio in one FFmpeg call without touching the speech, and measures the result. The pipeline calls it after synthesis; QA reports the pause profile and flags overruns. The direction drops its numeric pause sentence and the adaptation prompt limits paragraph breaks and ellipses.

**Tech Stack:** Python 3.11+, FFmpeg 4.2 (`silencedetect`, `atrim`, `aevalsrc`, `concat`), OpenAI SDK (`audio.transcriptions` with `verbose_json` + `timestamp_granularities=["word"]`), difflib, pytest. Spec: `docs/superpowers/specs/2026-09-06-natural-pauses-design.md`. Branch: `natural-pauses`.

Run tests with `.venv/bin/python -m pytest -p no:warnings`. No test makes a paid call; the paid checks are the final task, run by the controller.

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `app/pauses.py` | Tokenise script breaks, align words, detect/classify/target pauses, rebuild audio, profile and issues, `shape`, `AlignmentError`. | Create |
| `tests/test_pauses.py` | Unit tests for all of the above with synthetic audio. | Create |
| `app/ai.py` | `AIClient.word_timestamps`; adaptation prompt limits. | Modify |
| `app/direction.py` | Drop the numeric pause sentence. | Modify |
| `app/config.py`, `app/schemas.py`, `app/main.py` | `ALIGN_MODEL`, `SHAPE_PAUSES`, request overrides, job config. | Modify |
| `app/pipeline.py` | `_shape_pauses`; raw→shaped in `_narrate_segment` and regeneration; pause profile into QA. | Modify |
| `scripts/pause_listening_set.py` | Raw vs shaped listening set with pause tables. | Create |
| `README.md`, `.env.example` | Document. | Modify |
| `tests/test_ai.py`, `tests/test_direction.py`, `tests/test_api.py`, `tests/test_pipeline.py` | Updated. | Modify |

---

### Task 1: `app/pauses.py`

**Files:**
- Create: `app/pauses.py`
- Create: `tests/test_pauses.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pauses.py`:

```python
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


def test_classify_gives_none_to_an_unmapped_word():
    tokens = pauses.script_tokens("Hello there.")
    heard = spoken(("Goodbye", 0.0, 0.5), ("there", 0.9, 1.3))

    classified = pauses.classify([(0.5, 0.9)], heard, pauses.align(tokens, heard), tokens)

    assert classified == [(0.5, 0.9, "none")]


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
    assert len(gaps) == 3
    assert abs(gaps[0] - 250) <= 30 and abs(gaps[1] - 1000) <= 30 and abs(gaps[2] - 60) <= 30
    runs = [found[0][0], found[1][0] - found[0][1], found[2][0] - found[1][1]]
    assert all(abs(run - 1.0) < 0.05 for run in runs)


def test_profile_and_issues(tmp_path):
    path = bursts(tmp_path / "in.wav", [(1.0, 0.35), (1.0, 0.5), (1.0, 2.2), (1.0, 0.05)])

    profile = pauses.pause_profile(path)

    assert profile["count"] == 3  # the 50 ms tail is below the detection floor
    assert profile["max_ms"] >= 2150
    assert profile["micro"] == 0
    issues = pauses.profile_issues({"profile": profile, "unpunctuated_over_cap": 1, "pace": "moderate"})
    assert any("longest pause" in issue for issue in issues)
    assert any("unpunctuated" in issue for issue in issues)
    assert pauses.profile_issues({"profile": {"max_ms": 900, "micro": 0}, "unpunctuated_over_cap": 0, "pace": "moderate"}) == []


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
    assert clause_low * 0.85 - 30 <= gaps[0] <= clause_high * 0.85 + 30
    assert sentence_low * 0.85 - 30 <= gaps[1] <= sentence_high * 0.85 + 30
    assert gaps[2] <= pauses.EDGE_SILENCE_MS + 30
    assert result["classes"] == {"clause": 1, "sentence": 1, "edge": 1}
    assert result["profile"]["count"] == 2


def test_shape_falls_back_to_punctuation_order_without_timestamps(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    result = pauses.shape(source, "First part, second part. Third part.", [], "moderate", tmp_path / "shaped.wav", seed="seg")

    assert result["method"] == "order"
    assert result["classes"] == {"clause": 1, "sentence": 1, "edge": 1}


def test_shape_refuses_when_nothing_lines_up(tmp_path):
    source = bursts(tmp_path / "raw.wav", [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    with pytest.raises(pauses.AlignmentError):
        pauses.shape(source, "One sentence with no breaks at all", [], "moderate", tmp_path / "shaped.wav", seed="seg")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pauses.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pauses'`

- [ ] **Step 3: Create `app/pauses.py`**

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pauses.py`
Expected: 17 passed. If a gap measured by `detect_pauses` on the rebuilt file is off by more than the 30 ms tolerance, print the measured values and check the `aevalsrc` duration and the `-sample_fmt s16` output before changing anything; the tone bursts are full-scale so silence edges are sharp.

- [ ] **Step 5: Commit**

```bash
git add app/pauses.py tests/test_pauses.py
git commit -m "Add pause shaping: align pauses to the script and re-time them to human bands"
```

---

### Task 2: Word timestamps, settings, direction and prompt changes

**Files:**
- Modify: `app/ai.py` (`word_timestamps`; adaptation prompt limits)
- Modify: `app/direction.py` (`_character` drops the numeric pause sentence)
- Modify: `app/config.py`, `app/schemas.py`, `app/main.py`, `.env.example`
- Test: `tests/test_ai.py`, `tests/test_direction.py`, `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ai.py`:

```python
def test_word_timestamps_ask_whisper_for_word_granularity(tmp_path):
    captured = {}

    class Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(words=[
                SimpleNamespace(word="On", start=0.0, end=0.24),
                SimpleNamespace(word="May", start=0.24, end=0.5),
            ])

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
    raw = tmp_path / "raw.wav"
    raw.write_bytes(b"RIFFraw")

    words = ai.word_timestamps(raw, "whisper-1")

    assert captured["model"] == "whisper-1"
    assert captured["response_format"] == "verbose_json"
    assert captured["timestamp_granularities"] == ["word"]
    assert captured["language"] == "en"
    assert words == [{"word": "On", "start": 0.0, "end": 0.24}, {"word": "May", "start": 0.24, "end": 0.5}]


def test_adaptation_limits_paragraph_breaks_and_ellipses():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=NarrationAdaptation(narration_text="ok"))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.adapt("x", "m", {})

    system = captured["input"][0]["content"]
    assert "at most twice per passage" in system
    assert "at most once per passage" in system
```

In `tests/test_direction.py`, change `test_brief_carries_the_narrator_profile_and_the_moment` so the line `assert "1321" in text and "3547" in text` becomes `assert "1321" not in text and "3547" not in text  # pause lengths are shaped after synthesis, not requested`, and in `test_brief_uses_measurements_when_the_tone_analysis_failed` replace `assert "1321" in text` with `assert "1321" not in text`. In `test_brief_fills_gaps_in_a_partial_profile`, delete the line `assert "the longest pause is about 900ms" in text`.

Append to `tests/test_api.py`:

```python
def test_job_configuration_shapes_pauses_by_default():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["align_model"] == "whisper-1"
    assert config["shape_pauses"] is True
    assert job_configuration(ProcessRequest(shape_pauses=False, align_model="w2"), Settings(openai_api_key="unused"))["shape_pauses"] is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_ai.py tests/test_direction.py tests/test_api.py`
Expected: the new and changed tests FAIL.

- [ ] **Step 3: `app/ai.py`**

Add after `diarize`:

```python
    def word_timestamps(self, audio_path: Path, model: str) -> list[dict]:
        """When each word of a voiced segment is spoken, so pauses can be tied to the script."""
        with Path(audio_path).open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
                language="en",
            )
        return [
            {
                "word": _field(item, "word") or "",
                "start": float(_field(item, "start") or 0.0),
                "end": float(_field(item, "end") or 0.0),
            }
            for item in (_field(result, "words") or [])
        ]
```

In the adaptation system prompt, replace the ellipsis bullet and the paragraph bullet so they read:

```
"- Put an ellipsis (…) where the narrator should hold a beat, at most once per passage, and a dash "
"(—) for a change of thought or an afterthought, at most twice. Use the characters … and — "
"themselves, not ... or --, and only where a person telling the story would actually pause.\n"
"- Start a new paragraph (blank line) only at a real shift in the story, at most twice per passage.\n"
```

- [ ] **Step 4: `app/direction.py`**

In `_character`, delete the whole `if measured.get("mean_pause_ms"):` block (the sentence quoting the average and longest pause). Leave `measured` in use only if still referenced; if it is no longer referenced, delete the `measured = ...` line too.

- [ ] **Step 5: settings, request, job config, env example**

`app/config.py`, after `detect_recordings`:

```python
    align_model: str = field(default_factory=lambda: os.getenv("ALIGN_MODEL", "whisper-1"))
    shape_pauses: bool = field(
        default_factory=lambda: os.getenv("SHAPE_PAUSES", "1").strip().lower() not in {"0", "false", "no"}
    )
```

`app/schemas.py`, `ProcessRequest` after `detect_recordings`:

```python
    align_model: str | None = None
    shape_pauses: bool | None = None
```

`app/main.py`, `job_configuration` after `"detect_recordings"`:

```python
        "align_model": payload.align_model or config.align_model,
        "shape_pauses": payload.shape_pauses if payload.shape_pauses is not None else config.shape_pauses,
```

`.env.example`, after `DETECT_RECORDINGS=1`:

```
ALIGN_MODEL=whisper-1
SHAPE_PAUSES=1
```

- [ ] **Step 6: Run the suite**

Run: `.venv/bin/python -m pytest -p no:warnings`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add app/ai.py app/direction.py app/config.py app/schemas.py app/main.py .env.example tests/test_ai.py tests/test_direction.py tests/test_api.py
git commit -m "Add word timestamps and pause-shaping settings; stop asking the voice for pause lengths"
```

---

### Task 3: Pipeline integration

**Files:**
- Modify: `app/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update the fakes and write the failing tests**

In `tests/test_pipeline.py`:

Add to `FakeAI`:

```python
    def word_timestamps(self, _audio_path, _model):
        return []
```

Add a helper near `create_source`:

```python
def bursts(path: Path, pattern: list[tuple[float, float]]) -> None:
    """Tone bursts separated by silences: pattern is [(tone_s, silence_after_s), ...]."""
    inputs, labels = [], []
    for tone_s, gap_s in pattern:
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
```

Add a fake after `BoundaryFakeAI`:

```python
class PausingFakeAI(ProfilingFakeAI):
    """Voices a three-part script as tone bursts with a 0.9 s and a 1.4 s pause, and knows the word times."""

    def __init__(self, timestamps_error: Exception | None = None):
        super().__init__()
        self.timestamps_error = timestamps_error
        self.timestamp_calls = 0

    def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration})
        return NarrationAdaptation(narration_text="First part, second part. Third part.", beat="build", emotion="calm")

    def synthesize(self, text, model, voice, style, output_path, profile=None, previous_style=None, persona=None, speed=1.0):
        self.synthesis_calls.append({"text": text, "voice": voice, "style": style, "output_path": output_path,
                                     "previous_style": previous_style, "persona": persona, "speed": speed})
        bursts(Path(output_path), [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    def word_timestamps(self, _audio_path, _model):
        self.timestamp_calls += 1
        if self.timestamps_error:
            raise self.timestamps_error
        return [{"word": w, "start": s, "end": e} for w, s, e in
                (("First", 0.0, 0.5), ("part", 0.5, 1.0), ("second", 1.9, 2.4), ("part", 2.4, 2.9), ("Third", 4.3, 4.8), ("part", 4.8, 5.3))]
```

In `start_job`, add `"align_model": "fake-align", "shape_pauses": shape_pauses,` to `job_config`, and give `start_job` a keyword `shape_pauses: bool = True`.

Append these tests:

```python
def measured_gaps(path: Path) -> list[int]:
    from app import pauses
    found = pauses.detect_pauses(path)
    duration = pauses._duration_s(path)
    return [round((e - s) * 1000) for s, e in found if s > 0.005 and e < duration - 0.005]


def test_pauses_are_shaped_after_synthesis(tmp_path):
    from app import pauses

    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["tts_audio_path"].endswith("0001_r01.wav")
    assert Path(segment["tts_audio_path"]).with_name("0001_r01_raw.wav").exists()
    gaps = measured_gaps(Path(segment["tts_audio_path"]))
    # A pure-tone source is a "continuous" narrator, so bands are scaled by 0.85.
    assert pauses.BANDS["clause"][0] * 0.85 - 30 <= gaps[0] <= pauses.BANDS["clause"][1] * 0.85 + 30
    assert pauses.BANDS["sentence"][0] * 0.85 - 30 <= gaps[1] <= pauses.BANDS["sentence"][1] * 0.85 + 30
    assert segment["qa"]["pauses"]["method"] == "words"
    assert segment["qa"]["pauses"]["classes"] == {"clause": 1, "sentence": 1, "edge": 1}
    assert ai.timestamp_calls == 1
    assert db.one("SELECT stage FROM model_calls WHERE job_id=? AND stage='alignment'", (segment["job_id"],))


def test_shaping_can_be_switched_off(tmp_path):
    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id, shape_pauses=False))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert ai.timestamp_calls == 0
    assert measured_gaps(Path(segment["tts_audio_path"])) == pytest.approx([900, 1400], abs=40)
    assert "pauses" not in segment["qa"]


def test_timestamp_failure_falls_back_to_punctuation_order(tmp_path):
    ai = PausingFakeAI(timestamps_error=RuntimeError("no whisper"))
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["qa"]["pauses"]["method"] == "order"
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Word timestamps failed on segment 1" in w for w in warnings)


def test_unshapeable_audio_keeps_the_raw_voice_with_a_warning(tmp_path):
    class LonelyFakeAI(PausingFakeAI):
        def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
            return NarrationAdaptation(narration_text="One sentence with no breaks at all")

    ai = LonelyFakeAI(timestamps_error=RuntimeError("no whisper"))
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert measured_gaps(Path(segment["tts_audio_path"])) == pytest.approx([900, 1400], abs=40)
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Pauses left as voiced on segment 1" in w for w in warnings)


def test_regenerating_tts_shapes_pauses_too(tmp_path):
    from app import pauses

    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id))
    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(segment["id"], "tts")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
    assert fresh["tts_audio_path"].endswith("0001_r02.wav")
    gaps = measured_gaps(Path(fresh["tts_audio_path"]))
    assert pauses.BANDS["clause"][0] * 0.85 - 30 <= gaps[0] <= pauses.BANDS["clause"][1] * 0.85 + 30
    assert fresh["qa"]["pauses"]["method"] == "words"
```

Add `import pytest` at the top of `tests/test_pipeline.py` if it is not already imported.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pipeline.py`
Expected: the five new tests FAIL; existing tests pass.

- [ ] **Step 3: Edit `app/pipeline.py`**

Add `import shutil` to the imports and `from . import pauses, recordings, speaking_profile` in place of the existing `from . import recordings, speaking_profile`.

Add this method after `_qa_verdict`:

```python
    def _shape_pauses(
        self, ai: AIClient, job: dict, segment: dict, raw_path: Path, final_path: Path, script: str, profile: dict | None,
    ) -> dict | None:
        """Turn the voice's pauses into shaped ones; on any failure the raw voice is kept and a warning says why."""
        config = job["config"]
        if not config.get("shape_pauses", self.config.shape_pauses):
            raw_path.replace(final_path)
            return None
        pace = ((profile or {}).get("derived") or {}).get("pace") or "moderate"
        align_model = config.get("align_model", self.config.align_model)
        spoken: list[dict] = []
        try:
            spoken = self._tracked_call(
                job["id"], segment["id"], "alignment", align_model, str(raw_path),
                lambda: ai.word_timestamps(raw_path, align_model),
            )
        except Exception as exc:  # noqa: BLE001 - alignment is an enhancement; shaping can still try by order
            traceback.print_exc()
            self._warn(job["id"], f"Word timestamps failed on segment {segment['segment_index']}: {str(exc)[:200]}")
        try:
            return pauses.shape(raw_path, script, spoken, pace, final_path, seed=segment["id"], ffmpeg=self.config.ffmpeg, ffprobe=self.config.ffprobe)
        except pauses.AlignmentError as exc:
            self._warn(job["id"], f"Pauses left as voiced on segment {segment['segment_index']}: {exc}")
            shutil.copyfile(raw_path, final_path)
            return None
```

Change `_qa_verdict`'s signature to `def _qa_verdict(self, qa: QAEvaluation, transcript: str, tts_duration_ms: int | None, segment: dict, shaped: dict | None = None) -> dict:` and add before `return payload`:

```python
        if shaped:
            payload["pauses"] = {key: shaped[key] for key in ("method", "pace", "classes", "profile")}
            for issue in pauses.profile_issues(shaped):
                payload["passed"] = False
                payload["issues"] = payload["issues"] + [issue]
```

In `_narrate_segment`, replace the TTS block (from `tts_path = ...` through the `UPDATE segments SET tts_audio_path=?` execute) with:

```python
        stem = f"{segment['segment_index']:04d}_r{segment['revision']:02d}"
        generated = self._generated_dir(project["id"], job_id)
        raw_path, tts_path = generated / f"{stem}_raw.wav", generated / f"{stem}.wav"
        self._tracked_call(
            job_id, segment_id, "tts", config["tts_model"], adaptation.narration_text,
            lambda: ai.synthesize(
                adaptation.narration_text, config["tts_model"], config["voice"], style, raw_path,
                profile, previous_style, persona, speed,
            ),
        )
        shaped = self._shape_pauses(ai, job, segment, raw_path, tts_path, adaptation.narration_text, profile)
        tts_duration = self.audio.probe(tts_path).duration_ms
        self.db.execute(
            "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, status='generated', updated_at=? WHERE id=?",
            (str(tts_path), tts_duration, utc_now(), segment_id),
        )
```

and pass `shaped` into the verdict: `qa_payload = self._qa_verdict(qa, transcript, tts_duration, segment, shaped)`.

In `regenerate_segment`, in the `if stage == "tts":` block, replace the path and synthesize lines with:

```python
                revision = segment["revision"] + 1
                stem = f"{segment['segment_index']:04d}_r{revision:02d}"
                generated = self._generated_dir(project["id"], job["id"])
                raw_path, tts_path = generated / f"{stem}_raw.wav", generated / f"{stem}.wav"
                self._tracked_call(
                    job["id"], segment_id, "tts", config["tts_model"], narration_text,
                    lambda: ai.synthesize(
                        narration_text, config["tts_model"], config["voice"], style, raw_path,
                        project.get("speaking_profile"), previous_style, persona, speed,
                    ),
                )
                shaped = self._shape_pauses(ai, job, segment, raw_path, tts_path, narration_text, project.get("speaking_profile"))
```

keeping the following `duration = self.audio.probe(tts_path).duration_ms` and UPDATE. Initialise `shaped = None` before the `if stage == "translation":` chain, and pass it to the verdict at the end: `qa_payload = self._qa_verdict(qa, fresh["transcript_si"], fresh.get("tts_duration_ms"), fresh, shaped)`.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pipeline.py` then the full suite.
Expected: all pass. If `test_pauses_are_shaped_after_synthesis` fails on the gap bands, print `segment["qa"]["pauses"]` and the measured gaps and report; do not widen the bands.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "Shape pauses after synthesis and report the pause profile in QA"
```

---

### Task 4: Listening set script and README

**Files:**
- Create: `scripts/pause_listening_set.py`
- Modify: `README.md`

- [ ] **Step 1: Create `scripts/pause_listening_set.py`**

```python
"""Copy raw and shaped voice files for chosen segments so the owner can compare the pauses by ear.

Usage:
    .venv/bin/python scripts/pause_listening_set.py <job_id> <segment_index> [<segment_index> ...]

Writes data/pause_listening/<job_id>/NN_raw.wav, NN_shaped.wav, and pauses.md with a pause table
for each, plus the reference file the owner named. Makes no API calls.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pauses  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402

REFERENCE = Path.home() / "Downloads" / "final.mp3"


def table(path: Path) -> str:
    profile = pauses.pause_profile(path, settings.ffmpeg, settings.ffprobe)
    return (f"| {path.name} | {profile['count']} | {profile['median_ms']} | {profile['p90_ms']} | "
            f"{profile['max_ms']} | {profile['micro']} | {profile['per_minute']} |")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    job_id, indexes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    db = Database(settings.database_path)
    out = settings.data_dir / "pause_listening" / job_id
    out.mkdir(parents=True, exist_ok=True)
    lines = ["| file | pauses | median ms | p90 ms | max ms | under 300 ms | per minute |", "|---|---|---|---|---|---|---|"]
    if REFERENCE.exists():
        lines.append(table(REFERENCE))
    for index in indexes:
        segment = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index))
        if not segment or segment.get("kind") != "narration" or not segment.get("tts_audio_path"):
            print("skipping", index)
            continue
        shaped = Path(segment["tts_audio_path"])
        raw = shaped.with_name(shaped.stem + "_raw.wav")
        for source, name in ((raw, f"{index:02d}_raw.wav"), (shaped, f"{index:02d}_shaped.wav")):
            if source.exists():
                shutil.copyfile(source, out / name)
                lines.append(table(out / name))
        lines.append(f"| script {index} | {segment['narration_en'][:120]!r} | | | | | |")
    (out / "pauses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("written to", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: README**

Configuration table: add `ALIGN_MODEL` (`whisper-1`) and `SHAPE_PAUSES` (`1`) after `DETECT_RECORDINGS`. In "Narration direction", replace the paragraph beginning "Pace is driven through the brief." with:

```
Pace is driven through the brief. `TTS_SPEED` (or `speed` on the process request) is passed to the
API and defaults to `1.0`. The brief no longer quotes pause lengths: pauses are shaped after synthesis
(see below).
```

Add a section after "Embedded recordings":

```markdown
## Natural pauses

The voice model decides how the words are spoken; the pipeline decides the silences. After each
narration segment is voiced, the raw file is kept as `NNNN_rNN_raw.wav` and the pauses are detected
with FFmpeg. One `whisper-1` call with word timestamps ties every pause to its place in the script,
and the script's punctuation says what kind of break it is: a clause break after a comma, a dash, a
sentence end, a paragraph shift after a blank line, or a held beat after an ellipsis. Each kind gets
a length drawn from a band tuned for an intimate English storyteller (clause 150–280 ms, dash
250–400, sentence 420–650, paragraph 850–1200, beat 1200–1600), scaled shorter for a continuous
narrator and longer for a measured one, with a little variation so nothing is metronomic. A break the
script did not ask for is capped at 200 ms. Leading and trailing silence are trimmed to 60 ms so the
assembly gap is the only gap at a join. Speech itself is never stretched or re-levelled. The bands
live at the top of `app/pauses.py`.

QA records the pause profile of every shaped segment and flags one whose longest pause exceeds the
natural ceiling or that still has unpunctuated breaks over 200 ms. If word timestamps fail, clear
pauses are matched in order to the script's breaks; if that does not line up either, the raw voice is
kept and the job says so. `SHAPE_PAUSES=0` (or `shape_pauses` on the process request) turns shaping
off.

`scripts/pause_listening_set.py <job_id> <segment_index...>` copies the raw and shaped files for
chosen segments to `data/pause_listening/` with a pause table for each, for listening.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/pause_listening_set.py README.md
git commit -m "Add the pause listening set and document natural pauses"
```

---

### Task 5: Real-model verification (controller)

- [ ] Run `set -a && source .env && set +a && .venv/bin/python scripts/end_to_end_check.py 5220183a-664a-4f9b-a534-2c3cc2ba8ff7 0 130`.
- [ ] Run `.venv/bin/python scripts/pause_listening_set.py <job_id> 1 2 5` on the new job and read `pauses.md`.
- [ ] Compare the shaped profiles against the 5 September reference and the bands; report the table; deliver the files for listening.

## Final verification

- [ ] `.venv/bin/python -m pytest -p no:warnings` — all pass.
- [ ] `git status --short` — clean.

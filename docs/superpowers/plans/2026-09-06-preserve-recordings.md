# Preserve Embedded Recordings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect evidence audio embedded in the creator's recording (911 calls, interrogation tape), keep it untouched in the English export, and voice only the narrator's own speech.

**Architecture:** A new pure module `app/recordings.py` turns diarization spans into recording spans and a cut plan. `AIClient.diarize` calls OpenAI's `gpt-4o-transcribe-diarize` with a narrator reference clip chosen once per project. The pipeline re-cuts each silence-based chunk into `narration` and `original` segments (new `segments.kind` column); original segments skip every AI stage, are cut from the normalised source with loudness alignment, and go to the reviewer as needs-review until confirmed. Two endpoints let the reviewer confirm a recording or flip a segment's kind. Assembly is unchanged except for a fixed 300 ms gap next to recordings.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI SDK 2.x (`audio.transcriptions` with `response_format="diarized_json"`, `known_speaker_names`, `known_speaker_references`), FFmpeg, SQLite, pytest. Spec: `docs/superpowers/specs/2026-09-06-preserve-recordings-design.md`. Branch: `preserve-recordings` (built on `natural-narration`).

Run tests with `.venv/bin/python -m pytest -p no:warnings` (the project's pyproject already adds `-q`). Test fixture `tests/fixtures/gilgo_probe_spans.json` holds real diarization output from the Gilgo Beach project: key `sample` (the delivery sample, no reference, one speaker `A`) and key `chunk2` (the 911-call chunk with a narrator reference; chunk length 37878 ms). No test makes a paid call except Task 7, which is the approved end-to-end check.

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `app/recordings.py` | Pure span logic: script test, reference choice, recording spans, narration spans, cut plan, clip text, English-run safety net. | Create |
| `tests/test_recordings.py` | Unit tests for the above, including the real probe fixture. | Create |
| `app/ai.py` | `AIClient.diarize`. | Modify |
| `app/audio.py` | `extract` and `extract_levelled`; `split` uses `extract`. | Modify |
| `app/database.py` | `segments.kind`, `jobs.warnings_json` (schema + migration). | Modify |
| `app/config.py`, `app/schemas.py`, `app/main.py` | `DIARIZE_MODEL`, `DETECT_RECORDINGS`, request overrides, `KindUpdate`, two endpoints. | Modify |
| `app/pipeline.py` | Narrator reference, detection and re-cut, `_narrate_segment`, `_qa_verdict`, `set_segment_kind`, `confirm_segment`, gaps next to recordings. | Modify |
| `app/static/app.js`, `app/static/index.html` | Recording cards, confirm/flip buttons, warnings in the progress line. | Modify |
| `README.md` | Document. | Modify |
| `scripts/end_to_end_check.py` | Real-model run on a slice of an existing project. | Create |
| `tests/test_ai.py`, `tests/test_audio.py`, `tests/test_database.py`, `tests/test_api.py`, `tests/test_pipeline.py` | Updated. | Modify |

---

### Task 1: Recording span logic in `app/recordings.py`

**Files:**
- Create: `app/recordings.py`
- Create: `tests/test_recordings.py`
- Uses: `tests/fixtures/gilgo_probe_spans.json` (already committed)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_recordings.py`:

```python
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
    assert recordings.is_english("Hello? Do you need the police?")
    assert not recordings.is_english(SINHALA)
    assert not recordings.is_english("")


def test_reference_is_the_dominant_speakers_longest_span_trimmed_to_ten_seconds():
    assert recordings.choose_reference(SAMPLE_SPANS) == (16.43, 26.43)


def test_reference_prefers_the_speaker_with_the_most_speech():
    spans = [span("A", 0, 3, SINHALA), span("B", 3, 8, SINHALA), span("A", 8, 12, SINHALA)]

    assert recordings.choose_reference(spans) == (8.0, 12.0)


def test_reference_needs_at_least_two_seconds():
    assert recordings.choose_reference([span("A", 0, 1.5, SINHALA), span("A", 2, 3.4, SINHALA)]) is None
    assert recordings.choose_reference([]) is None


def test_the_real_911_call_chunk_yields_one_recording():
    assert recordings.original_spans(CHUNK2_SPANS, CHUNK2_MS) == [(13750, 36640)]


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
    text = recordings.clip_text(CHUNK2_SPANS, 13750, 36640)

    assert text.startswith("Hi, how can I assist you? Hello? Hello?")
    assert text.endswith("Do you need the police? Where?")
    assert "නිව්" not in text


def test_english_run_flags_a_recording_left_in_a_transcript():
    transcript = SINHALA + " 9-1-1, how can I assist you? Hello? Hello? Hello, you dialed into the 911 system."

    assert recordings.english_run(transcript)


def test_english_run_ignores_names_and_short_phrases_inside_narration():
    assert not recordings.english_run(SINHALA + " Rex Heuermann " + SINHALA + " Gilgo Beach Killer " + SINHALA)
    assert not recordings.english_run("")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_recordings.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.recordings'`

- [ ] **Step 3: Create `app/recordings.py`**

```python
"""Finds evidence audio embedded in a creator recording so it can be kept instead of re-voiced.

Pure functions over diarization spans: dictionaries with `speaker`, `start`, `end` (seconds) and
`text`. When a narrator reference clip was supplied, the narrator's spans are labelled NARRATOR.
The narrator speaks Sinhala; embedded recordings are English. The diarizer sometimes writes Sinhala
in another Indic script and sometimes mislabels a short English word inside a recording as the
narrator, so the rules below lean on script, not on the speaker label alone.
"""

from __future__ import annotations

import re


NARRATOR = "narrator"

# The narrator reference clip cut from the delivery sample; the API accepts 2-10 s.
REFERENCE_MIN_S = 2.0
REFERENCE_MAX_S = 10.0
# A recording stays open across its own silences; only narrator speech or an implausible gap ends it.
MAX_INTERNAL_GAP_MS = 10_000
# Anything shorter is a stray label, not a recording worth keeping.
MIN_SPAN_MS = 1500
# Room into the surrounding silence so a cut never clips a word.
PAD_MS = 150

Piece = tuple[int, int, str]

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_PUNCTUATION = ".,!?;:\"'“”‘’()[]—–-"


def has_foreign_script(text: str) -> bool:
    """True when the text has letters outside the Latin range: the narrator speaking Sinhala."""
    return any(ch.isalpha() and ord(ch) >= 0x0250 for ch in text or "")


def is_english(text: str) -> bool:
    return bool(_LATIN_WORD.search(text or "")) and not has_foreign_script(text)


def _ms(seconds) -> int:
    return round(float(seconds) * 1000)


def choose_reference(spans: list[dict]) -> tuple[float, float] | None:
    """The dominant speaker's longest span, trimmed to what the API accepts as a reference."""
    totals: dict[str, float] = {}
    for item in spans:
        totals[item["speaker"]] = totals.get(item["speaker"], 0.0) + max(0.0, float(item["end"]) - float(item["start"]))
    if not totals:
        return None
    dominant = max(totals, key=totals.get)
    longest = max(
        (item for item in spans if item["speaker"] == dominant),
        key=lambda item: float(item["end"]) - float(item["start"]),
    )
    start, end = float(longest["start"]), float(longest["end"])
    if end - start < REFERENCE_MIN_S:
        return None
    return start, min(end, start + REFERENCE_MAX_S)


def narration_spans(spans: list[dict]) -> list[tuple[int, int]]:
    """Where the narrator is actually heard, in ms relative to the chunk."""
    return sorted(
        (_ms(item["start"]), _ms(item["end"]))
        for item in spans
        if has_foreign_script(item.get("text") or "")
    )


def original_spans(spans: list[dict], chunk_ms: int) -> list[tuple[int, int]]:
    """Embedded recordings in ms relative to the chunk.

    A recording opens at a Latin-only or empty span by someone other than the narrator and stays
    open across silences and short mislabelled spans until the narrator speaks or the gap is
    implausible.
    """
    found: list[list[int]] = []
    current: list[int] | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current[1] - current[0] >= MIN_SPAN_MS:
            found.append(current)
        current = None

    for item in sorted(spans, key=lambda item: float(item["start"])):
        start, end = _ms(item["start"]), _ms(item["end"])
        if has_foreign_script(item.get("text") or ""):
            close()
            continue
        other = item.get("speaker") != NARRATOR
        if current is None:
            if other:
                current = [start, end]
            continue
        if start - current[1] > MAX_INTERNAL_GAP_MS:
            close()
            if other:
                current = [start, end]
            continue
        current[1] = max(current[1], end)
    close()
    return [(max(0, start - PAD_MS), min(chunk_ms, end + PAD_MS)) for start, end in found]


def cut_plan(
    chunk_start_ms: int,
    chunk_end_ms: int,
    originals: list[tuple[int, int]],
    narration: list[tuple[int, int]],
) -> list[Piece]:
    """Split a chunk into ordered narration and original pieces, in absolute source time.

    Narration pieces survive only where the narrator is heard; silence next to a recording belongs
    to the recording.
    """
    length = chunk_end_ms - chunk_start_ms
    if not originals:
        return [(chunk_start_ms, chunk_end_ms, "narration")]

    def has_narration(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in narration)

    pieces: list[list] = []
    cursor = 0
    for start, end in sorted(originals):
        start, end = max(0, start), min(length, end)
        if end <= start:
            continue
        if start > cursor:
            pieces.append([cursor, start, "narration"])
        pieces.append([start, end, "original"])
        cursor = end
    if cursor < length:
        pieces.append([cursor, length, "narration"])

    merged: list[list] = []
    for start, end, kind in pieces:
        if kind == "narration" and not has_narration(start, end):
            if merged and merged[-1][2] == "original":
                merged[-1][1] = end  # trailing silence joins the recording before it
                continue
            kind = "original"  # leading silence joins the recording after it
        if kind == "original" and merged and merged[-1][2] == "original":
            merged[-1][1] = end
            continue
        merged.append([start, end, kind])
    return [(chunk_start_ms + s, chunk_start_ms + e, k) for s, e, k in merged]


def clip_text(spans: list[dict], start_ms: int, end_ms: int) -> str:
    """What is said inside a recording, for the reviewer."""
    parts: list[str] = []
    for item in sorted(spans, key=lambda item: float(item["start"])):
        inside = _ms(item["end"]) > start_ms and _ms(item["start"]) < end_ms
        text = (item.get("text") or "").strip()
        if inside and text and not has_foreign_script(text):
            parts.append(text)
    return " ".join(parts)


def english_run(text: str, minimum_words: int = 8) -> bool:
    """True when a transcript contains a run of English words long enough to be a recording."""
    run = 0
    for token in (text or "").split():
        word = token.strip(_PUNCTUATION)
        if has_foreign_script(word):
            run = 0
        elif _LATIN_WORD.fullmatch(word):
            run += 1
            if run >= minimum_words:
                return True
    return False
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_recordings.py`
Expected: 20 passed. If a fixture-based assertion is off by a few ms, print the actual value and check the arithmetic against the fixture (13.9 s − 150 ms = 13750; 36.49 s + 150 ms = 36640) before touching the rule.

- [ ] **Step 5: Commit**

```bash
git add app/recordings.py tests/test_recordings.py
git commit -m "Add recording span logic: reference choice, recording spans, cut plan, safety net"
```

---

### Task 2: `AIClient.diarize` and audio extraction

**Files:**
- Modify: `app/ai.py` (add `diarize` and a `_field` helper)
- Modify: `app/audio.py` (add `extract`, `extract_levelled`; `split` uses `extract`)
- Test: `tests/test_ai.py`, `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ai.py`:

```python
def test_diarize_sends_the_narrator_reference_as_a_data_url(tmp_path):
    captured = {}

    class Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(segments=[
                SimpleNamespace(speaker="narrator", start=0.0, end=9.25, text="කතාව", id="s1", type="transcript.text.segment"),
                SimpleNamespace(speaker="A", start=13.9, end=15.4, text="Hi, how can I assist you?", id="s2", type="transcript.text.segment"),
            ])

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"RIFFchunk")
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"RIFFreference")

    spans = ai.diarize(chunk, "diarize-model", reference)

    assert captured["model"] == "diarize-model"
    assert captured["response_format"] == "diarized_json"
    assert captured["chunking_strategy"] == "auto"
    assert captured["known_speaker_names"] == ["narrator"]
    import base64
    assert captured["known_speaker_references"] == ["data:audio/wav;base64," + base64.b64encode(b"RIFFreference").decode("ascii")]
    assert "prompt" not in captured
    assert spans == [
        {"speaker": "narrator", "start": 0.0, "end": 9.25, "text": "කතාව"},
        {"speaker": "A", "start": 13.9, "end": 15.4, "text": "Hi, how can I assist you?"},
    ]


def test_diarize_without_a_reference_omits_the_speaker_fields(tmp_path):
    captured = {}

    class Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(segments=[])

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFsample")

    assert ai.diarize(sample, "diarize-model") == []
    assert "known_speaker_names" not in captured
    assert "known_speaker_references" not in captured
```

Append to `tests/test_audio.py`:

```python
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


def test_extract_levelled_cuts_exactly_and_normalises_loudness(tmp_path):
    audio = AudioService()
    source = tmp_path / "quiet.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
         "-af", "volume=-30dB", "-ar", "24000", "-ac", "1", str(source)],
        check=True,
    )

    info = audio.extract_levelled(source, 500, 3500, tmp_path / "out" / "clip.wav")

    assert info.channels == 1 and info.sample_rate == 24000
    assert 2900 <= info.duration_ms <= 3100
    assert mean_volume_db(tmp_path / "out" / "clip.wav", 0.5, 2.5) > -25  # lifted from about -33 dB
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_ai.py tests/test_audio.py`
Expected: the five new tests FAIL with `AttributeError` (no `diarize`, no `extract`, no `extract_levelled`).

- [ ] **Step 3: Add `diarize` to `app/ai.py`**

Add `from .recordings import NARRATOR` to the imports. Add this module-level helper after `PROMPT_VERSION`:

```python
def _field(item, name: str):
    """Read a field from either an SDK object or a plain dict."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)
```

Add this method to `AIClient` after `transcribe`:

```python
    def diarize(self, audio_path: Path, model: str, reference_path: Path | None = None) -> list[dict]:
        """Label who speaks when. With a reference clip, the narrator's spans are labelled NARRATOR.

        The diarization model does not accept a prompt, so glossary terms are not passed.
        """
        extra: dict = {}
        if reference_path is not None:
            encoded = base64.b64encode(Path(reference_path).read_bytes()).decode("ascii")
            extra = {
                "known_speaker_names": [NARRATOR],
                "known_speaker_references": [f"data:audio/wav;base64,{encoded}"],
            }
        with Path(audio_path).open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="diarized_json",
                chunking_strategy="auto",
                **extra,
            )
        return [
            {
                "speaker": _field(segment, "speaker"),
                "start": float(_field(segment, "start") or 0.0),
                "end": float(_field(segment, "end") or 0.0),
                "text": _field(segment, "text") or "",
            }
            for segment in (_field(result, "segments") or [])
        ]
```

- [ ] **Step 4: Add `extract` and `extract_levelled` to `app/audio.py`; make `split` use `extract`**

Add after `plan_segments`:

```python
    def extract(self, source: Path, start_ms: int, end_ms: int, destination: Path, pad_ms: int = 200) -> AudioInfo:
        """Cut [start_ms, end_ms] from the source as mono 24 kHz, padded into the neighbours except at the edges."""
        info = self.probe(source)
        begin = max(0, start_ms - (pad_ms if start_ms else 0))
        finish = min(info.duration_ms, end_ms + (pad_ms if end_ms < info.duration_ms else 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-y", "-v", "error",
            "-ss", f"{begin / 1000:.3f}", "-to", f"{finish / 1000:.3f}",
            "-i", str(source), "-ac", "1", "-ar", "24000", str(destination),
        ])
        return self.probe(destination)

    def extract_levelled(self, source: Path, start_ms: int, end_ms: int, destination: Path) -> AudioInfo:
        """Cut a recording out of the source, exactly, and align its loudness with the narration target."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-y", "-v", "error",
            "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
            "-i", str(source), "-ac", "1", "-ar", "24000",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", str(destination),
        ])
        return self.probe(destination)
```

Replace the body of `split` so the loop reads:

```python
    def split(self, source: Path, destination_dir: Path) -> list[tuple[int, int, Path]]:
        info = self.probe(source)
        plan = self.plan_segments(info.duration_ms, self.silence_boundaries(source))
        destination_dir.mkdir(parents=True, exist_ok=True)
        created: list[tuple[int, int, Path]] = []
        for index, (start_ms, end_ms) in enumerate(plan, start=1):
            output = destination_dir / f"{index:04d}_source.wav"
            self.extract(source, start_ms, end_ms, output)
            created.append((start_ms, end_ms, output))
        return created
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings`
Expected: all pass (previous 77 plus 20 from Task 1 plus 5 here = 102).

- [ ] **Step 6: Commit**

```bash
git add app/ai.py app/audio.py tests/test_ai.py tests/test_audio.py
git commit -m "Add diarization call and source extraction helpers"
```

---

### Task 3: Schema columns, settings, request fields, job configuration

**Files:**
- Modify: `app/database.py` (SCHEMA and `ADDED_COLUMNS`)
- Modify: `app/config.py`, `app/schemas.py`, `app/main.py` (`job_configuration` only)
- Test: `tests/test_database.py`, `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_database.py`:

```python
def test_initialize_adds_kind_and_warnings_to_an_older_database(tmp_path):
    import sqlite3

    from app.database import Database

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE segments (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, project_id TEXT NOT NULL, "
            "segment_index INTEGER NOT NULL, start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, "
            "source_audio_path TEXT NOT NULL, style_json TEXT NOT NULL DEFAULT '{}', qa_json TEXT NOT NULL DEFAULT '{}', "
            "updated_at TEXT NOT NULL);"
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, "
            "progress INTEGER NOT NULL DEFAULT 0, config_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);"
            "INSERT INTO jobs VALUES ('j1', 'p1', 'queued', 'queued', 0, '{}', 'now');"
            "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, updated_at) "
            "VALUES ('s1', 'j1', 'p1', 1, 0, 1000, 'x.wav', 'now');"
        )

    db = Database(path)
    db.initialize()

    assert db.one("SELECT * FROM segments WHERE id='s1'")["kind"] == "narration"
    assert db.one("SELECT * FROM jobs WHERE id='j1'")["warnings"] == []
```

Append to `tests/test_api.py`:

```python
def test_job_configuration_enables_recording_detection_by_default():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["diarize_model"] == "gpt-4o-transcribe-diarize"
    assert config["detect_recordings"] is True


def test_job_configuration_lets_a_request_turn_recording_detection_off():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(
        ProcessRequest(detect_recordings=False, diarize_model="diarize-next"),
        Settings(openai_api_key="unused"),
    )

    assert config["detect_recordings"] is False
    assert config["diarize_model"] == "diarize-next"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_database.py tests/test_api.py`
Expected: the three new tests FAIL (`KeyError: 'kind'`, `KeyError: 'diarize_model'`).

- [ ] **Step 3: Update `app/database.py`**

In `SCHEMA`, add to the `jobs` table after `error TEXT,`:

```sql
    warnings_json TEXT NOT NULL DEFAULT '[]',
```

and to the `segments` table after `source_audio_path TEXT NOT NULL,`:

```sql
    kind TEXT NOT NULL DEFAULT 'narration',
```

Replace `ADDED_COLUMNS` with:

```python
# Columns added after the first release, applied to databases created before them.
ADDED_COLUMNS = [
    ("projects", "speaking_profile_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("jobs", "warnings_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("segments", "kind", "TEXT NOT NULL DEFAULT 'narration'"),
]
```

- [ ] **Step 4: Update `app/config.py`**

Add after `tts_speed`:

```python
    diarize_model: str = field(default_factory=lambda: os.getenv("DIARIZE_MODEL", "gpt-4o-transcribe-diarize"))
    detect_recordings: bool = field(
        default_factory=lambda: os.getenv("DETECT_RECORDINGS", "1").strip().lower() not in {"0", "false", "no"}
    )
```

- [ ] **Step 5: Update `app/schemas.py`**

In `ProcessRequest`, add after `speed`:

```python
    diarize_model: str | None = None
    detect_recordings: bool | None = None
```

Add after `RegenerateRequest`:

```python
class KindUpdate(BaseModel):
    kind: Literal["narration", "original"]
```

- [ ] **Step 6: Update `job_configuration` in `app/main.py`**

Add after the `"speed"` entry:

```python
        "diarize_model": payload.diarize_model or config.diarize_model,
        "detect_recordings": payload.detect_recordings if payload.detect_recordings is not None else config.detect_recordings,
```

Also add `DIARIZE_MODEL=gpt-4o-transcribe-diarize` and `DETECT_RECORDINGS=1` to `.env.example` after `TTS_SPEED=1.0`.

- [ ] **Step 7: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings`
Expected: all pass (105).

- [ ] **Step 8: Commit**

```bash
git add app/database.py app/config.py app/schemas.py app/main.py .env.example tests/test_database.py tests/test_api.py
git commit -m "Add segment kind, job warnings, and recording-detection settings"
```

---

### Task 4: Pipeline: narrator reference, detection, re-cut, kept segments, overrides

**Files:**
- Modify: `app/pipeline.py` (substantial; the full target file is given below)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update the fakes and write the failing tests**

In `tests/test_pipeline.py`:

Add to `FakeAI` (after `describe_delivery`):

```python
    def diarize(self, audio_path, _model, reference=None):
        # The sample has one speaker; chunks are narrator-only unless a subclass says otherwise.
        speaker = "narrator" if reference else "A"
        return [{"speaker": speaker, "start": 0.0, "end": 3.0, "text": "මෙය පරීක්ෂණයකි."}]
```

Change `ProfilingFakeAI.__init__` to also set `self.diarize_calls: list[dict] = []` and add:

```python
    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        return super().diarize(audio_path, model, reference)
```

Add a new fake after `ProfilingFakeAI`:

```python
SINHALA = "ඇය කියනවා කවුරුහරි එනවා කියලා."
CALL = "Hello? Do you need the police?"


class RecordingFakeAI(ProfilingFakeAI):
    """First chunk has a 911 call from 20 s to 30 s; later chunks are narrator only."""

    def __init__(self, fail_chunks: bool = False, fail_sample: bool = False):
        super().__init__()
        self.fail_chunks = fail_chunks
        self.fail_sample = fail_sample

    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        if reference is None:
            if self.fail_sample:
                raise RuntimeError("diarization unavailable")
            return [{"speaker": "A", "start": 0.0, "end": 9.0, "text": SINHALA}]
        if self.fail_chunks:
            raise RuntimeError("diarization failed")
        if Path(audio_path).name == "0001_source.wav":
            return [
                {"speaker": "narrator", "start": 0.0, "end": 20.0, "text": SINHALA},
                {"speaker": "A", "start": 20.0, "end": 24.0, "text": "Hello?"},
                {"speaker": "A", "start": 27.0, "end": 30.0, "text": "Do you need the police?"},
                {"speaker": "narrator", "start": 30.0, "end": 45.0, "text": SINHALA},
            ]
        return [{"speaker": "narrator", "start": 0.0, "end": 20.0, "text": SINHALA}]
```

In `start_job`, add `"diarize_model": "fake-diarize", "detect_recordings": True,` to `job_config` after `"speed": 0.95,`, and give `start_job` a keyword `detect_recordings: bool = True` that sets that key.

Append these tests:

```python
def run_recording_job(tmp_path, ai=None):
    ai = ai or RecordingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)
    segments = db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
    return pipeline, db, project_id, job_id, ai, segments


def test_a_detected_recording_becomes_its_own_kept_segment(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert [s["kind"] for s in segments] == ["narration", "original", "narration", "narration"]
    assert [(s["start_ms"], s["end_ms"]) for s in segments] == [(0, 19850), (19850, 30150), (30150, 45000), (45000, 65000)]
    original = segments[1]
    assert original["transcript_si"] == CALL
    assert original["narration_en"] == ""
    assert original["tts_audio_path"].endswith("0002_original.wav")
    assert 10100 <= pipeline.audio.probe(Path(original["tts_audio_path"])).duration_ms <= 10500
    assert original["status"] == "kept" and original["qa_status"] == "needs_review"
    assert "Recorded audio kept as is. Confirm." in original["qa"]["issues"]
    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "awaiting_review"


def test_kept_segments_skip_every_model_stage(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert len(ai.synthesis_calls) == 3
    assert len(ai.adapt_calls) == 3
    stages = db.all("SELECT stage, segment_id FROM model_calls WHERE job_id=?", (job_id,))
    assert not [row for row in stages if row["segment_id"] == segments[1]["id"]]


def test_the_narration_after_a_recording_knows_what_was_heard(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert "[Recording plays: " + CALL + "]" in ai.adapt_calls[1]["previous_narration"]
    # Continuity skips the recording: segment 3 inherits segment 1's style.
    assert ai.synthesis_calls[1]["previous_style"] == segments[0]["style"]


def test_narrator_reference_is_cut_once_and_reused(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    profile = db.one("SELECT * FROM projects WHERE id=?", (project_id,))["speaking_profile"]
    reference = Path(profile["narrator_reference"]["path"])
    assert reference.exists() and reference.name == "narrator_reference.wav"
    assert profile["narrator_reference"]["start_s"] == 0.0 and profile["narrator_reference"]["end_s"] == 9.0
    first_run = len(ai.diarize_calls)

    pipeline.process_job(start_job(db, project_id))

    without_reference = [c for c in ai.diarize_calls if c["reference"] is None]
    assert len(without_reference) == 1  # the sample was diarized only once
    assert all(c["reference"] == reference for c in ai.diarize_calls[first_run:])


def test_assembly_uses_the_minimum_gap_next_to_a_recording(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    captured = {}
    real_assemble = pipeline.audio.assemble

    def spy(paths, wav, mp3, gaps_ms=None):
        captured["gaps"] = gaps_ms
        real_assemble(paths, wav, mp3, gaps_ms)

    pipeline.audio.assemble = spy

    artifacts = pipeline.assemble_project(project_id, job_id)

    assert captured["gaps"] == [300, 300, 800]
    exported = json.loads(Path(artifacts["wav"]).parent.joinpath("script_en.json").read_text(encoding="utf-8"))
    assert [entry["kind"] for entry in exported] == ["narration", "original", "narration", "narration"]


def test_detection_failure_on_a_chunk_keeps_it_as_narration_with_a_warning(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, RecordingFakeAI(fail_chunks=True))

    assert [s["kind"] for s in segments] == ["narration", "narration"]
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Recording detection failed on chunk 1" in w for w in warnings)
    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "awaiting_review"


def test_detection_failure_on_the_sample_disables_detection_with_a_warning(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, RecordingFakeAI(fail_sample=True))

    assert [s["kind"] for s in segments] == ["narration", "narration"]
    assert all(c["reference"] is None for c in ai.diarize_calls) and len(ai.diarize_calls) == 1
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Recording detection unavailable" in w for w in warnings)


def test_detection_can_be_switched_off_per_job(tmp_path):
    ai = RecordingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id, detect_recordings=False))

    assert ai.diarize_calls == []
    assert [s["kind"] for s in db.all("SELECT * FROM segments WHERE project_id=? ORDER BY segment_index", (project_id,))] == ["narration", "narration"]


def test_an_english_run_in_a_transcript_is_flagged_for_review(tmp_path):
    class LeakyFakeAI(ProfilingFakeAI):
        def transcribe(self, *_args, **_kwargs):
            return SINHALA + " 9-1-1, how can I assist you? Hello? Hello? Hello, you dialed into the 911 system."

    ai = LeakyFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["qa_status"] == "needs_review"
    assert any("Possible recorded audio" in issue for issue in segment["qa"]["issues"])


def test_a_recording_can_be_confirmed(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    pipeline.confirm_segment(segments[1]["id"])

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["qa_status"] == "passed" and fresh["status"] == "kept"
    assert "Recorded audio kept as is. Confirm." not in fresh["qa"]["issues"]


def test_confirming_a_narration_segment_is_refused(tmp_path):
    import pytest

    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    with pytest.raises(RuntimeError):
        pipeline.confirm_segment(segments[0]["id"])


def test_a_recording_can_be_turned_into_narration(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    before = len(ai.synthesis_calls)

    pipeline.set_segment_kind(segments[1]["id"], "narration")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["kind"] == "narration"
    assert fresh["tts_audio_path"].endswith("0002_r02.wav")
    assert fresh["narration_en"] == "This is a test."
    assert fresh["qa_status"] == "passed"
    assert len(ai.synthesis_calls) == before + 1
    assert ai.adapt_calls[-1]["previous_narration"] == segments[0]["narration_en"]


def test_a_narration_segment_can_be_kept_as_recorded(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    pipeline.set_segment_kind(segments[2]["id"], "original")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[2]["id"],))
    assert fresh["kind"] == "original"
    assert fresh["narration_en"] == "" and fresh["style"] == {}
    assert fresh["tts_audio_path"].endswith("0003_original_r02.wav")
    assert 14600 <= fresh["tts_duration_ms"] <= 15100  # 30150-45000 ms cut exactly
    assert fresh["qa_status"] == "needs_review" and fresh["status"] == "kept"


def test_regenerating_a_kept_segment_recuts_it_without_model_calls(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    calls_before = len(ai.synthesis_calls) + len(ai.adapt_calls)

    pipeline.regenerate_segment(segments[1]["id"], "tts")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["tts_audio_path"].endswith("0002_original_r02.wav")
    assert fresh["revision"] == 2 and fresh["qa_status"] == "needs_review"
    assert len(ai.synthesis_calls) + len(ai.adapt_calls) == calls_before
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pipeline.py`
Expected: the new tests FAIL (no `kind` handling, no `confirm_segment`, no `set_segment_kind`); the existing tests may also fail until the pipeline accepts `diarize_model`/`detect_recordings` keys, which it ignores today, so they should still pass.

- [ ] **Step 3: Rewrite `app/pipeline.py`**

Replace the whole file with the following. Everything from the `natural-narration` branch is preserved; the additions are `_warn`, `_normalized_path`, `_generated_dir`, `_narrator_reference`, `_create_segments`, `_qa_verdict`, `_narrate_segment`, `_recut_original`, `set_segment_kind`, `confirm_segment`, and the kind-aware loop, regeneration, and assembly.

```python
from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from . import recordings, speaking_profile
from .ai import AIClient, PROMPT_VERSION
from .audio import AudioService
from .config import Settings
from .database import Database, utc_now
from .direction import MIN_GAP_MS, segment_gap_ms
from .schemas import QAEvaluation


T = TypeVar("T")

CONFIRM_NOTE = "Recorded audio kept as is. Confirm."
ENGLISH_NOTE = "Possible recorded audio (English speech in transcript). Mark as original if so."
DURATION_RANGE = (0.55, 1.45)


def kept_qa() -> dict:
    """QA payload for a recording kept from the source: nothing to check, but a reviewer must confirm."""
    return {
        "passed": False,
        "severity": "low",
        "issues": [CONFIRM_NOTE],
        "recommended_stage_to_retry": "none",
        "duration_ratio": 1.0,
    }


class Pipeline:
    def __init__(self, db: Database, config: Settings, audio: AudioService | None = None):
        self.db = db
        self.config = config
        self.audio = audio or AudioService(config.ffmpeg, config.ffprobe)

    def _ai(self) -> AIClient:
        return AIClient(self.config.openai_api_key)

    def _job_update(self, job_id: str, *, status: str, stage: str, progress: int, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE jobs SET status=?, stage=?, progress=?, error=? WHERE id=?",
            (status, stage, progress, error, job_id),
        )

    def _warn(self, job_id: str, message: str) -> None:
        """Record a non-fatal problem on the job so the reviewer sees it."""
        job = self.db.one("SELECT warnings_json FROM jobs WHERE id=?", (job_id,))
        warnings = list((job or {}).get("warnings") or [])
        warnings.append(message)
        self.db.execute("UPDATE jobs SET warnings_json=? WHERE id=?", (json.dumps(warnings), job_id))
        print(f"Job {job_id}: {message}")

    def _direction_inputs(self, project: dict, config: dict) -> tuple[str | None, float]:
        """The project-level persona override (strings only) and the job's speaking speed."""
        persona = (project.get("narrator_profile") or {}).get("persona")
        if not isinstance(persona, str) or not persona.strip():
            persona = None
        return persona, config.get("speed", self.config.tts_speed)

    def _normalized_path(self, project_id: str) -> Path:
        return self.config.data_dir / "projects" / project_id / "original" / "source_normalized.wav"

    def _generated_dir(self, project_id: str, job_id: str) -> Path:
        return self.config.data_dir / "projects" / project_id / "generated" / job_id

    def _tracked_call(
        self,
        job_id: str,
        segment_id: str | None,
        stage: str,
        model: str,
        input_value: str,
        operation: Callable[[], T],
    ) -> T:
        digest = hashlib.sha256(input_value.encode("utf-8")).hexdigest()
        call_id = self.db.execute(
            "INSERT INTO model_calls(job_id, segment_id, stage, model, prompt_version, input_hash, status, started_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (job_id, segment_id, stage, model, PROMPT_VERSION, digest, "running", utc_now()),
        )
        try:
            result = operation()
        except Exception as exc:
            self.db.execute(
                "UPDATE model_calls SET status='failed', error=?, completed_at=? WHERE id=?",
                (str(exc)[:2000], utc_now(), call_id),
            )
            raise
        self.db.execute(
            "UPDATE model_calls SET status='completed', completed_at=? WHERE id=?",
            (utc_now(), call_id),
        )
        return result

    def _speaking_profile(self, job_id: str, project: dict, normalized: Path, config: dict) -> dict:
        """Describe how the source narrator speaks, so the English can be directed to match.

        Analysed once per project and reused afterwards. A failure here degrades the
        narration slightly but must never fail the job.
        """
        existing = project.get("speaking_profile")
        if existing and existing.get("measured"):
            return existing

        measured = speaking_profile.measure(normalized, self.config.ffmpeg, self.config.ffprobe)
        described = None
        audio_model = config.get("audio_model", self.config.audio_model)
        try:
            sample = normalized.parent / "delivery_sample.wav"
            speaking_profile.extract_sample(normalized, sample, measured["duration_s"], self.config.ffmpeg)
            ai = self._ai()
            described = self._tracked_call(
                job_id, None, "speaking_profile", audio_model, str(sample),
                lambda: ai.describe_delivery(sample, audio_model),
            )
        except Exception as exc:  # noqa: BLE001 - style analysis is an enhancement, not a gate
            traceback.print_exc()
            print(f"Speaking profile description failed, continuing with measurements only: {exc}")

        profile = {
            "measured": measured,
            "derived": speaking_profile.derive(measured),
            "described": described,
            "analyzed_at": utc_now(),
        }
        self.db.execute(
            "UPDATE projects SET speaking_profile_json=?, updated_at=? WHERE id=?",
            (json.dumps(profile), utc_now(), project["id"]),
        )
        return profile

    def _narrator_reference(self, job_id: str, project: dict, normalized: Path, profile: dict, config: dict) -> Path | None:
        """A 2-10 s clip of the narrator, so diarization can label their speech by name.

        Chosen once per project from the delivery sample and stored in the speaking profile. Any
        failure disables recording detection for this job with a warning; it never fails the job.
        """
        stored = (profile.get("narrator_reference") or {}) if profile else {}
        if stored.get("path") and Path(stored["path"]).exists():
            return Path(stored["path"])
        model = config.get("diarize_model", self.config.diarize_model)
        try:
            sample = normalized.parent / "delivery_sample.wav"
            if not sample.exists():
                duration_s = (profile.get("measured") or {}).get("duration_s") or self.audio.probe(normalized).duration_ms / 1000
                speaking_profile.extract_sample(normalized, sample, duration_s, self.config.ffmpeg)
            ai = self._ai()
            spans = self._tracked_call(job_id, None, "diarization", model, str(sample), lambda: ai.diarize(sample, model))
            window = recordings.choose_reference(spans)
            if window is None:
                self._warn(job_id, "Recording detection unavailable: no clear narrator speech in the delivery sample")
                return None
            start_s, end_s = window
            reference = normalized.parent / "narrator_reference.wav"
            self.audio.extract(sample, round(start_s * 1000), round(end_s * 1000), reference, pad_ms=0)
            profile["narrator_reference"] = {"path": str(reference), "start_s": start_s, "end_s": end_s}
            self.db.execute(
                "UPDATE projects SET speaking_profile_json=?, updated_at=? WHERE id=?",
                (json.dumps(profile), utc_now(), project["id"]),
            )
            return reference
        except Exception as exc:  # noqa: BLE001 - detection is an enhancement, not a gate
            traceback.print_exc()
            self._warn(job_id, f"Recording detection unavailable: {str(exc)[:200]}")
            return None

    def _create_segments(
        self,
        job_id: str,
        project: dict,
        normalized: Path,
        chunks: list[tuple[int, int, Path]],
        reference: Path | None,
        config: dict,
    ) -> list[str]:
        """One row per piece: narration to be voiced, or a recording kept from the source."""
        self.db.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
        model = config.get("diarize_model", self.config.diarize_model)
        ai = self._ai() if reference is not None else None
        generated = self._generated_dir(project["id"], job_id)
        segment_ids: list[str] = []
        index = 0
        for chunk_index, (chunk_start, chunk_end, chunk_path) in enumerate(chunks, start=1):
            spans: list[dict] = []
            pieces: list[recordings.Piece] = [(chunk_start, chunk_end, "narration")]
            if reference is not None:
                try:
                    spans = self._tracked_call(
                        job_id, None, "diarization", model, str(chunk_path),
                        lambda: ai.diarize(chunk_path, model, reference),
                    )
                    originals = recordings.original_spans(spans, chunk_end - chunk_start)
                    pieces = recordings.cut_plan(chunk_start, chunk_end, originals, recordings.narration_spans(spans))
                except Exception as exc:  # noqa: BLE001 - one chunk's detection must not fail the job
                    traceback.print_exc()
                    self._warn(job_id, f"Recording detection failed on chunk {chunk_index}: {str(exc)[:200]}")
                    spans, pieces = [], [(chunk_start, chunk_end, "narration")]
            for start_ms, end_ms, kind in pieces:
                index += 1
                segment_id = str(uuid.uuid4())
                segment_ids.append(segment_id)
                if kind == "original":
                    clip = generated / f"{index:04d}_original.wav"
                    info = self.audio.extract_levelled(normalized, start_ms, end_ms, clip)
                    text = recordings.clip_text(spans, start_ms - chunk_start, end_ms - chunk_start)
                    self.db.execute(
                        "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, kind, "
                        "transcript_si, tts_audio_path, tts_duration_ms, qa_json, qa_status, status, updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (segment_id, job_id, project["id"], index, start_ms, end_ms, str(clip), "original",
                         text, str(clip), info.duration_ms, json.dumps(kept_qa()), "needs_review", "kept", utc_now()),
                    )
                    continue
                source_path = chunk_path
                if (start_ms, end_ms) != (chunk_start, chunk_end):
                    source_path = chunk_path.parent / f"{index:04d}_piece.wav"
                    self.audio.extract(normalized, start_ms, end_ms, source_path)
                self.db.execute(
                    "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, kind, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (segment_id, job_id, project["id"], index, start_ms, end_ms, str(source_path), "narration", utc_now()),
                )
        return segment_ids

    def _qa_verdict(self, qa: QAEvaluation, transcript: str, tts_duration_ms: int | None, segment: dict) -> dict:
        """Model QA plus the mechanical checks: duration ratio, and an English run that looks like a recording."""
        payload = qa.model_dump()
        if tts_duration_ms:
            ratio = tts_duration_ms / max(1, segment["end_ms"] - segment["start_ms"])
            payload["duration_ratio"] = round(ratio, 3)
            low, high = DURATION_RANGE
            if not low <= ratio <= high:
                payload["passed"] = False
                payload["issues"] = payload["issues"] + [f"Duration ratio {ratio:.2f} is outside {low}-{high}"]
                payload["recommended_stage_to_retry"] = "tts"
        if recordings.english_run(transcript):
            payload["passed"] = False
            payload["issues"] = payload["issues"] + [ENGLISH_NOTE]
        return payload

    def _narrate_segment(
        self,
        ai: AIClient,
        job: dict,
        project: dict,
        segment: dict,
        profile: dict | None,
        previous_context: str,
        previous_style: dict | None,
        persona: str | None,
        speed: float,
    ) -> tuple[str, dict]:
        """Run every narration stage for one segment and return the context the next one inherits."""
        job_id, segment_id, config = job["id"], segment["id"], job["config"]
        transcript = self._tracked_call(
            job_id, segment_id, "transcription", config["stt_model"], str(segment["source_audio_path"]),
            lambda: ai.transcribe(Path(segment["source_audio_path"]), config["stt_model"], project["topic"], project["glossary"]),
        )
        self.db.execute(
            "UPDATE segments SET transcript_si=?, status='transcribed', updated_at=? WHERE id=?",
            (transcript, utc_now(), segment_id),
        )

        faithful = self._tracked_call(
            job_id, segment_id, "translation", config["text_model"], transcript,
            lambda: ai.translate(transcript, config["text_model"], project["topic"], project["glossary"], previous_context[-1000:]),
        )
        self.db.execute(
            "UPDATE segments SET faithful_en=?, entities_json=?, numbers_json=?, status='translated', updated_at=? WHERE id=?",
            (faithful.english_faithful, json.dumps(faithful.entities), json.dumps(faithful.numbers), utc_now(), segment_id),
        )

        adaptation = self._tracked_call(
            job_id, segment_id, "adaptation", config["text_model"], faithful.english_faithful,
            lambda: ai.adapt(faithful.english_faithful, config["text_model"], project["narrator_profile"] or {}, previous_context[-600:]),
        )
        style = adaptation.model_dump(exclude={"narration_text"})
        self.db.execute(
            "UPDATE segments SET narration_en=?, style_json=?, status='adapted', updated_at=? WHERE id=?",
            (adaptation.narration_text, json.dumps(style), utc_now(), segment_id),
        )

        tts_path = self._generated_dir(project["id"], job_id) / f"{segment['segment_index']:04d}_r{segment['revision']:02d}.wav"
        self._tracked_call(
            job_id, segment_id, "tts", config["tts_model"], adaptation.narration_text,
            lambda: ai.synthesize(
                adaptation.narration_text, config["tts_model"], config["voice"], style, tts_path,
                profile, previous_style, persona, speed,
            ),
        )
        tts_duration = self.audio.probe(tts_path).duration_ms
        self.db.execute(
            "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, status='generated', updated_at=? WHERE id=?",
            (str(tts_path), tts_duration, utc_now(), segment_id),
        )

        qa = self._tracked_call(
            job_id, segment_id, "qa", config["qa_model"], faithful.english_faithful + adaptation.narration_text,
            lambda: ai.evaluate(faithful.english_faithful, adaptation.narration_text, config["qa_model"]),
        )
        qa_payload = self._qa_verdict(qa, transcript, tts_duration, segment)
        qa_status = "passed" if qa_payload["passed"] else "needs_review"
        self.db.execute(
            "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
            (json.dumps(qa_payload), qa_status, qa_status, utc_now(), segment_id),
        )
        return adaptation.narration_text, style

    def process_job(self, job_id: str) -> None:
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            return
        project = self.db.one("SELECT * FROM projects WHERE id=?", (job["project_id"],))
        if not project or not project.get("source_path"):
            self._job_update(job_id, status="failed", stage="validation", progress=0, error="Project has no uploaded audio")
            return
        config = job["config"]
        project_dir = self.config.data_dir / "projects" / project["id"]
        try:
            self.db.execute("UPDATE jobs SET started_at=? WHERE id=?", (utc_now(), job_id))
            self._job_update(job_id, status="running", stage="normalizing", progress=3)
            normalized = project_dir / "original" / "source_normalized.wav"
            self.audio.normalize(Path(project["source_path"]), normalized)

            self._job_update(job_id, status="running", stage="analyzing", progress=5)
            profile = self._speaking_profile(job_id, project, normalized, config)

            reference = None
            if config.get("detect_recordings", self.config.detect_recordings):
                self._job_update(job_id, status="running", stage="finding recordings", progress=7)
                reference = self._narrator_reference(job_id, project, normalized, profile, config)

            self._job_update(job_id, status="running", stage="segmenting", progress=8)
            chunks = self.audio.split(normalized, project_dir / "segments" / job_id)
            segment_ids = self._create_segments(job_id, project, normalized, chunks, reference, config)

            ai = self._ai()
            previous_context = ""
            previous_style: dict | None = None
            persona, speed = self._direction_inputs(project, config)
            total = len(segment_ids)
            for position, segment_id in enumerate(segment_ids):
                base_progress = 10 + round(80 * position / max(total, 1))
                segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
                assert segment
                if segment["kind"] == "original":
                    self._job_update(job_id, status="running", stage=f"keeping recording {position + 1}/{total}", progress=base_progress)
                    previous_context += f"\n[Recording plays: {segment['transcript_si']}]"
                    continue
                self._job_update(job_id, status="running", stage=f"transcribing {position + 1}/{total}", progress=base_progress)
                previous_context, previous_style = self._narrate_segment(
                    ai, job, project, segment, profile, previous_context, previous_style, persona, speed,
                )

            failed = self.db.one("SELECT COUNT(*) AS count FROM segments WHERE job_id=? AND qa_status!='passed'", (job_id,))["count"]
            if config.get("human_review_gate", True) or failed:
                self._job_update(job_id, status="awaiting_review", stage="review", progress=95)
            else:
                self.assemble_project(project["id"], job_id)
                self._job_update(job_id, status="completed", stage="completed", progress=100)
                self.db.execute("UPDATE jobs SET completed_at=? WHERE id=?", (utc_now(), job_id))
        except Exception as exc:
            traceback.print_exc()
            self._job_update(job_id, status="failed", stage="failed", progress=0, error=str(exc)[:2000])
            self.db.execute("UPDATE projects SET status='failed', updated_at=? WHERE id=?", (utc_now(), project["id"]))

    def _previous_narration(self, job_id: str, segment_index: int) -> tuple[str, dict | None]:
        """The nearest earlier narration segment's script and style, skipping kept recordings."""
        predecessor = self.db.one(
            "SELECT narration_en, style_json FROM segments WHERE job_id=? AND kind='narration' AND segment_index<? "
            "ORDER BY segment_index DESC LIMIT 1",
            (job_id, segment_index),
        )
        if not predecessor:
            return "", None
        return predecessor["narration_en"], predecessor["style"]

    def _recut_original(self, segment: dict, job: dict, project: dict) -> None:
        """Cut the recording from the source again; nothing is generated for it."""
        try:
            revision = segment["revision"] + 1
            clip = self._generated_dir(project["id"], job["id"]) / f"{segment['segment_index']:04d}_original_r{revision:02d}.wav"
            info = self.audio.extract_levelled(self._normalized_path(project["id"]), segment["start_ms"], segment["end_ms"], clip)
            self.db.execute(
                "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, revision=?, qa_json=?, qa_status='needs_review', "
                "status='kept', updated_at=? WHERE id=?",
                (str(clip), info.duration_ms, revision, json.dumps(kept_qa()), utc_now(), segment["id"]),
            )
        except Exception as exc:  # noqa: BLE001 - reported on the segment, as regeneration does
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment["id"]),
            )

    def regenerate_segment(self, segment_id: str, stage: str) -> None:
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment:
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (segment["job_id"],))
        project = self.db.one("SELECT * FROM projects WHERE id=?", (segment["project_id"],))
        assert job and project
        if segment["kind"] == "original":
            self._recut_original(segment, job, project)
            return
        ai = self._ai()
        config = job["config"]
        previous_narration, previous_style = self._previous_narration(job["id"], segment["segment_index"])
        persona, speed = self._direction_inputs(project, config)
        try:
            faithful_text = segment["faithful_en"]
            narration_text = segment["narration_en"]
            style = segment["style"] or {}
            if stage == "translation":
                faithful = self._tracked_call(
                    job["id"], segment_id, "translation", config["text_model"], segment["transcript_si"],
                    lambda: ai.translate(segment["transcript_si"], config["text_model"], project["topic"], project["glossary"], ""),
                )
                faithful_text = faithful.english_faithful
                self.db.execute(
                    "UPDATE segments SET faithful_en=?, entities_json=?, numbers_json=? WHERE id=?",
                    (faithful_text, json.dumps(faithful.entities), json.dumps(faithful.numbers), segment_id),
                )
                stage = "adaptation"
            if stage == "adaptation":
                adaptation = self._tracked_call(
                    job["id"], segment_id, "adaptation", config["text_model"], faithful_text,
                    lambda: ai.adapt(faithful_text, config["text_model"], project["narrator_profile"] or {}, previous_narration[-600:]),
                )
                narration_text = adaptation.narration_text
                style = adaptation.model_dump(exclude={"narration_text"})
                self.db.execute(
                    "UPDATE segments SET narration_en=?, style_json=? WHERE id=?",
                    (narration_text, json.dumps(style), segment_id),
                )
                stage = "tts"
            if stage == "tts":
                revision = segment["revision"] + 1
                tts_path = self._generated_dir(project["id"], job["id"]) / f"{segment['segment_index']:04d}_r{revision:02d}.wav"
                self._tracked_call(
                    job["id"], segment_id, "tts", config["tts_model"], narration_text,
                    lambda: ai.synthesize(
                        narration_text, config["tts_model"], config["voice"], style, tts_path,
                        project.get("speaking_profile"), previous_style, persona, speed,
                    ),
                )
                duration = self.audio.probe(tts_path).duration_ms
                self.db.execute(
                    "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, revision=?, qa_status='pending' WHERE id=?",
                    (str(tts_path), duration, revision, segment_id),
                )
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            qa = self._tracked_call(
                job["id"], segment_id, "qa", config["qa_model"], fresh["faithful_en"] + fresh["narration_en"],
                lambda: ai.evaluate(fresh["faithful_en"], fresh["narration_en"], config["qa_model"]),
            )
            qa_payload = self._qa_verdict(qa, fresh["transcript_si"], fresh.get("tts_duration_ms"), fresh)
            status = "passed" if qa_payload["passed"] else "needs_review"
            self.db.execute(
                "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
                (json.dumps(qa_payload), status, status, utc_now(), segment_id),
            )
        except Exception as exc:
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment_id),
            )

    def set_segment_kind(self, segment_id: str, kind: str) -> None:
        """Reviewer override: keep a segment as recorded, or voice one that detection kept."""
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment:
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (segment["job_id"],))
        project = self.db.one("SELECT * FROM projects WHERE id=?", (segment["project_id"],))
        assert job and project
        if kind == "original":
            self.db.execute(
                "UPDATE segments SET kind='original', faithful_en='', narration_en='', style_json='{}', updated_at=? WHERE id=?",
                (utc_now(), segment_id),
            )
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            self._recut_original(fresh, job, project)
            return
        self.db.execute(
            "UPDATE segments SET kind='narration', tts_audio_path=NULL, tts_duration_ms=NULL, qa_json='{}', "
            "qa_status='pending', status='created', revision=revision+1, updated_at=? WHERE id=?",
            (utc_now(), segment_id),
        )
        try:
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            previous_narration, previous_style = self._previous_narration(job["id"], fresh["segment_index"])
            persona, speed = self._direction_inputs(project, job["config"])
            self._narrate_segment(
                self._ai(), job, project, fresh, project.get("speaking_profile"),
                previous_narration, previous_style, persona, speed,
            )
        except Exception as exc:  # noqa: BLE001 - reported on the segment, as regeneration does
            traceback.print_exc()
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment_id),
            )

    def confirm_segment(self, segment_id: str) -> None:
        """The reviewer has listened to a kept recording and accepts it."""
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment or segment.get("kind") != "original":
            raise RuntimeError("Only recorded-audio segments need confirming")
        qa = dict(segment.get("qa") or kept_qa())
        qa["passed"] = True
        qa["issues"] = [issue for issue in qa.get("issues", []) if issue != CONFIRM_NOTE]
        self.db.execute(
            "UPDATE segments SET qa_json=?, qa_status='passed', status='kept', updated_at=? WHERE id=?",
            (json.dumps(qa), utc_now(), segment_id),
        )

    def assemble_project(self, project_id: str, job_id: str | None = None) -> dict[str, str]:
        if not job_id:
            latest = self.db.one("SELECT id FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            if not latest:
                raise RuntimeError("No processing job exists for this project")
            job_id = latest["id"]
        segments = self.db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
        if not segments or any(not segment.get("tts_audio_path") for segment in segments):
            raise RuntimeError("Every segment must have generated audio before assembly")
        export_dir = self.config.data_dir / "projects" / project_id / "exports"
        wav_path = export_dir / "final_en.wav"
        mp3_path = export_dir / "final_en.mp3"
        project = self.db.one("SELECT speaking_profile_json FROM projects WHERE id=?", (project_id,))
        profile = project.get("speaking_profile") if project else None
        gaps = [
            MIN_GAP_MS if "original" in (earlier.get("kind"), later.get("kind"))
            else segment_gap_ms(earlier.get("style"), later.get("style"), profile)
            for earlier, later in zip(segments, segments[1:])
        ]
        self.audio.assemble([Path(s["tts_audio_path"]) for s in segments], wav_path, mp3_path, gaps)
        (export_dir / "transcript_si.json").write_text(
            json.dumps([{"index": s["segment_index"], "kind": s.get("kind", "narration"), "start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["transcript_si"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (export_dir / "script_en.json").write_text(
            json.dumps([{"index": s["segment_index"], "kind": s.get("kind", "narration"), "faithful": s["faithful_en"], "narration": s["narration_en"], "qa": s["qa"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.db.execute("UPDATE projects SET status='completed', updated_at=? WHERE id=?", (utc_now(), project_id))
        self.db.execute(
            "UPDATE jobs SET status='completed', stage='completed', progress=100, completed_at=? WHERE id=?",
            (utc_now(), job_id),
        )
        return {"wav": str(wav_path), "mp3": str(mp3_path)}
```

- [ ] **Step 4: Run the pipeline tests**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_pipeline.py`
Expected: all pass (15 existing plus 14 new = 29). If `test_a_detected_recording_becomes_its_own_kept_segment` fails on the time ranges, print the segments and check `recordings.cut_plan` inputs (originals `[(19850, 30150)]`, narration `[(0, 20000), (30000, 45000)]` for chunk 1) before changing anything.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -p no:warnings`
Expected: all pass (119).

- [ ] **Step 6: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "Detect embedded recordings, keep them as their own segments, and let reviewers confirm or flip them"
```

---

### Task 5: Endpoints and review-page controls

**Files:**
- Modify: `app/main.py` (two endpoints; import `KindUpdate`)
- Modify: `app/static/app.js`, `app/static/index.html`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_api.py`:

```python
def seed_job_with_segments(module, kinds):
    """Insert a project, a job, and one segment per kind directly into the app's database."""
    import json
    import uuid

    from app.database import utc_now

    project_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = utc_now()
    module.db.execute(
        "INSERT INTO projects(id,title,topic,created_at,updated_at) VALUES(?,?,?,?,?)",
        (project_id, "Seeded", "History", now, now),
    )
    module.db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,warnings_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (job_id, project_id, "awaiting_review", "review", 95, "{}", json.dumps(["Recording detection failed on chunk 2: timeout"]), now),
    )
    ids = []
    for index, kind in enumerate(kinds, start=1):
        segment_id = str(uuid.uuid4())
        ids.append(segment_id)
        module.db.execute(
            "INSERT INTO segments(id,job_id,project_id,segment_index,start_ms,end_ms,source_audio_path,kind,qa_json,qa_status,status,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (segment_id, job_id, project_id, index, 0, 1000, "x.wav", kind,
             json.dumps({"passed": False, "issues": ["Recorded audio kept as is. Confirm."]}) if kind == "original" else "{}",
             "needs_review" if kind == "original" else "passed", "kept" if kind == "original" else "passed", now),
        )
    return job_id, ids


def test_kind_and_confirm_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.config
    import app.main

    importlib.reload(app.config)
    module = importlib.reload(app.main)
    from fastapi.testclient import TestClient

    client = TestClient(module.app)
    job_id, (narration_id, original_id) = seed_job_with_segments(module, ["narration", "original"])
    flips = []
    monkeypatch.setattr(module.pipeline, "set_segment_kind", lambda segment_id, kind: flips.append((segment_id, kind)))

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["warnings"] == ["Recording detection failed on chunk 2: timeout"]
    segments = client.get(f"/api/jobs/{job_id}/segments").json()
    assert [s["kind"] for s in segments] == ["narration", "original"]

    flipped = client.patch(f"/api/segments/{narration_id}/kind", json={"kind": "original"})
    assert flipped.status_code == 202
    assert flips == [(narration_id, "original")]
    assert client.get(f"/api/jobs/{job_id}/segments").json()[0]["status"] == "regenerating"

    assert client.patch(f"/api/segments/{original_id}/kind", json={"kind": "bogus"}).status_code == 422

    confirmed = client.post(f"/api/segments/{original_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["qa_status"] == "passed"
    assert "Recorded audio kept as is. Confirm." not in confirmed.json()["qa"]["issues"]

    assert client.post(f"/api/segments/{narration_id}/confirm").status_code == 409
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest -p no:warnings tests/test_api.py`
Expected: the new test FAILS with a 404/405 on the kind endpoint.

- [ ] **Step 3: Add the endpoints to `app/main.py`**

Add `KindUpdate` to the `from .schemas import ...` line. Add after `regenerate_segment`:

```python
@app.patch("/api/segments/{segment_id}/kind", status_code=202)
def update_segment_kind(segment_id: str, payload: KindUpdate, background: BackgroundTasks) -> dict:
    """Keep a segment as recorded audio, or voice one that detection kept."""
    require_segment(segment_id)
    if payload.kind == "narration" and not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY is not configured")
    db.execute("UPDATE segments SET status='regenerating', updated_at=? WHERE id=?", (utc_now(), segment_id))
    background.add_task(pipeline.set_segment_kind, segment_id, payload.kind)
    return {"segment_id": segment_id, "kind": payload.kind, "status": "regenerating"}


@app.post("/api/segments/{segment_id}/confirm")
def confirm_segment(segment_id: str) -> dict:
    """The reviewer accepts a kept recording."""
    require_segment(segment_id)
    try:
        pipeline.confirm_segment(segment_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    return present_segment(require_segment(segment_id))
```

- [ ] **Step 4: Update the review page**

In `app/static/app.js`, in `refreshJob`, replace the `$("progress-text").textContent = ...` line with:

```js
  const warnings = (job.warnings || []).length ? ` · ${job.warnings.join(" · ")}` : "";
  $("progress-text").textContent = `${job.stage} · ${job.progress}%${job.error ? ` · ${job.error}` : ""}${warnings}`;
```

Replace the whole `loadSegments` function with:

```js
function segmentCard(segment) {
  const qa = `<span class="${segment.qa_status === "passed" ? "qa-pass" : "qa-review"}">${segment.qa_status}</span>`;
  const issues = `<span class="muted">${escapeHtml((segment.qa?.issues || []).join(" · "))}</span>`;
  const player = segment.tts_audio_url ? `<audio controls src="${segment.tts_audio_url}"></audio>` : "";
  if (segment.kind === "original") {
    return `
    <article class="segment" data-id="${segment.id}" data-kind="original">
      <div class="segment-head"><strong>Segment ${segment.segment_index}</strong>
        <span class="pill">Recorded audio, kept as is</span>${qa}</div>
      <p class="muted">${escapeHtml(segment.transcript_si || "(no speech recognised in this recording)")}</p>
      ${player}
      <div class="actions">
        <button class="confirm">Confirm recording</button>
        <button class="to-narration secondary">Treat as narration</button>
        ${issues}
      </div>
    </article>`;
  }
  return `
    <article class="segment" data-id="${segment.id}" data-kind="narration">
      <div class="segment-head"><strong>Segment ${segment.segment_index}</strong>${qa}</div>
      <div class="segment-grid">
        <label>Sinhala transcript<textarea class="si">${escapeHtml(segment.transcript_si || "")}</textarea></label>
        <label>English narration<textarea class="en">${escapeHtml(segment.narration_en || "")}</textarea></label>
      </div>
      ${player}
      <div class="actions">
        <button class="save secondary">Save edits</button>
        <button class="regen">Regenerate this segment</button>
        <button class="to-original secondary">Keep as recorded</button>
        ${issues}
      </div>
    </article>`;
}

async function loadSegments() {
  if (!currentJob) return;
  const segments = await request(`/api/jobs/${currentJob}/segments`);
  $("segments").innerHTML = segments.map(segmentCard).join("");
  document.querySelectorAll(".segment").forEach(card => {
    const id = card.dataset.id;
    const later = (ms) => setTimeout(loadSegments, ms);
    card.querySelector(".save")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/transcript`, jsonOptions("PATCH", {text:card.querySelector(".si").value}));
      await request(`/api/segments/${id}/script`, jsonOptions("PATCH", {text:card.querySelector(".en").value}));
      card.querySelector(".qa-review, .qa-pass").textContent = "pending";
    });
    card.querySelector(".regen")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/regenerate`, jsonOptions("POST", {stage:"tts"}));
      card.querySelector(".regen").textContent = "Regenerating…"; later(2500);
    });
    card.querySelector(".to-original")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"original"})); later(1500);
    });
    card.querySelector(".to-narration")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/kind`, jsonOptions("PATCH", {kind:"narration"}));
      card.querySelector(".to-narration").textContent = "Voicing…"; later(4000);
    });
    card.querySelector(".confirm")?.addEventListener("click", async () => {
      await request(`/api/segments/${id}/confirm`, {method:"POST"}); await loadSegments();
    });
  });
}
```

In `app/static/index.html`, change the footer to:

```html
  <footer>English narration is AI-generated; recordings marked "kept as is" are the original audio. Review meaning, pronunciation, and pacing before publishing.</footer>
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest -p no:warnings`
Expected: all pass (120).

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/static/app.js app/static/index.html tests/test_api.py
git commit -m "Add confirm and kind endpoints and recording controls on the review page"
```

---

### Task 6: README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Configuration table**

Add rows `DIARIZE_MODEL` (default `gpt-4o-transcribe-diarize`) and `DETECT_RECORDINGS` (default `1`) after `TTS_SPEED`.

- [ ] **Step 2: API overview**

Add after the regenerate line:

```
- `PATCH /api/segments/{id}/kind`  (`original` keeps the segment as recorded audio; `narration` voices it)
- `POST /api/segments/{id}/confirm`  (accept a kept recording)
```

- [ ] **Step 3: Pipeline line and "What is included"**

Change the pipeline line near the top to:

```
`upload → validate → normalize → analyse delivery → find recordings → segment → Sinhala transcript → faithful translation → narration adaptation → TTS → QA → human review → assembly`
```

Add a bullet to "What is included" after the speaking-pattern bullet:

```
- embedded recordings (911 calls, interrogation tape, news clips) are detected by speaker
  diarization against a narrator reference clip, kept from the source with loudness alignment,
  and confirmed by the reviewer instead of being translated and re-voiced
```

- [ ] **Step 4: New section after "Narration direction"**

```markdown
## Embedded recordings

Creator videos often play real evidence audio: a 911 call, an interrogation, a news clip. That audio
must not be translated or re-voiced. Once per project the pipeline diarizes the delivery sample,
takes the dominant speaker's longest span (2–10 s) as the narrator reference, and stores it in the
speaking profile. Every chunk is then diarized with that reference. A recording starts at speech by
someone other than the narrator in a Latin-only script and continues, across its own silences,
until the narrator speaks again in Sinhala. Recordings under 1.5 s are ignored; kept ones are padded
150 ms into silence.

Each recording becomes its own segment of kind `original`. It skips transcription, translation,
adaptation, and TTS; its audio is cut from the normalised source and loudness-aligned; the
narration that follows is told what was heard so it can refer to it. Original segments arrive on the
review page as needs review with "Recorded audio kept as is. Confirm." Confirm them, or choose
"Treat as narration" to voice one. A narration segment can be kept with "Keep as recorded". At
assembly, the gap next to a recording is the 300 ms minimum.

If diarization is unavailable the job continues as narration only and says so in the progress line.
Independently, any narration segment whose transcript contains eight or more consecutive English
words is flagged for review as a possible recording. `DETECT_RECORDINGS=0` (or `detect_recordings`
on the process request) turns detection off.

`scripts/end_to_end_check.py` runs the real pipeline on a slice of an existing project's source with
the configured models and prints every segment's kind and text. It spends API credit.
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "Document embedded recording detection and review controls"
```

---

### Task 7: End-to-end check with the real models (paid, approved)

Runs the finished pipeline on the first 130 seconds of the Gilgo Beach source, which contains the intro, the 911 call, and its continuation, with the configured OpenAI models.

**Files:**
- Create: `scripts/end_to_end_check.py`
- Output: `data/e2e_check/<new project id>/report.md` and the project's exports

- [ ] **Step 1: Create `scripts/end_to_end_check.py`**

```python
"""Run the real pipeline on a slice of an existing project's source with the actual models.

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/end_to_end_check.py <source_project_id> <start_s> <end_s>

Cuts [start_s, end_s] from the source project's normalised audio, creates a new project through
the HTTP API, uploads the slice, runs a processing job to completion (the review gate is on, so
it stops at awaiting_review), prints every segment, confirms every kept recording, assembles,
and writes data/e2e_check/<project_id>/report.md. Spends API credit in proportion to the slice.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app, db, pipeline  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    source_project, start_s, end_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    source = settings.data_dir / "projects" / source_project / "original" / "source_normalized.wav"
    if not source.exists():
        sys.exit(f"No normalised source at {source}")
    out_dir = settings.data_dir / "e2e_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_path = out_dir / f"slice_{int(start_s)}_{int(end_s)}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(source),
         "-ac", "1", "-ar", "24000", str(slice_path)],
        check=True,
    )
    original = db.one("SELECT * FROM projects WHERE id=?", (source_project,)) or {}

    client = TestClient(app)
    created = client.post("/api/projects", json={
        "title": f"e2e check {int(start_s)}-{int(end_s)}s",
        "topic": original.get("topic") or "true-crime documentary narration",
        "glossary": original.get("glossary") or {},
    })
    created.raise_for_status()
    project_id = created.json()["id"]
    with slice_path.open("rb") as handle:
        uploaded = client.post(f"/api/projects/{project_id}/upload", files={"file": (slice_path.name, handle, "audio/wav")})
    uploaded.raise_for_status()
    print("project", project_id, "uploaded", uploaded.json()["duration_ms"], "ms")

    # TestClient runs background tasks before returning, so this call blocks until the job finishes.
    started = client.post(f"/api/projects/{project_id}/process", json={"human_review_gate": True})
    started.raise_for_status()
    job_id = started.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    segments = client.get(f"/api/jobs/{job_id}/segments").json()

    lines = [f"# End-to-end check: project {project_id}, job {job_id}", "",
             f"Job status: {job['status']} at stage {job['stage']}; error: {job.get('error')}",
             f"Warnings: {job.get('warnings')}", ""]
    for s in segments:
        lines += [f"## Segment {s['segment_index']} [{s['kind']}] {s['start_ms']}-{s['end_ms']} ms  qa={s['qa_status']}",
                  f"- audio: {s.get('tts_audio_path')} ({s.get('tts_duration_ms')} ms)",
                  f"- transcript: {s.get('transcript_si')}",
                  f"- narration: {s.get('narration_en')}",
                  f"- style: {s.get('style')}",
                  f"- qa: {s.get('qa')}", ""]
    for s in segments:
        if s["kind"] == "original":
            client.post(f"/api/segments/{s['id']}/confirm").raise_for_status()
    if job["status"] == "awaiting_review":
        assembled = client.post(f"/api/projects/{project_id}/assemble")
        lines += ["## Assembly", f"status {assembled.status_code}: {assembled.json()}", ""]
        for name, path in pipeline.assemble_project(project_id, job_id).items():
            info = pipeline.audio.probe(Path(path))
            lines.append(f"- {name}: {path} ({info.duration_ms} ms, {info.channels} ch, {info.sample_rate} Hz)")
    calls = db.all("SELECT stage, model, status, COUNT(*) AS n FROM model_calls WHERE job_id=? GROUP BY stage, model, status", (job_id,))
    lines += ["", "## Model calls", *[f"- {c['stage']} {c['model']} {c['status']}: {c['n']}" for c in calls]]
    report = out_dir / project_id / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("report written to", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it on the first 130 s of the Gilgo Beach source**

```bash
set -a && source .env && set +a && .venv/bin/python scripts/end_to_end_check.py 5220183a-664a-4f9b-a534-2c3cc2ba8ff7 0 130
```

Expected: a project with roughly five segments. The intro (0–45 s) is narration; the 911 call is one `original` segment starting near 58.75 s (45 s + 13.75 s) and running to the end of that chunk, with a transcript beginning "Hi, how can I assist you?"; the following chunk begins with an `original` segment for the call's continuation ("Where? Tell me. Where on Long Island are you?…") followed by narration; the job stops at `awaiting_review`; the report lists the model calls per stage including `diarization` with status `completed`. Assembly produces a WAV close to the sum of segment durations plus 300 ms gaps around recordings.

- [ ] **Step 3: Judge the result**

Read the report. Check: (a) every original segment's transcript is English evidence audio and no narration segment's transcript contains a long English run (if one does, the safety net must have flagged it); (b) the narration segment after the call refers to it without repeating it; (c) listen-ready files: the exports under `data/projects/<project_id>/exports/`. If the detection boundary is off by more than a second, or a narration segment was kept, record the exact spans in the final report rather than tuning thresholds blind.

- [ ] **Step 4: Commit the script**

```bash
git add scripts/end_to_end_check.py
git commit -m "Add real-model end-to-end check on a source slice"
```

`data/` is git-ignored; the report and the new project stay local.

---

## Final verification

- [ ] `.venv/bin/python -m pytest -p no:warnings` — all pass.
- [ ] `git status --short` — clean.
- [ ] Report to the owner: the end-to-end project id and export paths, the segment table with kinds and time ranges, the diarization spans found, warnings if any, and the reminder that existing projects need a fresh run for detection.

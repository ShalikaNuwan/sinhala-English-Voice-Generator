# Source-Matched Breaks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hand the owner a reviewed English script whose breaks are their own measured Sinhala pauses, written as `<break time="x.xs" />` marks, as ordered parts around the kept recordings, from a job that voices nothing.

**Architecture:** `_create_segments` already has every narrator phrase with times from the diarizer; it now stores them per narration segment (`phrases_json`). A new pure module `app/breaks.py` groups phrases into flowing runs (gap under 150 ms merges), turns each run's trailing silence into a break value, validates that the adaptation's lines cover the runs, and renders and splits parts under ElevenLabs' limits. `AIClient.adapt` accepts the numbered runs and returns `lines`. The pipeline stores `lines_json`, flags misaligned lines in QA, and skips synthesis when `voice_stage` is off. `export_script_parts` writes the parts with marks, the recording clips, and an order sheet. Spec: `docs/superpowers/specs/2026-09-21-source-breaks-design.md`. Branch: `source-breaks`.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI SDK, SQLite, pytest. Tests: `.venv/bin/python -m pytest -p no:warnings`. No test makes a paid call.

**Already on the branch (uncommitted draft, becomes the base commit):** `voice_stage` setting/request/job key and the skip in `_narrate_segment`; `app/script_parts.py` (text-only grouping, replaced in Task 4); a first `export_script_parts` in `app/pipeline.py` (rewritten in Task 4); the endpoint `POST /api/projects/{id}/export/parts`; the "Text only" checkbox and "Export script parts" button; `scripts/export_script_parts.py`; tests in `tests/test_script_parts.py`, `tests/test_pipeline.py`, `tests/test_api.py`; a README section "Voicing outside the pipeline".

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `app/breaks.py` | Runs from phrases, break table, marks, line validation and enrichment, default lines, rendering, part splitting. | Create |
| `tests/test_breaks.py` | Unit tests. | Create |
| `app/schemas.py` | `ScriptLine`, `NarrationAdaptation.lines`. | Modify |
| `app/database.py` | `segments.phrases_json`, `segments.lines_json`. | Modify |
| `app/ai.py` | `adapt(..., runs=None)`. | Modify |
| `app/pipeline.py` | Store phrases; runs → adapt → lines; QA flag; `_adapt_lines` helper used by processing and regeneration; export rewrite. | Modify |
| `app/script_parts.py` | Lines-based parts and the order sheet. | Rewrite |
| `README.md` | Section rewrite. | Modify |
| `tests/test_ai.py`, `tests/test_database.py`, `tests/test_pipeline.py`, `tests/test_script_parts.py` | Updated. | Modify |

---

### Task 1: `app/breaks.py`

**Files:** Create `app/breaks.py`, `tests/test_breaks.py`.

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from app import breaks


def phrase(start, end, text="ඇය කියනවා"):
    return {"start_ms": start, "end_ms": end, "text": text}


def test_phrases_that_run_on_form_one_run_and_a_real_pause_ends_it():
    runs = breaks.runs([phrase(0, 8000, "A"), phrase(8100, 15000, "B"), phrase(15600, 30000, "C")], 45000)

    assert runs == [
        {"start_ms": 0, "end_ms": 15000, "text": "A B", "gap_after_ms": 600},
        {"start_ms": 15600, "end_ms": 30000, "text": "C", "gap_after_ms": 15000},
    ]


def test_the_last_gap_reaches_the_next_segments_first_phrase_when_known():
    runs = breaks.runs([phrase(0, 44700)], 45000, next_phrase_start_ms=45300)

    assert runs[-1]["gap_after_ms"] == 600


def test_runs_ignore_empty_phrases_and_sort_by_time():
    runs = breaks.runs([phrase(5000, 6000, "later"), phrase(0, 1000, "first"), phrase(2000, 3000, "   ")], 7000)

    assert [r["text"] for r in runs] == ["first", "later"]


def test_break_table():
    assert breaks.break_seconds(100, "and then,") is None
    assert breaks.break_seconds(200, "and then,") == 0.2
    assert breaks.break_seconds(200, "She never came home.") == 0.4  # a sentence end is lifted to the minimum
    assert breaks.break_seconds(760, "Frightened.”") == 0.8
    assert breaks.break_seconds(2600, "Long Island…") == 2.6
    assert breaks.break_seconds(5000, "anything") == 3.0
    assert breaks.break_seconds(444, "no punctuation") == 0.4


def test_mark_renders_the_elevenlabs_tag():
    assert breaks.mark(0.8) == '<break time="0.8s" />'
    assert breaks.mark(3.0) == '<break time="3.0s" />'


def test_lines_must_cover_every_run_once_in_order():
    assert breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [2, 3]}], 3)
    assert not breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [3]}], 3)
    assert not breaks.covers_runs([{"text": "a", "runs": [2]}, {"text": "b", "runs": [1]}], 2)
    assert not breaks.covers_runs([{"text": "a", "runs": [1]}, {"text": "b", "runs": [1, 2]}], 2)
    assert not breaks.covers_runs([], 1)


def test_attach_breaks_gives_each_line_the_silence_of_its_last_run():
    run_list = breaks.runs([phrase(0, 8000, "A"), phrase(8600, 15000, "B"), phrase(15300, 20000, "C")], 20000, 21000)
    lines = [{"text": "First line.", "runs": [1]}, {"text": "Second, then third.", "runs": [2, 3]}]

    enriched = breaks.attach_breaks(lines, run_list)

    assert enriched[0] == {"text": "First line.", "runs": [1], "start_ms": 0, "end_ms": 8000, "gap_after_ms": 600, "break_s": 0.6}
    assert enriched[1]["start_ms"] == 8600 and enriched[1]["end_ms"] == 20000
    assert enriched[1]["gap_after_ms"] == 1000 and enriched[1]["break_s"] == 1.0


def test_default_lines_share_the_span_by_length_and_use_default_breaks():
    lines = breaks.default_lines("Short one.\n\nA much longer paragraph here.", 10000, 20000)

    assert [line["text"] for line in lines] == ["Short one.", "A much longer paragraph here."]
    assert lines[0]["start_ms"] == 10000 and lines[1]["end_ms"] == 20000
    assert lines[0]["gap_after_ms"] == breaks.DEFAULT_PARAGRAPH_MS and lines[0]["break_s"] == 0.9
    assert lines[1]["gap_after_ms"] == breaks.DEFAULT_SENTENCE_MS and lines[1]["break_s"] == 0.5
    assert breaks.default_lines("   ", 0, 1000) == []


def test_render_writes_a_mark_after_every_line_that_has_one():
    text = breaks.render([{"text": "One.", "break_s": 0.6}, {"text": "Two", "break_s": None}, {"text": "Three.", "break_s": 3.0}])

    assert text == 'One. <break time="0.6s" />\nTwo\nThree. <break time="3.0s" />\n'


def test_parts_split_by_characters_and_by_marks():
    long_line = {"text": "x" * 5000, "break_s": 0.5}
    by_chars = breaks.split_parts([long_line, long_line, long_line])
    assert [len(p) for p in by_chars] == [1, 1, 1]

    marked = [{"text": "m.", "break_s": 0.4} for _ in range(85)]
    by_marks = breaks.split_parts(marked)
    assert [len(p) for p in by_marks] == [40, 40, 5]

    unmarked = [{"text": "m", "break_s": None} for _ in range(85)]
    assert len(breaks.split_parts(unmarked)) == 1
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest -p no:warnings tests/test_breaks.py` → `ImportError`.

- [ ] **Step 3: Create `app/breaks.py`**

```python
"""Turns the narrator's own silences into break marks for a script voiced elsewhere.

The diarizer hears the Sinhala as phrases with times. Phrases that follow one another with almost
no gap are one flowing run; a real pause ends a run. Each English line renders one run (or a merged
pair), so the silence after the run becomes the break after the line, written the way ElevenLabs
Multilingual v2 reads it: <break time="0.8s" />. Pure functions; the pipeline stores, the export
writes.
"""

from __future__ import annotations


# Phrases closer than this are one flowing run.
FLOW_GAP_MS = 150
# An English sentence end carries more weight than a Sinhala run-on; give it at least this much.
SENTENCE_MIN_MS = 400
# ElevenLabs' ceiling for one break tag.
BREAK_MAX_MS = 3000
# For segments without phrase data.
DEFAULT_SENTENCE_MS = 500
DEFAULT_PARAGRAPH_MS = 900
# Under ElevenLabs v2's 10,000-character generation limit, and few enough marks to stay stable.
PART_MAX_CHARS = 9000
PART_MAX_BREAKS = 40

_CLOSERS = "\"'”’)]"


def runs(phrases: list[dict], segment_end_ms: int, next_phrase_start_ms: int | None = None) -> list[dict]:
    """Group phrases into flowing runs and measure the silence after each."""
    ordered = sorted((p for p in phrases if (p.get("text") or "").strip()), key=lambda p: p["start_ms"])
    result: list[dict] = []
    for item in ordered:
        text = item["text"].strip()
        if result and item["start_ms"] - result[-1]["end_ms"] < FLOW_GAP_MS:
            result[-1]["end_ms"] = max(result[-1]["end_ms"], item["end_ms"])
            result[-1]["text"] = f"{result[-1]['text']} {text}"
        else:
            result.append({"start_ms": item["start_ms"], "end_ms": item["end_ms"], "text": text, "gap_after_ms": 0})
    for earlier, later in zip(result, result[1:]):
        earlier["gap_after_ms"] = max(0, later["start_ms"] - earlier["end_ms"])
    if result:
        tail_to = next_phrase_start_ms if next_phrase_start_ms is not None else segment_end_ms
        result[-1]["gap_after_ms"] = max(0, tail_to - result[-1]["end_ms"])
    return result


def ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip(_CLOSERS).endswith((".", "!", "?", "…"))


def break_seconds(gap_ms: int, line_text: str) -> float | None:
    """The break to write after a line, from the measured silence; None when punctuation is enough."""
    if gap_ms < FLOW_GAP_MS:
        return None
    if ends_sentence(line_text):
        gap_ms = max(gap_ms, SENTENCE_MIN_MS)
    return round(min(gap_ms, BREAK_MAX_MS) / 1000, 1)


def mark(seconds: float) -> str:
    return f'<break time="{seconds:.1f}s" />'


def covers_runs(lines: list[dict], run_count: int) -> bool:
    """Every run rendered exactly once, in order."""
    used = [number for line in lines for number in line.get("runs", [])]
    return used == list(range(1, run_count + 1))


def attach_breaks(lines: list[dict], run_list: list[dict]) -> list[dict]:
    """Give each line the times and trailing silence of the runs it covers, and the break to write."""
    result = []
    for line in lines:
        covered = [run_list[number - 1] for number in line.get("runs", []) if 1 <= number <= len(run_list)]
        gap = covered[-1]["gap_after_ms"] if covered else 0
        text = (line.get("text") or "").strip()
        result.append({
            "text": text,
            "runs": list(line.get("runs", [])),
            "start_ms": covered[0]["start_ms"] if covered else None,
            "end_ms": covered[-1]["end_ms"] if covered else None,
            "gap_after_ms": gap,
            "break_s": break_seconds(gap, text),
        })
    return result


def default_lines(narration_text: str, start_ms: int, end_ms: int) -> list[dict]:
    """Lines for a segment without phrase data: one per paragraph, default breaks, time shared by length."""
    paragraphs = [p.strip() for p in (narration_text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    total = sum(len(p) for p in paragraphs) or 1
    cursor = start_ms
    result = []
    for position, text in enumerate(paragraphs):
        span = round((end_ms - start_ms) * len(text) / total)
        gap = DEFAULT_SENTENCE_MS if position == len(paragraphs) - 1 else DEFAULT_PARAGRAPH_MS
        result.append({"text": text, "runs": [], "start_ms": cursor, "end_ms": cursor + span, "gap_after_ms": gap, "break_s": break_seconds(gap, text)})
        cursor += span
    if result:
        result[-1]["end_ms"] = end_ms
    return result


def render(lines: list[dict]) -> str:
    """Script text with a mark after every line that has a break."""
    return "".join(
        f"{line['text']} {mark(line['break_s'])}\n" if line.get("break_s") else f"{line['text']}\n"
        for line in lines
    )


def split_parts(lines: list[dict]) -> list[list[dict]]:
    """Split lines into generation-sized parts by character count and by number of marks."""
    parts: list[list[dict]] = []
    current: list[dict] = []
    chars = marks = 0
    for line in lines:
        has_mark = 1 if line.get("break_s") else 0
        cost = len(line["text"]) + 1 + (20 if has_mark else 0)
        if current and (chars + cost > PART_MAX_CHARS or marks + has_mark > PART_MAX_BREAKS):
            parts.append(current)
            current, chars, marks = [], 0, 0
        current.append(line)
        chars += cost
        marks += has_mark
    if current:
        parts.append(current)
    return parts
```

- [ ] **Step 4: Run** — expect 10 passed; full suite unchanged plus 10. Note `default_lines` sets the last line's `end_ms` to `end_ms` exactly (rounding).

- [ ] **Step 5: Commit** — `git add app/breaks.py tests/test_breaks.py && git commit -m "Add break rules: runs from phrases, the break table, marks, line checks, parts"`.

---

### Task 2: Schema, columns, adaptation with runs

**Files:** `app/schemas.py`, `app/database.py`, `app/ai.py`; tests `tests/test_ai.py`, `tests/test_database.py`.

- [ ] **Step 1: Failing tests**

Append to `tests/test_ai.py`:

```python
def test_adaptation_with_runs_asks_for_one_line_per_run():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=NarrationAdaptation(narration_text="ok"))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.adapt("She said it. He left.", "m", {}, runs=["ඇය කිව්වා", "එයා ගියා"])

    system = captured["input"][0]["content"]
    user = captured["input"][1]["content"]
    assert "one line per run" in system
    assert "Spoken runs, numbered, in order:\n1. ඇය කිව්වා\n2. එයා ගියා" in user


def test_adaptation_without_runs_keeps_the_passage_prompt():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=NarrationAdaptation(narration_text="ok"))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.adapt("x", "m", {})

    assert "one line per run" not in captured["input"][0]["content"]
    assert "Spoken runs" not in captured["input"][1]["content"]


def test_adaptation_lines_parse_and_default_to_empty():
    from app.schemas import ScriptLine

    adaptation = NarrationAdaptation(narration_text="A.\n\nB.", lines=[ScriptLine(text="A.", runs=[1]), ScriptLine(text="B.", runs=[2, 3])])
    assert [line.runs for line in adaptation.lines] == [[1], [2, 3]]
    assert NarrationAdaptation(narration_text="x").lines == []
    style = adaptation.model_dump(exclude={"narration_text", "lines"})
    assert "lines" not in style
```

Append to `tests/test_database.py`:

```python
def test_initialize_adds_phrases_and_lines_to_an_older_database(tmp_path):
    import sqlite3

    from app.database import Database

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE segments (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, project_id TEXT NOT NULL, "
            "segment_index INTEGER NOT NULL, start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, "
            "source_audio_path TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'narration', style_json TEXT NOT NULL DEFAULT '{}', "
            "qa_json TEXT NOT NULL DEFAULT '{}', updated_at TEXT NOT NULL);"
            "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, updated_at) "
            "VALUES ('s1', 'j1', 'p1', 1, 0, 1000, 'x.wav', 'now');"
        )

    db = Database(path)
    db.initialize()

    row = db.one("SELECT * FROM segments WHERE id='s1'")
    assert row["phrases"] == [] and row["lines"] == []
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: `app/schemas.py`** — add before `NarrationAdaptation`:

```python
class ScriptLine(BaseModel):
    text: str = Field(description="One line of the script, rendering the spoken run or runs it names.")
    runs: list[int] = Field(default_factory=list, description="The 1-based numbers of the spoken runs this line renders, in order.")
```

and in `NarrationAdaptation`, after `narration_text`:

```python
    lines: list[ScriptLine] = Field(
        default_factory=list,
        description="The script as one line per spoken run, in order, when runs were given. Empty otherwise.",
    )
```

Everywhere the pipeline does `adaptation.model_dump(exclude={"narration_text"})` it must now exclude `lines` too; Task 3 changes those call sites.

- [ ] **Step 4: `app/database.py`** — in `SCHEMA`'s `segments` table add after `style_json`:

```sql
    phrases_json TEXT NOT NULL DEFAULT '[]',
    lines_json TEXT NOT NULL DEFAULT '[]',
```

and extend `ADDED_COLUMNS`:

```python
    ("segments", "phrases_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("segments", "lines_json", "TEXT NOT NULL DEFAULT '[]'"),
```

- [ ] **Step 5: `app/ai.py` `adapt`** — signature `def adapt(self, faithful_en: str, model: str, narrator_profile: dict, previous_narration: str = "", runs: list[str] | None = None)`. Build the user content as today, then if `runs`: append

```python
        if runs:
            numbered = "\n".join(f"{number}. {text}" for number, text in enumerate(runs, start=1))
            user_content += f"\nSpoken runs, numbered, in order:\n{numbered}"
```

and append to the system prompt string (keep the existing text; add this as the final paragraph, only when `runs`):

```python
        system = SYSTEM_PROMPT_TEXT  # the existing literal, unchanged
        if runs:
            system += (
                "\nThe Sinhala was spoken as the numbered runs in the user message. Return lines: one line per run, "
                "in the same order, each rendering exactly that run's content, with runs listing the number it "
                "covers. Merge two neighbouring runs into one line only when English grammar leaves no other "
                "choice, and list both numbers. Every run must be covered exactly once. narration_text must be "
                "the lines joined by blank lines."
            )
```

(Implement by pulling the current system literal into a local variable first; do not change its wording.)

- [ ] **Step 6: Run the suite** — all pass. **Commit** `git add app/schemas.py app/database.py app/ai.py tests/test_ai.py tests/test_database.py && git commit -m "Adapt with spoken runs; store phrases and lines per segment"`.

---

### Task 3: Pipeline: phrases, runs, lines, QA flag, regeneration

**Files:** `app/pipeline.py`, `tests/test_pipeline.py`.

- [ ] **Step 1: Fakes and failing tests**

`FakeAI.adapt` becomes:

```python
    def adapt(self, *_args, runs=None, **_kwargs):
        from app.schemas import ScriptLine
        if runs:
            lines = [ScriptLine(text=f"Line {number}.", runs=[number]) for number in range(1, len(runs) + 1)]
            return NarrationAdaptation(narration_text="\n\n".join(line.text for line in lines), lines=lines,
                                       beat="reveal", emotion="tense", pause_after_ms=800)
        return NarrationAdaptation(narration_text="This is a test.", beat="reveal", emotion="tense", pause_after_ms=800)
```

`ProfilingFakeAI.adapt` records `runs`:

```python
    def adapt(self, faithful_en, model, narrator_profile, previous_narration="", runs=None):
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration, "runs": runs})
        return super().adapt(faithful_en, model, narrator_profile, previous_narration=previous_narration, runs=runs)
```

Any other fake `adapt` override in the file (`PausingFakeAI`, `LonelyFakeAI`, `CeilingFakeAI`) gains `runs=None` in its signature and ignores it.

Add a fake after `BoundaryFakeAI`:

```python
class PhrasedFakeAI(ProfilingFakeAI):
    """Chunk 1 is narrator only, spoken as four phrases: A and B run on, then real pauses before C and D."""

    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        if reference is None:
            return [{"speaker": "A", "start": 0.0, "end": 9.0, "text": SINHALA}]
        if Path(audio_path).name == "0001_source.wav":
            return [
                {"speaker": "narrator", "start": 0.0, "end": 8.0, "text": "A " + SINHALA},
                {"speaker": "narrator", "start": 8.1, "end": 15.0, "text": "B " + SINHALA},
                {"speaker": "narrator", "start": 15.6, "end": 30.0, "text": "C " + SINHALA},
                {"speaker": "narrator", "start": 30.3, "end": 44.7, "text": "D " + SINHALA},
            ]
        return [{"speaker": "narrator", "start": 0.5, "end": 20.2, "text": "E " + SINHALA}]


class MisalignedFakeAI(PhrasedFakeAI):
    """Returns lines that do not cover the runs."""

    def adapt(self, faithful_en, model, narrator_profile, previous_narration="", runs=None):
        from app.schemas import ScriptLine
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration, "runs": runs})
        if runs:
            lines = [ScriptLine(text="Only one.", runs=[1]), ScriptLine(text="Wrong.", runs=[1, 2])]
            return NarrationAdaptation(narration_text="Only one.\n\nWrong.", lines=lines)
        return NarrationAdaptation(narration_text="This is a test.")
```

Append tests:

```python
def test_narrator_phrases_are_stored_per_narration_segment(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert segments[0]["phrases"] == [{"start_ms": 0, "end_ms": 19850, "text": SINHALA}]
    assert segments[1]["phrases"] == []  # the kept recording
    assert segments[2]["phrases"] == [{"start_ms": 30150, "end_ms": 45000, "text": SINHALA}]
    assert segments[3]["phrases"] == [{"start_ms": 45000, "end_ms": 65000, "text": SINHALA}]


def test_runs_reach_the_adaptation_and_lines_carry_the_measured_breaks(tmp_path):
    ai = PhrasedFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    first, second = db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
    assert ai.adapt_calls[0]["runs"] == ["A " + SINHALA + " B " + SINHALA, "C " + SINHALA, "D " + SINHALA]
    assert first["narration_en"] == "Line 1.\n\nLine 2.\n\nLine 3."
    lines = first["lines"]
    assert [line["runs"] for line in lines] == [[1], [2], [3]]
    assert [line["gap_after_ms"] for line in lines] == [600, 300, 800]  # 45300 (next phrase, chunk 2 file 0.5 s minus 0.2 s lead) - 44700
    assert [line["break_s"] for line in lines] == [0.6, 0.4, 0.8]
    assert lines[0]["start_ms"] == 0 and lines[0]["end_ms"] == 15000
    assert second["lines"][0]["runs"] == [1] and second["lines"][0]["gap_after_ms"] == 0 and second["lines"][0]["break_s"] is None
    assert first["qa_status"] == "passed"


def test_lines_that_do_not_cover_the_runs_go_to_review(tmp_path):
    ai = MisalignedFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    first = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=1", (job_id,))
    assert first["qa_status"] == "needs_review"
    assert "Script lines do not follow the spoken runs" in first["qa"]["issues"]
    assert [line["runs"] for line in first["lines"]] == [[1], [1, 2]]


def test_a_segment_without_phrases_adapts_as_a_passage_with_a_warning(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE job_id=?", (job_id,))
    # FakeAI's sample-and-chunk spans give this 1 s source a narrator phrase 0-1000 ms, so phrases exist; switch detection off to get none.
    assert ai.adapt_calls[0]["runs"] == [SINHALA_TEST]


def test_detection_off_means_no_phrases_no_runs_and_no_warning_about_them(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id, detect_recordings=False)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE job_id=?", (job_id,))
    assert segment["phrases"] == [] and segment["lines"] == []
    assert ai.adapt_calls[0]["runs"] is None
    assert not any("spoken phrases" in w for w in db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"])


def test_regenerating_the_adaptation_uses_the_stored_phrases(tmp_path):
    ai = PhrasedFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)
    first = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=1", (job_id,))

    pipeline.regenerate_segment(first["id"], "adaptation")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (first["id"],))
    assert ai.adapt_calls[-1]["runs"] == ai.adapt_calls[0]["runs"]
    assert [line["break_s"] for line in fresh["lines"]] == [0.6, 0.4, 0.8]
```

Define `SINHALA_TEST = "මෙය පරීක්ෂණයකි."` near `SINHALA` (it is the text `FakeAI.diarize` returns).

- [ ] **Step 2: Run** — the new tests fail; note which existing tests fail because `FakeAI.adapt` now returns lines (none should).

- [ ] **Step 3: Edit `app/pipeline.py`**

Imports: `from . import breaks, pauses, recordings, script_parts, speaking_profile`.

In `_create_segments`, for narration pieces, build the phrases before the INSERT and store them:

```python
                phrases = [
                    {
                        "start_ms": max(start_ms, chunk_start + round(float(item["start"]) * 1000)),
                        "end_ms": min(end_ms, chunk_start + round(float(item["end"]) * 1000)),
                        "text": (item.get("text") or "").strip(),
                    }
                    for item in spans
                    if recordings.is_narrator_speech(item)
                    and chunk_start + round(float(item["end"]) * 1000) > start_ms
                    and chunk_start + round(float(item["start"]) * 1000) < end_ms
                ]
                self.db.execute(
                    "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, kind, phrases_json, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (segment_id, job_id, project["id"], index, start_ms, end_ms, str(source_path), "narration", json.dumps(phrases), utc_now()),
                )
```

(`spans` is the chunk-relative, lead-shifted list; when detection failed it is `[]` and phrases is `[]`.)

Add two helpers after `_direction_inputs`:

```python
    def _next_phrase_start(self, job_id: str, segment_index: int) -> int | None:
        """Where the narrator next speaks after this segment, if the next segment is narration with phrases."""
        following = self.db.one(
            "SELECT kind, phrases_json FROM segments WHERE job_id=? AND segment_index=?", (job_id, segment_index + 1),
        )
        if following and following.get("kind") == "narration" and following.get("phrases"):
            return following["phrases"][0]["start_ms"]
        return None

    def _adapt_lines(self, ai: AIClient, job: dict, project: dict, segment: dict, faithful_text: str, previous_narration: str):
        """Adapt a segment as lines over its spoken runs; fall back to a passage when it has no phrases.

        Returns (narration_text, style, lines, lines_ok, run_count).
        """
        config = job["config"]
        phrases = segment.get("phrases") or []
        run_list = breaks.runs(phrases, segment["end_ms"], self._next_phrase_start(job["id"], segment["segment_index"])) if phrases else []
        if not run_list and config.get("detect_recordings", self.config.detect_recordings):
            self._warn(job["id"], f"No spoken phrases for segment {segment['segment_index']}; its breaks will use defaults")
        run_texts = [run["text"] for run in run_list] or None
        adaptation = self._tracked_call(
            job["id"], segment["id"], "adaptation", config["text_model"], faithful_text,
            lambda: ai.adapt(faithful_text, config["text_model"], project["narrator_profile"] or {}, previous_narration[-600:], runs=run_texts),
        )
        style = adaptation.model_dump(exclude={"narration_text", "lines"})
        raw_lines = [line.model_dump() for line in adaptation.lines]
        lines_ok = True
        lines: list[dict] = []
        if run_list:
            lines_ok = bool(raw_lines) and breaks.covers_runs(raw_lines, len(run_list))
            lines = breaks.attach_breaks(raw_lines, run_list)
        narration_text = "\n\n".join(line["text"] for line in lines) if lines else adaptation.narration_text
        return narration_text, style, lines, lines_ok, len(run_list)
```

In `_narrate_segment`, replace the adaptation block (from `adaptation = self._tracked_call(` through the `UPDATE segments SET narration_en=?, style_json=?` execute) with:

```python
        narration_text, style, lines, lines_ok, _ = self._adapt_lines(ai, job, project, segment, faithful.english_faithful, previous_context)
        self.db.execute(
            "UPDATE segments SET narration_en=?, style_json=?, lines_json=?, status='adapted', updated_at=? WHERE id=?",
            (narration_text, json.dumps(style), json.dumps(lines), utc_now(), segment_id),
        )
```

and use `narration_text` wherever `adaptation.narration_text` was used below (synthesis input, QA input, the returned context). After `qa_payload = self._qa_verdict(...)` add:

```python
        if not lines_ok:
            qa_payload["passed"] = False
            qa_payload["issues"] = qa_payload["issues"] + ["Script lines do not follow the spoken runs"]
```

In `regenerate_segment`'s `if stage == "adaptation":` block, replace the adapt call and its UPDATE with:

```python
                narration_text, style, lines, lines_ok, _ = self._adapt_lines(ai, job, project, segment, faithful_text, previous_narration)
                self.db.execute(
                    "UPDATE segments SET narration_en=?, style_json=?, lines_json=? WHERE id=?",
                    (narration_text, json.dumps(style), json.dumps(lines), segment_id),
                )
```

initialise `lines_ok = True` before the stage chain, and apply the same QA addition after the verdict there.

- [ ] **Step 4: Run** `tests/test_pipeline.py`, then the full suite. If `test_narrator_phrases_are_stored_per_narration_segment` fails on segment 3's start, check the clamp to the piece bounds (`max(start_ms, …)`).

- [ ] **Step 5: Commit** — `git add app/pipeline.py tests/test_pipeline.py && git commit -m "Store spoken phrases, adapt as lines over runs, flag misaligned lines"`.

---

### Task 4: Export with marks, parts module, README

**Files:** `app/script_parts.py` (rewrite), `app/pipeline.py` (`export_script_parts` rewrite), `README.md`; tests `tests/test_script_parts.py` (rewrite), `tests/test_pipeline.py` (export tests updated).

- [ ] **Step 1: Rewrite `tests/test_script_parts.py`**

```python
from __future__ import annotations

from app import breaks, script_parts


def seg(index, kind="narration", text="", lines=None, start=None, end=None, audio=None):
    return {"segment_index": index, "kind": kind, "narration_en": text, "lines": lines or [], "transcript_si": "call text",
            "start_ms": start if start is not None else index * 1000, "end_ms": end if end is not None else index * 1000 + 1000,
            "tts_audio_path": audio}


def line(text, break_s, start, end):
    return {"text": text, "runs": [1], "start_ms": start, "end_ms": end, "gap_after_ms": round((break_s or 0) * 1000), "break_s": break_s}


def test_parts_alternate_narration_runs_and_recordings_and_carry_lines():
    parts = script_parts.plan_parts([
        seg(1, lines=[line("Intro one.", 0.6, 1000, 1500), line("Intro two.", 0.4, 1600, 2000)]),
        seg(2, kind="original", audio="/a/0002_original.wav"),
        seg(3, text="After the call.\n\nAnd more.", start=3000, end=4000),
    ])

    assert [p["kind"] for p in parts] == ["narration", "recording", "narration"]
    assert [l["text"] for l in parts[0]["lines"]] == ["Intro one.", "Intro two."]
    assert parts[0]["segments"] == [1] and parts[0]["start_ms"] == 1000 and parts[0]["end_ms"] == 2000
    # A segment without lines gets default lines with default breaks.
    assert [l["text"] for l in parts[2]["lines"]] == ["After the call.", "And more."]
    assert [l["break_s"] for l in parts[2]["lines"]] == [0.9, 0.5]


def test_narration_parts_are_split_under_the_generation_limits():
    big = [line("x" * 5000, 0.5, i * 10, i * 10 + 5) for i in range(3)]
    parts = script_parts.plan_parts([seg(1, lines=big)])
    pieces = script_parts.expand(parts)

    assert [p["kind"] for p in pieces] == ["narration", "narration", "narration"]
    assert all(len(p["lines"]) == 1 for p in pieces)
    assert [p["number"] for p in pieces] == [1, 2, 3]


def test_expand_numbers_files_in_playing_order():
    parts = script_parts.plan_parts([seg(1, lines=[line("A.", 0.5, 0, 500)]), seg(2, kind="original", audio="/a/x.wav"), seg(3, lines=[line("B.", None, 2000, 2500)])])

    pieces = script_parts.expand(parts)

    assert [(p["number"], p["kind"]) for p in pieces] == [(1, "narration"), (2, "recording"), (3, "narration")]
    assert script_parts.file_name(pieces[0]) == "01_narration.txt"
    assert script_parts.file_name(pieces[1]) == "02_recording_segment002.wav"


def test_index_lines_report_marks_and_target_length():
    parts = script_parts.plan_parts([seg(1, lines=[line("A.", 0.6, 0, 8000), line("B.", None, 8600, 12000)]), seg(2, kind="original", audio="/a/x.wav", start=12000, end=20000)])
    pieces = script_parts.expand(parts)

    lines = script_parts.index_lines(pieces)

    assert lines[-2] == "- `01_narration.txt`: narration, segments 1, source 0.0-12.0 s, target 12.0 s, 4 characters, 1 break mark"
    assert lines[-1] == "- `02_recording_segment002.wav`: kept recording, segment 2, source 12.0-20.0 s"


def test_render_of_a_piece_matches_breaks_render():
    piece = {"kind": "narration", "lines": [line("A.", 0.6, 0, 1), line("B", None, 2, 3)]}
    assert script_parts.render(piece) == breaks.render(piece["lines"])
```

- [ ] **Step 2: Rewrite `app/script_parts.py`**

```python
"""Splits a reviewed project into narration parts around the kept recordings, ready to voice elsewhere.

Every run of narration between two kept recordings is one part, carrying its lines and the break
after each line; a part is split further under the generation limits. Pure functions; the pipeline
writes the files.
"""

from __future__ import annotations

from . import breaks


def plan_parts(segments: list[dict]) -> list[dict]:
    """Alternating narration parts (with lines) and recordings, in playing order."""
    parts: list[dict] = []
    current: dict | None = None
    for segment in sorted(segments, key=lambda item: item["segment_index"]):
        if segment.get("kind") == "original":
            if current:
                parts.append(current)
                current = None
            parts.append({
                "kind": "recording",
                "segment_index": segment["segment_index"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "audio_path": segment.get("tts_audio_path"),
                "transcript": segment.get("transcript_si") or "",
            })
            continue
        lines = list(segment.get("lines") or []) or breaks.default_lines(segment.get("narration_en") or "", segment["start_ms"], segment["end_ms"])
        if not lines:
            continue
        if current is None:
            current = {"kind": "narration", "segments": [], "lines": [], "start_ms": segment["start_ms"], "end_ms": segment["end_ms"]}
        current["segments"].append(segment["segment_index"])
        current["lines"].extend(lines)
        current["end_ms"] = segment["end_ms"]
    if current:
        parts.append(current)
    return parts


def expand(parts: list[dict]) -> list[dict]:
    """Split narration parts under the generation limits and number every piece in playing order."""
    pieces: list[dict] = []
    for part in parts:
        if part["kind"] == "recording":
            pieces.append(dict(part, number=len(pieces) + 1))
            continue
        for chunk in breaks.split_parts(part["lines"]):
            timed = [line for line in chunk if line.get("start_ms") is not None]
            pieces.append({
                "kind": "narration",
                "number": len(pieces) + 1,
                "segments": part["segments"],
                "lines": chunk,
                "start_ms": timed[0]["start_ms"] if timed else part["start_ms"],
                "end_ms": timed[-1]["end_ms"] if timed else part["end_ms"],
            })
    return pieces


def file_name(piece: dict) -> str:
    if piece["kind"] == "recording":
        return f"{piece['number']:02d}_recording_segment{piece['segment_index']:03d}.wav"
    return f"{piece['number']:02d}_narration.txt"


def render(piece: dict) -> str:
    return breaks.render(piece["lines"])


def index_lines(pieces: list[dict]) -> list[str]:
    """A human-readable order sheet, one line per file."""
    lines = ["# Script parts, in playing order", "",
             "Voice each narration file on its own (ElevenLabs Multilingual v2 reads the break marks), then place",
             "the recording files between the voiced parts. Target is the length of the original passage.", ""]
    for piece in pieces:
        span = f"{piece['start_ms'] / 1000:.1f}-{piece['end_ms'] / 1000:.1f} s"
        if piece["kind"] == "recording":
            lines.append(f"- `{file_name(piece)}`: kept recording, segment {piece['segment_index']}, source {span}")
        else:
            segs = ", ".join(str(index) for index in piece["segments"])
            chars = sum(len(line["text"]) for line in piece["lines"])
            marks = sum(1 for line in piece["lines"] if line.get("break_s"))
            target = (piece["end_ms"] - piece["start_ms"]) / 1000
            lines.append(f"- `{file_name(piece)}`: narration, segments {segs}, source {span}, target {target:.1f} s, {chars} characters, {marks} break mark{'s' if marks != 1 else ''}")
    return lines
```

- [ ] **Step 3: Rewrite `Pipeline.export_script_parts`**

```python
    def export_script_parts(self, project_id: str, job_id: str | None = None) -> dict[str, str]:
        """Write the reviewed script as ordered parts with break marks, around the kept recordings."""
        if not job_id:
            latest = self.db.one("SELECT id FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            if not latest:
                raise RuntimeError("No processing job exists for this project")
            job_id = latest["id"]
        segments = self.db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
        pieces = script_parts.expand(script_parts.plan_parts(segments))
        if not pieces:
            raise RuntimeError("No reviewed narration to export yet")
        out_dir = self.config.data_dir / "projects" / project_id / "exports" / "script_parts"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        written: dict[str, str] = {}
        for piece in pieces:
            name = script_parts.file_name(piece)
            if piece["kind"] == "recording":
                source = Path(piece["audio_path"]) if piece["audio_path"] else None
                if source is None or not source.is_file():
                    raise RuntimeError(f"Kept recording for segment {piece['segment_index']} has no audio; regenerate it first")
                shutil.copyfile(source, out_dir / name)
            else:
                (out_dir / name).write_text(script_parts.render(piece), encoding="utf-8")
            written[name] = str(out_dir / name)
        (out_dir / "index.md").write_text("\n".join(script_parts.index_lines(pieces)) + "\n", encoding="utf-8")
        written["index.md"] = str(out_dir / "index.md")
        return written
```

- [ ] **Step 4: Update the two export tests in `tests/test_pipeline.py`**

`test_script_parts_export_splits_the_reviewed_script_around_recordings` becomes:

```python
def test_script_parts_export_writes_marked_lines_around_recordings(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    written = pipeline.export_script_parts(project_id)

    assert list(written) == ["01_narration.txt", "02_recording_segment002.wav", "03_narration.txt", "index.md"]
    # Segment 1's single run ends at the recording, so its line has no break; segments 3 and 4 each have one run.
    assert Path(written["01_narration.txt"]).read_text(encoding="utf-8") == "Line 1.\n"
    assert Path(written["03_narration.txt"]).read_text(encoding="utf-8") == "Line 1.\nLine 1.\n"
    assert 10100 <= pipeline.audio.probe(Path(written["02_recording_segment002.wav"])).duration_ms <= 10500
    index = Path(written["index.md"]).read_text(encoding="utf-8")
    assert "`01_narration.txt`: narration, segments 1, source 0.0-19.9 s, target 19.9 s" in index
    assert "`02_recording_segment002.wav`: kept recording, segment 2, source 19.9-30.2 s" in index
    assert "`03_narration.txt`: narration, segments 3, 4, source 30.2-65.0 s" in index


def test_script_parts_export_writes_measured_breaks(tmp_path):
    ai = PhrasedFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)

    written = pipeline.export_script_parts(project_id)

    text = Path(written["01_narration.txt"]).read_text(encoding="utf-8")
    assert text == 'Line 1. <break time="0.6s" />\nLine 2. <break time="0.4s" />\nLine 3. <break time="0.8s" />\nLine 1.\n'
    assert "1 break mark" not in Path(written["index.md"]).read_text(encoding="utf-8")
    assert "3 break marks" in Path(written["index.md"]).read_text(encoding="utf-8")
```

Keep `test_script_parts_export_is_refused_when_a_recording_has_no_audio` as it is. Add:

```python
def test_old_segments_without_lines_export_with_default_breaks(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    db.execute("UPDATE segments SET lines_json='[]', narration_en='Old one.\n\nOld two.' WHERE id=?", (segments[0]["id"],))

    written = pipeline.export_script_parts(project_id)

    assert Path(written["01_narration.txt"]).read_text(encoding="utf-8") == 'Old one. <break time="0.9s" />\nOld two. <break time="0.5s" />\n'
```

- [ ] **Step 5: README** — replace the "Voicing outside the pipeline" section with:

```markdown
## Voicing outside the pipeline

Tick "Text only (voice elsewhere)" before starting processing (or send `voice_stage: false`, or set
`VOICE_STAGE=0`) and a job stops after transcription, detection, translation, adaptation and QA:
nothing is voiced and no voice credit is spent. Review the scripts as usual, then click "Export
script parts" (`POST /api/projects/{id}/export/parts`, or `scripts/export_script_parts.py`).

The breaks in the exported script are the narrator's own. The diarizer hears the Sinhala as phrases
with times; phrases that run on with under 150 ms between them are one flowing run, and a real
pause ends a run. The adaptation writes the English as one line per run, in order (merging two runs
only when English grammar forces it), so every line has a measured silence after it. That silence is
written after the line as `<break time="0.8s" />`, which ElevenLabs Multilingual v2 honours (v3 does
not read break tags). A line that ends a sentence gets at least 400 ms; nothing exceeds 3.0 s; a
gap under 150 ms writes no mark. The table lives at the top of `app/breaks.py`. When a segment's
lines do not cover its runs, QA sends it to review with "Script lines do not follow the spoken
runs". Segments without phrase data (detection off, or older jobs) export their paragraphs with
default breaks of 0.9 s between paragraphs and 0.5 s at the end.

The export is written to `exports/script_parts/`: every stretch of narration between two kept
recordings becomes one or more numbered text files (split at 9,000 characters or 40 break marks,
since too many marks in one generation destabilise the voice), every kept recording is copied as a
numbered WAV, and `index.md` lists the playing order with source times, a target length per part,
and the character and mark counts. Voice each narration file on its own in ElevenLabs, then place the
recording files between the voiced parts in your editor.
```

- [ ] **Step 6: Run the suite; commit** — `git add app/script_parts.py app/pipeline.py README.md tests/test_script_parts.py tests/test_pipeline.py && git commit -m "Export script parts with the narrator's measured breaks as ElevenLabs marks"`.

---

### Task 5: Real-model check (controller)

- [ ] Run a text-only job on the 130-second Gilgo slice with the real models (a variant of `scripts/end_to_end_check.py` that posts `voice_stage: false` and skips assembly), then `scripts/export_script_parts.py <project_id>`; read `index.md` and the narration files; confirm the lines cover the runs (no QA flag), the breaks look like the measured gaps, and the recordings sit between parts. Deliver the folder path to the owner.

## Final verification

- [ ] `.venv/bin/python -m pytest -p no:warnings` — all pass.
- [ ] `git status --short` — clean.

# Natural True-Crime Narration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the generated English narration sound like one real storyteller telling a true-crime case, consistent across the whole video, instead of a per-segment synthetic read.

**Architecture:** The adaptation stage writes a spoken performance script plus a story beat and a director's note. A new `app/direction.py` turns a fixed narrator persona, the source speaker's measured profile, the segment's style, and the previous segment's style into one voice-direction brief for OpenAI TTS. The pipeline threads the previous segment's style and narration through synthesis and regeneration, generates WAV segments, and the assembler inserts real pauses between segments derived from the adaptation's pause fields and the source narrator's measured pause habit.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, OpenAI Python SDK 2.x (`audio.speech`, `responses.parse`), FFmpeg 4.2 (`concat`, `anullsrc`, `loudnorm`), pytest. Spec: `docs/superpowers/specs/2026-09-05-natural-narration-design.md`.

Run every test command from the project root with `.venv/bin/python -m pytest`. Tests never call the paid API; they use the fake clients already in `tests/`.

---

## File map

| File | Responsibility | Action |
|---|---|---|
| `app/direction.py` | Narrator persona, delivery rules, `build_instructions`, `speaking_speed`, `segment_gap_ms`. Pure functions, no I/O. | Create |
| `tests/test_direction.py` | Unit tests for everything in `app/direction.py`. | Create |
| `app/schemas.py` | `NarrationAdaptation` gains `beat` and `delivery`. | Modify |
| `app/ai.py` | `adapt()` writes a performance script and takes previous narration; `synthesize()` uses the brief, `speed`, WAV output; `narration_instructions` removed; prompt version bump. | Modify |
| `app/config.py` | Default voice `cedar`, new `tts_speed`. | Modify |
| `app/main.py` | Job configuration carries `speed`. | Modify |
| `app/audio.py` | `assemble()` accepts `gaps_ms` and inserts silence. | Modify |
| `app/pipeline.py` | Threads previous style and narration, persona, speed; WAV names; gaps in assembly. | Modify |
| `app/static/index.html` | Voice field default `cedar`. | Modify |
| `.env.example`, `README.md` | Document `cedar`, `TTS_SPEED`, the direction model. | Modify |
| `scripts/listening_test.py` | Voices real segments old vs new and asks the audio model to critique them. Paid. | Create |
| `tests/test_ai.py`, `tests/test_pipeline.py`, `tests/test_audio.py`, `tests/test_api.py` | Updated for the new signatures and behaviour. | Modify |

---

### Task 1: Voice-direction brief in `app/direction.py`

Moves the instruction builder out of `app/ai.py` and gives it the persona, delivery rules, beat, director's note, and continuity.

**Files:**
- Create: `app/direction.py`
- Create: `tests/test_direction.py`
- Modify: `tests/test_ai.py` (remove the three `test_instructions_*` tests and the `PROFILE`/`STYLE` constants; they move here)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_direction.py`:

```python
from __future__ import annotations

from app import direction


PROFILE = {
    "measured": {"mean_pause_ms": 1321, "longest_pause_ms": 3547},
    "derived": {"pace": "moderate", "pause_style": "deliberate", "dynamics": "moderately dynamic"},
    "described": "Steady, moderate energy. Neutral and factual tone with little fluctuation.",
}
STYLE = {
    "pace": "moderate",
    "emotion": "calm, suspenseful",
    "emphasis": ["dead inside her house"],
    "beat": "reveal",
    "delivery": "Lower your voice as the call cuts out and let the last line hang.",
}
PREVIOUS = {"beat": "build", "emotion": "tense"}


def test_brief_opens_with_the_storyteller_persona():
    text = direction.build_instructions(STYLE, PROFILE)

    assert text.startswith(direction.DEFAULT_PERSONA)


def test_brief_carries_the_narrator_profile_and_the_moment():
    text = direction.build_instructions(STYLE, PROFILE)

    # The original speaker's character.
    assert "Steady, moderate energy" in text
    assert "deliberate" in text
    assert "moderately dynamic" in text
    assert "1321" in text and "3547" in text
    # What this passage needs.
    assert "reveal beat" in text
    assert "calm, suspenseful" in text
    assert "Lower your voice as the call cuts out" in text
    assert "dead inside her house" in text


def test_brief_includes_every_delivery_rule():
    text = direction.build_instructions(STYLE, PROFILE)

    for rule in direction.DELIVERY_RULES:
        assert rule in text


def test_brief_adds_continuity_only_when_a_previous_passage_exists():
    without = direction.build_instructions(STYLE, PROFILE)
    with_previous = direction.build_instructions(STYLE, PROFILE, previous_style=PREVIOUS)

    assert "Continuity" not in without
    assert "Continuity" in with_previous
    assert "build beat" in with_previous
    assert "tense" in with_previous
    assert "do not reset to neutral" in with_previous


def test_brief_falls_back_to_persona_rules_and_moment_without_a_profile():
    text = direction.build_instructions(STYLE, None)

    assert direction.DEFAULT_PERSONA in text
    assert "calm, suspenseful" in text
    assert "Voice character" not in text


def test_brief_uses_measurements_when_the_tone_analysis_failed():
    profile = {**PROFILE, "described": None}

    text = direction.build_instructions(STYLE, profile)

    assert "deliberate" in text
    assert "1321" in text
    assert "None" not in text


def test_brief_never_prints_none_for_missing_style_fields():
    text = direction.build_instructions({"narration_text": "x"}, None, previous_style={})

    assert "None" not in text
    assert "build beat" in text  # default beat


def test_brief_uses_a_project_persona_when_one_is_given():
    text = direction.build_instructions(STYLE, PROFILE, persona="You are a calm history lecturer.")

    assert text.startswith("You are a calm history lecturer.")
    assert direction.DEFAULT_PERSONA not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_direction.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.direction'`

- [ ] **Step 3: Create `app/direction.py`**

```python
"""Directs the voice: who the narrator is, how they speak, and what this passage needs.

Everything here is a pure function of dictionaries so it can be unit-tested without
touching the network. The pipeline passes the result to the TTS model as `instructions`.
"""

from __future__ import annotations


DEFAULT_PERSONA = (
    "You are a real person telling one listener, late at night, about a murder that actually "
    "happened. You are not a presenter and nothing is performed: you are remembering the case "
    "and telling it carefully because the people in it were real. You care about the victims. "
    "Your voice is quiet, close to the microphone, and unhurried."
)

DELIVERY_RULES = [
    "Talk, do not read. Let the rhythm be slightly uneven, the way real speech is.",
    "End statements low and settled. Never lift the pitch at the end of a sentence and never sing-song.",
    "Breathe. Take a short breath before a long sentence. An ellipsis is a beat you hold. A dash is a change of thought.",
    "Stay quiet and close. Intensity comes from getting quieter and slower, not louder.",
    "Say names, dates, times, and numbers carefully, as if you want the listener to remember them.",
    "Deliver quoted speech from 911 calls and witnesses as restrained reportage. Do not act it out.",
    "Never sound like an announcer, a voice assistant, or an advert. No brightness, no polish, no smile in the voice.",
]


def _character(profile: dict) -> str | None:
    """Describe the original speaker so the English narrator can match them."""
    derived = profile.get("derived") or {}
    measured = profile.get("measured") or {}
    parts: list[str] = []
    if profile.get("described"):
        parts.append(profile["described"])
    if derived:
        parts.append(
            f"Baseline pace is {derived.get('pace') or 'moderate'}, "
            f"with {derived.get('pause_style') or 'deliberate'} pauses and "
            f"{derived.get('dynamics') or 'controlled'} delivery."
        )
    if measured.get("mean_pause_ms"):
        parts.append(
            f"Leave roughly {measured['mean_pause_ms']}ms between sentences, "
            f"stretching to about {measured.get('longest_pause_ms') or measured['mean_pause_ms']}ms "
            "at the heaviest beats."
        )
    return " ".join(parts) if parts else None


def _moment(style: dict) -> str:
    """What this passage needs, from the adaptation model."""
    text = (
        f"This passage is a {style.get('beat') or 'build'} beat: "
        f"{style.get('emotion') or 'neutral'} in tone, pace {style.get('pace') or 'moderate'}."
    )
    if style.get("delivery"):
        text += f" Direction: {style['delivery']}"
    emphasis = ", ".join(style.get("emphasis") or [])
    if emphasis:
        text += f" Emphasise: {emphasis}."
    return text


def _continuity(previous_style: dict) -> str:
    return (
        f"Continuity: the previous passage was a {previous_style.get('beat') or 'build'} beat and "
        f"ended {previous_style.get('emotion') or 'neutral'} in tone. Carry that mood into your "
        "first line; do not reset to neutral."
    )


def build_instructions(
    style: dict,
    profile: dict | None,
    previous_style: dict | None = None,
    persona: str | None = None,
) -> str:
    """Build the TTS direction: persona, the original speaker's character, how to speak, this moment."""
    sections = [persona or DEFAULT_PERSONA]
    character = _character(profile) if profile else None
    if character:
        sections.append("Voice character, matched to the original narrator: " + character)
    sections.append("How to speak:\n- " + "\n- ".join(DELIVERY_RULES))
    sections.append(_moment(style))
    if previous_style:
        sections.append(_continuity(previous_style))
    return "\n\n".join(sections)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_direction.py -v`
Expected: 8 passed

- [ ] **Step 5: Remove the moved tests from `tests/test_ai.py`**

Delete from `tests/test_ai.py` the `PROFILE` and `STYLE` constants and the three functions `test_instructions_carry_the_narrator_profile_and_the_segment_moment`, `test_instructions_fall_back_to_segment_style_when_no_profile_exists`, and `test_instructions_use_measurements_when_the_tone_analysis_failed`. Keep `test_transcription_prompts_for_sinhala_without_unsupported_language_code` and `test_delivery_description_sends_the_audio_and_ignores_word_meaning`.

Run: `.venv/bin/python -m pytest tests/test_ai.py tests/test_direction.py -q`
Expected: 10 passed

- [ ] **Step 6: Commit**

```bash
git add app/direction.py tests/test_direction.py tests/test_ai.py
git commit -m "Add voice-direction brief with persona, delivery rules, and continuity"
```

---

### Task 2: Speed clamp and segment gap rule in `app/direction.py`

**Files:**
- Modify: `app/direction.py`
- Modify: `tests/test_direction.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_direction.py`:

```python
MEASURED_PROFILE = {"measured": {"mean_pause_ms": 773, "longest_pause_ms": 2732}}


def test_gap_is_the_pause_after_plus_the_pause_before():
    gap = direction.segment_gap_ms({"pause_after_ms": 800}, {"pause_before_ms": 400}, MEASURED_PROFILE)

    assert gap == 1200


def test_gap_falls_back_to_the_source_narrators_mean_pause_when_styles_say_nothing():
    gap = direction.segment_gap_ms({"pause_after_ms": 0}, {"pause_before_ms": 0}, MEASURED_PROFILE)

    assert gap == 773


def test_gap_falls_back_to_a_default_without_a_profile():
    assert direction.segment_gap_ms({}, {}, None) == direction.DEFAULT_GAP_MS
    assert direction.segment_gap_ms(None, None, {"measured": {"mean_pause_ms": 0}}) == direction.DEFAULT_GAP_MS


def test_gap_never_drops_below_the_minimum():
    gap = direction.segment_gap_ms({"pause_after_ms": 100}, {"pause_before_ms": 50}, MEASURED_PROFILE)

    assert gap == direction.MIN_GAP_MS


def test_gap_never_exceeds_the_source_narrators_longest_pause():
    gap = direction.segment_gap_ms({"pause_after_ms": 3000}, {"pause_before_ms": 3000}, MEASURED_PROFILE)

    assert gap == 2732


def test_gap_uses_the_default_ceiling_without_a_measured_longest_pause():
    gap = direction.segment_gap_ms({"pause_after_ms": 3000}, {"pause_before_ms": 3000}, None)

    assert gap == direction.DEFAULT_MAX_GAP_MS


def test_speed_is_clamped_to_the_api_range():
    assert direction.speaking_speed(1.0) == 1.0
    assert direction.speaking_speed(0.92) == 0.92
    assert direction.speaking_speed(0.1) == 0.25
    assert direction.speaking_speed(9.0) == 4.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_direction.py -v -k "gap or speed"`
Expected: FAIL with `AttributeError: module 'app.direction' has no attribute 'segment_gap_ms'`

- [ ] **Step 3: Add the functions to `app/direction.py`**

Add these constants directly under `DELIVERY_RULES`:

```python
# Silence between assembled segments.
MIN_GAP_MS = 300
DEFAULT_GAP_MS = 600
DEFAULT_MAX_GAP_MS = 2500
# The TTS endpoint accepts speed 0.25-4.0.
SPEED_RANGE = (0.25, 4.0)
```

Append to the end of the file:

```python
def segment_gap_ms(previous_style: dict | None, next_style: dict | None, profile: dict | None) -> int:
    """How much silence to leave between two assembled segments.

    The adaptation model's pause fields win. When they say nothing, fall back to the source
    narrator's measured mean pause, and never leave a gap longer than their longest pause.
    """
    previous_style = previous_style or {}
    next_style = next_style or {}
    measured = (profile or {}).get("measured") or {}

    gap = int(previous_style.get("pause_after_ms") or 0) + int(next_style.get("pause_before_ms") or 0)
    if gap == 0:
        gap = int(measured.get("mean_pause_ms") or 0) or DEFAULT_GAP_MS

    longest = int(measured.get("longest_pause_ms") or 0)
    ceiling = max(longest, MIN_GAP_MS) if longest else DEFAULT_MAX_GAP_MS
    return max(MIN_GAP_MS, min(gap, ceiling))


def speaking_speed(requested: float) -> float:
    """Clamp a configured speed to what the API accepts."""
    low, high = SPEED_RANGE
    return min(high, max(low, float(requested)))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_direction.py -v`
Expected: 15 passed

- [ ] **Step 5: Commit**

```bash
git add app/direction.py tests/test_direction.py
git commit -m "Add segment gap rule and speed clamp for narration direction"
```

---

### Task 3: `beat` and `delivery` on `NarrationAdaptation`

**Files:**
- Modify: `app/schemas.py:490-496`
- Test: `tests/test_ai.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ai.py`:

```python
def test_adaptation_defaults_beat_and_delivery_when_the_model_omits_them():
    from app.schemas import NarrationAdaptation

    adaptation = NarrationAdaptation(narration_text="She never came home.")

    assert adaptation.beat == "build"
    assert adaptation.delivery == ""
    style = adaptation.model_dump(exclude={"narration_text"})
    assert set(style) == {"beat", "delivery", "pace", "emphasis", "pause_before_ms", "pause_after_ms", "emotion"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ai.py::test_adaptation_defaults_beat_and_delivery_when_the_model_omits_them -v`
Expected: FAIL with `AttributeError: 'NarrationAdaptation' object has no attribute 'beat'`

- [ ] **Step 3: Replace the `NarrationAdaptation` class in `app/schemas.py`**

```python
class NarrationAdaptation(BaseModel):
    narration_text: str = Field(description="The script to be spoken aloud, written for the ear.")
    beat: Literal["setup", "build", "reveal", "aftermath", "reflection"] = Field(
        default="build",
        description="Where this passage sits in the story arc.",
    )
    delivery: str = Field(
        default="",
        description="One sentence of direction to the voice actor for this passage.",
    )
    pace: Literal["slow", "moderate", "fast"] = "moderate"
    emphasis: list[str] = Field(
        default_factory=list,
        description="Exact phrases from narration_text to give weight to.",
    )
    pause_before_ms: int = Field(default=0, ge=0, le=3000)
    pause_after_ms: int = Field(default=250, ge=0, le=3000)
    emotion: str = Field(default="neutral", description="The emotional tone of this passage, in a few words.")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add app/schemas.py tests/test_ai.py
git commit -m "Add story beat and delivery note to narration adaptation"
```

---

### Task 4: Performance script and directed synthesis in `app/ai.py`

**Files:**
- Modify: `app/ai.py` (remove `narration_instructions` at lines 14-48; replace `adapt()` at 104-132 and `synthesize()` at 134-152; bump `PROMPT_VERSION` at line 11)
- Test: `tests/test_ai.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ai.py`:

```python
def test_adaptation_asks_for_a_spoken_performance_script_with_previous_context():
    from app.schemas import NarrationAdaptation

    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=NarrationAdaptation(narration_text="ok"))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.adapt("She was found dead.", "text-model", {"tone": "calm"}, previous_narration="Earlier that night…")

    system = captured["input"][0]["content"]
    user = captured["input"][1]["content"]
    assert "spoken" in system.lower() or "speak" in system.lower()
    assert "ellipsis" in system.lower()
    assert "beat" in system.lower() and "delivery" in system.lower()
    assert "Do not add, remove, weaken, or strengthen" in system
    assert "Earlier that night…" in user
    assert "She was found dead." in user
    assert captured["text_format"] is NarrationAdaptation


def test_synthesis_sends_the_brief_speed_and_wav_format(tmp_path):
    from app import direction

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def stream_to_file(self, path):
            captured["path"] = path

    class Speech:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(with_streaming_response=Speech()))
    )
    output = tmp_path / "generated" / "0002_r01.wav"
    style = {"beat": "reveal", "emotion": "tense", "emphasis": ["911"]}
    previous = {"beat": "setup", "emotion": "calm"}

    ai.synthesize(
        "Hello? Hello?", "tts-model", "cedar", style, output,
        profile=None, previous_style=previous, persona=None, speed=0.92,
    )

    assert captured["model"] == "tts-model"
    assert captured["voice"] == "cedar"
    assert captured["input"] == "Hello? Hello?"
    assert captured["speed"] == 0.92
    assert captured["response_format"] == "wav"
    assert captured["instructions"] == direction.build_instructions(style, None, previous, None)
    assert captured["path"] == output
    assert output.parent.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_ai.py -v`
Expected: the two new tests FAIL. The adaptation test fails because `adapt()` has no `previous_narration` argument; the synthesis test fails with `TypeError` on the unexpected `previous_style` keyword.

- [ ] **Step 3: Edit `app/ai.py`**

Replace the imports and the top of the file (lines 1-48) with:

```python
from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from .direction import build_instructions, speaking_speed
from .schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation


PROMPT_VERSION = "2026-09-mvp2"
```

Replace the `adapt` method with:

```python
    def adapt(
        self,
        faithful_en: str,
        model: str,
        narrator_profile: dict,
        previous_narration: str = "",
    ) -> NarrationAdaptation:
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You turn faithful English into a script that a storyteller will speak aloud as the "
                        "voice-over of a true-crime documentary. Write for the ear, not the page:\n"
                        "- Short sentences, one idea each. Use contractions. Avoid formal written-English "
                        "constructions.\n"
                        "- Put an ellipsis (…) where the narrator should hold a beat, and a dash (—) for a "
                        "change of thought or an afterthought. Use them only where a person telling the story "
                        "would actually pause.\n"
                        "- Start a new paragraph (blank line) before a shift in the story.\n"
                        "- Write dates, times, and numbers the way they are said aloud, for example "
                        "'May 1st, 2010' and '4:51 in the morning'.\n"
                        "- Keep quoted speech from 911 calls and witnesses as plain spoken lines.\n"
                        "- Collapse accidental repeated starts, duplicated phrases, and incomplete false starts.\n"
                        "Do not add, remove, weaken, or strengthen any factual claim. Preserve every distinct "
                        "name, date, number, place, and causal link. Preserve supported suspense and emphasis. "
                        "Use the narrator profile.\n"
                        "Also return: beat (where this passage sits in the story: setup, build, reveal, "
                        "aftermath, or reflection); delivery (one sentence of direction to the voice actor for "
                        "this passage); emotion; pace; emphasis (exact phrases to weight); and pause_before_ms "
                        "and pause_after_ms (silence the narrator would leave before and after this passage)."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Narrator profile: {narrator_profile}\n"
                        f"Previous narration, for context only (do not repeat it): {previous_narration}\n"
                        f"Faithful English: {faithful_en}"
                    ),
                },
            ],
            text_format=NarrationAdaptation,
        )
        if not response.output_parsed:
            raise RuntimeError("Adaptation model did not return a structured result")
        return response.output_parsed
```

Replace the `synthesize` method with:

```python
    def synthesize(
        self,
        text: str,
        model: str,
        voice: str,
        style: dict,
        output_path: Path,
        profile: dict | None = None,
        previous_style: dict | None = None,
        persona: str | None = None,
        speed: float = 1.0,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        instructions = build_instructions(style, profile, previous_style, persona)
        with self.client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            speed=speaking_speed(speed),
            response_format="wav",
        ) as response:
            response.stream_to_file(output_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_ai.py tests/test_direction.py -v`
Expected: 20 passed

- [ ] **Step 5: Run the whole suite to see what the signature change broke**

Run: `.venv/bin/python -m pytest -q`
Expected: all pass. The fakes in `tests/test_pipeline.py` accept `*_args, **_kwargs` for `adapt`, and the pipeline does not yet pass the new keyword arguments to `synthesize`, so nothing breaks here. If anything fails, fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add app/ai.py tests/test_ai.py
git commit -m "Write a spoken performance script and direct synthesis with the full brief"
```

---

### Task 5: `cedar` voice and `TTS_SPEED` in configuration, job config, and the web form

**Files:**
- Modify: `app/config.py:21` (voice default) and add `tts_speed`
- Modify: `app/schemas.py` (`ProcessRequest` gains `speed`)
- Modify: `app/main.py:536-546` (`job_configuration`)
- Modify: `app/static/index.html:39`
- Modify: `.env.example`
- Test: `tests/test_api.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_api.py`, replace `test_job_configuration_defaults_to_the_configured_male_voice` with:

```python
def test_job_configuration_defaults_to_cedar_at_normal_speed():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["voice"] == "cedar"
    assert config["speed"] == 1.0
    assert config["audio_model"] == "gpt-audio"
    assert config["human_review_gate"] is True
```

And extend `test_job_configuration_lets_a_request_override_the_voice_and_audio_model` so the request is `ProcessRequest(voice="ballad", audio_model="gpt-audio-mini", speed=0.9)` and add `assert config["speed"] == 0.9` after the existing assertions.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_api.py -v -k job_configuration`
Expected: both FAIL (`config["voice"] == "onyx"`, and `ProcessRequest` rejects `speed`).

- [ ] **Step 3: Update `app/config.py`**

Change the `tts_voice` line and add `tts_speed` directly after it:

```python
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "cedar"))
    tts_speed: float = field(default_factory=lambda: float(os.getenv("TTS_SPEED", "1.0")))
```

- [ ] **Step 4: Update `ProcessRequest` in `app/schemas.py`**

Add after the `voice` field:

```python
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
```

- [ ] **Step 5: Update `job_configuration` in `app/main.py`**

Add after the `"voice"` entry:

```python
        "speed": payload.speed if payload.speed is not None else config.tts_speed,
```

- [ ] **Step 6: Update the web form and `.env.example`**

In `app/static/index.html` line 39 change `value="onyx"` to `value="cedar"`.

In `.env.example` change `TTS_VOICE=onyx` to `TTS_VOICE=cedar` and add a line `TTS_SPEED=1.0` directly after it.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_api.py -v`
Expected: 3 passed

- [ ] **Step 8: Commit**

```bash
git add app/config.py app/schemas.py app/main.py app/static/index.html .env.example tests/test_api.py
git commit -m "Default to the cedar voice and add a configurable narration speed"
```

---

### Task 6: Silence between segments in `AudioService.assemble()`

**Files:**
- Modify: `app/audio.py:122-140`
- Test: `tests/test_audio.py`

- [ ] **Step 1: Write the failing tests**

Add the imports below to the top of `tests/test_audio.py` and append the helper and tests:

```python
import subprocess
from pathlib import Path

import pytest

from app.audio import AudioError


def tone(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-ar", "24000", "-ac", "1", str(path)],
        check=True,
    )
    return path


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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v -k assembly`
Expected: FAIL with `TypeError: AudioService.assemble() got an unexpected keyword argument 'gaps_ms'` for two tests; the no-gaps test passes already.

- [ ] **Step 3: Replace `assemble` in `app/audio.py`**

```python
    def assemble(
        self,
        segment_paths: list[Path],
        wav_path: Path,
        mp3_path: Path,
        gaps_ms: list[int] | None = None,
    ) -> None:
        """Join the segments in order, leaving `gaps_ms[i]` of silence after segment i."""
        if not segment_paths:
            raise AudioError("No generated segments are available for assembly")
        gaps = list(gaps_ms or [])
        if gaps and len(gaps) != len(segment_paths) - 1:
            raise AudioError("Assembly needs exactly one gap between each pair of segments")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        inputs: list[str] = []
        labels: list[str] = []
        for index, path in enumerate(segment_paths):
            if index and gaps and gaps[index - 1] > 0:
                inputs.extend([
                    "-f", "lavfi", "-t", f"{gaps[index - 1] / 1000:.3f}",
                    "-i", "anullsrc=r=24000:cl=mono",
                ])
                labels.append(f"[{len(labels)}:a]")
            inputs.extend(["-i", str(path)])
            labels.append(f"[{len(labels)}:a]")
        self._run([
            self.ffmpeg, "-y", "-v", "error", *inputs,
            "-filter_complex", f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map", "[out]", "-ar", "24000", str(wav_path),
        ])
        self._run([
            self.ffmpeg, "-y", "-v", "error", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path),
        ])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_audio.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add app/audio.py tests/test_audio.py
git commit -m "Insert silence between segments during assembly"
```

---

### Task 7: Thread continuity, persona, speed, WAV names, and gaps through the pipeline

**Files:**
- Modify: `app/pipeline.py` (`process_job` lines 134-196, `regenerate_segment` lines 217-257, `assemble_project` lines 289-295, imports)
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Update the fakes and write the failing tests**

In `tests/test_pipeline.py`, change `FakeAI.adapt` and `FakeAI.synthesize` to:

```python
    def adapt(self, *_args, **_kwargs):
        return NarrationAdaptation(
            narration_text="This is a test.", beat="reveal", emotion="tense", pause_after_ms=800,
        )

    def synthesize(self, _text, _model, _voice, _style, output_path: Path, profile=None,
                   previous_style=None, persona=None, speed=1.0):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(output_path)],
            check=True,
        )
```

Change `create_source` to take a length:

```python
def create_source(path: Path, seconds: float = 1.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}", str(path)],
        check=True,
    )
```

Change `ProfilingFakeAI` to record everything synthesis receives, and give `build_project` a `seconds` parameter:

```python
class ProfilingFakeAI(FakeAI):
    """Fake that records how it was called, so profile plumbing can be asserted."""

    def __init__(self, describe_error: Exception | None = None):
        self.describe_calls = 0
        self.describe_error = describe_error
        self.synthesis_profiles: list[dict | None] = []
        self.synthesis_calls: list[dict] = []
        self.adapt_calls: list[dict] = []

    def describe_delivery(self, _audio_path, _model):
        self.describe_calls += 1
        if self.describe_error:
            raise self.describe_error
        return "Steady, moderate energy with a factual tone."

    def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration})
        return super().adapt(faithful_en, model, narrator_profile, previous_narration)

    def synthesize(self, text, model, voice, style, output_path, profile=None,
                   previous_style=None, persona=None, speed=1.0):
        self.synthesis_profiles.append(profile)
        self.synthesis_calls.append({
            "text": text, "voice": voice, "style": style, "output_path": output_path,
            "previous_style": previous_style, "persona": persona, "speed": speed,
        })
        super().synthesize(text, model, voice, style, output_path, profile, previous_style, persona, speed)


def build_project(tmp_path, ai, seconds: float = 1.0, narrator_profile: dict | None = None):
    data_dir = tmp_path / "data"
    config = Settings(data_dir=data_dir, database_path=data_dir / "app.db", openai_api_key="not-used")
    db = Database(config.database_path)
    db.initialize()
    project_id = str(uuid.uuid4())
    source = data_dir / "projects" / project_id / "original" / "source.wav"
    source.parent.mkdir(parents=True)
    create_source(source, seconds)
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,glossary_json,narrator_profile_json,status,source_path,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (project_id, "Test", "History", "{}", json.dumps(narrator_profile or {}), "uploaded", str(source), now, now),
    )
    pipeline = Pipeline(db, config)
    pipeline._ai = lambda: ai
    return pipeline, db, project_id
```

In `start_job`, add `"speed": 0.95,` to `job_config` after `"voice": "onyx",`.

Append these tests. A 65-second source splits into two segments (45 s + 20 s) because a pure tone has no silence to cut at.

```python
def test_each_segment_is_voiced_with_the_previous_segments_style(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id))

    assert len(ai.synthesis_calls) == 2
    assert ai.synthesis_calls[0]["previous_style"] is None
    assert ai.synthesis_calls[1]["previous_style"]["beat"] == "reveal"
    assert ai.synthesis_calls[1]["previous_style"]["emotion"] == "tense"


def test_adaptation_sees_the_previous_narration(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id))

    assert ai.adapt_calls[0]["previous_narration"] == ""
    assert ai.adapt_calls[1]["previous_narration"] == "This is a test."


def test_synthesis_receives_the_job_speed_and_the_project_persona(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, narrator_profile={"persona": "You are a calm lecturer."})

    pipeline.process_job(start_job(db, project_id))

    assert ai.synthesis_calls[0]["speed"] == 0.95
    assert ai.synthesis_calls[0]["persona"] == "You are a calm lecturer."


def test_generated_segments_are_wav_files(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["tts_audio_path"].endswith("0001_r01.wav")
    assert Path(segment["tts_audio_path"]).exists()


def test_regenerating_a_segment_uses_its_predecessors_style(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    pipeline.process_job(start_job(db, project_id))
    second = db.one("SELECT * FROM segments WHERE project_id=? AND segment_index=2", (project_id,))
    first = db.one("SELECT * FROM segments WHERE project_id=? AND segment_index=1", (project_id,))

    pipeline.regenerate_segment(second["id"], "tts")

    call = ai.synthesis_calls[-1]
    assert call["previous_style"] == first["style"]
    assert str(call["output_path"]).endswith("0002_r02.wav")


def test_regenerating_the_first_segment_has_no_predecessor(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id))
    first = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(first["id"], "tts")

    assert ai.synthesis_calls[-1]["previous_style"] is None


def test_assembly_leaves_the_adaptations_pause_between_segments(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)

    artifacts = pipeline.assemble_project(project_id, job_id)

    # Two 1 s fake segments plus the fake adaptation's 800 ms pause_after.
    duration = pipeline.audio.probe(Path(artifacts["wav"])).duration_ms
    assert 2650 <= duration <= 2950
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: the seven new tests FAIL. The continuity, persona, and speed tests fail because the pipeline never passes those arguments (the fake records `None` / `1.0`); the WAV tests fail on `.mp3` paths; the assembly test measures about 2000 ms.

- [ ] **Step 3: Edit `app/pipeline.py`**

Add to the imports:

```python
from .direction import segment_gap_ms
```

In `process_job`, replace the block from `ai = self._ai()` through `previous_context = adaptation.narration_text` with:

```python
            ai = self._ai()
            previous_context = ""
            previous_style: dict | None = None
            persona = (project.get("narrator_profile") or {}).get("persona")
            speed = config.get("speed", self.config.tts_speed)
            for position, segment_id in enumerate(segment_ids):
                base_progress = 10 + round(80 * position / max(len(segment_ids), 1))
                segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
                assert segment
                self._job_update(job_id, status="running", stage=f"transcribing {position + 1}/{len(segment_ids)}", progress=base_progress)
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

                tts_path = project_dir / "generated" / job_id / f"{position + 1:04d}_r01.wav"
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
                source_duration = segment["end_ms"] - segment["start_ms"]
                duration_ratio = tts_duration / source_duration if source_duration else 0
                qa_payload = qa.model_dump() | {"duration_ratio": round(duration_ratio, 3)}
                if not 0.55 <= duration_ratio <= 1.45:
                    qa_payload["passed"] = False
                    qa_payload["issues"] = qa_payload["issues"] + [f"Duration ratio {duration_ratio:.2f} is outside 0.55-1.45"]
                    qa_payload["recommended_stage_to_retry"] = "tts"
                qa_status = "passed" if qa_payload["passed"] else "needs_review"
                self.db.execute(
                    "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
                    (json.dumps(qa_payload), qa_status, qa_status, utc_now(), segment_id),
                )
                previous_context = adaptation.narration_text
                previous_style = style
```

In `regenerate_segment`, replace the block from `ai = self._ai()` through the end of the `if stage == "tts":` block with:

```python
        ai = self._ai()
        config = job["config"]
        predecessor = self.db.one(
            "SELECT narration_en, style_json FROM segments WHERE job_id=? AND segment_index=?",
            (job["id"], segment["segment_index"] - 1),
        )
        previous_narration = predecessor["narration_en"] if predecessor else ""
        previous_style = predecessor["style"] if predecessor else None
        persona = (project.get("narrator_profile") or {}).get("persona")
        speed = config.get("speed", self.config.tts_speed)
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
                tts_path = self.config.data_dir / "projects" / project["id"] / "generated" / job["id"] / f"{segment['segment_index']:04d}_r{revision:02d}.wav"
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
```

The QA block that follows (`fresh = self.db.one(...)` onward) is unchanged.

In `assemble_project`, replace the single line `self.audio.assemble([Path(s["tts_audio_path"]) for s in segments], wav_path, mp3_path)` with:

```python
        project = self.db.one("SELECT speaking_profile_json FROM projects WHERE id=?", (project_id,))
        profile = project.get("speaking_profile") if project else None
        gaps = [
            segment_gap_ms(earlier.get("style"), later.get("style"), profile)
            for earlier, later in zip(segments, segments[1:])
        ]
        self.audio.assemble([Path(s["tts_audio_path"]) for s in segments], wav_path, mp3_path, gaps)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: 12 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all passed, no failures.

- [ ] **Step 6: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "Carry narration continuity, persona, and speed through the pipeline; assemble with pauses"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the configuration table**

Change the default shown for `TTS_VOICE` from `onyx` to `cedar`, and add a row for `TTS_SPEED` with default `1.0` directly after it.

- [ ] **Step 2: Update the "What is included" list**

Replace the bullet `separate Sinhala transcription, faithful translation, and spoken-English adaptation stages` with:

```
- separate Sinhala transcription, faithful translation, and spoken-English adaptation stages; the
  adaptation writes a performance script (short sentences, held beats, spoken dates) and labels each
  passage with a story beat and a director's note
```

Replace the bullet `final WAV/MP3 plus Sinhala transcript and English script JSON exports` with:

```
- lossless WAV segments, assembled with real pauses between passages, exported as final WAV/MP3 plus
  Sinhala transcript and English script JSON
```

- [ ] **Step 3: Add a "Narration direction" section after "Speaking-pattern matching"**

```markdown
## Narration direction

Every TTS call is directed with one brief, built in `app/direction.py` from four parts:

1. **Persona.** A fixed intimate true-crime storyteller: one real person telling one listener about a
   case that actually happened. A project can replace it by putting a `persona` string in its
   `narrator_profile` when the project is created.
2. **Voice character.** The measured and described speaking profile of the original narrator.
3. **How to speak.** Rules aimed at the usual synthetic tells: talk rather than read, end statements
   low, breathe before long sentences, get quieter rather than louder for intensity, say names and
   dates carefully, report quoted speech rather than act it, never sound like an announcer.
4. **This passage and continuity.** The beat, emotion, director's note, and emphasis from the
   adaptation stage, plus the beat and emotion the previous passage ended on so the voice does not
   reset at segment boundaries. Regenerating a segment looks up its predecessor for the same reason.

Pace is driven through the brief. `TTS_SPEED` (or `speed` on the process request) is passed to the
API unchanged and defaults to `1.0`.

At assembly, the gap between two segments is the adaptation's `pause_after_ms` plus the next
`pause_before_ms`; when both are zero it falls back to the source narrator's mean pause, and it is
clamped between 300 ms and the source narrator's longest measured pause.

`scripts/listening_test.py` voices real segments from an existing job the old way and the new way and
asks the audio model to say which sounds more human and why. It spends API credit; use it when tuning
the persona or rules.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "Document narration direction, cedar default, and speed setting"
```

---

### Task 9: Listening test (paid, approved by the owner)

Voices two real segments from the Gilgo Beach project both ways, asks the audio model to critique each file, and leaves the files for the owner to hear. The critique is the evidence for tuning `DEFAULT_PERSONA` and `DELIVERY_RULES`.

**Files:**
- Create: `scripts/listening_test.py`
- Output: `data/listening_test/*.wav`, `data/listening_test/critique.md`

- [ ] **Step 1: Create `scripts/listening_test.py`**

```python
"""Voice real segments the old way and the new way, then ask the audio model which sounds human.

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/listening_test.py <job_id> <segment_index> [<segment_index> ...]

For each segment index this writes, under data/listening_test/:
    NN_old_onyx.wav          stored narration text, the pre-change one-line direction, voice onyx
    NN_new_cedar.wav         re-adapted performance script, the full brief, voice cedar, speed 1.0
    NN_new_cedar_s092.wav    same as above at speed 0.92, to judge whether the speed setting helps
and appends the audio model's critique of every file to critique.md.

It spends API credit: two text-model calls and up to three TTS calls per segment, plus one audio-model
call per file.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import AIClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.direction import build_instructions  # noqa: E402


OUT_DIR = settings.data_dir / "listening_test"
CRITIQUE_PROMPT = (
    "You are judging a true-crime documentary voice-over. Ignore the words; judge only the delivery. "
    "Rate from 1 to 5 how much this sounds like a real human storyteller rather than text-to-speech, "
    "then list the specific things that give away a synthetic voice (for example: pitch lifting at "
    "sentence ends, evenness of rhythm, no breaths, brightness, announcer tone, reset between "
    "passages). Finish with one sentence of direction that would make it more human. Be concrete."
)


def old_instructions(style: dict, profile: dict | None) -> str:
    """The pre-change direction, reproduced so the comparison is honest."""
    lines = ["Natural English documentary narration."]
    if profile:
        derived = profile.get("derived") or {}
        measured = profile.get("measured") or {}
        character = ["Narrator character, matched to the original speaker:"]
        if profile.get("described"):
            character.append(profile["described"])
        if derived:
            character.append(
                f"Baseline pace is {derived.get('pace', 'moderate')}, with "
                f"{derived.get('pause_style', 'deliberate')} pauses and "
                f"{derived.get('dynamics', 'controlled')} delivery."
            )
        if measured.get("mean_pause_ms"):
            character.append(
                f"Leave roughly {measured['mean_pause_ms']}ms between sentences, stretching to about "
                f"{measured['longest_pause_ms']}ms at the most dramatic beats."
            )
        lines.append(" ".join(character))
    moment = f"This passage: {style.get('emotion', 'neutral')} in tone, pace {style.get('pace', 'moderate')}."
    emphasis = ", ".join(style.get("emphasis") or [])
    if emphasis:
        moment += f" Emphasise: {emphasis}."
    lines.append(moment)
    return " ".join(lines)


def speak(ai: AIClient, text: str, voice: str, instructions: str, speed: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ai.client.audio.speech.with_streaming_response.create(
        model=settings.tts_model, voice=voice, input=text, instructions=instructions,
        speed=speed, response_format="wav",
    ) as response:
        response.stream_to_file(path)


def critique(ai: AIClient, path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    response = ai.client.chat.completions.create(
        model=settings.audio_model,
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CRITIQUE_PROMPT},
                {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
            ],
        }],
    )
    return (response.choices[0].message.content or "").strip()


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    job_id, indexes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    db = Database(settings.database_path)
    ai = AIClient(settings.openai_api_key)
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        sys.exit(f"No job {job_id}")
    project = db.one("SELECT * FROM projects WHERE id=?", (job["project_id"],))
    profile = project.get("speaking_profile") or None
    persona = (project.get("narrator_profile") or {}).get("persona")
    report = OUT_DIR / "critique.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for index in indexes:
        segment = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index))
        previous = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index - 1))
        if not segment:
            sys.exit(f"No segment {index} in job {job_id}")
        old_style = segment["style"] or {}

        old_path = OUT_DIR / f"{index:02d}_old_onyx.wav"
        speak(ai, segment["narration_en"], "onyx", old_instructions(old_style, profile), 1.0, old_path)
        print("wrote", old_path)

        # Re-adapt with the new prompt so the new voice reads the new performance script.
        previous_narration = previous["narration_en"] if previous else ""
        adaptation = ai.adapt(segment["faithful_en"], settings.text_model, project["narrator_profile"] or {}, previous_narration[-600:])
        new_style = adaptation.model_dump(exclude={"narration_text"})
        # The predecessor's stored style predates the change, but beat/emotion are what continuity needs.
        previous_style = (previous["style"] or {}) if previous else None
        brief = build_instructions(new_style, profile, previous_style, persona)
        (OUT_DIR / f"{index:02d}_new_script.txt").write_text(
            f"# beat: {new_style['beat']}\n# delivery: {new_style['delivery']}\n# emotion: {new_style['emotion']}\n\n"
            f"{adaptation.narration_text}\n\n---- brief ----\n{brief}\n", encoding="utf-8",
        )
        for speed, suffix in ((1.0, ""), (0.92, "_s092")):
            path = OUT_DIR / f"{index:02d}_new_cedar{suffix}.wav"
            speak(ai, adaptation.narration_text, "cedar", brief, speed, path)
            print("wrote", path)

        with report.open("a", encoding="utf-8") as out:
            for path in sorted(OUT_DIR.glob(f"{index:02d}_*.wav")):
                out.write(f"## {path.name}\n\n{critique(ai, path)}\n\n")
                print("critiqued", path.name)
    print("critique written to", report)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against the Gilgo Beach job, segments 1 and 2**

Run (the key lives in `.env`, which the app does not read automatically):

```bash
set -a; source .env; set +a
.venv/bin/python scripts/listening_test.py 2fe1200c-87b4-4def-a111-e417ee20810b 1 2
```

Expected: six `wrote` lines, six `critiqued` lines, and `critique written to data/listening_test/critique.md`. If the API rejects `speed` for the TTS model, remove the `speed=speed` argument in `speak()` and the `_s092` variant, note that in the final report, and rerun.

- [ ] **Step 3: Read `data/listening_test/critique.md` and the two `_new_script.txt` files**

Compare the critique of `old_onyx` with `new_cedar` for both segments. Look for: does the new brief remove the tells the model names for the old file? Does the model name new tells introduced by the brief (over-whispering, too slow, flatness)? Does `s092` sound stretched or more natural?

- [ ] **Step 4: Tune and re-run once if needed**

If the critique names a specific remaining tell, adjust the matching line of `DELIVERY_RULES` or `DEFAULT_PERSONA` in `app/direction.py`, run `.venv/bin/python -m pytest tests/test_direction.py -q` (expected: 15 passed), and re-run Step 2 for one segment. If `s092` is judged better than `1.0`, set `TTS_SPEED=0.92` in `.env.example` and the README table and change the `tts_speed` default in `app/config.py` to `"0.92"`. Do at most one tuning round; the owner listens and decides further changes.

- [ ] **Step 5: Commit**

```bash
git add scripts/listening_test.py app/direction.py .env.example README.md app/config.py
git commit -m "Add listening test that voices real segments old and new and critiques them"
```

`data/` is git-ignored, so the audio and the critique stay local.

---

## Final verification

- [ ] Run: `.venv/bin/python -m pytest -q` — expected: all passed.
- [ ] Run: `git status --short` — expected: clean.
- [ ] Report to the owner: the four (or six) files in `data/listening_test/`, the critique summary, any tuning made, and the reminder that existing projects need a fresh processing run to get the new script and voice.

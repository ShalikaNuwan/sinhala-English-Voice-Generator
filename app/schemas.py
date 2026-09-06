from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProjectCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    topic: str = Field(default="", max_length=1000)
    glossary: dict[str, str] = Field(default_factory=dict)
    narrator_profile: dict[str, str | int | float | list[str]] = Field(default_factory=dict)


class ProcessRequest(BaseModel):
    stt_model: str | None = None
    text_model: str | None = None
    qa_model: str | None = None
    tts_model: str | None = None
    audio_model: str | None = None
    voice: str | None = None
    # Same range as direction.SPEED_RANGE; kept literal so schemas stays import-free.
    speed: float | None = Field(default=None, ge=0.25, le=4.0)
    diarize_model: str | None = None
    detect_recordings: bool | None = None
    align_model: str | None = None
    shape_pauses: bool | None = None
    human_review_gate: bool = True


class TextUpdate(BaseModel):
    text: str = Field(min_length=1)


class RegenerateRequest(BaseModel):
    stage: Literal["translation", "adaptation", "tts", "qa"] = "tts"


class KindUpdate(BaseModel):
    kind: Literal["narration", "original"]


class FaithfulTranslation(BaseModel):
    english_faithful: str
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    uncertain_spans: list[str] = Field(default_factory=list)


class NarrationAdaptation(BaseModel):
    narration_text: str = Field(description="The script to be spoken aloud, written for the ear.")
    beat: Literal["setup", "build", "reveal", "aftermath", "reflection"] = Field(
        default="build",
        description="Where this passage sits in the story arc.",
    )
    delivery: str = Field(
        default="",
        max_length=300,
        description="One sentence of direction to the voice actor for this passage.",
    )
    pace: Literal["slow", "moderate", "fast"] = Field(
        default="moderate",
        description="How fast to speak this passage, relative to the narrator's baseline.",
    )
    emphasis: list[str] = Field(
        default_factory=list,
        description="Exact phrases from narration_text to give weight to.",
    )
    pause_before_ms: int = Field(
        default=0, ge=0, le=3000,
        description="Silence before this passage in milliseconds. 0 means use the narrator's usual gap.",
    )
    pause_after_ms: int = Field(
        default=0, ge=0, le=3000,
        description="Silence after this passage in milliseconds. 0 means use the narrator's usual gap.",
    )
    emotion: str = Field(default="neutral", description="The emotional tone of this passage, in a few words.")


class QAEvaluation(BaseModel):
    passed: bool
    severity: Literal["low", "medium", "high"] = "low"
    issues: list[str] = Field(default_factory=list)
    recommended_stage_to_retry: Literal["none", "translation", "adaptation", "tts"] = "none"


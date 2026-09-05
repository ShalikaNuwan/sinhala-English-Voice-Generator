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
    human_review_gate: bool = True


class TextUpdate(BaseModel):
    text: str = Field(min_length=1)


class RegenerateRequest(BaseModel):
    stage: Literal["translation", "adaptation", "tts", "qa"] = "tts"


class FaithfulTranslation(BaseModel):
    english_faithful: str
    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    uncertain_spans: list[str] = Field(default_factory=list)


class NarrationAdaptation(BaseModel):
    narration_text: str
    pace: Literal["slow", "moderate", "fast"] = "moderate"
    emphasis: list[str] = Field(default_factory=list)
    pause_before_ms: int = Field(default=0, ge=0, le=3000)
    pause_after_ms: int = Field(default=250, ge=0, le=3000)
    emotion: str = "neutral"


class QAEvaluation(BaseModel):
    passed: bool
    severity: Literal["low", "medium", "high"] = "low"
    issues: list[str] = Field(default_factory=list)
    recommended_stage_to_retry: Literal["none", "translation", "adaptation", "tts"] = "none"


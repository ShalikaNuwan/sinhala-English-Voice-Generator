from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from .schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation


PROMPT_VERSION = "2026-09-mvp1"


def narration_instructions(style: dict, profile: dict | None) -> str:
    """Build the TTS direction: who the narrator is, then what this moment needs.

    The profile describes the original speaker and stays constant across the project.
    The style comes from the adaptation model and changes segment to segment.
    """
    lines = ["Natural English documentary narration."]

    if profile:
        derived = profile.get("derived") or {}
        measured = profile.get("measured") or {}
        described = profile.get("described")
        character = ["Narrator character, matched to the original speaker:"]
        if described:
            character.append(described)
        if derived:
            character.append(
                f"Baseline pace is {derived.get('pace', 'moderate')}, "
                f"with {derived.get('pause_style', 'deliberate')} pauses and "
                f"{derived.get('dynamics', 'controlled')} delivery."
            )
        if measured.get("mean_pause_ms"):
            character.append(
                f"Leave roughly {measured['mean_pause_ms']}ms between sentences, "
                f"stretching to about {measured['longest_pause_ms']}ms at the most dramatic beats."
            )
        lines.append(" ".join(character))

    emphasis = ", ".join(style.get("emphasis") or [])
    moment = f"This passage: {style.get('emotion', 'neutral')} in tone, pace {style.get('pace', 'moderate')}."
    if emphasis:
        moment += f" Emphasise: {emphasis}."
    lines.append(moment)
    return " ".join(lines)


class AIClient:
    def __init__(self, api_key: str | None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required before processing audio")
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, audio_path: Path, model: str, topic: str, glossary: dict[str, str]) -> str:
        terms = ", ".join(f"{key}: {value}" for key, value in glossary.items())
        prompt = (
            f"The spoken language is Sinhala. Transcribe it in Sinhala script. "
            f"Topic: {topic}. Important names and terms: {terms}"
        ).strip()
        with audio_path.open("rb") as audio_file:
            result = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                prompt=prompt[:1000],
            )
        return result.text.strip()

    def translate(
        self,
        transcript_si: str,
        model: str,
        topic: str,
        glossary: dict[str, str],
        previous_context: str,
    ) -> FaithfulTranslation:
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a bilingual Sinhala-English translation specialist. Translate the supplied "
                        "Sinhala narration into accurate English. Preserve every factual claim, name, date, "
                        "number, place, uncertainty, and causal relationship. Do not add context and do not "
                        "optimize for dramatic style."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Topic: {topic}\nGlossary: {glossary}\nPrevious context: {previous_context}\n"
                        f"Sinhala transcript: {transcript_si}"
                    ),
                },
            ],
            text_format=FaithfulTranslation,
        )
        if not response.output_parsed:
            raise RuntimeError("Translation model did not return a structured result")
        return response.output_parsed

    def adapt(
        self,
        faithful_en: str,
        model: str,
        narrator_profile: dict,
    ) -> NarrationAdaptation:
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite faithful English into natural spoken narration. Use the narrator profile. "
                        "Do not add, remove, weaken, or strengthen factual claims. Preserve supported suspense "
                        "and emphasis, and avoid formal written-English constructions. Collapse accidental "
                        "repeated starts, duplicated phrases, and incomplete false starts, while preserving "
                        "every distinct factual detail."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Narrator profile: {narrator_profile}\nFaithful English: {faithful_en}",
                },
            ],
            text_format=NarrationAdaptation,
        )
        if not response.output_parsed:
            raise RuntimeError("Adaptation model did not return a structured result")
        return response.output_parsed

    def synthesize(
        self,
        text: str,
        model: str,
        voice: str,
        style: dict,
        output_path: Path,
        profile: dict | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        instructions = narration_instructions(style, profile)
        with self.client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=text,
            instructions=instructions,
            response_format="mp3",
        ) as response:
            response.stream_to_file(output_path)

    def describe_delivery(self, audio_path: Path, model: str) -> str:
        """Describe how the source narrator speaks, ignoring what they are saying."""
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        response = self.client.chat.completions.create(
            model=model,
            modalities=["text"],
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyse a narrator's vocal delivery. The spoken language is usually Sinhala. "
                        "Ignore what the words mean and describe only HOW the person speaks."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this narrator's delivery: energy level, emotional tone, and how "
                                "they use pauses and emphasis. Answer in three short sentences that could "
                                "direct a voice actor."
                            ),
                        },
                        {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
                    ],
                },
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def evaluate(self, faithful_en: str, narration_en: str, model: str) -> QAEvaluation:
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Compare the faithful translation with the narration script. Identify omissions, "
                        "additions, changed certainty, and incorrect entities, dates, numbers, locations, or "
                        "causal relationships. Pass only when meaning is preserved."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Faithful translation:\n{faithful_en}\n\nNarration script:\n{narration_en}",
                },
            ],
            text_format=QAEvaluation,
        )
        if not response.output_parsed:
            raise RuntimeError("QA model did not return a structured result")
        return response.output_parsed

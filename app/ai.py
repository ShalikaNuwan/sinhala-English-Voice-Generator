from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from .direction import build_instructions, speaking_speed
from .schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation


PROMPT_VERSION = "2026-09-mvp2"


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
                        "and pause_after_ms (silence in milliseconds the narrator would leave before and after this "
                        "passage; leave both at 0 unless this passage needs a longer or shorter pause than the "
                        "narrator's usual gap)."
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

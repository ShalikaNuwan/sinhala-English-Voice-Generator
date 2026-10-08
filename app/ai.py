from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from . import elevenlabs, turns
from .direction import build_instructions, speaking_speed
from .recordings import NARRATOR
from .schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation

# The teller the script is written for when a project sets no narrator profile of its own.
STORYTELLER_PROFILE = {
    "persona": (
        "A composed true-crime documentary narrator telling the story of a real case to one listener, "
        "late at night. Intimate, steady and plain-spoken; never breathless, never bookish. Lets the "
        "facts carry the dread and trusts the listener to feel it."
    ),
    "style": "spoken beats, one idea per sentence, natural English idiom, names and numbers said carefully",
    "avoid": "translation-ese, filler transitions, instructions to the audience, invented drama",
}


PROMPT_VERSION = "2026-09-mvp3"

# A Realtime transcription session takes raw 16-bit mono PCM at this rate, in small pieces.
TURN_SAMPLE_RATE = 24000
TURN_PIECE_MS = 100
# Events stop arriving long before the socket closes, so the session ends on quiet, or on the cap.
TURN_RECV_TIMEOUT_S = 5.0
TURN_QUIET_LIMIT = 3
TURN_SESSION_MAX_S = 360.0


def _field(item, name: str):
    """Read a field from either an SDK object or a plain dict."""
    return item.get(name) if isinstance(item, dict) else getattr(item, name, None)


def _run_coroutine(coroutine):
    """Run a coroutine from synchronous code, in a worker thread when this one already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


class AIClient:
    def __init__(self, api_key: str | None, elevenlabs_config: dict | None = None):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required before processing audio")
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        # Every text stage stays on OpenAI. Only voicing can be handed to another provider.
        self.elevenlabs = elevenlabs_config or None

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

    def _async_client(self) -> AsyncOpenAI:
        """A client for one Realtime session: each session runs its own loop, and a pool cannot cross loops."""
        return AsyncOpenAI(api_key=self.api_key)

    def speech_turns(
        self, pcm_path: Path, duration_ms: int, threshold: float, silence_ms: int, model: str,
    ) -> list[dict]:
        """When the speaker starts and stops, heard by a Realtime session's voice detection.

        Server VAD is the only source of millisecond speech boundaries in the API: the diarizer's
        spans move between calls on the same audio. Times are file-relative; pairing lives in `turns`.
        """
        events = _run_coroutine(self._listen(Path(pcm_path), threshold, silence_ms, model))
        return turns.pair_events(events, silence_ms, duration_ms)

    async def _listen(self, pcm_path: Path, threshold: float, silence_ms: int, model: str) -> list[dict]:
        """Stream raw PCM into one transcription session and collect its events in arrival order."""
        audio = pcm_path.read_bytes()
        events: list[dict] = []
        client = self._async_client()
        try:
            await self._stream(client, audio, events, threshold, silence_ms, model)
        except Exception:  # noqa: BLE001 - a dropped connection keeps the turns already heard
            if not any(event["type"] == "start" for event in events):
                raise
        finally:
            await client.close()
        return events

    async def _stream(self, client: AsyncOpenAI, audio: bytes, events: list[dict], threshold: float, silence_ms: int, model: str) -> None:
        """Send the audio and collect every event until the session goes quiet; an error event is noted, not fatal."""
        failure: str | None = None
        # No model in the URL: the session carries it, and a model in the query is rejected.
        async with client.realtime.connect(extra_query={"intent": "transcription"}) as connection:
            await connection.session.update(session={
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": TURN_SAMPLE_RATE},
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": threshold,
                            "prefix_padding_ms": 0,
                            "silence_duration_ms": silence_ms,
                        },
                        "transcription": {"model": model},
                    },
                },
            })
            piece = TURN_SAMPLE_RATE * 2 * TURN_PIECE_MS // 1000
            for offset in range(0, len(audio), piece):
                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(audio[offset:offset + piece]).decode("ascii"),
                )
            await connection.input_audio_buffer.commit()
            quiet = 0
            deadline = time.monotonic() + TURN_SESSION_MAX_S
            while quiet < TURN_QUIET_LIMIT and time.monotonic() < deadline:
                try:
                    event = await asyncio.wait_for(connection.recv(), timeout=TURN_RECV_TIMEOUT_S)
                except asyncio.TimeoutError:
                    quiet += 1
                    continue
                kind = _field(event, "type")
                if kind == "input_audio_buffer.speech_started":
                    events.append({"type": "start", "ms": event.audio_start_ms, "item_id": event.item_id, "text": None})
                elif kind == "input_audio_buffer.speech_stopped":
                    events.append({"type": "stop", "ms": event.audio_end_ms, "item_id": event.item_id, "text": None})
                elif kind == "conversation.item.input_audio_transcription.completed":
                    events.append({"type": "text", "ms": None, "item_id": event.item_id, "text": event.transcript or ""})
                elif kind == "error":
                    # Transcripts arrive last and out of order, so keep listening; the commit after
                    # server VAD has already taken the buffer reports an empty commit every time.
                    failure = _field(_field(event, "error"), "message") or "Realtime session failed"
        if failure and not any(event["type"] == "start" for event in events):
            raise RuntimeError(failure)

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
                        "optimize for dramatic style. Names of real people and places in the narration are "
                        "English names written phonetically in Sinhala: recover the standard English spelling "
                        "when the case, the glossary or the previous context identifies the person or place "
                        "(for example a Sinhala rendering of Byers), and list any name you are not sure of in "
                        "uncertain_spans rather than inventing a spelling."
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
        runs: list[str] | None = None,
    ) -> NarrationAdaptation:
        context = ""
        if previous_narration:
            context = (
                "Previous narration, for context only. Use it to know where you are in the story; "
                f"do not repeat it and do not rewrite it:\n<<<\n{previous_narration}\n>>>\n"
            )
        profile = narrator_profile or STORYTELLER_PROFILE
        user_content = (
            f"Narrator profile: {profile}\n"
            f"{context}"
            f"Faithful English to adapt:\n<<<\n{faithful_en}\n>>>"
        )
        if runs:
            numbered = "\n".join(f"{number}. {text}" for number, text in enumerate(runs, start=1))
            user_content += f"\nSpoken runs, numbered, in order:\n{numbered}"
        system = (
            "You turn faithful English into the script a storyteller speaks aloud as the voice-over of a "
            "true-crime documentary. The faithful text gives you the facts; you give them a teller. "
            "Retell, do not transliterate: say what the Sinhala narrator meant the way a fluent English "
            "storyteller would say it to one listener, with the sentence shapes, idioms and word order of "
            "spoken English, never the Sinhala sentence built word by word in English. Write for the ear, "
            "not the page:\n"
            "- Short sentences, one idea each, in spoken beats. Use contractions. Avoid formal written-English "
            "constructions, and avoid phrases that only exist in translation (say 'the cash register', not "
            "'the cashier counter'; 'she was seventeen', not 'her age was seventeen').\n"
            "- When the narrator speaks to the audience, render the intent in natural English, not the words: "
            "an invitation to settle down and listen becomes something like 'Settle in, and listen closely', "
            "never an instruction such as 'Start listening to the story'.\n"
            "- Keep reported speech reported; do not turn 'he said that he was late' into a direct quote.\n"
            "- No filler transitions the narrator did not build ('Anyway', 'So,' 'Well,'), no summarising, "
            "no editorialising, no added drama. The tension comes from the facts and their order.\n"
            "- Say it the way the story is told in English true-crime: name the person rather than 'this "
            "boy', keep the running names the narrator uses ('Rob'), and keep each fact on the beat where "
            "the narrator placed it.\n"
            "- Put an ellipsis (…) where the narrator should hold a beat, at most once per passage, and a dash "
            "(—) for a change of thought or an afterthought, at most twice. Use the characters … and — "
            "themselves, not ... or --, and only where a person telling the story would actually pause.\n"
            "- Start a new paragraph (blank line) only at a real shift in the story, at most twice per passage.\n"
            "- Write dates, times, and numbers the way they are said aloud, for example "
            "'May 1st, 2010' and '4:51 in the morning'.\n"
            "- Keep quoted speech from 911 calls and witnesses as plain spoken lines.\n"
            "- Collapse incomplete false starts and repetitions the speaker clearly did not intend.\n"
            "- A sentence the segment boundary cut off is not a cliffhanger: never end it with an ellipsis. "
            "Fold its words into the sentence before it when they fit; otherwise write only the words that "
            "are there, as the plain opening of a sentence, and let the next passage finish the thought.\n"
            "Hard limit, which overrides every rule above. Do not add, remove, weaken, or strengthen any "
            "factual claim. Preserve every distinct name, date, number, place, and causal link. If a rule "
            "above would cost you a detail or soften a hedge, break the rule and keep the fact. Preserve "
            "supported suspense and emphasis. Use the narrator profile.\n"
            "Always: narration_text is only the words to be spoken. No stage directions, no bracketed or "
            "parenthetical cues, no markdown, no headings, no speaker labels, no sound-effect notes. "
            "Every note about performance goes in the delivery field instead.\n"
            "Also return: beat (where this passage sits in the story: setup, build, reveal, aftermath, or "
            "reflection); delivery (one sentence of direction to the voice actor for this passage, under "
            "300 characters); emotion (the tone, in two or three words); pace; emphasis (two or three short "
            "phrases copied from narration_text character for character, never a paraphrase, never a phrase "
            "that is not in the script); and pause_before_ms and pause_after_ms (silence in milliseconds; "
            "your number is added to the neighbouring passage's request, and anything under 300 is treated "
            "as 300 and nothing exceeds the narrator's longest measured pause, so leave both at 0 unless this "
            "passage really needs a longer pause than the narrator's usual gap)."
        )
        if runs:
            system += (
                "\nThe Sinhala was spoken as the numbered runs in the user message. Return lines: one line per run, "
                "in the same order, each rendering exactly that run's content, with runs listing the number it "
                "covers. Merge two neighbouring runs into one line only when English grammar leaves no other "
                "choice, and list both numbers. Every run must be covered exactly once. The run texts are rough "
                "machine transcripts of the Sinhala, often garbled or loosely translated; they show only where each "
                "run starts and ends. Every fact and sentence comes from the faithful translation: never add "
                "anything that appears only in a run text, and never drop part of the faithful translation. The "
                "paragraph-break limit above does not apply in this mode: narration_text must be the lines joined "
                "by blank lines, one line per paragraph."
            )
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": system,
                },
                {
                    "role": "user",
                    "content": user_content,
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
        previous_text: str | None = None,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.elevenlabs:
            # ElevenLabs has no field for the brief. Delivery comes from the voice settings, and the
            # only continuity lever is telling it what was said immediately before.
            elevenlabs.synthesize(
                text, output_path,
                api_key=self.elevenlabs.get("api_key"),
                voice=self.elevenlabs.get("voice"),
                model=self.elevenlabs.get("model"),
                stability=self.elevenlabs.get("stability"),
                similarity=self.elevenlabs.get("similarity"),
                previous_text=previous_text,
            )
            return
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
                        "Compare the faithful translation with the narration script. Identify omissions, additions, "
                        "changed certainty, and incorrect entities, dates, numbers, locations, or causal relationships. "
                        "Dates, times, and numbers written the way they are spoken ('4:51 in the morning' for "
                        "'4:51 a.m.', 'May 1st, 2010' for '2010-05-01') are correct, not changes. Contractions, "
                        "reordering within a sentence, and an ellipsis or dash used for pacing are style, not meaning. Pass only when "
                        "meaning is preserved."
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

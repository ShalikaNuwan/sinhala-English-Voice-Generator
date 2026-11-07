from __future__ import annotations

import asyncio
import base64
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

from . import elevenlabs, fact_check, turns
from .direction import build_instructions, speaking_speed
from .recordings import NARRATOR
from .schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation
from .sentences import SplitDecision


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


def _break_marks(a_narration: str, b_narration: str) -> str:
    """A plain note when the narration shows its own break mark (… or ...) at the cut, so the model
    knows that sentence is broken and has to be mended."""
    notes = []
    if a_narration.rstrip().endswith(("…", "...")):
        notes.append("The narration of A ends with a break mark: its last sentence is broken at the cut and must be mended.")
    if b_narration.lstrip().startswith(("…", "...")):
        notes.append("The narration of B starts with a break mark: its first sentence is broken at the cut and must be mended.")
    return "".join(f"{note}\n" for note in notes)


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
        runs: list[str] | None = None,
    ) -> NarrationAdaptation:
        context = ""
        if previous_narration:
            context = (
                "Previous narration, for context only. Use it to know where you are in the story; "
                f"do not repeat it and do not rewrite it:\n<<<\n{previous_narration}\n>>>\n"
            )
        user_content = (
            f"Narrator profile: {narrator_profile}\n"
            f"{context}"
            f"Faithful English to adapt:\n<<<\n{faithful_en}\n>>>"
        )
        if runs:
            numbered = "\n".join(f"{number}. {text}" for number, text in enumerate(runs, start=1))
            user_content += f"\nSpoken runs, numbered, in order:\n{numbered}"
        system = (
            "You turn faithful English into a script that a storyteller will speak aloud as the "
            "voice-over of a true-crime documentary. Write for the ear, not the page:\n"
            "- Short sentences, one idea each. Use contractions. Avoid formal written-English "
            "constructions.\n"
            "- Put an ellipsis (…) where the narrator should hold a beat, at most once per passage, and a dash "
            "(—) for a change of thought or an afterthought, at most twice. Use the characters … and — "
            "themselves, not ... or --, and only where a person telling the story would actually pause.\n"
            "- Start a new paragraph (blank line) only at a real shift in the story, at most twice per passage.\n"
            "- Write dates, times, and numbers the way they are said aloud, for example "
            "'May 1st, 2010' and '4:51 in the morning'.\n"
            "- Keep quoted speech from 911 calls and witnesses as plain spoken lines.\n"
            "- Collapse incomplete false starts and repetitions the speaker clearly did not intend.\n"
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

    def check_facts(self, text: str, context: str, topic: str, model: str) -> tuple[fact_check.FactReport, set[str]]:
        """Check a passage's facts with web search: the report and the links the search returned."""
        return fact_check.ask_model(text, context, topic, model, self.client)

    def split_sentence(self, a_narration: str, a_faithful: str, b_narration: str, b_faithful: str, model: str) -> SplitDecision:
        """Whether a sentence crosses the cut between segments A and B, decided for each text on its own.

        The answer names exact words; `sentences` checks that they are an exact edge of the text before
        moving them, and that a joined sentence uses only the words it replaces.
        """
        response = self.client.responses.parse(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": (
                        "A narrated story was cut into audio segments at silences, and each segment was translated "
                        "on its own. Sometimes a cut falls inside a sentence: the sentence starts at the end of "
                        "segment A and finishes at the start of segment B. You see the end of A and the start of B "
                        "twice: as the narration script and as the faithful English translation. Decide each text on "
                        "its own; they are often cut differently.\n"
                        "- When the text on both sides already reads as whole sentences, that text is no split: "
                        "leave its direction and fragment empty. The narration is often already whole when the "
                        "faithful English is split, because the script writer carried the missing words across.\n"
                        "- Never move a fragment whose words, or their meaning, already appear on the other side of "
                        "the cut; that would say them twice.\n"
                        "- When a sentence does cross the cut cleanly, move its smaller half. direction back: the "
                        "fragment is the very start of B and joins the end of A. direction forward: the fragment is the "
                        "very end of A and joins the start of B. Copy that fragment exactly, character for character.\n"
                        "- Often the cut left A ending in a broken sentence (frequently marked with \u2026 or ...) and "
                        "B starts by finishing it or saying it again in other words, e.g. A ends \"At that time, Ted had "
                        "never\u2026\" and B starts \"He'd never been caught by police for anything.\" Then use direction "
                        "merge: a_tail = the broken sentence at the very end of A, b_head = the words at the very start "
                        "of B that finish or repeat it (both copied exactly, character for character), fragment = one "
                        "clean sentence that says it once, and place = a or b, the segment that already holds more of "
                        "it (a when even). Write the joined sentence only from the words of a_tail and b_head: you may "
                        "drop repeated words, the \u2026 marks and fix the grammar where they meet, but never add a "
                        "name, number, date, place, or any word or fact that is not in them.\n"
                        "- Copy every quoted piece from the text it comes from: the narration pieces from the "
                        "narration, the faithful pieces from the faithful English. Never skip words in between.\n"
                        "- reason: one short sentence."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{_break_marks(a_narration, b_narration)}"
                        f"End of segment A, narration:\n<<<\n{a_narration}\n>>>\n"
                        f"End of segment A, faithful English:\n<<<\n{a_faithful}\n>>>\n"
                        f"Start of segment B, narration:\n<<<\n{b_narration}\n>>>\n"
                        f"Start of segment B, faithful English:\n<<<\n{b_faithful}\n>>>"
                    ),
                },
            ],
            text_format=SplitDecision,
        )
        if not response.output_parsed:
            raise RuntimeError("Sentence fix model did not return a structured result")
        return response.output_parsed

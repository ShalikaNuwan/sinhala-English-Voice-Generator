from __future__ import annotations

from types import SimpleNamespace
from typing import get_args

from app.ai import AIClient
from app.schemas import NarrationAdaptation


def test_transcription_prompts_for_sinhala_without_unsupported_language_code(tmp_path):
    captured = {}

    class Transcriptions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(text=" පරීක්ෂණය ")

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(audio=SimpleNamespace(transcriptions=Transcriptions()))
    audio_file = tmp_path / "segment.wav"
    audio_file.write_bytes(b"fake")

    result = ai.transcribe(audio_file, "gpt-4o-transcribe", "History", {"Kevin": "Kevin"})

    assert result == "පරීක්ෂණය"
    assert "language" not in captured
    assert "spoken language is Sinhala" in captured["prompt"]
    assert "Kevin" in captured["prompt"]


def test_delivery_description_sends_the_audio_and_ignores_word_meaning(tmp_path):
    from app.ai import AIClient

    captured = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="  Steady and factual.  "))]
            )

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    sample = tmp_path / "sample.wav"
    sample.write_bytes(b"RIFFfake-audio-bytes")

    result = ai.describe_delivery(sample, "gpt-audio")

    assert result == "Steady and factual."
    assert captured["model"] == "gpt-audio"
    system = captured["messages"][0]["content"]
    assert "Sinhala" in system
    audio_part = [p for p in captured["messages"][1]["content"] if p["type"] == "input_audio"][0]
    assert audio_part["input_audio"]["format"] == "wav"
    # The audio must actually be attached, base64 encoded.
    import base64
    assert base64.b64decode(audio_part["input_audio"]["data"]) == b"RIFFfake-audio-bytes"


def test_adaptation_has_documented_defaults_and_a_stable_style_contract():
    adaptation = NarrationAdaptation(narration_text="She never came home.")

    assert adaptation.beat == "build"
    assert adaptation.delivery == ""
    # 0 means: no specific request, use the narrator's usual gap at assembly.
    assert adaptation.pause_before_ms == 0
    assert adaptation.pause_after_ms == 0
    # app/direction.py reads these keys from the stored style; the set must not drift.
    style = adaptation.model_dump(exclude={"narration_text"})
    assert set(style) == {"beat", "delivery", "pace", "emphasis", "pause_before_ms", "pause_after_ms", "emotion"}
    # The adaptation prompt names these beats by hand; keep prompt and schema in sync.
    assert set(get_args(NarrationAdaptation.model_fields["beat"].annotation)) == {
        "setup", "build", "reveal", "aftermath", "reflection",
    }


def test_adaptation_asks_for_a_spoken_performance_script_with_previous_context():
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
    # Every structured field the schema demands must be explained to the model.
    for name in NarrationAdaptation.model_fields:
        assert name in system, name
    assert "No stage directions" in system
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
    ai.elevenlabs = None
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


def test_adaptation_works_with_three_positional_arguments_and_omits_empty_context():
    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=NarrationAdaptation(narration_text="ok"))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.adapt("She was found dead.", "text-model", {})

    user = captured["input"][1]["content"]
    assert "Previous narration" not in user
    assert "She was found dead." in user


def test_synthesis_clamps_speed_and_opens_with_the_persona(tmp_path):
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
    ai.elevenlabs = None

    ai.synthesize("Hello.", "tts-model", "cedar", {}, tmp_path / "x.wav", speed=9.0)

    assert captured["speed"] == 4.0
    assert captured["instructions"].startswith(direction.DEFAULT_PERSONA[:40])


def test_qa_accepts_spoken_renderings_of_dates_and_numbers():
    from app.schemas import QAEvaluation

    captured = {}

    class Responses:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_parsed=QAEvaluation(passed=True))

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(responses=Responses())

    ai.evaluate("At 4:51 a.m.", "At 4:51 in the morning.", "qa-model")

    system = captured["input"][0]["content"]
    assert "4:51 in the morning" in system
    assert "style, not meaning" in system


def test_the_legacy_instruction_builder_is_gone():
    import app.ai

    assert not hasattr(app.ai, "narration_instructions")


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


def test_synthesis_routes_to_elevenlabs_when_that_provider_is_configured(tmp_path, monkeypatch):
    """The OpenAI speech endpoint must not be touched at all when ElevenLabs is the provider."""
    from app import elevenlabs

    captured = {}

    def fake_synthesize(text, output_path, **kwargs):
        captured["text"] = text
        captured["path"] = output_path
        captured.update(kwargs)

    monkeypatch.setattr(elevenlabs, "synthesize", fake_synthesize)

    def explode(**_kwargs):
        raise AssertionError("the OpenAI speech endpoint was called")

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(with_streaming_response=SimpleNamespace(create=explode)))
    )
    ai.elevenlabs = {"api_key": "el-key", "voice": "uju3wxzG5OhpWcoi3SMy",
                     "model": "eleven_multilingual_v2", "stability": 0.5, "similarity": 0.75}

    output = tmp_path / "0003_r01.wav"
    ai.synthesize("She says someone is coming after her.", "gpt-4o-mini-tts", "cedar", {}, output,
                  previous_text="The call comes in at 4:51.")

    assert captured["text"] == "She says someone is coming after her."
    assert captured["path"] == output
    assert captured["voice"] == "uju3wxzG5OhpWcoi3SMy"
    assert captured["model"] == "eleven_multilingual_v2"
    assert captured["stability"] == 0.5
    assert captured["api_key"] == "el-key"
    assert captured["previous_text"] == "The call comes in at 4:51."


def test_synthesis_still_uses_openai_when_no_other_provider_is_configured(tmp_path):
    captured = {}

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def stream_to_file(self, path): captured["path"] = path

    class Speech:
        def create(self, **kwargs):
            captured.update(kwargs)
            return Response()

    ai = AIClient.__new__(AIClient)
    ai.client = SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(with_streaming_response=Speech()))
    )
    ai.elevenlabs = None

    ai.synthesize("Testing.", "gpt-4o-mini-tts", "cedar", {}, tmp_path / "a.wav", previous_text="ignored here")

    assert captured["voice"] == "cedar"
    assert captured["model"] == "gpt-4o-mini-tts"
    assert "instructions" in captured  # the full brief still goes to OpenAI

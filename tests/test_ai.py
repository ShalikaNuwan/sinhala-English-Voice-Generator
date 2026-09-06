from __future__ import annotations

from types import SimpleNamespace

from app.ai import AIClient


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


def test_adaptation_defaults_beat_and_delivery_when_the_model_omits_them():
    from app.schemas import NarrationAdaptation

    adaptation = NarrationAdaptation(narration_text="She never came home.")

    assert adaptation.beat == "build"
    assert adaptation.delivery == ""
    assert adaptation.pause_after_ms == 0  # 0 means: use the narrator's usual gap at assembly
    style = adaptation.model_dump(exclude={"narration_text"})
    assert set(style) == {"beat", "delivery", "pace", "emphasis", "pause_before_ms", "pause_after_ms", "emotion"}

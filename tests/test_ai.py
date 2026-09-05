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


PROFILE = {
    "measured": {"mean_pause_ms": 1321, "longest_pause_ms": 3547},
    "derived": {"pace": "moderate", "pause_style": "deliberate", "dynamics": "moderately dynamic"},
    "described": "Steady, moderate energy. Neutral and factual tone with little fluctuation.",
}
STYLE = {"pace": "moderate", "emotion": "calm, suspenseful", "emphasis": ["dead inside her house"]}


def test_instructions_carry_the_narrator_profile_and_the_segment_moment():
    from app.ai import narration_instructions

    text = narration_instructions(STYLE, PROFILE)

    # The narrator's character, taken from the source recording.
    assert "Steady, moderate energy" in text
    assert "deliberate" in text
    assert "moderately dynamic" in text
    assert "1321" in text and "3547" in text
    # What this particular moment needs.
    assert "calm, suspenseful" in text
    assert "dead inside her house" in text


def test_instructions_fall_back_to_segment_style_when_no_profile_exists():
    from app.ai import narration_instructions

    text = narration_instructions(STYLE, None)

    assert "calm, suspenseful" in text
    assert "dead inside her house" in text
    assert "Narrator character" not in text


def test_instructions_use_measurements_when_the_tone_analysis_failed():
    from app.ai import narration_instructions

    profile = {**PROFILE, "described": None}

    text = narration_instructions(STYLE, profile)

    assert "deliberate" in text
    assert "1321" in text
    assert "None" not in text


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

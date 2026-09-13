from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from app import elevenlabs


def tone_pcm(samples: int = 24_000) -> bytes:
    """Raw signed 16-bit little-endian mono, which is what pcm_24000 returns."""
    return b"".join(struct.pack("<h", (index % 200) * 100 - 10_000) for index in range(samples))


class Recorder:
    """Stands in for the network so the request can be inspected."""

    def __init__(self, payload: bytes | None = None, error: Exception | None = None):
        self.payload = payload if payload is not None else tone_pcm()
        self.error = error
        self.url: str | None = None
        self.headers: dict = {}
        self.body: dict = {}

    def __call__(self, url: str, headers: dict, body: bytes) -> bytes:
        self.url, self.headers, self.body = url, headers, json.loads(body)
        if self.error:
            raise self.error
        return self.payload


def call(tmp_path, transport, **overrides):
    options = dict(
        api_key="test-key", voice="uju3wxzG5OhpWcoi3SMy", model="eleven_multilingual_v2",
        stability=0.5, similarity=0.75,
    )
    options.update(overrides)
    destination = tmp_path / "out" / "segment.wav"
    elevenlabs.synthesize("She says someone is coming after her.", destination, transport=transport, **options)
    return destination


def test_request_targets_the_chosen_voice_and_asks_for_pipeline_audio(tmp_path):
    """24 kHz PCM is what the rest of the pipeline works in, so no transcode is needed."""
    recorder = Recorder()

    call(tmp_path, recorder)

    assert "uju3wxzG5OhpWcoi3SMy" in recorder.url
    assert "output_format=pcm_24000" in recorder.url
    assert recorder.headers["xi-api-key"] == "test-key"
    assert recorder.body["model_id"] == "eleven_multilingual_v2"
    assert recorder.body["voice_settings"]["stability"] == 0.5
    assert recorder.body["voice_settings"]["similarity_boost"] == 0.75


def test_the_returned_pcm_becomes_a_24k_mono_wav(tmp_path):
    recorder = Recorder(payload=tone_pcm(12_000))

    destination = call(tmp_path, recorder)

    with wave.open(str(destination)) as handle:
        assert handle.getframerate() == 24_000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 12_000


def test_the_previous_line_is_sent_so_the_voice_does_not_reset(tmp_path):
    """ElevenLabs has no instructions field; previous_text is the one continuity lever it does have."""
    recorder = Recorder()

    call(tmp_path, recorder, previous_text="The call comes in at 4:51 in the morning.")

    assert recorder.body["previous_text"] == "The call comes in at 4:51 in the morning."


def test_no_previous_line_means_the_field_is_omitted(tmp_path):
    recorder = Recorder()

    call(tmp_path, recorder)

    assert "previous_text" not in recorder.body


def test_a_missing_key_is_refused_before_any_request(tmp_path):
    recorder = Recorder()

    with pytest.raises(elevenlabs.ElevenLabsError):
        call(tmp_path, recorder, api_key="")

    assert recorder.url is None


def test_an_empty_response_is_an_error_rather_than_a_silent_file(tmp_path):
    recorder = Recorder(payload=b"")

    with pytest.raises(elevenlabs.ElevenLabsError):
        call(tmp_path, recorder)


def test_a_service_failure_surfaces_its_message(tmp_path):
    recorder = Recorder(error=RuntimeError("paid_plan_required: upgrade your subscription"))

    with pytest.raises(elevenlabs.ElevenLabsError) as caught:
        call(tmp_path, recorder)

    assert "upgrade your subscription" in str(caught.value)

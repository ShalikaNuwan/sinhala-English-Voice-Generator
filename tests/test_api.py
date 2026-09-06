from __future__ import annotations

import importlib
import io
import math
import struct
import wave


def make_wav(seconds: float = 0.2) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        frames = bytearray()
        for index in range(round(seconds * 16_000)):
            sample = round(1500 * math.sin(2 * math.pi * 440 * index / 16_000))
            frames.extend(struct.pack("<h", sample))
        audio.writeframes(frames)
    return output.getvalue()


def test_project_upload_and_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import app.config
    import app.main

    importlib.reload(app.config)
    module = importlib.reload(app.main)

    from fastapi.testclient import TestClient

    client = TestClient(module.app)
    created = client.post("/api/projects", json={"title": "Test story", "topic": "History"})
    assert created.status_code == 201
    project_id = created.json()["id"]

    uploaded = client.post(
        f"/api/projects/{project_id}/upload",
        files={"file": ("sample.wav", make_wav(), "audio/wav")},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["duration_ms"] > 0

    project = client.get(f"/api/projects/{project_id}").json()
    assert project["status"] == "uploaded"
    assert project["source_filename"] == "sample.wav"

    blocked = client.post(f"/api/projects/{project_id}/process", json={})
    assert blocked.status_code == 503



def test_job_configuration_defaults_to_cedar_at_normal_speed():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["voice"] == "cedar"
    assert config["speed"] == 1.0
    assert config["audio_model"] == "gpt-audio"
    assert config["human_review_gate"] is True


def test_job_configuration_lets_a_request_override_the_voice_and_audio_model():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(
        ProcessRequest(voice="ballad", audio_model="gpt-audio-mini", speed=0.9),
        Settings(openai_api_key="unused"),
    )

    assert config["voice"] == "ballad"
    assert config["audio_model"] == "gpt-audio-mini"
    assert config["speed"] == 0.9


def test_process_request_rejects_a_speed_outside_the_tts_range():
    import pytest
    from pydantic import ValidationError

    from app.schemas import ProcessRequest

    with pytest.raises(ValidationError):
        ProcessRequest(speed=5.0)
    with pytest.raises(ValidationError):
        ProcessRequest(speed=0.1)
    assert ProcessRequest(speed=0.25).speed == 0.25
    assert ProcessRequest(speed=4.0).speed == 4.0

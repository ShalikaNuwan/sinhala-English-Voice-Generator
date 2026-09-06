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


def test_job_configuration_enables_recording_detection_by_default():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["diarize_model"] == "gpt-4o-transcribe-diarize"
    assert config["detect_recordings"] is True


def test_job_configuration_lets_a_request_turn_recording_detection_off():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(
        ProcessRequest(detect_recordings=False, diarize_model="diarize-next"),
        Settings(openai_api_key="unused"),
    )

    assert config["detect_recordings"] is False
    assert config["diarize_model"] == "diarize-next"


def seed_job_with_segments(module, kinds):
    """Insert a project, a job, and one segment per kind directly into the app's database."""
    import json
    import uuid

    from app.database import utc_now

    project_id, job_id = str(uuid.uuid4()), str(uuid.uuid4())
    now = utc_now()
    module.db.execute(
        "INSERT INTO projects(id,title,topic,created_at,updated_at) VALUES(?,?,?,?,?)",
        (project_id, "Seeded", "History", now, now),
    )
    module.db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,warnings_json,created_at) VALUES(?,?,?,?,?,?,?,?)",
        (job_id, project_id, "awaiting_review", "review", 95, "{}", json.dumps(["Recording detection failed on chunk 2: timeout"]), now),
    )
    ids = []
    for index, kind in enumerate(kinds, start=1):
        segment_id = str(uuid.uuid4())
        ids.append(segment_id)
        module.db.execute(
            "INSERT INTO segments(id,job_id,project_id,segment_index,start_ms,end_ms,source_audio_path,tts_audio_path,kind,qa_json,qa_status,status,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (segment_id, job_id, project_id, index, 0, 1000, "x.wav", "x_tts.wav", kind,
             json.dumps({"passed": False, "issues": ["Recorded audio kept as is. Confirm."]}) if kind == "original" else "{}",
             "needs_review" if kind == "original" else "passed", "kept" if kind == "original" else "passed", now),
        )
    return job_id, ids


def test_kind_and_confirm_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    import app.config
    import app.main

    importlib.reload(app.config)
    module = importlib.reload(app.main)
    from fastapi.testclient import TestClient

    client = TestClient(module.app)
    job_id, (narration_id, original_id) = seed_job_with_segments(module, ["narration", "original"])
    flips = []
    monkeypatch.setattr(module.pipeline, "set_segment_kind", lambda segment_id, kind: flips.append((segment_id, kind)))

    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["warnings"] == ["Recording detection failed on chunk 2: timeout"]
    segments = client.get(f"/api/jobs/{job_id}/segments").json()
    assert [s["kind"] for s in segments] == ["narration", "original"]

    flipped = client.patch(f"/api/segments/{narration_id}/kind", json={"kind": "original"})
    assert flipped.status_code == 202
    assert flips == [(narration_id, "original")]
    assert client.get(f"/api/jobs/{job_id}/segments").json()[0]["status"] == "regenerating"

    assert client.patch(f"/api/segments/{original_id}/kind", json={"kind": "bogus"}).status_code == 422

    confirmed = client.post(f"/api/segments/{original_id}/confirm")
    assert confirmed.status_code == 200
    assert confirmed.json()["qa_status"] == "passed"
    assert "Recorded audio kept as is. Confirm." not in confirmed.json()["qa"]["issues"]

    assert client.post(f"/api/segments/{narration_id}/confirm").status_code == 409


def test_job_configuration_shapes_pauses_by_default():
    from app.config import Settings
    from app.main import job_configuration
    from app.schemas import ProcessRequest

    config = job_configuration(ProcessRequest(), Settings(openai_api_key="unused"))

    assert config["align_model"] == "whisper-1"
    assert config["shape_pauses"] is True
    assert job_configuration(ProcessRequest(shape_pauses=False, align_model="w2"), Settings(openai_api_key="unused"))["shape_pauses"] is False

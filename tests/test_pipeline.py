from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

from app.config import Settings
from app.database import Database, utc_now
from app.pipeline import Pipeline
from app.schemas import FaithfulTranslation, NarrationAdaptation, QAEvaluation


class FakeAI:
    def transcribe(self, *_args, **_kwargs):
        return "මෙය පරීක්ෂණයකි."

    def translate(self, *_args, **_kwargs):
        return FaithfulTranslation(english_faithful="This is a test.")

    def adapt(self, *_args, **_kwargs):
        return NarrationAdaptation(narration_text="This is a test.")

    def describe_delivery(self, _audio_path, _model):
        return "Steady, moderate energy with a factual tone."

    def synthesize(self, _text, _model, _voice, _style, output_path: Path, _profile=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(output_path)],
            check=True,
        )

    def evaluate(self, *_args, **_kwargs):
        return QAEvaluation(passed=True)


def create_source(path: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=330:duration=1", str(path)],
        check=True,
    )


def test_full_pipeline_with_fake_ai(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    config = Settings(data_dir=data_dir, database_path=data_dir / "app.db", openai_api_key="not-used")
    db = Database(config.database_path)
    db.initialize()
    project_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())
    source = data_dir / "projects" / project_id / "original" / "source.wav"
    source.parent.mkdir(parents=True)
    create_source(source)
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,glossary_json,narrator_profile_json,status,source_path,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (project_id, "Test", "History", "{}", "{}", "uploaded", str(source), now, now),
    )
    job_config = {
        "stt_model": "fake-stt",
        "text_model": "fake-text",
        "qa_model": "fake-qa",
        "tts_model": "fake-tts",
        "voice": "fake",
        "human_review_gate": True,
    }
    db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,created_at) VALUES(?,?,?,?,?,?,?)",
        (job_id, project_id, "queued", "queued", 0, json.dumps(job_config), now),
    )

    pipeline = Pipeline(db, config)
    monkeypatch.setattr(pipeline, "_ai", lambda: FakeAI())
    pipeline.process_job(job_id)

    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    segment = db.one("SELECT * FROM segments WHERE job_id=?", (job_id,))
    assert job["status"] == "awaiting_review"
    assert segment["qa_status"] == "passed"
    assert segment["transcript_si"] == "මෙය පරීක්ෂණයකි."

    artifacts = pipeline.assemble_project(project_id, job_id)
    assert Path(artifacts["wav"]).exists()
    assert Path(artifacts["mp3"]).exists()
    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "completed"


class ProfilingFakeAI(FakeAI):
    """Fake that records how it was called, so profile plumbing can be asserted."""

    def __init__(self, describe_error: Exception | None = None):
        self.describe_calls = 0
        self.describe_error = describe_error
        self.synthesis_profiles: list[dict | None] = []

    def describe_delivery(self, _audio_path, _model):
        self.describe_calls += 1
        if self.describe_error:
            raise self.describe_error
        return "Steady, moderate energy with a factual tone."

    def synthesize(self, text, model, voice, style, output_path, profile=None):
        self.synthesis_profiles.append(profile)
        super().synthesize(text, model, voice, style, output_path, profile)


def build_project(tmp_path, ai):
    data_dir = tmp_path / "data"
    config = Settings(data_dir=data_dir, database_path=data_dir / "app.db", openai_api_key="not-used")
    db = Database(config.database_path)
    db.initialize()
    project_id = str(uuid.uuid4())
    source = data_dir / "projects" / project_id / "original" / "source.wav"
    source.parent.mkdir(parents=True)
    create_source(source)
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,glossary_json,narrator_profile_json,status,source_path,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (project_id, "Test", "History", "{}", "{}", "uploaded", str(source), now, now),
    )
    pipeline = Pipeline(db, config)
    pipeline._ai = lambda: ai
    return pipeline, db, project_id


def start_job(db, project_id):
    job_id = str(uuid.uuid4())
    job_config = {
        "stt_model": "fake-stt", "text_model": "fake-text", "qa_model": "fake-qa",
        "tts_model": "fake-tts", "audio_model": "fake-audio", "voice": "onyx",
        "human_review_gate": True,
    }
    db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,created_at) VALUES(?,?,?,?,?,?,?)",
        (job_id, project_id, "queued", "queued", 0, json.dumps(job_config), utc_now()),
    )
    return job_id


def test_process_job_stores_a_speaking_profile_on_the_project(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    profile = db.one("SELECT * FROM projects WHERE id=?", (project_id,))["speaking_profile"]
    assert profile["described"] == "Steady, moderate energy with a factual tone."
    assert profile["measured"]["duration_s"] > 0
    assert profile["derived"]["pace"] in {"continuous", "moderate", "measured"}


def test_speaking_profile_reaches_the_speech_synthesis(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    assert ai.synthesis_profiles
    assert ai.synthesis_profiles[0]["described"] == "Steady, moderate energy with a factual tone."


def test_speaking_profile_is_analysed_once_and_reused_by_later_jobs(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))
    pipeline.process_job(start_job(db, project_id))

    assert ai.describe_calls == 1


def test_job_completes_when_the_tone_analysis_fails(tmp_path):
    ai = ProfilingFakeAI(describe_error=RuntimeError("audio model unavailable"))
    pipeline, db, project_id = build_project(tmp_path, ai)

    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)

    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "awaiting_review"
    profile = db.one("SELECT * FROM projects WHERE id=?", (project_id,))["speaking_profile"]
    assert profile["described"] is None
    assert profile["measured"]["duration_s"] > 0

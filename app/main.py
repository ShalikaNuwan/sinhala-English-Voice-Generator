from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .audio import AudioError, AudioService
from .config import Settings, settings
from .database import Database, utc_now
from .pipeline import Pipeline
from .schemas import ProcessRequest, ProjectCreate, RegenerateRequest, TextUpdate


ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac"}
STATIC_DIR = Path(__file__).parent / "static"

db = Database(settings.database_path)
db.initialize()
audio = AudioService(settings.ffmpeg, settings.ffprobe)
pipeline = Pipeline(db, settings, audio)

app = FastAPI(title="Sinhala-to-English Voice Production", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")


def job_configuration(payload: ProcessRequest, config: Settings) -> dict:
    """Resolve the models and voice a job runs with: request overrides, else configured defaults."""
    return {
        "stt_model": payload.stt_model or config.stt_model,
        "text_model": payload.text_model or config.text_model,
        "qa_model": payload.qa_model or config.qa_model,
        "tts_model": payload.tts_model or config.tts_model,
        "audio_model": payload.audio_model or config.audio_model,
        "voice": payload.voice or config.tts_voice,
        "speed": payload.speed if payload.speed is not None else config.tts_speed,
        "diarize_model": payload.diarize_model or config.diarize_model,
        "detect_recordings": payload.detect_recordings if payload.detect_recordings is not None else config.detect_recordings,
        "human_review_gate": payload.human_review_gate,
    }


def require_project(project_id: str) -> dict:
    project = db.one("SELECT * FROM projects WHERE id=?", (project_id,))
    if not project:
        raise HTTPException(404, "Project not found")
    return project


def require_segment(segment_id: str) -> dict:
    segment = db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
    if not segment:
        raise HTTPException(404, "Segment not found")
    return segment


def media_url(path: str | None) -> str | None:
    if not path:
        return None
    try:
        relative = Path(path).resolve().relative_to(settings.data_dir)
    except ValueError:
        return None
    return "/media/" + relative.as_posix()


def present_segment(segment: dict) -> dict:
    return segment | {
        "source_audio_url": media_url(segment.get("source_audio_path")),
        "tts_audio_url": media_url(segment.get("tts_audio_path")),
    }


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "ffmpeg": settings.ffmpeg, "openai_configured": bool(settings.openai_api_key)}


@app.post("/api/projects", status_code=201)
def create_project(payload: ProjectCreate) -> dict:
    project_id = str(uuid.uuid4())
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,glossary_json,narrator_profile_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
        (project_id, payload.title, payload.topic, json.dumps(payload.glossary), json.dumps(payload.narrator_profile), now, now),
    )
    return require_project(project_id)


@app.get("/api/projects")
def list_projects() -> list[dict]:
    return db.all("SELECT * FROM projects ORDER BY created_at DESC")


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    project = require_project(project_id)
    project["jobs"] = db.all("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC", (project_id,))
    return project


@app.post("/api/projects/{project_id}/upload")
def upload_audio(project_id: str, file: UploadFile = File(...)) -> dict:
    require_project(project_id)
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    destination_dir = settings.data_dir / "projects" / project_id / "original"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"source{extension}"
    temporary = destination.with_suffix(destination.suffix + ".uploading")
    size = 0
    try:
        with temporary.open("wb") as output:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB")
                output.write(chunk)
        info = audio.probe(temporary)
        if info.duration_ms > settings.max_audio_minutes * 60_000:
            raise HTTPException(400, f"Audio is longer than the configured {settings.max_audio_minutes}-minute limit")
        temporary.replace(destination)
    except (AudioError, HTTPException) as exc:
        temporary.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(400, str(exc)) from exc
    db.execute(
        "UPDATE projects SET source_path=?, source_filename=?, source_duration_ms=?, status='uploaded', updated_at=? WHERE id=?",
        (str(destination), file.filename, info.duration_ms, utc_now(), project_id),
    )
    return {"project_id": project_id, "filename": file.filename, "duration_ms": info.duration_ms, "codec": info.codec, "sample_rate": info.sample_rate}


@app.post("/api/projects/{project_id}/process", status_code=202)
def process_project(project_id: str, payload: ProcessRequest, background: BackgroundTasks) -> dict:
    project = require_project(project_id)
    if not project.get("source_path"):
        raise HTTPException(400, "Upload audio before starting processing")
    if not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY is not configured")
    active = db.one("SELECT id FROM jobs WHERE project_id=? AND status IN ('queued','running') LIMIT 1", (project_id,))
    if active:
        raise HTTPException(409, "This project already has a running job")
    job_id = str(uuid.uuid4())
    config = job_configuration(payload, settings)
    db.execute(
        "INSERT INTO jobs(id,project_id,status,stage,progress,config_json,created_at) VALUES(?,?,?,?,?,?,?)",
        (job_id, project_id, "queued", "queued", 0, json.dumps(config), utc_now()),
    )
    db.execute("UPDATE projects SET status='processing', updated_at=? WHERE id=?", (utc_now(), project_id))
    background.add_task(pipeline.process_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        raise HTTPException(404, "Job not found")
    counts = db.one(
        "SELECT COUNT(*) AS total, SUM(CASE WHEN qa_status='passed' THEN 1 ELSE 0 END) AS passed, "
        "SUM(CASE WHEN qa_status='needs_review' THEN 1 ELSE 0 END) AS needs_review FROM segments WHERE job_id=?",
        (job_id,),
    )
    return job | {"segment_counts": counts}


@app.get("/api/jobs/{job_id}/segments")
def list_segments(job_id: str) -> list[dict]:
    if not db.one("SELECT id FROM jobs WHERE id=?", (job_id,)):
        raise HTTPException(404, "Job not found")
    return [present_segment(s) for s in db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))]


@app.patch("/api/segments/{segment_id}/transcript")
def update_transcript(segment_id: str, payload: TextUpdate) -> dict:
    require_segment(segment_id)
    db.execute(
        "UPDATE segments SET transcript_si=?, status='edited', qa_status='pending', revision=revision+1, updated_at=? WHERE id=?",
        (payload.text, utc_now(), segment_id),
    )
    return present_segment(require_segment(segment_id))


@app.patch("/api/segments/{segment_id}/script")
def update_script(segment_id: str, payload: TextUpdate) -> dict:
    require_segment(segment_id)
    db.execute(
        "UPDATE segments SET narration_en=?, status='edited', qa_status='pending', revision=revision+1, updated_at=? WHERE id=?",
        (payload.text, utc_now(), segment_id),
    )
    return present_segment(require_segment(segment_id))


@app.post("/api/segments/{segment_id}/regenerate", status_code=202)
def regenerate_segment(segment_id: str, payload: RegenerateRequest, background: BackgroundTasks) -> dict:
    require_segment(segment_id)
    if not settings.openai_api_key:
        raise HTTPException(503, "OPENAI_API_KEY is not configured")
    db.execute("UPDATE segments SET status='regenerating', updated_at=? WHERE id=?", (utc_now(), segment_id))
    background.add_task(pipeline.regenerate_segment, segment_id, payload.stage)
    return {"segment_id": segment_id, "status": "regenerating", "stage": payload.stage}


@app.post("/api/projects/{project_id}/assemble")
def assemble_project(project_id: str) -> dict:
    require_project(project_id)
    try:
        artifacts = pipeline.assemble_project(project_id)
    except (RuntimeError, AudioError) as exc:
        raise HTTPException(409, str(exc)) from exc
    return {name: media_url(path) for name, path in artifacts.items()}


@app.get("/api/projects/{project_id}/export")
def export_project(project_id: str) -> dict:
    require_project(project_id)
    export_dir = settings.data_dir / "projects" / project_id / "exports"
    names = ["final_en.wav", "final_en.mp3", "transcript_si.json", "script_en.json"]
    return {name: media_url(str(export_dir / name)) for name in names if (export_dir / name).exists()}

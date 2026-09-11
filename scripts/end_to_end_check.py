"""Run the real pipeline on a slice of an existing project's source with the actual models.

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/end_to_end_check.py <source_project_id> <start_s> <end_s>

Cuts [start_s, end_s] from the source project's normalised audio, creates a new project through
the HTTP API, uploads the slice, runs a processing job to completion (the review gate is on, so
it stops at awaiting_review), prints every segment, confirms every kept recording, assembles,
and writes data/e2e_check/<project_id>/report.md. Spends API credit in proportion to the slice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import settings  # noqa: E402
from app.main import app, db, pipeline  # noqa: E402


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    source_project, start_s, end_s = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    source = settings.data_dir / "projects" / source_project / "original" / "source_normalized.wav"
    if not source.exists():
        sys.exit(f"No normalised source at {source}")
    out_dir = settings.data_dir / "e2e_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    slice_path = out_dir / f"slice_{int(start_s)}_{int(end_s)}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", str(source),
         "-ac", "1", "-ar", "24000", str(slice_path)],
        check=True,
    )
    original = db.one("SELECT * FROM projects WHERE id=?", (source_project,)) or {}

    client = TestClient(app)
    created = client.post("/api/projects", json={
        "title": f"e2e check {int(start_s)}-{int(end_s)}s",
        "topic": original.get("topic") or "true-crime documentary narration",
        "glossary": original.get("glossary") or {},
    })
    created.raise_for_status()
    project_id = created.json()["id"]
    with slice_path.open("rb") as handle:
        uploaded = client.post(f"/api/projects/{project_id}/upload", files={"file": (slice_path.name, handle, "audio/wav")})
    uploaded.raise_for_status()
    print("project", project_id, "uploaded", uploaded.json()["duration_ms"], "ms", flush=True)

    # TestClient runs background tasks before returning, so this call blocks until the job finishes.
    started = client.post(f"/api/projects/{project_id}/process", json={"human_review_gate": True})
    started.raise_for_status()
    job_id = started.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}").json()
    segments = client.get(f"/api/jobs/{job_id}/segments").json()

    lines = [f"# End-to-end check: project {project_id}, job {job_id}", "",
             f"Job status: {job['status']} at stage {job['stage']}; error: {job.get('error')}",
             f"Warnings: {job.get('warnings')}", ""]
    for s in segments:
        lines += [f"## Segment {s['segment_index']} [{s['kind']}] {s['start_ms']}-{s['end_ms']} ms  qa={s['qa_status']}",
                  f"- audio: {s.get('tts_audio_path')} ({s.get('tts_duration_ms')} ms)",
                  f"- transcript: {s.get('transcript_si')}",
                  f"- narration: {s.get('narration_en')}",
                  f"- style: {s.get('style')}",
                  f"- qa: {s.get('qa')}", ""]
    for s in segments:
        if s["kind"] == "original":
            client.post(f"/api/segments/{s['id']}/confirm").raise_for_status()
    if job["status"] == "awaiting_review":
        assembled = client.post(f"/api/projects/{project_id}/assemble")
        lines += ["## Assembly", f"status {assembled.status_code}: {assembled.json()}", ""]
        export_dir = settings.data_dir / "projects" / project_id / "exports"
        for name in ("final_en.wav", "final_en.mp3"):
            path = export_dir / name
            if path.exists():
                info = pipeline.audio.probe(path)
                lines.append(f"- {name}: {path} ({info.duration_ms} ms, {info.channels} ch, {info.sample_rate} Hz)")
    calls = db.all("SELECT stage, model, status, COUNT(*) AS n FROM model_calls WHERE job_id=? GROUP BY stage, model, status", (job_id,))
    lines += ["", "## Model calls", *[f"- {c['stage']} {c['model']} {c['status']}: {c['n']}" for c in calls]]
    report = out_dir / project_id / "report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("report written to", report)


if __name__ == "__main__":
    main()

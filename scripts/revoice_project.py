"""Re-voice every narration segment with the configured TTS provider, then assemble.

    .venv/bin/python scripts/revoice_project.py            # voice everything, then assemble
    .venv/bin/python scripts/revoice_project.py --dry-run  # count characters and cost, generate nothing
    .venv/bin/python scripts/revoice_project.py --limit 3  # the first three, to hear it before committing

The scripts are already written and reviewed, so this re-runs only the voice: synthesis, pause
shaping and mastering, exactly as `_narrate_segment` does. QA is deliberately NOT re-run. Its
verdicts are about the text, which has not changed, and re-running it would clear the approvals a
reviewer gave by hand.

Kept recordings are never touched: they are real evidence audio, not narration.

Safe to re-run. A segment whose new audio already exists is skipped, so an interrupted run
continues rather than paying to generate the same words twice.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pauses  # noqa: E402
from app.audio import AudioService  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database, utc_now  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402

CREATOR_PER_CHAR = 22 / 121_000  # what a character costs on the ElevenLabs Creator plan


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])

    db = Database(settings.database_path)
    pipeline = Pipeline(db, settings)
    audio = AudioService(settings.ffmpeg, settings.ffprobe)

    project = db.one("SELECT * FROM projects ORDER BY created_at DESC LIMIT 1")
    if not project:
        sys.exit("No project in the local database")
    job = db.one("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project["id"],))
    if not job:
        sys.exit("That project has no job")
    segments = db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job["id"],))
    narration = [s for s in segments if s["kind"] == "narration" and (s["narration_en"] or "").strip()]
    recordings = sum(1 for s in segments if s["kind"] == "original")
    if limit:
        narration = narration[:limit]

    characters = sum(len(s["narration_en"]) for s in narration)
    print(f"project:  {project['title']}")
    print(f"provider: {settings.tts_provider}"
          + (f"  voice: {settings.elevenlabs_voice}  stability: {settings.elevenlabs_stability}"
             if settings.tts_provider == "elevenlabs" else f"  voice: {settings.tts_voice}"))
    print(f"segments: {len(narration)} narration, {recordings} recordings left alone")
    print(f"text:     {characters:,} characters"
          + (f"  about ${characters * CREATOR_PER_CHAR:.2f} of ElevenLabs credit"
             if settings.tts_provider == "elevenlabs" else ""))
    if dry_run:
        print("\ndry run, nothing generated")
        return

    ai = pipeline._ai()
    profile = project.get("speaking_profile")
    pace = ((profile or {}).get("derived") or {}).get("pace") or "moderate"
    generated = settings.data_dir / "projects" / project["id"] / "generated" / job["id"]
    generated.mkdir(parents=True, exist_ok=True)

    started, done, skipped, failed, spent = time.time(), 0, 0, 0, 0
    previous_text, previous_style = "", None

    for position, segment in enumerate(narration, start=1):
        index = segment["segment_index"]
        revision = segment["revision"] + 1
        stem = f"{index:04d}_r{revision:02d}"
        raw_path, final_path = generated / f"{stem}_raw.wav", generated / f"{stem}.wav"
        text, style = segment["narration_en"], segment["style"] or {}

        if final_path.exists() and final_path.stat().st_size > 1000 and segment["tts_audio_path"] == str(final_path):
            skipped += 1
            previous_text, previous_style = text, style
            continue

        try:
            ai.synthesize(text, job["config"].get("tts_model", settings.tts_model),
                          job["config"].get("voice", settings.tts_voice), style, raw_path,
                          profile, previous_style, None, 1.0, previous_text=previous_text)
            spent += len(text)
            try:
                spoken = ai.word_timestamps(raw_path, settings.align_model)
                pauses.shape(raw_path, text, spoken, pace, final_path, seed=segment["id"],
                             ffmpeg=settings.ffmpeg, ffprobe=settings.ffprobe)
            except Exception as exc:  # noqa: BLE001 - shaping is an enhancement, the voice is not
                final_path.write_bytes(raw_path.read_bytes())
                print(f"  [{index}] pauses left as voiced: {str(exc)[:70]}")
            if job["config"].get("master_voice", settings.master_voice):
                audio.master(final_path, final_path)
            duration = audio.probe(final_path).duration_ms
            db.execute(
                "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, revision=?, updated_at=? WHERE id=?",
                (str(final_path), duration, revision, utc_now(), segment["id"]),
            )
            done += 1
            rate = (time.time() - started) / max(done, 1)
            left = (len(narration) - position) * rate
            print(f"  [{position:>3}/{len(narration)}] segment {index:<4} {duration:>6} ms"
                  f"   {spent:>6,} chars   ~{left/60:.0f} min left")
        except Exception as exc:  # noqa: BLE001 - one bad segment must not end the run
            failed += 1
            print(f"  [{index}] FAILED: {str(exc)[:160]}")
            traceback.print_exc(limit=1)
        previous_text, previous_style = text, style

    print(f"\nvoiced {done}, skipped {skipped}, failed {failed}, {spent:,} characters used")
    if failed:
        print("re-run to retry the failures; finished segments are skipped")
        return
    if limit:
        print("partial run, not assembling")
        return
    artifacts = pipeline.assemble_project(project["id"], job["id"])
    for name, path in artifacts.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

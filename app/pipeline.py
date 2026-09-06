from __future__ import annotations

import hashlib
import json
import traceback
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from . import speaking_profile
from .ai import AIClient, PROMPT_VERSION
from .audio import AudioService
from .config import Settings
from .database import Database, utc_now
from .direction import segment_gap_ms


T = TypeVar("T")


class Pipeline:
    def __init__(self, db: Database, config: Settings, audio: AudioService | None = None):
        self.db = db
        self.config = config
        self.audio = audio or AudioService(config.ffmpeg, config.ffprobe)

    def _ai(self) -> AIClient:
        return AIClient(self.config.openai_api_key)

    def _job_update(self, job_id: str, *, status: str, stage: str, progress: int, error: str | None = None) -> None:
        self.db.execute(
            "UPDATE jobs SET status=?, stage=?, progress=?, error=? WHERE id=?",
            (status, stage, progress, error, job_id),
        )

    def _direction_inputs(self, project: dict, config: dict) -> tuple[str | None, float]:
        """The project-level persona override (strings only) and the job's speaking speed."""
        persona = (project.get("narrator_profile") or {}).get("persona")
        if not isinstance(persona, str) or not persona.strip():
            persona = None
        return persona, config.get("speed", self.config.tts_speed)

    def _tracked_call(
        self,
        job_id: str,
        segment_id: str,
        stage: str,
        model: str,
        input_value: str,
        operation: Callable[[], T],
    ) -> T:
        digest = hashlib.sha256(input_value.encode("utf-8")).hexdigest()
        call_id = self.db.execute(
            "INSERT INTO model_calls(job_id, segment_id, stage, model, prompt_version, input_hash, status, started_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (job_id, segment_id, stage, model, PROMPT_VERSION, digest, "running", utc_now()),
        )
        try:
            result = operation()
        except Exception as exc:
            self.db.execute(
                "UPDATE model_calls SET status='failed', error=?, completed_at=? WHERE id=?",
                (str(exc)[:2000], utc_now(), call_id),
            )
            raise
        self.db.execute(
            "UPDATE model_calls SET status='completed', completed_at=? WHERE id=?",
            (utc_now(), call_id),
        )
        return result

    def _speaking_profile(self, job_id: str, project: dict, normalized: Path, config: dict) -> dict:
        """Describe how the source narrator speaks, so the English can be directed to match.

        Analysed once per project and reused afterwards. A failure here degrades the
        narration slightly but must never fail the job.
        """
        existing = project.get("speaking_profile")
        if existing and existing.get("measured"):
            return existing

        measured = speaking_profile.measure(normalized, self.config.ffmpeg, self.config.ffprobe)
        described = None
        audio_model = config.get("audio_model", self.config.audio_model)
        try:
            sample = normalized.parent / "delivery_sample.wav"
            speaking_profile.extract_sample(normalized, sample, measured["duration_s"], self.config.ffmpeg)
            ai = self._ai()
            described = self._tracked_call(
                job_id, None, "speaking_profile", audio_model, str(sample),
                lambda: ai.describe_delivery(sample, audio_model),
            )
        except Exception as exc:  # noqa: BLE001 - style analysis is an enhancement, not a gate
            traceback.print_exc()
            print(f"Speaking profile description failed, continuing with measurements only: {exc}")

        profile = {
            "measured": measured,
            "derived": speaking_profile.derive(measured),
            "described": described,
            "analyzed_at": utc_now(),
        }
        self.db.execute(
            "UPDATE projects SET speaking_profile_json=?, updated_at=? WHERE id=?",
            (json.dumps(profile), utc_now(), project["id"]),
        )
        return profile

    def process_job(self, job_id: str) -> None:
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not job:
            return
        project = self.db.one("SELECT * FROM projects WHERE id=?", (job["project_id"],))
        if not project or not project.get("source_path"):
            self._job_update(job_id, status="failed", stage="validation", progress=0, error="Project has no uploaded audio")
            return
        config = job["config"]
        project_dir = self.config.data_dir / "projects" / project["id"]
        try:
            self.db.execute("UPDATE jobs SET started_at=? WHERE id=?", (utc_now(), job_id))
            self._job_update(job_id, status="running", stage="normalizing", progress=3)
            normalized = project_dir / "original" / "source_normalized.wav"
            self.audio.normalize(Path(project["source_path"]), normalized)

            self._job_update(job_id, status="running", stage="analyzing", progress=5)
            profile = self._speaking_profile(job_id, project, normalized, config)

            self._job_update(job_id, status="running", stage="segmenting", progress=8)
            chunks = self.audio.split(normalized, project_dir / "segments" / job_id)
            self.db.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            segment_ids: list[str] = []
            for index, (start_ms, end_ms, chunk_path) in enumerate(chunks, start=1):
                segment_id = str(uuid.uuid4())
                segment_ids.append(segment_id)
                self.db.execute(
                    "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (segment_id, job_id, project["id"], index, start_ms, end_ms, str(chunk_path), utc_now()),
                )

            ai = self._ai()
            previous_context = ""
            previous_style: dict | None = None
            persona, speed = self._direction_inputs(project, config)
            for position, segment_id in enumerate(segment_ids):
                base_progress = 10 + round(80 * position / max(len(segment_ids), 1))
                segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
                assert segment
                self._job_update(job_id, status="running", stage=f"transcribing {position + 1}/{len(segment_ids)}", progress=base_progress)
                transcript = self._tracked_call(
                    job_id, segment_id, "transcription", config["stt_model"], str(segment["source_audio_path"]),
                    lambda: ai.transcribe(Path(segment["source_audio_path"]), config["stt_model"], project["topic"], project["glossary"]),
                )
                self.db.execute(
                    "UPDATE segments SET transcript_si=?, status='transcribed', updated_at=? WHERE id=?",
                    (transcript, utc_now(), segment_id),
                )

                faithful = self._tracked_call(
                    job_id, segment_id, "translation", config["text_model"], transcript,
                    lambda: ai.translate(transcript, config["text_model"], project["topic"], project["glossary"], previous_context[-1000:]),
                )
                self.db.execute(
                    "UPDATE segments SET faithful_en=?, entities_json=?, numbers_json=?, status='translated', updated_at=? WHERE id=?",
                    (faithful.english_faithful, json.dumps(faithful.entities), json.dumps(faithful.numbers), utc_now(), segment_id),
                )

                adaptation = self._tracked_call(
                    job_id, segment_id, "adaptation", config["text_model"], faithful.english_faithful,
                    lambda: ai.adapt(faithful.english_faithful, config["text_model"], project["narrator_profile"] or {}, previous_context[-600:]),
                )
                style = adaptation.model_dump(exclude={"narration_text"})
                self.db.execute(
                    "UPDATE segments SET narration_en=?, style_json=?, status='adapted', updated_at=? WHERE id=?",
                    (adaptation.narration_text, json.dumps(style), utc_now(), segment_id),
                )

                tts_path = project_dir / "generated" / job_id / f"{segment['segment_index']:04d}_r01.wav"
                self._tracked_call(
                    job_id, segment_id, "tts", config["tts_model"], adaptation.narration_text,
                    lambda: ai.synthesize(
                        adaptation.narration_text, config["tts_model"], config["voice"], style, tts_path,
                        profile, previous_style, persona, speed,
                    ),
                )
                tts_duration = self.audio.probe(tts_path).duration_ms
                self.db.execute(
                    "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, status='generated', updated_at=? WHERE id=?",
                    (str(tts_path), tts_duration, utc_now(), segment_id),
                )

                qa = self._tracked_call(
                    job_id, segment_id, "qa", config["qa_model"], faithful.english_faithful + adaptation.narration_text,
                    lambda: ai.evaluate(faithful.english_faithful, adaptation.narration_text, config["qa_model"]),
                )
                source_duration = segment["end_ms"] - segment["start_ms"]
                duration_ratio = tts_duration / source_duration if source_duration else 0
                qa_payload = qa.model_dump() | {"duration_ratio": round(duration_ratio, 3)}
                if not 0.55 <= duration_ratio <= 1.45:
                    qa_payload["passed"] = False
                    qa_payload["issues"] = qa_payload["issues"] + [f"Duration ratio {duration_ratio:.2f} is outside 0.55-1.45"]
                    qa_payload["recommended_stage_to_retry"] = "tts"
                qa_status = "passed" if qa_payload["passed"] else "needs_review"
                self.db.execute(
                    "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
                    (json.dumps(qa_payload), qa_status, qa_status, utc_now(), segment_id),
                )
                previous_context = adaptation.narration_text
                previous_style = style

            failed = self.db.one("SELECT COUNT(*) AS count FROM segments WHERE job_id=? AND qa_status!='passed'", (job_id,))["count"]
            if config.get("human_review_gate", True) or failed:
                self._job_update(job_id, status="awaiting_review", stage="review", progress=95)
            else:
                self.assemble_project(project["id"], job_id)
                self._job_update(job_id, status="completed", stage="completed", progress=100)
                self.db.execute("UPDATE jobs SET completed_at=? WHERE id=?", (utc_now(), job_id))
        except Exception as exc:
            traceback.print_exc()
            self._job_update(job_id, status="failed", stage="failed", progress=0, error=str(exc)[:2000])
            self.db.execute("UPDATE projects SET status='failed', updated_at=? WHERE id=?", (utc_now(), project["id"]))

    def regenerate_segment(self, segment_id: str, stage: str) -> None:
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment:
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (segment["job_id"],))
        project = self.db.one("SELECT * FROM projects WHERE id=?", (segment["project_id"],))
        assert job and project
        ai = self._ai()
        config = job["config"]
        predecessor = self.db.one(
            "SELECT narration_en, style_json FROM segments WHERE job_id=? AND segment_index=?",
            (job["id"], segment["segment_index"] - 1),
        )
        previous_narration = predecessor["narration_en"] if predecessor else ""
        previous_style = predecessor["style"] if predecessor else None
        persona, speed = self._direction_inputs(project, config)
        try:
            faithful_text = segment["faithful_en"]
            narration_text = segment["narration_en"]
            style = segment["style"] or {}
            if stage == "translation":
                faithful = self._tracked_call(
                    job["id"], segment_id, "translation", config["text_model"], segment["transcript_si"],
                    lambda: ai.translate(segment["transcript_si"], config["text_model"], project["topic"], project["glossary"], ""),
                )
                faithful_text = faithful.english_faithful
                self.db.execute(
                    "UPDATE segments SET faithful_en=?, entities_json=?, numbers_json=? WHERE id=?",
                    (faithful_text, json.dumps(faithful.entities), json.dumps(faithful.numbers), segment_id),
                )
                stage = "adaptation"
            if stage == "adaptation":
                adaptation = self._tracked_call(
                    job["id"], segment_id, "adaptation", config["text_model"], faithful_text,
                    lambda: ai.adapt(faithful_text, config["text_model"], project["narrator_profile"] or {}, previous_narration[-600:]),
                )
                narration_text = adaptation.narration_text
                style = adaptation.model_dump(exclude={"narration_text"})
                self.db.execute(
                    "UPDATE segments SET narration_en=?, style_json=? WHERE id=?",
                    (narration_text, json.dumps(style), segment_id),
                )
                stage = "tts"
            if stage == "tts":
                revision = segment["revision"] + 1
                tts_path = self.config.data_dir / "projects" / project["id"] / "generated" / job["id"] / f"{segment['segment_index']:04d}_r{revision:02d}.wav"
                self._tracked_call(
                    job["id"], segment_id, "tts", config["tts_model"], narration_text,
                    lambda: ai.synthesize(
                        narration_text, config["tts_model"], config["voice"], style, tts_path,
                        project.get("speaking_profile"), previous_style, persona, speed,
                    ),
                )
                duration = self.audio.probe(tts_path).duration_ms
                self.db.execute(
                    "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, revision=?, qa_status='pending' WHERE id=?",
                    (str(tts_path), duration, revision, segment_id),
                )
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            qa = self._tracked_call(
                job["id"], segment_id, "qa", config["qa_model"], fresh["faithful_en"] + fresh["narration_en"],
                lambda: ai.evaluate(fresh["faithful_en"], fresh["narration_en"], config["qa_model"]),
            )
            qa_payload = qa.model_dump()
            if fresh.get("tts_duration_ms"):
                ratio = fresh["tts_duration_ms"] / max(1, fresh["end_ms"] - fresh["start_ms"])
                qa_payload["duration_ratio"] = round(ratio, 3)
                if not 0.55 <= ratio <= 1.45:
                    qa_payload["passed"] = False
                    qa_payload["issues"].append(f"Duration ratio {ratio:.2f} is outside 0.55-1.45")
                    qa_payload["recommended_stage_to_retry"] = "tts"
            status = "passed" if qa_payload["passed"] else "needs_review"
            self.db.execute(
                "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
                (json.dumps(qa_payload), status, status, utc_now(), segment_id),
            )
        except Exception as exc:
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment_id),
            )

    def assemble_project(self, project_id: str, job_id: str | None = None) -> dict[str, str]:
        if not job_id:
            latest = self.db.one("SELECT id FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            if not latest:
                raise RuntimeError("No processing job exists for this project")
            job_id = latest["id"]
        segments = self.db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
        if not segments or any(not segment.get("tts_audio_path") for segment in segments):
            raise RuntimeError("Every segment must have generated audio before assembly")
        export_dir = self.config.data_dir / "projects" / project_id / "exports"
        wav_path = export_dir / "final_en.wav"
        mp3_path = export_dir / "final_en.mp3"
        project = self.db.one("SELECT speaking_profile_json FROM projects WHERE id=?", (project_id,))
        profile = project.get("speaking_profile") if project else None
        gaps = [
            segment_gap_ms(earlier.get("style"), later.get("style"), profile)
            for earlier, later in zip(segments, segments[1:])
        ]
        self.audio.assemble([Path(s["tts_audio_path"]) for s in segments], wav_path, mp3_path, gaps)
        (export_dir / "transcript_si.json").write_text(
            json.dumps([{"index": s["segment_index"], "start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["transcript_si"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (export_dir / "script_en.json").write_text(
            json.dumps([{"index": s["segment_index"], "faithful": s["faithful_en"], "narration": s["narration_en"], "qa": s["qa"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.db.execute("UPDATE projects SET status='completed', updated_at=? WHERE id=?", (utc_now(), project_id))
        self.db.execute(
            "UPDATE jobs SET status='completed', stage='completed', progress=100, completed_at=? WHERE id=?",
            (utc_now(), job_id),
        )
        return {"wav": str(wav_path), "mp3": str(mp3_path)}

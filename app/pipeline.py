from __future__ import annotations

import hashlib
import json
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Callable, TypeVar

from . import pauses, recordings, speaking_profile
from .ai import AIClient, PROMPT_VERSION
from .audio import AudioService
from .config import Settings
from .database import Database, utc_now
from .direction import MIN_GAP_MS, segment_gap_ms
from .schemas import QAEvaluation


T = TypeVar("T")

CONFIRM_NOTE = "Recorded audio kept as is. Confirm."
ENGLISH_NOTE = "Possible recorded audio (English speech in transcript). Mark as original if so."
DURATION_RANGE = (0.55, 1.45)


def kept_qa() -> dict:
    """QA payload for a recording kept from the source: nothing to check, but a reviewer must confirm."""
    return {
        "passed": False,
        "severity": "low",
        "issues": [CONFIRM_NOTE],
        "recommended_stage_to_retry": "none",
        "duration_ratio": 1.0,
    }


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

    def _warn(self, job_id: str, message: str) -> None:
        """Record a non-fatal problem on the job so the reviewer sees it."""
        job = self.db.one("SELECT warnings_json FROM jobs WHERE id=?", (job_id,))
        warnings = list((job or {}).get("warnings") or [])
        warnings.append(message)
        self.db.execute("UPDATE jobs SET warnings_json=? WHERE id=?", (json.dumps(warnings), job_id))
        print(f"Job {job_id}: {message}")

    def _direction_inputs(self, project: dict, config: dict) -> tuple[str | None, float]:
        """The project-level persona override (strings only) and the job's speaking speed."""
        persona = (project.get("narrator_profile") or {}).get("persona")
        if not isinstance(persona, str) or not persona.strip():
            persona = None
        return persona, config.get("speed", self.config.tts_speed)

    def _normalized_path(self, project_id: str) -> Path:
        return self.config.data_dir / "projects" / project_id / "original" / "source_normalized.wav"

    def _generated_dir(self, project_id: str, job_id: str) -> Path:
        return self.config.data_dir / "projects" / project_id / "generated" / job_id

    def _tracked_call(
        self,
        job_id: str,
        segment_id: str | None,
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

    def _narrator_reference(self, job_id: str, project: dict, normalized: Path, profile: dict, config: dict) -> Path | None:
        """A 2-10 s clip of the narrator, so diarization can label their speech by name.

        Chosen once per project from the delivery sample and stored in the speaking profile. Any
        failure disables recording detection for this job with a warning; it never fails the job.
        """
        stored = (profile.get("narrator_reference") or {}) if profile else {}
        if stored.get("path") and Path(stored["path"]).exists():
            return Path(stored["path"])
        model = config.get("diarize_model", self.config.diarize_model)
        try:
            sample = normalized.parent / "delivery_sample.wav"
            if not sample.exists():
                duration_s = (profile.get("measured") or {}).get("duration_s") or self.audio.probe(normalized).duration_ms / 1000
                speaking_profile.extract_sample(normalized, sample, duration_s, self.config.ffmpeg)
            ai = self._ai()
            spans = self._tracked_call(job_id, None, "diarization", model, str(sample), lambda: ai.diarize(sample, model))
            window = recordings.choose_reference(spans)
            if window is None:
                self._warn(job_id, "Recording detection unavailable: no clear narrator speech in the delivery sample")
                return None
            start_s, end_s = window
            reference = normalized.parent / "narrator_reference.wav"
            self.audio.extract(sample, round(start_s * 1000), round(end_s * 1000), reference, pad_ms=0)
            profile["narrator_reference"] = {"path": str(reference), "start_s": start_s, "end_s": end_s}
            self.db.execute(
                "UPDATE projects SET speaking_profile_json=?, updated_at=? WHERE id=?",
                (json.dumps(profile), utc_now(), project["id"]),
            )
            return reference
        except Exception as exc:  # noqa: BLE001 - detection is an enhancement, not a gate
            traceback.print_exc()
            self._warn(job_id, f"Recording detection unavailable: {str(exc)[:200]}")
            return None

    def _create_segments(
        self,
        job_id: str,
        project: dict,
        normalized: Path,
        chunks: list[tuple[int, int, Path]],
        reference: Path | None,
        config: dict,
    ) -> list[str]:
        """One row per piece: narration to be voiced, or a recording kept from the source."""
        self.db.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
        model = config.get("diarize_model", self.config.diarize_model)
        ai = self._ai() if reference is not None else None
        generated = self._generated_dir(project["id"], job_id)
        segment_ids: list[str] = []
        index = 0
        for chunk_index, (chunk_start, chunk_end, chunk_path) in enumerate(chunks, start=1):
            spans: list[dict] = []
            pieces: list[recordings.Piece] = [(chunk_start, chunk_end, "narration")]
            if reference is not None:
                try:
                    spans = self._tracked_call(
                        job_id, None, "diarization", model, str(chunk_path),
                        lambda: ai.diarize(chunk_path, model, reference),
                    )
                    # The chunk file starts CHUNK_PAD_MS before the planned chunk; diarizer times are file-relative.
                    lead_s = self.audio.chunk_lead_ms(chunk_start) / 1000
                    spans = [dict(item, start=float(item["start"]) - lead_s, end=float(item["end"]) - lead_s) for item in spans]
                    originals = recordings.original_spans(spans, chunk_end - chunk_start)
                    pieces = recordings.cut_plan(chunk_start, chunk_end, originals, recordings.narration_spans(spans))
                except Exception as exc:  # noqa: BLE001 - one chunk's detection must not fail the job
                    traceback.print_exc()
                    self._warn(job_id, f"Recording detection failed on chunk {chunk_index}: {str(exc)[:200]}")
                    spans, pieces = [], [(chunk_start, chunk_end, "narration")]
            for start_ms, end_ms, kind in pieces:
                index += 1
                segment_id = str(uuid.uuid4())
                segment_ids.append(segment_id)
                if kind == "original":
                    clip = generated / f"{index:04d}_original.wav"
                    info = self.audio.extract_levelled(normalized, start_ms, end_ms, clip)
                    text = recordings.clip_text(spans, start_ms - chunk_start, end_ms - chunk_start)
                    self.db.execute(
                        "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, kind, "
                        "transcript_si, tts_audio_path, tts_duration_ms, qa_json, qa_status, status, updated_at) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (segment_id, job_id, project["id"], index, start_ms, end_ms, str(clip), "original",
                         text, str(clip), info.duration_ms, json.dumps(kept_qa()), "needs_review", "kept", utc_now()),
                    )
                    continue
                source_path = chunk_path
                if (start_ms, end_ms) != (chunk_start, chunk_end):
                    source_path = chunk_path.parent / f"{index:04d}_piece.wav"
                    self.audio.extract(normalized, start_ms, end_ms, source_path, pad_ms=0)
                self.db.execute(
                    "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, kind, updated_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (segment_id, job_id, project["id"], index, start_ms, end_ms, str(source_path), "narration", utc_now()),
                )
        return segment_ids

    def _qa_verdict(
        self, qa: QAEvaluation, transcript: str, tts_duration_ms: int | None, segment: dict, shaped: dict | None = None,
    ) -> dict:
        """Model QA plus the mechanical checks: duration ratio, an English run that looks like a recording, and pauses."""
        payload = qa.model_dump()
        if tts_duration_ms is not None:
            ratio = tts_duration_ms / max(1, segment["end_ms"] - segment["start_ms"])
            payload["duration_ratio"] = round(ratio, 3)
            low, high = DURATION_RANGE
            if not low <= ratio <= high:
                payload["passed"] = False
                payload["issues"] = payload["issues"] + [f"Duration ratio {ratio:.2f} is outside {low}-{high}"]
                payload["recommended_stage_to_retry"] = "tts"
        if recordings.english_run(transcript):
            payload["passed"] = False
            payload["issues"] = payload["issues"] + [ENGLISH_NOTE]
        shaped = shaped or (segment.get("qa") or {}).get("pauses")
        if shaped:
            payload["pauses"] = {key: shaped.get(key) for key in ("method", "pace", "classes", "profile", "unpunctuated_over_cap")}
            for issue in pauses.profile_issues(shaped):
                payload["passed"] = False
                payload["issues"] = payload["issues"] + [issue]
        return payload

    def _shape_pauses(
        self, ai: AIClient, job: dict, segment: dict, raw_path: Path, final_path: Path, script: str, profile: dict | None,
    ) -> dict | None:
        """Turn the voice's pauses into shaped ones; on any failure the raw voice is kept and a warning says why."""
        config = job["config"]
        if not config.get("shape_pauses", self.config.shape_pauses):
            raw_path.replace(final_path)
            return None
        pace = ((profile or {}).get("derived") or {}).get("pace") or "moderate"
        align_model = config.get("align_model", self.config.align_model)
        spoken: list[dict] = []
        try:
            spoken = self._tracked_call(
                job["id"], segment["id"], "alignment", align_model, str(raw_path),
                lambda: ai.word_timestamps(raw_path, align_model),
            )
        except Exception as exc:  # noqa: BLE001 - alignment is an enhancement; shaping can still try by order
            traceback.print_exc()
            self._warn(job["id"], f"Word timestamps failed on segment {segment['segment_index']}: {str(exc)[:200]}")
        try:
            return pauses.shape(raw_path, script, spoken, pace, final_path, seed=segment["id"], ffmpeg=self.config.ffmpeg, ffprobe=self.config.ffprobe)
        except pauses.AlignmentError as exc:
            self._warn(job["id"], f"Pauses left as voiced on segment {segment['segment_index']}: {exc}")
            shutil.copyfile(raw_path, final_path)
            return None

    def _narrate_segment(
        self,
        ai: AIClient,
        job: dict,
        project: dict,
        segment: dict,
        profile: dict | None,
        previous_context: str,
        previous_style: dict | None,
        persona: str | None,
        speed: float,
    ) -> tuple[str, dict]:
        """Run every narration stage for one segment and return the context the next one inherits."""
        job_id, segment_id, config = job["id"], segment["id"], job["config"]
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

        stem = f"{segment['segment_index']:04d}_r{segment['revision']:02d}"
        generated = self._generated_dir(project["id"], job_id)
        raw_path, tts_path = generated / f"{stem}_raw.wav", generated / f"{stem}.wav"
        self._tracked_call(
            job_id, segment_id, "tts", config["tts_model"], adaptation.narration_text,
            lambda: ai.synthesize(
                adaptation.narration_text, config["tts_model"], config["voice"], style, raw_path,
                profile, previous_style, persona, speed,
            ),
        )
        shaped = self._shape_pauses(ai, job, segment, raw_path, tts_path, adaptation.narration_text, profile)
        tts_duration = self.audio.probe(tts_path).duration_ms
        self.db.execute(
            "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, status='generated', updated_at=? WHERE id=?",
            (str(tts_path), tts_duration, utc_now(), segment_id),
        )

        qa = self._tracked_call(
            job_id, segment_id, "qa", config["qa_model"], faithful.english_faithful + adaptation.narration_text,
            lambda: ai.evaluate(faithful.english_faithful, adaptation.narration_text, config["qa_model"]),
        )
        qa_payload = self._qa_verdict(qa, transcript, tts_duration, segment, shaped)
        qa_status = "passed" if qa_payload["passed"] else "needs_review"
        self.db.execute(
            "UPDATE segments SET qa_json=?, qa_status=?, status=?, updated_at=? WHERE id=?",
            (json.dumps(qa_payload), qa_status, qa_status, utc_now(), segment_id),
        )
        return adaptation.narration_text, style

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

            reference = None
            if config.get("detect_recordings", self.config.detect_recordings):
                self._job_update(job_id, status="running", stage="finding recordings", progress=7)
                reference = self._narrator_reference(job_id, project, normalized, profile, config)

            self._job_update(job_id, status="running", stage="segmenting", progress=8)
            chunks = self.audio.split(normalized, project_dir / "segments" / job_id)
            segment_ids = self._create_segments(job_id, project, normalized, chunks, reference, config)

            ai = self._ai()
            previous_context = ""
            previous_style: dict | None = None
            persona, speed = self._direction_inputs(project, config)
            total = len(segment_ids)
            for position, segment_id in enumerate(segment_ids):
                base_progress = 10 + round(80 * position / max(total, 1))
                segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
                assert segment
                if segment["kind"] == "original":
                    self._job_update(job_id, status="running", stage=f"keeping recording {position + 1}/{total}", progress=base_progress)
                    previous_context += f"\n[Recording plays: {(segment['transcript_si'] or '')[:200]}]"
                    continue
                self._job_update(job_id, status="running", stage=f"transcribing {position + 1}/{total}", progress=base_progress)
                previous_context, previous_style = self._narrate_segment(
                    ai, job, project, segment, profile, previous_context, previous_style, persona, speed,
                )

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

    def _previous_narration(self, job_id: str, segment_index: int) -> tuple[str, dict | None]:
        """The nearest earlier narration segment's script and style, skipping kept recordings."""
        predecessor = self.db.one(
            "SELECT narration_en, style_json FROM segments WHERE job_id=? AND kind='narration' AND segment_index<? "
            "ORDER BY segment_index DESC LIMIT 1",
            (job_id, segment_index),
        )
        if not predecessor:
            return "", None
        return predecessor["narration_en"], predecessor["style"]

    def _recut_original(self, segment: dict, job: dict, project: dict) -> None:
        """Cut the recording from the source again; nothing is generated for it."""
        try:
            revision = segment["revision"] + 1
            clip = self._generated_dir(project["id"], job["id"]) / f"{segment['segment_index']:04d}_original_r{revision:02d}.wav"
            info = self.audio.extract_levelled(self._normalized_path(project["id"]), segment["start_ms"], segment["end_ms"], clip)
            self.db.execute(
                "UPDATE segments SET tts_audio_path=?, tts_duration_ms=?, revision=?, qa_json=?, qa_status='needs_review', "
                "status='kept', updated_at=? WHERE id=?",
                (str(clip), info.duration_ms, revision, json.dumps(kept_qa()), utc_now(), segment["id"]),
            )
        except Exception as exc:  # noqa: BLE001 - reported on the segment, as regeneration does
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment["id"]),
            )

    def regenerate_segment(self, segment_id: str, stage: str) -> None:
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment:
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (segment["job_id"],))
        project = self.db.one("SELECT * FROM projects WHERE id=?", (segment["project_id"],))
        assert job and project
        if segment["kind"] == "original":
            self._recut_original(segment, job, project)
            return
        ai = self._ai()
        config = job["config"]
        previous_narration, previous_style = self._previous_narration(job["id"], segment["segment_index"])
        persona, speed = self._direction_inputs(project, config)
        try:
            faithful_text = segment["faithful_en"]
            narration_text = segment["narration_en"]
            style = segment["style"] or {}
            shaped = None
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
                stem = f"{segment['segment_index']:04d}_r{revision:02d}"
                generated = self._generated_dir(project["id"], job["id"])
                raw_path, tts_path = generated / f"{stem}_raw.wav", generated / f"{stem}.wav"
                self._tracked_call(
                    job["id"], segment_id, "tts", config["tts_model"], narration_text,
                    lambda: ai.synthesize(
                        narration_text, config["tts_model"], config["voice"], style, raw_path,
                        project.get("speaking_profile"), previous_style, persona, speed,
                    ),
                )
                shaped = self._shape_pauses(ai, job, segment, raw_path, tts_path, narration_text, project.get("speaking_profile"))
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
            qa_payload = self._qa_verdict(qa, fresh["transcript_si"], fresh.get("tts_duration_ms"), fresh, shaped)
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

    def set_segment_kind(self, segment_id: str, kind: str) -> None:
        """Reviewer override: keep a segment as recorded, or voice one that detection kept."""
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment:
            return
        job = self.db.one("SELECT * FROM jobs WHERE id=?", (segment["job_id"],))
        project = self.db.one("SELECT * FROM projects WHERE id=?", (segment["project_id"],))
        assert job and project
        if kind == "original":
            self.db.execute(
                "UPDATE segments SET kind='original', faithful_en='', narration_en='', style_json='{}', "
                "tts_audio_path=NULL, tts_duration_ms=NULL, updated_at=? WHERE id=?",
                (utc_now(), segment_id),
            )
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            self._recut_original(fresh, job, project)
            return
        self.db.execute(
            "UPDATE segments SET kind='narration', tts_audio_path=NULL, tts_duration_ms=NULL, qa_json='{}', "
            "qa_status='pending', status='created', revision=revision+1, updated_at=? WHERE id=?",
            (utc_now(), segment_id),
        )
        try:
            fresh = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
            assert fresh
            previous_narration, previous_style = self._previous_narration(job["id"], fresh["segment_index"])
            persona, speed = self._direction_inputs(project, job["config"])
            self._narrate_segment(
                self._ai(), job, project, fresh, project.get("speaking_profile"),
                previous_narration, previous_style, persona, speed,
            )
        except Exception as exc:  # noqa: BLE001 - reported on the segment, as regeneration does
            traceback.print_exc()
            self.db.execute(
                "UPDATE segments SET status='failed', qa_status='failed', qa_json=?, updated_at=? WHERE id=?",
                (json.dumps({"passed": False, "issues": [str(exc)]}), utc_now(), segment_id),
            )

    def confirm_segment(self, segment_id: str) -> None:
        """The reviewer has listened to a kept recording and accepts it."""
        segment = self.db.one("SELECT * FROM segments WHERE id=?", (segment_id,))
        if not segment or segment.get("kind") != "original":
            raise RuntimeError("Only recorded-audio segments need confirming")
        if not segment.get("tts_audio_path") or segment.get("status") == "failed":
            raise RuntimeError("This recording has no audio to confirm; regenerate it first")
        qa = dict(segment.get("qa") or kept_qa())
        qa["passed"] = True
        qa["issues"] = [issue for issue in qa.get("issues", []) if issue != CONFIRM_NOTE]
        self.db.execute(
            "UPDATE segments SET qa_json=?, qa_status='passed', status='kept', updated_at=? WHERE id=?",
            (json.dumps(qa), utc_now(), segment_id),
        )

    def assemble_project(self, project_id: str, job_id: str | None = None) -> dict[str, str]:
        if not job_id:
            latest = self.db.one("SELECT id FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT 1", (project_id,))
            if not latest:
                raise RuntimeError("No processing job exists for this project")
            job_id = latest["id"]
        segments = self.db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
        if not segments or any(not segment.get("tts_audio_path") or segment.get("status") == "failed" for segment in segments):
            raise RuntimeError("Every segment must have generated audio before assembly")
        export_dir = self.config.data_dir / "projects" / project_id / "exports"
        wav_path = export_dir / "final_en.wav"
        mp3_path = export_dir / "final_en.mp3"
        project = self.db.one("SELECT speaking_profile_json FROM projects WHERE id=?", (project_id,))
        profile = project.get("speaking_profile") if project else None
        def gap(earlier: dict, later: dict) -> int:
            kinds = (earlier.get("kind"), later.get("kind"))
            if kinds == ("original", "original") and earlier["end_ms"] == later["start_ms"]:
                return 0
            if "original" in kinds:
                return MIN_GAP_MS
            return segment_gap_ms(earlier.get("style"), later.get("style"), profile)

        gaps = [gap(earlier, later) for earlier, later in zip(segments, segments[1:])]
        self.audio.assemble([Path(s["tts_audio_path"]) for s in segments], wav_path, mp3_path, gaps)
        (export_dir / "transcript_si.json").write_text(
            json.dumps([{"index": s["segment_index"], "kind": s.get("kind", "narration"), "start_ms": s["start_ms"], "end_ms": s["end_ms"], "text": s["transcript_si"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (export_dir / "script_en.json").write_text(
            json.dumps([{"index": s["segment_index"], "kind": s.get("kind", "narration"), "faithful": s["faithful_en"], "narration": s["narration_en"], "qa": s["qa"]} for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.db.execute("UPDATE projects SET status='completed', updated_at=? WHERE id=?", (utc_now(), project_id))
        self.db.execute(
            "UPDATE jobs SET status='completed', stage='completed', progress=100, completed_at=? WHERE id=?",
            (utc_now(), job_id),
        )
        return {"wav": str(wav_path), "mp3": str(mp3_path)}

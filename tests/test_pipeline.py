from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pytest

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
        return NarrationAdaptation(
            narration_text="This is a test.", beat="reveal", emotion="tense", pause_after_ms=800,
        )

    def describe_delivery(self, _audio_path, _model):
        return "Steady, moderate energy with a factual tone."

    def diarize(self, audio_path, _model, reference=None):
        # The sample has one speaker; chunks are narrator-only unless a subclass says otherwise.
        speaker = "narrator" if reference else "A"
        return [{"speaker": speaker, "start": 0.0, "end": 3.0, "text": "මෙය පරීක්ෂණයකි."}]

    def synthesize(self, _text, _model, _voice, _style, output_path: Path, profile=None,
                   previous_style=None, persona=None, speed=1.0):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1", str(output_path)],
            check=True,
        )

    def evaluate(self, *_args, **_kwargs):
        return QAEvaluation(passed=True)

    def word_timestamps(self, _audio_path, _model):
        return []


def create_source(path: Path, seconds: float = 1.0) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}", str(path)],
        check=True,
    )


def bursts(path: Path, pattern: list[tuple[float, float]]) -> None:
    """Tone bursts separated by silences: pattern is [(tone_s, silence_after_s), ...]."""
    inputs, labels = [], []
    for tone_s, gap_s in pattern:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={tone_s}"]
        labels.append(f"[{len(labels)}:a]")
        if gap_s > 0:
            inputs += ["-f", "lavfi", "-t", f"{gap_s}", "-i", "anullsrc=r=24000:cl=mono"]
            labels.append(f"[{len(labels)}:a]")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[out]",
         "-map", "[out]", "-ar", "24000", "-ac", "1", str(path)],
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
        self.synthesis_calls: list[dict] = []
        self.adapt_calls: list[dict] = []
        self.diarize_calls: list[dict] = []
        self.transcribe_paths: list[Path] = []

    def describe_delivery(self, _audio_path, _model):
        self.describe_calls += 1
        if self.describe_error:
            raise self.describe_error
        return "Steady, moderate energy with a factual tone."

    def transcribe(self, audio_path, *_args, **_kwargs):
        self.transcribe_paths.append(Path(audio_path))
        return super().transcribe(audio_path, *_args, **_kwargs)

    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        return super().diarize(audio_path, model, reference)

    def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration})
        return super().adapt(faithful_en, model, narrator_profile, previous_narration)

    def synthesize(self, text, model, voice, style, output_path, profile=None,
                   previous_style=None, persona=None, speed=1.0):
        self.synthesis_profiles.append(profile)
        self.synthesis_calls.append({
            "text": text, "voice": voice, "style": style, "output_path": output_path,
            "previous_style": previous_style, "persona": persona, "speed": speed,
        })
        super().synthesize(text, model, voice, style, output_path, profile, previous_style, persona, speed)


SINHALA = "ඇය කියනවා කවුරුහරි එනවා කියලා."
CALL = "Hello? Do you need the police?"


class RecordingFakeAI(ProfilingFakeAI):
    """First chunk has a 911 call from 20 s to 30 s; later chunks are narrator only."""

    def __init__(self, fail_chunks: bool = False, fail_sample: bool = False):
        super().__init__()
        self.fail_chunks = fail_chunks
        self.fail_sample = fail_sample

    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        if reference is None:
            if self.fail_sample:
                raise RuntimeError("diarization unavailable")
            return [{"speaker": "A", "start": 0.0, "end": 9.0, "text": SINHALA}]
        if self.fail_chunks:
            raise RuntimeError("diarization failed")
        if Path(audio_path).name == "0001_source.wav":
            return [
                {"speaker": "narrator", "start": 0.0, "end": 20.0, "text": SINHALA},
                {"speaker": "A", "start": 20.0, "end": 24.0, "text": "Hello?"},
                {"speaker": "A", "start": 27.0, "end": 30.0, "text": "Do you need the police?"},
                {"speaker": "narrator", "start": 30.0, "end": 45.0, "text": SINHALA},
            ]
        return [{"speaker": "narrator", "start": 0.2, "end": 20.2, "text": SINHALA}]


class BoundaryFakeAI(RecordingFakeAI):
    """A recording runs from 40 s to 50 s, across the 45 s chunk boundary."""

    def diarize(self, audio_path, model, reference=None):
        self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
        if reference is None:
            return [{"speaker": "A", "start": 0.0, "end": 9.0, "text": SINHALA}]
        if Path(audio_path).name == "0001_source.wav":
            return [
                {"speaker": "narrator", "start": 0.0, "end": 40.0, "text": SINHALA},
                {"speaker": "A", "start": 40.0, "end": 45.0, "text": "Hello?"},
            ]
        # Chunk 2's file begins 200 ms before 45 s, so a recording ending at source 50 s ends at file 5.2 s.
        return [
            {"speaker": "A", "start": 0.0, "end": 5.2, "text": "Do you need the police?"},
            {"speaker": "narrator", "start": 5.2, "end": 20.2, "text": SINHALA},
        ]


class PausingFakeAI(ProfilingFakeAI):
    """Voices a three-part script as tone bursts with a 0.9 s and a 1.4 s pause, and knows the word times."""

    def __init__(self, timestamps_error: Exception | None = None):
        super().__init__()
        self.timestamps_error = timestamps_error
        self.timestamp_calls = 0

    def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
        self.adapt_calls.append({"faithful": faithful_en, "previous_narration": previous_narration})
        return NarrationAdaptation(narration_text="First part, second part. Third part.", beat="build", emotion="calm")

    def synthesize(self, text, model, voice, style, output_path, profile=None, previous_style=None, persona=None, speed=1.0):
        self.synthesis_calls.append({"text": text, "voice": voice, "style": style, "output_path": output_path,
                                     "previous_style": previous_style, "persona": persona, "speed": speed})
        bursts(Path(output_path), [(1.0, 0.9), (1.0, 1.4), (1.0, 0.5)])

    def word_timestamps(self, _audio_path, _model):
        self.timestamp_calls += 1
        if self.timestamps_error:
            raise self.timestamps_error
        return [{"word": w, "start": s, "end": e} for w, s, e in
                (("First", 0.0, 0.5), ("part", 0.5, 1.0), ("second", 1.9, 2.4), ("part", 2.4, 2.9), ("Third", 4.3, 4.8), ("part", 4.8, 5.3))]


def build_project(tmp_path, ai, seconds: float = 1.0, narrator_profile: dict | None = None):
    data_dir = tmp_path / "data"
    config = Settings(data_dir=data_dir, database_path=data_dir / "app.db", openai_api_key="not-used")
    db = Database(config.database_path)
    db.initialize()
    project_id = str(uuid.uuid4())
    source = data_dir / "projects" / project_id / "original" / "source.wav"
    source.parent.mkdir(parents=True)
    create_source(source, seconds)
    now = utc_now()
    db.execute(
        "INSERT INTO projects(id,title,topic,glossary_json,narrator_profile_json,status,source_path,created_at,updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (project_id, "Test", "History", "{}", json.dumps(narrator_profile or {}), "uploaded", str(source), now, now),
    )
    pipeline = Pipeline(db, config)
    pipeline._ai = lambda: ai
    return pipeline, db, project_id


def start_job(db, project_id, detect_recordings: bool = True, shape_pauses: bool = True):
    job_id = str(uuid.uuid4())
    job_config = {
        "stt_model": "fake-stt", "text_model": "fake-text", "qa_model": "fake-qa",
        "tts_model": "fake-tts", "audio_model": "fake-audio", "voice": "onyx",
        "speed": 0.95,
        "diarize_model": "fake-diarize", "detect_recordings": detect_recordings,
        "align_model": "fake-align", "shape_pauses": shape_pauses,
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


def test_each_segment_is_voiced_with_the_previous_segments_style(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id))

    assert len(ai.synthesis_calls) == 2
    assert ai.synthesis_calls[0]["previous_style"] is None
    assert ai.synthesis_calls[1]["previous_style"]["beat"] == "reveal"
    assert ai.synthesis_calls[1]["previous_style"]["emotion"] == "tense"


def test_adaptation_sees_the_previous_narration(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id))

    assert ai.adapt_calls[0]["previous_narration"] == ""
    assert ai.adapt_calls[1]["previous_narration"] == "This is a test."


def test_synthesis_receives_the_job_speed_and_the_project_persona(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, narrator_profile={"persona": "You are a calm lecturer."})

    pipeline.process_job(start_job(db, project_id))

    assert ai.synthesis_calls[0]["speed"] == 0.95
    assert ai.synthesis_calls[0]["persona"] == "You are a calm lecturer."


def test_generated_segments_are_wav_files(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["tts_audio_path"].endswith("0001_r01.wav")
    assert Path(segment["tts_audio_path"]).exists()


def test_regenerating_a_segment_uses_its_predecessors_style(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    pipeline.process_job(start_job(db, project_id))
    second = db.one("SELECT * FROM segments WHERE project_id=? AND segment_index=2", (project_id,))
    first = db.one("SELECT * FROM segments WHERE project_id=? AND segment_index=1", (project_id,))

    pipeline.regenerate_segment(second["id"], "tts")

    call = ai.synthesis_calls[-1]
    assert call["previous_style"] == first["style"]
    # The voice writes to the raw path; pause shaping produces the final 0002_r02.wav from it.
    assert str(call["output_path"]).endswith("0002_r02_raw.wav")


def test_regenerating_the_first_segment_has_no_predecessor(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id))
    first = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(first["id"], "tts")

    assert ai.synthesis_calls[-1]["previous_style"] is None


def test_assembly_leaves_the_adaptations_pause_between_segments(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)

    captured = {}
    real_assemble = pipeline.audio.assemble

    def spy(paths, wav, mp3, gaps_ms=None):
        captured["gaps"] = gaps_ms
        real_assemble(paths, wav, mp3, gaps_ms)

    pipeline.audio.assemble = spy

    artifacts = pipeline.assemble_project(project_id, job_id)

    # Two 1 s fake segments plus the fake adaptation's 800 ms pause_after.
    duration = pipeline.audio.probe(Path(artifacts["wav"])).duration_ms
    assert 2650 <= duration <= 2950
    assert captured["gaps"] == [800]


def test_a_non_string_persona_is_ignored_rather_than_crashing(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, narrator_profile={"persona": ["not", "a", "string"]})

    pipeline.process_job(start_job(db, project_id))

    assert ai.synthesis_calls[0]["persona"] is None
    assert db.one("SELECT status FROM jobs WHERE project_id=?", (project_id,))["status"] == "awaiting_review"


def test_regenerating_the_adaptation_passes_the_predecessors_narration(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    pipeline.process_job(start_job(db, project_id))
    second = db.one("SELECT * FROM segments WHERE project_id=? AND segment_index=2", (project_id,))

    pipeline.regenerate_segment(second["id"], "adaptation")

    assert ai.adapt_calls[-1]["previous_narration"] == "This is a test."


def test_jobs_created_before_speed_existed_fall_back_to_the_configured_speed(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)
    config = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["config"]
    config.pop("speed")
    db.execute("UPDATE jobs SET config_json=? WHERE id=?", (json.dumps(config), job_id))

    pipeline.process_job(job_id)

    assert ai.synthesis_calls[0]["speed"] == pipeline.config.tts_speed


def test_assembly_caps_gaps_at_the_source_narrators_longest_pause(tmp_path):
    ai = ProfilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)
    # A pure-tone source measures no pauses; give the project a narrator whose longest pause is 500 ms.
    profile = db.one("SELECT speaking_profile_json FROM projects WHERE id=?", (project_id,))["speaking_profile"]
    profile["measured"]["longest_pause_ms"] = 500
    db.execute("UPDATE projects SET speaking_profile_json=? WHERE id=?", (json.dumps(profile), project_id))
    captured = {}
    real_assemble = pipeline.audio.assemble

    def spy(paths, wav, mp3, gaps_ms=None):
        captured["gaps"] = gaps_ms
        real_assemble(paths, wav, mp3, gaps_ms)

    pipeline.audio.assemble = spy

    pipeline.assemble_project(project_id, job_id)

    # The fake adaptation asks for 800 ms, but the narrator never pauses longer than 500 ms.
    assert captured["gaps"] == [500]


def run_recording_job(tmp_path, ai=None):
    ai = ai or RecordingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)
    job_id = start_job(db, project_id)
    pipeline.process_job(job_id)
    segments = db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
    return pipeline, db, project_id, job_id, ai, segments


def test_a_detected_recording_becomes_its_own_kept_segment(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert [s["kind"] for s in segments] == ["narration", "original", "narration", "narration"]
    assert [(s["start_ms"], s["end_ms"]) for s in segments] == [(0, 19850), (19850, 30150), (30150, 45000), (45000, 65000)]
    original = segments[1]
    assert original["transcript_si"] == CALL
    assert original["narration_en"] == ""
    assert original["tts_audio_path"].endswith("0002_original.wav")
    assert 10100 <= pipeline.audio.probe(Path(original["tts_audio_path"])).duration_ms <= 10500
    assert original["status"] == "kept" and original["qa_status"] == "needs_review"
    assert "Recorded audio kept as is. Confirm." in original["qa"]["issues"]
    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "awaiting_review"


def test_kept_segments_skip_every_model_stage(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert len(ai.synthesis_calls) == 3
    assert len(ai.adapt_calls) == 3
    stages = db.all("SELECT stage, segment_id FROM model_calls WHERE job_id=?", (job_id,))
    assert not [row for row in stages if row["segment_id"] == segments[1]["id"]]


def test_the_narration_after_a_recording_knows_what_was_heard(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert "[Recording plays: " + CALL + "]" in ai.adapt_calls[1]["previous_narration"]
    # Continuity skips the recording: segment 3 inherits segment 1's style.
    assert ai.synthesis_calls[1]["previous_style"] == segments[0]["style"]


def test_narrator_reference_is_cut_once_and_reused(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    profile = db.one("SELECT * FROM projects WHERE id=?", (project_id,))["speaking_profile"]
    reference = Path(profile["narrator_reference"]["path"])
    assert reference.exists() and reference.name == "narrator_reference.wav"
    assert profile["narrator_reference"]["start_s"] == 0.0 and profile["narrator_reference"]["end_s"] == 9.0
    first_run = len(ai.diarize_calls)

    pipeline.process_job(start_job(db, project_id))

    without_reference = [c for c in ai.diarize_calls if c["reference"] is None]
    assert len(without_reference) == 1  # the sample was diarized only once
    assert all(c["reference"] == reference for c in ai.diarize_calls[first_run:])


def test_assembly_uses_the_minimum_gap_next_to_a_recording(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    captured = {}
    real_assemble = pipeline.audio.assemble

    def spy(paths, wav, mp3, gaps_ms=None):
        captured["gaps"] = gaps_ms
        real_assemble(paths, wav, mp3, gaps_ms)

    pipeline.audio.assemble = spy

    artifacts = pipeline.assemble_project(project_id, job_id)

    assert captured["gaps"] == [300, 300, 800]
    exported = json.loads(Path(artifacts["wav"]).parent.joinpath("script_en.json").read_text(encoding="utf-8"))
    assert [entry["kind"] for entry in exported] == ["narration", "original", "narration", "narration"]


def test_detection_failure_on_a_chunk_keeps_it_as_narration_with_a_warning(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, RecordingFakeAI(fail_chunks=True))

    assert [s["kind"] for s in segments] == ["narration", "narration"]
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Recording detection failed on chunk 1" in w for w in warnings)
    assert db.one("SELECT status FROM jobs WHERE id=?", (job_id,))["status"] == "awaiting_review"


def test_detection_failure_on_the_sample_disables_detection_with_a_warning(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, RecordingFakeAI(fail_sample=True))

    assert [s["kind"] for s in segments] == ["narration", "narration"]
    assert all(c["reference"] is None for c in ai.diarize_calls) and len(ai.diarize_calls) == 1
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Recording detection unavailable" in w for w in warnings)


def test_detection_can_be_switched_off_per_job(tmp_path):
    ai = RecordingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai, seconds=65)

    pipeline.process_job(start_job(db, project_id, detect_recordings=False))

    assert ai.diarize_calls == []
    assert [s["kind"] for s in db.all("SELECT * FROM segments WHERE project_id=? ORDER BY segment_index", (project_id,))] == ["narration", "narration"]


def test_an_english_run_in_a_transcript_is_flagged_for_review(tmp_path):
    class LeakyFakeAI(ProfilingFakeAI):
        def transcribe(self, *_args, **_kwargs):
            return SINHALA + " 9-1-1, how can I assist you? Hello? Hello? Hello, you dialed into the 911 system."

    ai = LeakyFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["qa_status"] == "needs_review"
    assert any("Possible recorded audio" in issue for issue in segment["qa"]["issues"])


def test_a_recording_can_be_confirmed(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    pipeline.confirm_segment(segments[1]["id"])

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["qa_status"] == "passed" and fresh["status"] == "kept"
    assert "Recorded audio kept as is. Confirm." not in fresh["qa"]["issues"]


def test_confirming_a_narration_segment_is_refused(tmp_path):
    import pytest

    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    with pytest.raises(RuntimeError):
        pipeline.confirm_segment(segments[0]["id"])


def test_a_recording_can_be_turned_into_narration(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    before = len(ai.synthesis_calls)

    pipeline.set_segment_kind(segments[1]["id"], "narration")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["kind"] == "narration"
    assert fresh["tts_audio_path"].endswith("0002_r02.wav")
    assert fresh["narration_en"] == "This is a test."
    # The fake voice is 1 s long against a 10.3 s span, so only the duration check should complain.
    assert fresh["qa_status"] == "needs_review"
    assert [issue for issue in fresh["qa"]["issues"] if "Duration ratio" not in issue] == []
    assert len(ai.synthesis_calls) == before + 1
    assert ai.adapt_calls[-1]["previous_narration"] == segments[0]["narration_en"]


def test_a_narration_segment_can_be_kept_as_recorded(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    pipeline.set_segment_kind(segments[2]["id"], "original")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[2]["id"],))
    assert fresh["kind"] == "original"
    assert fresh["narration_en"] == "" and fresh["style"] == {}
    assert fresh["tts_audio_path"].endswith("0003_original_r02.wav")
    assert 14600 <= fresh["tts_duration_ms"] <= 15100  # 30150-45000 ms cut exactly
    assert fresh["qa_status"] == "needs_review" and fresh["status"] == "kept"


def test_regenerating_a_kept_segment_recuts_it_without_model_calls(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    calls_before = len(ai.synthesis_calls) + len(ai.adapt_calls)

    pipeline.regenerate_segment(segments[1]["id"], "tts")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[1]["id"],))
    assert fresh["tts_audio_path"].endswith("0002_original_r02.wav")
    assert fresh["revision"] == 2 and fresh["qa_status"] == "needs_review"
    assert len(ai.synthesis_calls) + len(ai.adapt_calls) == calls_before


def test_narration_pieces_are_transcribed_from_their_own_cut_not_the_chunk(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    assert segments[0]["source_audio_path"].endswith("0001_piece.wav")
    assert segments[2]["source_audio_path"].endswith("0003_piece.wav")
    assert segments[3]["source_audio_path"].endswith("0002_source.wav")  # untouched chunk keeps its file
    assert [p.name for p in ai.transcribe_paths] == ["0001_piece.wav", "0003_piece.wav", "0002_source.wav"]
    # Pieces are cut without padding so no recording audio bleeds into the narration transcript.
    assert 19700 <= pipeline.audio.probe(Path(segments[0]["source_audio_path"])).duration_ms <= 20000


def test_regenerating_the_segment_after_a_recording_skips_it_for_continuity(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)

    pipeline.regenerate_segment(segments[2]["id"], "adaptation")

    assert ai.adapt_calls[-1]["previous_narration"] == segments[0]["narration_en"]
    assert ai.synthesis_calls[-1]["previous_style"] == segments[0]["style"]


def test_a_recording_across_a_chunk_boundary_is_joined_without_a_gap(tmp_path):
    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, BoundaryFakeAI())
    assert [s["kind"] for s in segments] == ["narration", "original", "original", "narration"]
    assert segments[1]["end_ms"] == segments[2]["start_ms"] == 45000
    captured = {}
    real_assemble = pipeline.audio.assemble

    def spy(paths, wav, mp3, gaps_ms=None):
        captured["gaps"] = gaps_ms
        real_assemble(paths, wav, mp3, gaps_ms)

    pipeline.audio.assemble = spy

    pipeline.assemble_project(project_id, job_id)

    assert captured["gaps"] == [300, 0, 300]


def test_a_failed_flip_to_original_cannot_be_assembled(tmp_path):
    import pytest

    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    pipeline.audio.extract_levelled = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cut failed"))

    pipeline.set_segment_kind(segments[2]["id"], "original")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segments[2]["id"],))
    assert fresh["status"] == "failed" and fresh["tts_audio_path"] is None
    with pytest.raises(RuntimeError):
        pipeline.assemble_project(project_id, job_id)


def test_a_recording_without_audio_cannot_be_confirmed(tmp_path):
    import pytest

    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path)
    pipeline.audio.extract_levelled = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cut failed"))
    pipeline.set_segment_kind(segments[2]["id"], "original")

    with pytest.raises(RuntimeError, match="no audio"):
        pipeline.confirm_segment(segments[2]["id"])


def test_recording_times_on_later_chunks_account_for_the_split_padding(tmp_path):
    """Chunk files after the first start 200 ms early; diarizer times must be shifted before cutting."""

    class LateChunkFakeAI(RecordingFakeAI):
        def diarize(self, audio_path, model, reference=None):
            self.diarize_calls.append({"path": Path(audio_path), "reference": reference})
            if reference is None:
                return [{"speaker": "A", "start": 0.0, "end": 9.0, "text": SINHALA}]
            if Path(audio_path).name == "0001_source.wav":
                return [{"speaker": "narrator", "start": 0.0, "end": 45.0, "text": SINHALA}]
            # A recording from source 50 s to 60 s appears at file 5.2-15.2 s in the padded chunk-2 file.
            return [
                {"speaker": "narrator", "start": 0.2, "end": 5.2, "text": SINHALA},
                {"speaker": "A", "start": 5.2, "end": 15.2, "text": CALL},
                {"speaker": "narrator", "start": 15.2, "end": 20.2, "text": SINHALA},
            ]

    pipeline, db, project_id, job_id, ai, segments = run_recording_job(tmp_path, LateChunkFakeAI())

    assert [(s["kind"], s["start_ms"], s["end_ms"]) for s in segments] == [
        ("narration", 0, 45000), ("narration", 45000, 49850), ("original", 49850, 60150), ("narration", 60150, 65000),
    ]
    assert segments[2]["transcript_si"] == CALL


def measured_gaps(path: Path) -> list[int]:
    from app import pauses
    found = pauses.detect_pauses(path)
    duration = pauses._duration_s(path)
    return [round((e - s) * 1000) for s, e in found if s > 0.005 and e < duration - 0.005]


def test_pauses_are_shaped_after_synthesis(tmp_path):
    from app import pauses

    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["tts_audio_path"].endswith("0001_r01.wav")
    assert Path(segment["tts_audio_path"]).with_name("0001_r01_raw.wav").exists()
    assert measured_gaps(Path(segment["tts_audio_path"]).with_name("0001_r01_raw.wav")) == pytest.approx([900, 1400], abs=40)
    gaps = measured_gaps(Path(segment["tts_audio_path"]))
    # A pure-tone source is a "continuous" narrator, so bands are scaled by 0.85.
    assert pauses.BANDS["clause"][0] * 0.85 - 30 <= gaps[0] <= pauses.BANDS["clause"][1] * 0.85 + 30
    assert pauses.BANDS["sentence"][0] * 0.85 - 30 <= gaps[1] <= pauses.BANDS["sentence"][1] * 0.85 + 30
    assert segment["qa"]["pauses"]["method"] == "words"
    assert segment["qa"]["pauses"]["classes"] == {"clause": 1, "sentence": 1, "edge": 1}
    assert ai.timestamp_calls == 1
    assert db.one("SELECT stage FROM model_calls WHERE job_id=? AND stage='alignment'", (segment["job_id"],))


def test_shaping_can_be_switched_off(tmp_path):
    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id, shape_pauses=False))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert ai.timestamp_calls == 0
    assert measured_gaps(Path(segment["tts_audio_path"])) == pytest.approx([900, 1400], abs=40)
    assert "pauses" not in segment["qa"]


def test_timestamp_failure_falls_back_to_punctuation_order(tmp_path):
    ai = PausingFakeAI(timestamps_error=RuntimeError("no whisper"))
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["qa"]["pauses"]["method"] == "order"
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Word timestamps failed on segment 1" in w for w in warnings)


def test_unshapeable_audio_keeps_the_raw_voice_with_a_warning(tmp_path):
    class LonelyFakeAI(PausingFakeAI):
        def adapt(self, faithful_en, model, narrator_profile, previous_narration=""):
            return NarrationAdaptation(narration_text="One sentence with no breaks at all")

    ai = LonelyFakeAI(timestamps_error=RuntimeError("no whisper"))
    pipeline, db, project_id = build_project(tmp_path, ai)
    job_id = start_job(db, project_id)

    pipeline.process_job(job_id)

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert measured_gaps(Path(segment["tts_audio_path"])) == pytest.approx([900, 1400], abs=40)
    warnings = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))["warnings"]
    assert any("Pauses left as voiced on segment 1" in w for w in warnings)


def test_regenerating_tts_shapes_pauses_too(tmp_path):
    from app import pauses

    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id))
    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(segment["id"], "tts")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
    assert fresh["tts_audio_path"].endswith("0001_r02.wav")
    gaps = measured_gaps(Path(fresh["tts_audio_path"]))
    assert pauses.BANDS["clause"][0] * 0.85 - 30 <= gaps[0] <= pauses.BANDS["clause"][1] * 0.85 + 30
    assert fresh["qa"]["pauses"]["method"] == "words"


class CeilingFakeAI(PausingFakeAI):
    """A 2 s pause after a word the transcriber got wrong: it stays unknown, stays 2 s, and trips the ceiling."""

    def synthesize(self, text, model, voice, style, output_path, profile=None, previous_style=None, persona=None, speed=1.0):
        self.synthesis_calls.append({"text": text, "voice": voice, "style": style, "output_path": output_path,
                                     "previous_style": previous_style, "persona": persona, "speed": speed})
        bursts(Path(output_path), [(1.0, 0.9), (1.0, 2.0), (1.0, 0.5)])

    def word_timestamps(self, _audio_path, _model):
        self.timestamp_calls += 1
        # "extra" is an inserted word the script does not have; the pause after it cannot be tied to a break.
        return [{"word": w, "start": s, "end": e} for w, s, e in
                (("First", 0.0, 0.5), ("part", 0.5, 1.0), ("second", 1.9, 2.3), ("part", 2.3, 2.6),
                 ("extra", 2.6, 2.9), ("Third", 4.9, 5.4), ("part", 5.4, 5.9))]


def test_a_pause_over_the_ceiling_is_flagged_for_review(tmp_path):
    ai = CeilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)

    pipeline.process_job(start_job(db, project_id))

    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))
    assert segment["qa_status"] == "needs_review"
    assert any(issue.startswith("Pauses: longest pause") for issue in segment["qa"]["issues"])
    assert segment["qa"]["pauses"]["classes"].get("unknown") == 1


def test_a_qa_only_regeneration_keeps_the_pause_verdict(tmp_path):
    ai = CeilingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id))
    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(segment["id"], "qa")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
    assert fresh["qa_status"] == "needs_review"
    assert any(issue.startswith("Pauses: longest pause") for issue in fresh["qa"]["issues"])
    assert fresh["qa"]["pauses"]["profile"] == segment["qa"]["pauses"]["profile"]


def test_regenerating_tts_with_shaping_off_keeps_the_voice_as_is(tmp_path):
    ai = PausingFakeAI()
    pipeline, db, project_id = build_project(tmp_path, ai)
    pipeline.process_job(start_job(db, project_id, shape_pauses=False))
    segment = db.one("SELECT * FROM segments WHERE project_id=?", (project_id,))

    pipeline.regenerate_segment(segment["id"], "tts")

    fresh = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
    assert fresh["tts_audio_path"].endswith("0001_r02.wav")
    assert measured_gaps(Path(fresh["tts_audio_path"])) == pytest.approx([900, 1400], abs=40)
    assert "pauses" not in fresh["qa"] and ai.timestamp_calls == 0

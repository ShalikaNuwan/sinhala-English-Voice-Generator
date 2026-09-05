from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", ROOT_DIR / "data")).resolve()
    )
    database_path: Path | None = None
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    stt_model: str = field(default_factory=lambda: os.getenv("STT_MODEL", "gpt-transcribe"))
    text_model: str = field(default_factory=lambda: os.getenv("TEXT_MODEL", "gpt-5.6-terra"))
    qa_model: str = field(default_factory=lambda: os.getenv("QA_MODEL", "gpt-5.6-terra"))
    tts_model: str = field(default_factory=lambda: os.getenv("TTS_MODEL", "gpt-4o-mini-tts"))
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "onyx"))
    audio_model: str = field(default_factory=lambda: os.getenv("AUDIO_MODEL", "gpt-audio"))
    max_audio_minutes: int = field(default_factory=lambda: int(os.getenv("MAX_AUDIO_MINUTES", "120")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "500")))
    ffmpeg: str = field(default_factory=lambda: os.getenv("FFMPEG_BIN", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: os.getenv("FFPROBE_BIN", "ffprobe"))

    def __post_init__(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.database_path is None:
            self.database_path = self.data_dir / "app.db"


settings = Settings()

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    glossary_json TEXT NOT NULL DEFAULT '{}',
    narrator_profile_json TEXT NOT NULL DEFAULT '{}',
    speaking_profile_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'created',
    source_path TEXT,
    source_filename TEXT,
    source_duration_ms INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    status TEXT NOT NULL,
    stage TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    config_json TEXT NOT NULL DEFAULT '{}',
    error TEXT,
    warnings_json TEXT NOT NULL DEFAULT '[]',
    started_at TEXT,
    completed_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    segment_index INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    source_audio_path TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'narration',
    transcript_si TEXT NOT NULL DEFAULT '',
    faithful_en TEXT NOT NULL DEFAULT '',
    narration_en TEXT NOT NULL DEFAULT '',
    entities_json TEXT NOT NULL DEFAULT '[]',
    numbers_json TEXT NOT NULL DEFAULT '[]',
    style_json TEXT NOT NULL DEFAULT '{}',
    tts_audio_path TEXT,
    tts_duration_ms INTEGER,
    qa_json TEXT NOT NULL DEFAULT '{}',
    qa_status TEXT NOT NULL DEFAULT 'pending',
    status TEXT NOT NULL DEFAULT 'created',
    revision INTEGER NOT NULL DEFAULT 1,
    updated_at TEXT NOT NULL,
    UNIQUE(job_id, segment_index)
);

CREATE TABLE IF NOT EXISTS model_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    segment_id TEXT,
    stage TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_segments_job ON segments(job_id, segment_index);
"""


# Columns added after the first release, applied to databases created before them.
ADDED_COLUMNS = [
    ("projects", "speaking_profile_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("jobs", "warnings_json", "TEXT NOT NULL DEFAULT '[]'"),
    ("segments", "kind", "TEXT NOT NULL DEFAULT 'narration'"),
]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(SCHEMA)
            self._add_missing_columns(connection)

    @staticmethod
    def _add_missing_columns(connection: sqlite3.Connection) -> None:
        """CREATE TABLE IF NOT EXISTS skips existing tables, so new columns need adding by hand."""
        for table, column, definition in ADDED_COLUMNS:
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def execute(self, sql: str, values: tuple[Any, ...] = ()) -> int:
        with self.connect() as connection:
            cursor = connection.execute(sql, values)
            return cursor.lastrowid

    def one(self, sql: str, values: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(sql, values).fetchone()
        return self._decode(dict(row)) if row else None

    def all(self, sql: str, values: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(sql, values).fetchall()
        return [self._decode(dict(row)) for row in rows]

    @staticmethod
    def _decode(row: dict[str, Any]) -> dict[str, Any]:
        for key in tuple(row):
            if key.endswith("_json"):
                raw = row.pop(key)
                try:
                    row[key.removesuffix("_json")] = json.loads(raw or "null")
                except json.JSONDecodeError:
                    row[key.removesuffix("_json")] = None
        return row


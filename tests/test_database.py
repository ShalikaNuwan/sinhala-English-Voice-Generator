from __future__ import annotations

import sqlite3

from app.database import Database, utc_now


OLD_PROJECTS_TABLE = """
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    glossary_json TEXT NOT NULL DEFAULT '{}',
    narrator_profile_json TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'created',
    source_path TEXT,
    source_filename TEXT,
    source_duration_ms INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def test_existing_database_gains_the_speaking_profile_column_without_losing_data(tmp_path):
    """Databases created before the speaking profile must migrate, not break."""
    path = tmp_path / "app.db"
    connection = sqlite3.connect(path)
    connection.executescript(OLD_PROJECTS_TABLE)
    now = utc_now()
    connection.execute(
        "INSERT INTO projects(id,title,created_at,updated_at) VALUES(?,?,?,?)",
        ("existing-project", "Recorded before the upgrade", now, now),
    )
    connection.commit()
    connection.close()

    database = Database(path)
    database.initialize()

    project = database.one("SELECT * FROM projects WHERE id=?", ("existing-project",))
    assert project["title"] == "Recorded before the upgrade"
    assert project["speaking_profile"] == {}


def test_initialize_is_safe_to_run_repeatedly(tmp_path):
    database = Database(tmp_path / "app.db")
    database.initialize()
    database.initialize()

    assert database.all("SELECT * FROM projects") == []

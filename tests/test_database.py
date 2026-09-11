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


def test_initialize_adds_kind_and_warnings_to_an_older_database(tmp_path):
    import sqlite3

    from app.database import Database

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE segments (id TEXT PRIMARY KEY, job_id TEXT NOT NULL, project_id TEXT NOT NULL, "
            "segment_index INTEGER NOT NULL, start_ms INTEGER NOT NULL, end_ms INTEGER NOT NULL, "
            "source_audio_path TEXT NOT NULL, style_json TEXT NOT NULL DEFAULT '{}', qa_json TEXT NOT NULL DEFAULT '{}', "
            "updated_at TEXT NOT NULL);"
            "CREATE TABLE jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, "
            "progress INTEGER NOT NULL DEFAULT 0, config_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL);"
            "INSERT INTO jobs VALUES ('j1', 'p1', 'queued', 'queued', 0, '{}', 'now');"
            "INSERT INTO segments(id, job_id, project_id, segment_index, start_ms, end_ms, source_audio_path, updated_at) "
            "VALUES ('s1', 'j1', 'p1', 1, 0, 1000, 'x.wav', 'now');"
        )

    db = Database(path)
    db.initialize()

    assert db.one("SELECT * FROM segments WHERE id='s1'")["kind"] == "narration"
    assert db.one("SELECT * FROM jobs WHERE id='j1'")["warnings"] == []

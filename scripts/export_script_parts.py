"""Write the reviewed script of the latest project as ordered parts around the kept recordings.

    .venv/bin/python scripts/export_script_parts.py              # latest project
    .venv/bin/python scripts/export_script_parts.py <project_id>

Each narration part is a text file to voice on its own (for example in the ElevenLabs web app);
each kept recording is copied as a WAV to place between them. `index.md` lists the playing order.
Makes no API calls. Output: data/projects/<project_id>/exports/script_parts/
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.pipeline import Pipeline  # noqa: E402


def main() -> None:
    db = Database(settings.database_path)
    if len(sys.argv) > 1:
        project = db.one("SELECT * FROM projects WHERE id=?", (sys.argv[1],))
    else:
        project = db.one("SELECT * FROM projects ORDER BY created_at DESC LIMIT 1")
    if not project:
        sys.exit("No such project")
    written = Pipeline(db, settings).export_script_parts(project["id"])
    print(f"project: {project['title']}")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print((Path(written["index.md"])).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

"""Pull the reviewed English and Sinhala text from a running server into the local database.

    .venv/bin/python scripts/sync_from_server.py http://13.229.108.159:8000
    .venv/bin/python scripts/sync_from_server.py data/voice_previews/server_segments_snapshot.json

Takes a running server's base URL, or a saved snapshot of its segment list, so the server does not
have to be running (and billing) just to copy text across.

Review happens wherever the app is deployed, so the edited scripts live there and the local copy
falls behind. This copies the text and the QA verdicts back, matching on segment id, and touches
nothing else: audio paths, revisions and timings stay as they are locally. Makes no API calls and
costs nothing. Writes a timestamped backup of the local database first.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.database import Database, utc_now  # noqa: E402

# Text and the verdicts that were reached about it. Audio paths, revisions and timings are
# deliberately excluded: those describe local files, which the server knows nothing about.
FIELDS = ("transcript_si", "narration_en", "faithful_en")
# `status` carries "approved", the marker that a reviewer accepted a segment by hand, so it has to
# come across or the record of that decision is lost.
EXTRA = ("style_json", "status")


def fetch(url: str) -> list[dict]:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read())


def segments_from(source: str) -> tuple[str, list[dict]]:
    """Either a live server or a saved snapshot of one."""
    local_file = Path(source)
    if local_file.exists():
        return f"snapshot {local_file.name}", json.loads(local_file.read_text(encoding="utf-8"))
    base = source.rstrip("/")
    projects = fetch(f"{base}/api/projects")
    if not projects:
        sys.exit("The server has no projects")
    project = projects[0]
    job = fetch(f"{base}/api/projects/{project['id']}")["jobs"][0]
    return project["title"], fetch(f"{base}/api/jobs/{job['id']}/segments")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    label, remote = segments_from(sys.argv[1])
    print(f"source: {label} - {len(remote)} segments")

    backup = settings.database_path.with_name(
        f"app.db.before-sync-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    )
    shutil.copy2(settings.database_path, backup)
    print(f"backup: {backup}")

    db = Database(settings.database_path)
    changed = {field: 0 for field in FIELDS + EXTRA}
    qa_changed = missing = 0

    for segment in remote:
        local = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
        if not local:
            missing += 1
            continue
        updates, values = [], []
        for field in FIELDS:
            new = (segment.get(field) or "").strip()
            if new and new != (local.get(field) or "").strip():
                updates.append(f"{field}=?")
                values.append(new)
                changed[field] += 1
        remote_style = json.dumps(segment.get("style") or {}, sort_keys=True)
        if remote_style != json.dumps(json.loads(local.get("style_json") or "{}"), sort_keys=True):
            updates.append("style_json=?"); values.append(remote_style); changed["style_json"] += 1
        if segment.get("status") and segment["status"] != local.get("status"):
            updates.append("status=?"); values.append(segment["status"]); changed["status"] += 1
        if segment.get("qa_status") and segment["qa_status"] != local.get("qa_status"):
            updates += ["qa_status=?", "qa_json=?"]
            values += [segment["qa_status"], json.dumps(segment.get("qa") or {})]
            qa_changed += 1
        if updates:
            values += [utc_now(), segment["id"]]
            db.execute(f"UPDATE segments SET {', '.join(updates)}, updated_at=? WHERE id=?", tuple(values))

    for field, count in changed.items():
        print(f"  {field:<16} {count} updated")
    print(f"  qa verdicts      {qa_changed} updated")
    if missing:
        print(f"  {missing} server segments had no local match and were skipped")


if __name__ == "__main__":
    main()

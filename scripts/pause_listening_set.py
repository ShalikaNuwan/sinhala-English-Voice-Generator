"""Copy raw and shaped voice files for chosen segments so the owner can compare the pauses by ear.

Usage:
    .venv/bin/python scripts/pause_listening_set.py <job_id> <segment_index> [<segment_index> ...]

Writes data/pause_listening/<job_id>/NN_raw.wav, NN_shaped.wav, and pauses.md with a pause table
for each, plus the reference file the owner named. Makes no API calls.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pauses  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402

REFERENCE = Path.home() / "Downloads" / "final.mp3"


def table(path: Path) -> str:
    profile = pauses.pause_profile(path, settings.ffmpeg, settings.ffprobe)
    return (f"| {path.name} | {profile['count']} | {profile['median_ms']} | {profile['p90_ms']} | "
            f"{profile['max_ms']} | {profile['per_minute']} |")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    job_id, indexes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    db = Database(settings.database_path)
    out = settings.data_dir / "pause_listening" / job_id
    out.mkdir(parents=True, exist_ok=True)
    lines = ["| file | pauses | median ms | p90 ms | max ms | per minute |", "|---|---|---|---|---|---|"]
    if REFERENCE.exists():
        lines.append(table(REFERENCE))
    for index in indexes:
        segment = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index))
        if not segment or segment.get("kind") != "narration" or not segment.get("tts_audio_path"):
            print("skipping", index)
            continue
        shaped = Path(segment["tts_audio_path"])
        raw = shaped.with_name(shaped.stem + "_raw.wav")
        for source, name in ((raw, f"{index:02d}_raw.wav"), (shaped, f"{index:02d}_shaped.wav")):
            if source.exists():
                shutil.copyfile(source, out / name)
                lines.append(table(out / name))
        lines.append(f"| script {index} | {segment['narration_en'][:120]!r} | | | | |")
    (out / "pauses.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("written to", out)


if __name__ == "__main__":
    main()

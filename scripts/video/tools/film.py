"""Which film a tool works on: run a tool with FILM=<folder under data/video>, for example FILM=bundy."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAME = os.environ.get("FILM") or sys.exit("Set FILM to the film's folder under data/video, for example FILM=bundy")
D = ROOT / "data" / "video" / NAME
if not D.is_dir():
    sys.exit(f"No film folder at {D}")
FFMPEG = os.getenv("VIDEO_FFMPEG", "/opt/miniconda3/envs/video/bin/ffmpeg")
for path in (str(ROOT / "scripts" / "video"), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

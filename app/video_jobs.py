"""Background renders for the storyboard review: one process per film at a time, its state on disk.

A job is `build.py` run on the film (the 480p review cut, or the 1080p final film), started detached
through /bin/sh so that it outlives a server restart. Its output goes to <film>/logs/review_<kind>.log,
which ends with "__exit__ <code>" when the process finishes; <film>/review/job.json remembers which job
ran. Status is read back from those two files: done or failed from the exit line, running while the
process is alive, stopped when it died without one. Progress is the assembler's latest shot count.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts" / "video" / "build.py"
EXIT = "__exit__"
FULL_FILM_BYTES = 20e9          # free space a 90-minute final render needs
FULL_FILM_S = 90 * 60
# assemble.py's lines: "  shot 170/585 (#2) 6.5s ..." per fresh shot, and once "  169/585 shots already
# rendered or joined" for the cached ones. Not build.py's "527/527 shots have a picture".
SHOTS = re.compile(r"^\s*(?:shot )?(\d+)/(\d+) (?:\(#|shots already rendered)", re.M)


class JobBusy(Exception):
    pass


class NoSpace(Exception):
    pass


def commands(d: Path, kind: str, sound: bool = False) -> list[str]:
    """build.py as documentary.py runs it: the review stage's animatic, or render's final film."""
    if kind == "rebuild":
        return [sys.executable, "-u", str(BUILD), str(d), "--animatic", "--out", str(d / "animatic.mp4")]
    if kind == "render":
        return [sys.executable, "-u", str(BUILD), str(d), "--out", str(d / f"{d.name}_1080p.mp4"),
                *(["--sound"] if sound else [])]
    raise ValueError(f"unknown job {kind}")


def load_env() -> dict:
    """The environment for a job: this process's, with the project's .env (API keys) added."""
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def alive(pid: int) -> bool:
    # A shell this server started stays a zombie until reaped, and kill(0) succeeds on a zombie.
    try:
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped:
            return False
    except ChildProcessError:
        pass                    # not our child (the server restarted since): ask the system
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def progress(text: str) -> float:
    shots = SHOTS.findall(text)
    return int(shots[-1][0]) / int(shots[-1][1]) if shots else 0.0


def check_space(d: Path) -> None:
    duration = json.loads((d / "storyboard.json").read_text(encoding="utf-8")).get("duration_s", FULL_FILM_S)
    need = max(5e9, FULL_FILM_BYTES * duration / FULL_FILM_S)
    free = shutil.disk_usage(d).free
    if free < need:
        raise NoSpace(f"the final render needs about {need / 1e9:.0f} GB free; {free / 1e9:.0f} GB is free")


class Jobs:
    def state_path(self, d: Path) -> Path:
        return d / "review" / "job.json"

    def status(self, d: Path) -> dict | None:
        p = self.state_path(d)
        if not p.exists():
            return None
        job = json.loads(p.read_text(encoding="utf-8"))
        log = Path(job["log"])
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        exits = re.findall(rf"^{EXIT} (-?\d+)\s*$", text, re.M)
        if exits:
            state = "done" if exits[-1] == "0" else "failed"
        elif alive(job["pid"]):
            state = "running"
        else:
            state = "stopped"
        share = 1.0 if state == "done" else progress(text)
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith(EXIT)]
        tail = lines[-12:] if state in ("failed", "stopped") else lines[-1:]
        return {**job, "state": state, "progress": round(share, 3), "tail": tail}

    def start(self, d: Path, kind: str, sound: bool = False) -> dict:
        current = self.status(d)
        if current and current["state"] == "running":
            raise JobBusy(f"a {current['kind']} is already running for this film")
        if kind == "render":
            check_space(d)
        (d / "review").mkdir(parents=True, exist_ok=True)
        (d / "logs").mkdir(parents=True, exist_ok=True)
        log = d / "logs" / f"review_{kind}.log"
        command = " ".join(shlex.quote(c) for c in commands(d, kind, sound))
        started_at = time.time()
        shell = f"{command} > {shlex.quote(str(log))} 2>&1; echo \"{EXIT} $?\" >> {shlex.quote(str(log))}"
        proc = subprocess.Popen(["/bin/sh", "-c", shell], cwd=ROOT, env=load_env(), stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        job = {"kind": kind, "pid": proc.pid, "sound": sound, "log": str(log), "command": command,
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)), "started_at": started_at}
        self.state_path(d).write_text(json.dumps(job, indent=1), encoding="utf-8")
        return self.status(d)

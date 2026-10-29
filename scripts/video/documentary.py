"""From a narration to a documentary: every step of the video pipeline, in order, in one command.

    .venv/bin/python scripts/video/documentary.py prepare <video_dir> --audio narration.wav --script script.txt
    .venv/bin/python scripts/video/documentary.py render  <video_dir> [--sound]

prepare runs everything up to a review cut and stops, so the owner can look before the long render:

    align       whisper word times matched onto the script (align_script.py)
    sentences   sentence timing (sentences.py)
    case        the case profile: where this story's pictures are (case.py -> case.json; read it)
    harvest     real pictures: the case's Commons categories, named people and places, period
                photographs of its region and era, Wikipedia articles (harvest.py)
    caption     a vision model describes every picture (caption.py)
    footage     period film clips for the region and era (footage.py search + build)
    bible       the story bible: people, places, chapters, arc (story_bible.py)
    quotes      quotations and their cards (quotes.py)
    storyboard  scenes and beats, first pass (storyboard.py)
    needs       targeted searches for what the first pass lacked, then a second storyboard pass with
                them (search_plan.py, harvest.py --stages needs, caption.py, storyboard.py)
    scenarios   scene recipes, e.g. the victims' register (storyboard.py --scenarios)
    check       a vision check of every chosen picture (storyboard.py --recheck, check_pictures.py)
    ladder      reframes, holds, illustrations, date and loop rules (--ladder, --illustrate, --recheck)
    keyfacts    labels for the clues the narration stresses (--key-facts)
    review      storyboard.html, a 480p animatic.mp4 and critic.md (storyboard_page.py, build.py
                --animatic, critics.py)

render makes the 1080p film (build.py; --sound mixes music and effects under a bare narration).

Every stage's output goes to <video_dir>/logs/<stage>.log; <video_dir>/pipeline.json remembers the
stages done, so a rerun resumes where it stopped. --from STAGE reruns from that stage on; --only STAGE
runs one. Free-licence pictures only (public domain and CC0) unless --allow-fair-use.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIDEO = ROOT / "scripts" / "video"

PREPARE = ["align", "sentences", "case", "harvest", "caption", "footage", "bible", "quotes", "storyboard",
           "needs", "scenarios", "check", "ladder", "keyfacts", "review"]


def py(script: str, *args: str) -> list[str]:
    return [sys.executable, "-u", str(VIDEO / script), *args]


CLIP_EVERY_S = 45.0          # one period clip for about this much narration
CLIPS_MIN, CLIPS_MAX = 30, 120


def clip_target(d: Path) -> int:
    """How many footage segments to gather: enough for the narration's length, no more."""
    path = d / "sentences.json"
    if not path.exists():
        return CLIPS_MAX
    sentences = json.loads(path.read_text(encoding="utf-8"))
    length = sentences[-1]["end"] if sentences else 0.0
    return int(min(CLIPS_MAX, max(CLIPS_MIN, length / CLIP_EVERY_S)))


def steps(stage: str, d: Path, audio: Path | None, script: Path, free: list[str], events: list[str]) -> list[list[str]]:
    """The commands of one stage, in order."""
    s = str(d)
    storyboard = py("storyboard.py", s, *free, "--audio", str(audio or ""), *events)
    return {
        "align": [py("align_script.py", str(audio), str(script), s)],
        "sentences": [py("sentences.py", s, str(script))],
        "case": [py("case.py", s, str(script))],
        "harvest": [py("harvest.py", s, "--script", str(script), "--stages", "case,entities,period,wikipedia")],
        "caption": [py("caption.py", s)],
        "footage": [py("footage.py", "search", s), py("footage.py", "build", s, "--target", str(clip_target(d)))],
        "bible": [py("story_bible.py", s, *free)],
        "quotes": [py("quotes.py", s)],
        "storyboard": [storyboard],
        "needs": [py("search_plan.py", s), py("harvest.py", s, "--stages", "needs"), py("caption.py", s), storyboard],
        "scenarios": [py("storyboard.py", s, "--scenarios", *free)],
        "check": [py("storyboard.py", s, "--recheck", *free), py("check_pictures.py", s, *free)],
        "ladder": [py("storyboard.py", s, "--ladder", *free), py("storyboard.py", s, "--illustrate"),
                   py("storyboard.py", s, "--recheck", *free)],
        "keyfacts": [py("storyboard.py", s, "--key-facts")],
        "review": [py("storyboard_page.py", s), py("build.py", s, "--animatic", "--out", str(d / "animatic.mp4")),
                   py("critics.py", s)],
    }[stage]


def load_env() -> dict:
    """The environment for every step, with the project's .env (API keys) added if present."""
    env = dict(os.environ)
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return env


def run(stage: str, commands: list[list[str]], d: Path, env: dict) -> None:
    log = d / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as out:
        for cmd in commands:
            out.write(f"$ {' '.join(cmd[1:])}\n")
            out.flush()
            done = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
            if done.returncode != 0:
                tail = log.read_text(encoding="utf-8").splitlines()[-12:]
                raise SystemExit(f"{stage} failed (exit {done.returncode}); see {log}\n" + "\n".join(tail))
    lines = [x for x in log.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("$")]
    print(f"  {stage}: done in {time.monotonic() - started:.0f}s" + (f" | {lines[-1][:110]}" if lines else ""))


def state_of(d: Path) -> dict:
    path = d / "pipeline.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"done": []}


def save_state(d: Path, state: dict) -> None:
    (d / "pipeline.json").write_text(json.dumps(state, indent=1), encoding="utf-8")


def prepare(args) -> None:
    d = Path(args.video_dir)
    d.mkdir(parents=True, exist_ok=True)
    state = state_of(d)
    audio = Path(args.audio or state.get("audio") or "").expanduser().resolve() if (args.audio or state.get("audio")) else None
    script_src = Path(args.script or state.get("script_source") or "").expanduser()
    if audio is None or not audio.exists():
        raise SystemExit("--audio: the narration file is needed (once; it is remembered in pipeline.json)")
    script = d / "script.txt"
    if args.script:
        if not script_src.exists():
            raise SystemExit(f"--script: {script_src} not found")
        if script_src.resolve() != script.resolve():
            shutil.copyfile(script_src, script)
    if not script.exists():
        raise SystemExit("--script: the narration's text is needed (once)")
    state.update({"audio": str(audio), "script_source": str(script_src.resolve()) if args.script else state.get("script_source")})
    free = [] if args.allow_fair_use else ["--free-only"]
    events = ["--events", str(d / "events_mapped.json")] if (d / "events_mapped.json").exists() else []
    stages = PREPARE
    if args.only:
        stages = [args.only]
    elif args.from_stage:
        stages = PREPARE[PREPARE.index(args.from_stage):]
        state["done"] = [s for s in state["done"] if s not in stages]
    if args.skip_footage:
        stages = [s for s in stages if s != "footage"]
    env = load_env()
    print(f"preparing {d} from {audio.name} ({len(stages)} stages)")
    for stage in stages:
        if stage in state["done"] and not args.only:
            print(f"  {stage}: already done")
            continue
        run(stage, steps(stage, d, audio, script, free, events), d, env)
        state["done"] = [s for s in state["done"] if s != stage] + [stage]
        save_state(d, state)
        if stage == "case" and args.pause_after_case:
            print(f"case profile written: read and edit {d / 'case.json'}, then rerun prepare to continue")
            return
    print(f"review: {d / 'storyboard.html'}, {d / 'animatic.mp4'}, {d / 'critic.md'}; then: documentary.py render {d}")


def render(args) -> None:
    d = Path(args.video_dir)
    out = args.out or str(d / f"{d.name}_1080p.mp4")
    cmd = py("build.py", str(d), "--out", out, *(["--sound"] if args.sound else []),
             *(["--keep-intermediates"] if args.keep_intermediates else []))
    run("render", [cmd], d, load_env())
    print(f"film: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="everything up to the review cut")
    p.add_argument("video_dir")
    p.add_argument("--audio", default="", help="the narration (wav or mp3)")
    p.add_argument("--script", default="", help="the narration's text")
    p.add_argument("--from", dest="from_stage", choices=PREPARE, default=None, help="rerun from this stage on")
    p.add_argument("--only", choices=PREPARE, default=None, help="run only this stage")
    p.add_argument("--pause-after-case", action="store_true", help="stop after case.json so it can be edited")
    p.add_argument("--skip-footage", action="store_true", help="no period film clips (photographs only)")
    p.add_argument("--allow-fair-use", action="store_true", help="also use non-free pictures (default: free only)")
    p.set_defaults(func=prepare)
    r = sub.add_parser("render", help="the full-quality film")
    r.add_argument("video_dir")
    r.add_argument("--out", default="")
    r.add_argument("--sound", action="store_true", help="mix music and effects under a bare narration")
    r.add_argument("--keep-intermediates", action="store_true")
    r.set_defaults(func=render)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

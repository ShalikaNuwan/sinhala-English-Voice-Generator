"""Critics: check a rendered cut against its storyboard, so only what fails is fixed and re-rendered.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/critics.py <video_dir> [--video animatic.mp4] [--no-vision]

Code checks, on storyboard.json:
- continuity: a beat that names a person who has photographs in the bible, and shows a portrait of
  someone else, is flagged (a face must stay the face of the person being talked about);
- rhythm: a scene's beats against its intensity (a tense scene of long holds drags; a memorial of quick
  cuts rushes);
- repetition: the same picture back within a minute.

Vision check, on the rendered video: for each scene, frames from the middle of up to three of its beats go
to a creative-director model with the scene's purpose, mood and look, which says whether the scene works on
screen and, if not, what to change. Writes `critic.json` and `critic.md`. Fix the storyboard, then rebuild;
with `build.py --keep-intermediates` a rebuild only redraws the shots that changed.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402
from storyboard import _words, identifying_words, _first_name  # noqa: E402

FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
CRITIC_MODEL = "gpt-6-astra"
REPEAT_S = 60.0
# The longest a beat should usually hold at each intensity, and the shortest a memorial beat should be.
DRAG_S = {8: 13.0, 9: 12.0, 10: 11.0}
MEMORIAL_MIN_S = 6.0


# ------------------------------------------------------------------------------------- code checks

def continuity(beats: list[dict], bible: dict) -> list[dict]:
    """Beats that name exactly one person with photographs but show a portrait of someone else."""
    people = [p for p in bible.get("people") or [] if p.get("photo_ids")]
    firsts = [(_first_name(p), id(p)) for p in bible.get("people") or []]
    owner = {pid: p for p in people for pid in p["photo_ids"]}
    issues = []
    for b in beats:
        pid = b.get("library_id")
        if pid not in owner:
            continue
        said = set(_words(b.get("narration") or ""))
        named = [p for p in people
                 if identifying_words(p, {f for f, i in firsts if f and i != id(p)}) & said]
        if len(named) == 1 and owner[pid] is not named[0] and owner[pid] not in named:
            issues.append({"kind": "continuity", "beat": b["index"], "start": b["start"],
                           "note": f"names {named[0]['name']} but shows {owner[pid]['name']} ({pid})"})
    return issues


def rhythm(scenes: list[dict], beats: list[dict]) -> list[dict]:
    by_index = {b["index"]: b for b in beats}
    issues = []
    for s in scenes:
        lengths = [by_index[i]["end"] - by_index[i]["start"] for i in s["beats"] if i in by_index]
        if not lengths:
            continue
        mid = median(lengths)
        limit = DRAG_S.get(min(10, s["intensity"])) if s["intensity"] >= 8 else None
        if limit and mid > limit:
            issues.append({"kind": "rhythm", "scene": s["id"], "start": s["start"],
                           "note": f"intensity {s['intensity']} but beats hold {mid:.1f}s on average (under {limit:.0f}s reads tighter)"})
        if s["look"] == "memorial" and mid < MEMORIAL_MIN_S:
            issues.append({"kind": "rhythm", "scene": s["id"], "start": s["start"],
                           "note": f"memorial cuts every {mid:.1f}s; give each name at least {MEMORIAL_MIN_S:.0f}s"})
    return issues


def repetition(beats: list[dict]) -> list[dict]:
    last_seen: dict[str, float] = {}
    issues = []
    for b in beats:
        pid = b.get("library_id")
        if not pid or b.get("media") == "motion":
            continue
        if pid in last_seen and b["start"] - last_seen[pid] < REPEAT_S:
            issues.append({"kind": "repetition", "beat": b["index"], "start": b["start"],
                           "note": f"{pid} again after {b['start'] - last_seen[pid]:.0f}s"})
        last_seen[pid] = b["end"]
    return issues


# -------------------------------------------------------------------------------------- vision

class SceneVerdict(BaseModel):
    works: Literal["yes", "mostly", "no"]
    notes: list[str] = Field(default_factory=list, description="What to change, most important first; empty if it works.")


VISION = """You are the creative director of a true-crime documentary channel reviewing a rough cut. You see
frames from one scene, in order, with what the scene is meant to do. Judge whether the pictures carry the
scene's purpose and mood, whether the look (colour treatment) suits it, whether any frame is confusing,
repetitive or careless with the people in it (victims are shown with dignity; never remains). Be specific
and brief: name the frame number for any change."""


def frames_for(video: Path, scene: dict, beats: dict[int, dict], count: int = 3) -> list[bytes]:
    picks = scene["beats"][:: max(1, len(scene["beats"]) // count)][:count]
    out = []
    for index in picks:
        b = beats.get(index)
        if not b:
            continue
        at = (b["start"] + b["end"]) / 2
        done = subprocess.run([FFMPEG, "-v", "error", "-ss", f"{at:.2f}", "-i", str(video), "-frames:v", "1",
                               "-vf", "scale=640:-2", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True, timeout=60)
        if done.stdout:
            out.append(done.stdout)
    return out


def judge(client, model: str, scene: dict, frames: list[bytes]) -> SceneVerdict:
    content = [{"type": "input_text", "text": (
        f"Scene '{scene['title']}' ({scene['story_time']}, {scene['setting']}). Purpose: {scene['purpose']} "
        f"Mood: {scene['mood']}. Look: {scene['look']}. Visual idea: {scene['visual_idea']}")}]
    for n, frame in enumerate(frames, start=1):
        content += [{"type": "input_text", "text": f"Frame {n}:"},
                    {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(frame).decode(),
                     "detail": "low"}]
    response = client.responses.parse(model=model, input=[{"role": "system", "content": VISION},
                                                          {"role": "user", "content": content}], text_format=SceneVerdict)
    return response.output_parsed


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--video", default="animatic.mp4")
    parser.add_argument("--model", default=CRITIC_MODEL)
    parser.add_argument("--no-vision", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    beats = {b["index"]: b for b in sb["beats"]}
    issues = continuity(sb["beats"], sb["bible"]) + rhythm(sb["scenes"], sb["beats"]) + repetition(sb["beats"])
    verdicts: dict[int, dict] = {}
    video = video_dir / args.video
    if not args.no_vision and video.exists():
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)

        def one(scene):
            frames = frames_for(video, scene, beats)
            if not frames:
                return scene["id"], None
            try:
                return scene["id"], judge(client, args.model, scene, frames).model_dump()
            except Exception as exc:  # noqa: BLE001 - one scene's review may fail; the rest go on
                return scene["id"], {"works": "unknown", "notes": [str(exc)[:120]]}

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            verdicts = {sid: v for sid, v in pool.map(one, sb["scenes"]) if v}
    (video_dir / "critic.json").write_text(json.dumps({"issues": issues, "scenes": verdicts}, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    lines = ["# Critic's report", "", f"{len(issues)} code findings; "
             f"{sum(1 for v in verdicts.values() if v['works'] == 'no')} of {len(verdicts)} scenes judged not working.", ""]
    if issues:
        lines += ["## Code findings", ""] + [f"- {clock(i['start'])} {i['kind']}: {i['note']}" for i in issues] + [""]
    if verdicts:
        lines += ["## Scenes", ""]
        for s in sb["scenes"]:
            v = verdicts.get(s["id"])
            if v and v["works"] != "yes":
                lines.append(f"- {clock(s['start'])} **{s['title']}** ({v['works']}): " + "; ".join(v["notes"]))
    (video_dir / "critic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote critic.md: {len(issues)} code findings, {len(verdicts)} scenes reviewed")


if __name__ == "__main__":
    main()

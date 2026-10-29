"""Plan targeted picture searches from the storyboard: the story's own places, objects and moments, with years.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/search_plan.py <video_dir> [--per-scene 3]

The first harvest searches for people's names and for the region in its era. That leaves a story set in a suburb,
a small Iowa city or a shop with nothing of its own, so the planner falls back on unrelated city views. This
reads each scene of storyboard.json (its setting, story time, visual idea and opening narration) and the
story bible's places, and asks a model for the archive searches an picture researcher would run for it:
the place in its era ("<town> <state> 1970s"), the kinds of places ("1970s pharmacy interior"), the
objects and actions the narration turns on ("1970s cash register"). The searches are merged into
library/needs.json, which `harvest.py --stages needs` runs across Wikimedia Commons, Openverse and the
Internet Archive (public domain and CC0 only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402


class Search(BaseModel):
    scene: int
    query: str = Field(description="Two to six plain words an archive search engine understands.")
    kind: Literal["place", "setting", "object", "action"]


class Searches(BaseModel):
    searches: list[Search]


SYSTEM = """You are the picture researcher for a documentary built only from real archival photographs and film
(public domain archives: Wikimedia Commons, the Library of Congress and US National Archives on Flickr
Commons, the Internet Archive). For each scene, write the searches that would find pictures that truly
belong to it:
- place: the actual town, street, building or institution, with its era ("<town> <state> 1970s",
  "<city> downtown 1960s", "<county> jail 1970s"), using the names in the scene.
- setting: the kind of place the narration is in, of the right era, when the actual place is unlikely to be
  photographed ("1970s pharmacy interior", "1960s fast food restaurant", "suburban ranch house winter").
- object: an object the narration turns on ("1970s cash register", "rotary telephone", "evidence bag").
- action: what is happening, as a photographable thing ("police officers searching house", "car headlights
  snow night").
Never search for a victim or a private person by name; never for anything gruesome. Keep each query short
and concrete. Give up to the number of searches asked per scene, most useful first."""


def plan_searches(client, model: str, scenes: list[dict], sentences: list[dict], places: list[dict],
                  per_scene: int) -> list[Search]:
    blocks = []
    for s in scenes:
        text = " ".join(x["text"] for x in sentences[s["first"]:s["last"] + 1])[:500]
        blocks.append(f"[{s['id']}] {s['title']} | where: {s.get('setting')} | when: {s.get('story_time')} | "
                      f"visual idea: {s.get('visual_idea')} | narration: {text}")
    known = "\n".join(f"  {p['name']} ({p.get('era', '')})" for p in places)
    user = (f"Places of the story:\n{known}\n\nUp to {per_scene} searches for each of these scenes:\n\n"
            + "\n\n".join(blocks))
    response = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                          {"role": "user", "content": user}], text_format=Searches)
    return response.output_parsed.searches


def merge_needs(existing: list[dict], searches: list[Search]) -> list[dict]:
    """Searches merged into the needs list by query (case-insensitive), each remembering its scenes."""
    by_query = {n["query"].lower(): dict(n) for n in existing}
    for s in searches:
        key = s.query.strip().lower()
        if not key:
            continue
        need = by_query.setdefault(key, {"query": s.query.strip(), "beats": [], "scenes": [], "kind": s.kind})
        if s.scene not in need.setdefault("scenes", []):
            need["scenes"].append(s.scene)
    return list(by_query.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--per-scene", type=int, default=3)
    parser.add_argument("--scenes", default="", help="only these scene ids, e.g. 3,4,22")
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    wanted = {int(x) for x in args.scenes.split(",") if x.strip()}
    scenes = [s for s in sb["scenes"] if (not wanted or s["id"] in wanted) and s.get("scenario") != "roll_call"]
    client = OpenAI(api_key=settings.openai_api_key)
    searches: list[Search] = []
    for k in range(0, len(scenes), 25):                        # 25 scenes a call keeps each answer focused
        searches += plan_searches(client, args.model, scenes[k:k + 25], sentences, sb["bible"].get("places", []),
                                  args.per_scene)
    path = video_dir / "library" / "needs.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    merged = merge_needs(existing, searches)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(f"{len(searches)} searches for {len(scenes)} scenes ({dict(Counter(s.kind for s in searches))}); "
          f"needs.json now holds {len(merged)} distinct searches")


if __name__ == "__main__":
    main()

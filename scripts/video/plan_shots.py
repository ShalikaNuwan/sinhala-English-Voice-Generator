"""Turn an aligned narration into a shot list: what is on screen, second by second.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/plan_shots.py <video_dir> [--model gpt-6-sol]

Reads `lines.json` from the align stage, groups the lines into shots at the narrator's own pauses,
and asks a text model what each shot should show. Writes `shots.json` (the contract in
docs/superpowers/specs/2026-10-14-documentary-video-design.md) and `shots.md`, a readable plan for
the owner. Nothing is fetched or generated here: this is the cheap checkpoint before the expensive
stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings  # noqa: E402

MIN_SHOT_S = 7.0
MAX_SHOT_S = 12.0
BATCH = 12          # shots planned per model call, with the neighbours as context
CONTEXT_SHOTS = 3   # shots of story shown before and after the batch


class MapPoint(BaseModel):
    name: str
    x: float = Field(description="0 is the left edge of the panel, 1 the right.")
    y: float = Field(description="0 is the top of the panel, 1 the bottom.")


class TimelineEvent(BaseModel):
    label: str = Field(description="The date, short: 'Dec 11, 1978'.")
    text: str = Field(description="What happened, under eight words.")


class ChartNode(BaseModel):
    name: str
    role: str = Field(default="", description="Two or three words: 'contractor', 'missing teenager'.")


class ChartEdge(BaseModel):
    source: int = Field(description="Index into nodes, starting at 0.")
    target: int = Field(description="Index into nodes, starting at 0.")
    label: str = Field(default="", description="How they are connected, two or three words.")


class PlannedShot(BaseModel):
    index: int = Field(description="The shot number given in the prompt. Copy it exactly.")
    kind: Literal["archival", "stock", "generated", "card", "chart", "map"]
    query: str = Field(default="", description="Search terms, for archival or stock only.")
    prompt: str = Field(default="", description="Image prompt, for generated only. Never a real person's likeness.")
    text: str = Field(default="", description="On-screen words, for card only.")
    subtext: str = Field(default="", description="Smaller second line, for card only.")
    motion: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "still"] = "zoom_in"
    reason: str = Field(default="", description="Half a sentence: why this picture for these words.")
    points: list[MapPoint] = Field(default_factory=list, description="For map only: two to six places, arranged as they really lie.")
    events: list[TimelineEvent] = Field(default_factory=list, description="For a timeline chart only: two to eight events in order.")
    nodes: list[ChartNode] = Field(default_factory=list, description="For a relationship chart only: two to eight people.")
    edges: list[ChartEdge] = Field(default_factory=list, description="For a relationship chart only: how the nodes connect.")


class PlannedBatch(BaseModel):
    shots: list[PlannedShot]


SYSTEM = """You are the picture editor of a true-crime documentary. You are given the narration, cut
into numbered shots with their durations, and you decide what the viewer sees during each one.

Choose one kind per shot:
- archival: a real, documented photograph that Wikimedia Commons plausibly holds. Use it for the
  convicted killer, public buildings, cities, police departments, courts, newspapers of record,
  famous landmarks. Write `query` as the search terms. Do NOT choose archival for victims, private
  individuals, family members, crime scenes or anything a news agency would own: Commons will not
  have it and a press photo cannot be used.
- stock: ordinary, unowned imagery that carries the mood. Snow falling on a street, an empty
  pharmacy aisle, headlights on a wet road, a suburban house at dusk, bare winter trees, a corridor,
  a telephone, a clock. Write `query` as plain search terms, two to five words.
- generated: an atmospheric recreation when nothing real or stock would serve. Write `prompt` as a
  full image description, cinematic, 1970s or period-appropriate, documentary stills look.
  ABSOLUTE RULE: never describe a real named person, a face that is meant to be someone, a victim,
  or a recognisable crime scene. Places, objects, weather, hands, silhouettes, empty rooms only.
- card: words on screen. A date, a location, a chapter title, a short pulled quotation. Use for the
  opening, for each jump in time or place, and sparingly for a line that lands hard. Fill `text`
  and optionally `subtext`.
- chart: a timeline or a relationship diagram. Use rarely, only where the narration itself is
  laying out a sequence of dates or explaining how people connect. For a timeline fill `events`
  with two to eight entries in order. For a relationship diagram fill `nodes` with two to eight
  people and `edges` with how they connect. A chart with neither is useless, so fill one of them.
- map: a schematic map of the places just mentioned. Use rarely. Fill `points` with two to six
  places and give each an x and y between 0 and 1, arranged roughly as they really lie relative to
  one another. A map with no points is useless, so always fill them.

Rules:
- Match the picture to the words being spoken in that shot. If the narration says it is snowing in
  December, show snow. If it names a building, show that building.
- Vary the kinds. A run of more than three identical kinds is a failure. Aim overall for roughly
  half stock, a fifth generated, a fifth card, and the rest archival, chart and map.
- Motion: zoom_in for tension or arrival, zoom_out for reveal or aftermath, pans for landscapes and
  streets, still for cards and charts.
- Never invent a fact that is not in the narration.
- A `query` is what you would type into a photo search: plain words only. Never name the archive
  ("Wikimedia Commons", "Getty"), never add "photo" or "image", never add a licence word.
- Return exactly one entry per shot you were asked to plan, with the index copied exactly."""


def group_into_shots(lines: list[dict]) -> list[dict]:
    """Cut the narration into shots, preferring the narrator's longer pauses as the cut points."""
    shots: list[dict] = []
    current: list[dict] = []
    for position, line in enumerate(lines):
        current.append(line)
        span = current[-1]["end"] - current[0]["start"]
        if span < MIN_SHOT_S:
            continue
        gap_after = (lines[position + 1]["start"] - line["end"]) if position + 1 < len(lines) else 99.0
        # Cut here when the narrator takes a breath, or when the shot has run long enough anyway.
        if gap_after >= 0.9 or span >= MAX_SHOT_S:
            shots.append({"lines": current})
            current = []
    if current:
        shots.append({"lines": current})

    spans = []
    for shot in shots:
        start, end = shot["lines"][0]["start"], shot["lines"][-1]["end"]
        narration = " ".join(line["text"] for line in shot["lines"])
        # One long paragraph is still several pictures: hold nothing past MAX_SHOT_S.
        pieces = max(1, ceil((end - start) / MAX_SHOT_S))
        step = (end - start) / pieces
        for piece in range(pieces):
            spans.append({
                "start": round(start + piece * step, 2),
                "end": round(start + (piece + 1) * step, 2),
                "narration": narration,
                "part": (piece + 1, pieces),
            })

    result = []
    for number, span in enumerate(spans, start=1):
        result.append({"index": number, **span})
    # Shots butt against each other so the picture never drops out between them.
    for earlier, later in zip(result, result[1:]):
        earlier["end"] = later["start"]
    return result


def plan_batch(client, model: str, batch: list[dict], before: list[dict], after: list[dict]) -> list[PlannedShot]:
    def render(shots, label):
        if not shots:
            return ""
        body = "\n".join(f"  [{s['index']}] {s['narration'][:300]}" for s in shots)
        return f"{label}:\n{body}\n\n"

    user = (
        render(before, "Story just before (context only, do not plan these)")
        + "Plan these shots:\n"
        + "\n".join(
            f"  [{s['index']}] ({s['end'] - s['start']:.1f}s)"
            + (f" [picture {s['part'][0]} of {s['part'][1]} over this passage]" if s.get("part", (1, 1))[1] > 1 else "")
            + f" {s['narration']}"
            for s in batch
        )
        + "\n\n"
        + render(after, "Story just after (context only, do not plan these)")
    )
    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        text_format=PlannedBatch,
    )
    return response.output_parsed.shots


def to_contract(shot: dict, planned: PlannedShot | None) -> dict:
    entry = {
        "index": shot["index"],
        "start": shot["start"],
        "end": shot["end"],
        "narration": shot["narration"],
        "kind": "stock",
        "query": "",
        "prompt": "",
        "text": "",
        "subtext": "",
        "motion": "zoom_in",
        "reason": "",
        "points": [],
        "events": [],
        "nodes": [],
        "edges": [],
        "asset": None,
        "licence": None,
        "attribution": None,
        "source_url": None,
    }
    if planned:
        entry.update({
            "kind": planned.kind, "query": planned.query, "prompt": planned.prompt,
            "text": planned.text, "subtext": planned.subtext, "motion": planned.motion,
            "reason": planned.reason,
            "points": [p.model_dump() for p in planned.points],
            "events": [e.model_dump() for e in planned.events],
            "nodes": [n.model_dump() for n in planned.nodes],
            "edges": [[e.source, e.target, e.label] for e in planned.edges],
        })
        # A drawn shot with nothing to draw becomes a still instead of an empty frame.
        if planned.kind == "map" and not planned.points:
            entry.update({"kind": "stock", "query": "road map of illinois", "reason": "map planned without places"})
        if planned.kind == "chart" and not (planned.events or planned.nodes):
            entry.update({"kind": "stock", "query": "case files on a desk", "reason": "chart planned without content"})
    else:
        # Nothing came back for this shot; a quiet atmospheric still is the safe default.
        entry["query"] = "dark empty road at night"
        entry["reason"] = "planner returned nothing for this shot"
    return entry


def readable(shots: list[dict], duration: float) -> str:
    from collections import Counter
    kinds = Counter(s["kind"] for s in shots)
    out = [
        "# Shot plan", "",
        f"{len(shots)} shots over {duration / 60:.1f} minutes, "
        f"average {duration / max(1, len(shots)):.1f}s each.", "",
        "Kinds: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()), "",
        "| # | time | kind | on screen | narration |", "|---|---|---|---|---|",
    ]
    for shot in shots:
        m, s = divmod(int(shot["start"]), 60)
        shown = shot["query"] or shot["prompt"] or " / ".join(x for x in (shot["text"], shot["subtext"]) if x) or "-"
        narration = shot["narration"].replace("|", "/")[:110]
        out.append(f"| {shot['index']} | {m}:{s:02d} | {shot['kind']} | {shown[:80]} | {narration} |")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--audio", default="")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    lines = json.loads((video_dir / "lines.json").read_text(encoding="utf-8"))
    shots = group_into_shots(lines)
    duration = max(line["end"] for line in lines)
    shots[-1]["end"] = round(duration, 2)
    print(f"{len(lines)} lines -> {len(shots)} shots over {duration / 60:.1f} min; planning with {args.model}")

    client = OpenAI(api_key=settings.openai_api_key)
    batches = [shots[i:i + BATCH] for i in range(0, len(shots), BATCH)]

    def one(number_and_batch):
        number, batch = number_and_batch
        start = max(0, number * BATCH - CONTEXT_SHOTS)
        before = shots[start:number * BATCH]
        after = shots[number * BATCH + len(batch):number * BATCH + len(batch) + CONTEXT_SHOTS]
        for attempt in range(3):
            try:
                planned = plan_batch(client, args.model, batch, before, after)
                print(f"  planned shots {batch[0]['index']}-{batch[-1]['index']}", flush=True)
                return planned
            except Exception as exc:  # noqa: BLE001 - a batch may fail; the rest still plan
                if attempt == 2:
                    print(f"  FAILED shots {batch[0]['index']}-{batch[-1]['index']}: {str(exc)[:120]}", flush=True)
                    return []
        return []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        planned_batches = list(pool.map(one, enumerate(batches)))

    by_index = {p.index: p for batch in planned_batches for p in batch}
    entries = [to_contract(shot, by_index.get(shot["index"])) for shot in shots]
    missing = sum(1 for e in entries if e["reason"] == "planner returned nothing for this shot")

    audio = args.audio or str(Path("~/Downloads/ElevenLabs_Cacy.wav").expanduser())
    (video_dir / "shots.json").write_text(json.dumps({
        "audio": audio, "duration_s": round(duration, 2),
        "width": 1920, "height": 1080, "fps": 30, "subtitles": "sidecar",
        "shots": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "shots.md").write_text(readable(entries, duration), encoding="utf-8")
    print(f"wrote shots.json and shots.md; {len(entries)} shots, {missing} unplanned")


if __name__ == "__main__":
    main()

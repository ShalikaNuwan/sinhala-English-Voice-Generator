"""Describe every library picture with a vision model, so the planner can choose by content.

    .venv/bin/python scripts/video/caption.py data/video/gacy [--model gpt-6-sol] [--workers 6] [--redo]

Fills caption, subjects, places, era, colour, kind, graphic and quality for each item in
library.json that has no caption yet. Resumable: library.json is rewritten atomically after every
picture, re-read under a lock so nothing written meanwhile is lost. graphic: true marks a picture
that shows human remains or bodies; it stays on disk and is never planned.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.config import settings  # noqa: E402

CAPTION_FIELDS = ("caption", "subjects", "places", "era", "colour", "kind", "graphic", "quality")
FALLBACK_MODELS = ("gpt-5.5", "gpt-4.1")
MAX_SIDE = 1024

SYSTEM = (
    "You catalogue archival pictures for a true-crime documentary about John Wayne Gacy (Chicago and "
    "its north-west suburbs, 1960s-1994). Describe exactly what is visible. caption: one factual "
    "sentence saying what the picture shows (who or what, where, when if known). subjects: named "
    "people visibly shown, only when the title, filename or source makes their identity clear; never "
    "guess who a face belongs to; empty when unsure. places: named places shown or clearly the "
    "setting per the title or source. era: a year or a decade (e.g. '1978', '1970s', '2010s'). "
    "colour: bw or colour. kind: photo, document, newspaper, diagram or map. graphic: true if human "
    "remains, a body, a body bag being carried, or a corpse is visible; false otherwise. quality: "
    "1-5, how usable the picture is as full-screen material in a 1080p documentary (5 = sharp, "
    "large, well composed; 1 = tiny, illegible or irrelevant)."
)


class Caption(BaseModel):
    caption: str
    subjects: list[str] = Field(default_factory=list)
    places: list[str] = Field(default_factory=list)
    era: str
    colour: Literal["bw", "colour"]
    kind: Literal["photo", "document", "newspaper", "diagram", "map"]
    graphic: bool
    quality: int = Field(ge=1, le=5)


def merge_caption(item: dict, result: dict) -> dict:
    """A copy of item with the caption fields filled; harvest fields are never overwritten."""
    merged = dict(item)
    for key in (*CAPTION_FIELDS, "caption_model"):
        if key in result:
            merged[key] = result[key]
    return merged


def pending(items: list[dict], redo: bool = False) -> list[dict]:
    return [item for item in items if redo or not item.get("caption")]


def image_data_url(path: Path, max_side: int = MAX_SIDE) -> str:
    """The picture as a JPEG data URL, longest side at most max_side."""
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def write_atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def context_text(item: dict) -> str:
    return (
        f"Title: {item.get('title')}\nSource: {item.get('source')}\n"
        f"Source URL: {item.get('source_url')}\nAttribution: {item.get('attribution')}\n"
        f"Found while searching for: {item.get('found_by')}"
    )


def describe(client, model: str, item: dict, root: Path) -> dict:
    response = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "input_text", "text": context_text(item)},
                {"type": "input_image", "image_url": image_data_url(root / item["file"]), "detail": "high"},
            ]},
        ],
        text_format=Caption,
    )
    if not response.output_parsed:
        raise RuntimeError("no structured caption returned")
    return response.output_parsed.model_dump()


def choose_model(client, preferred: str, sample: Path) -> str:
    """The first model on this account that answers with an image attached."""
    url = image_data_url(sample, max_side=256)
    for model in (preferred, *FALLBACK_MODELS):
        try:
            client.responses.create(model=model, input=[{"role": "user", "content": [
                {"type": "input_text", "text": "Reply with one word: what is this?"},
                {"type": "input_image", "image_url": url},
            ]}])
            return model
        except Exception as exc:  # noqa: BLE001 - try the next model
            print(f"  {model} refused an image: {str(exc)[:120]}", flush=True)
    raise RuntimeError("no listed model accepts image input")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--only", default="", help="comma-separated ids")
    args = parser.parse_args()

    from openai import OpenAI

    root = Path(args.video_dir)
    path = root / "library.json"
    items = json.loads(path.read_text(encoding="utf-8"))["items"]
    todo = pending(items, redo=args.redo)
    if args.only:
        wanted = set(args.only.split(","))
        todo = [i for i in items if i["id"] in wanted]
    if not todo:
        print("nothing to caption")
        return

    client = OpenAI(api_key=settings.openai_api_key)
    model = choose_model(client, args.model, root / todo[0]["file"])
    print(f"captioning {len(todo)} of {len(items)} with {model}", flush=True)
    lock = threading.Lock()
    done = {"n": 0, "failed": 0}

    def one(item: dict) -> None:
        result = None
        for attempt in range(3):
            try:
                result = describe(client, model, item, root)
                break
            except Exception as exc:  # noqa: BLE001 - retried, then reported
                if attempt == 2:
                    with lock:
                        done["failed"] += 1
                    print(f"  FAILED {item['id']}: {str(exc)[:140]}", flush=True)
                    return
        result["caption_model"] = model
        with lock:
            # Re-read so a concurrent harvest rerun or another worker's write is never lost.
            current = json.loads(path.read_text(encoding="utf-8"))
            for n, entry in enumerate(current["items"]):
                if entry["id"] == item["id"]:
                    current["items"][n] = merge_caption(entry, result)
            write_atomic(path, current)
            done["n"] += 1
            flag = " GRAPHIC" if result["graphic"] else ""
            print(f"  {done['n']}/{len(todo)} {item['id']}{flag}: {result['caption'][:100]}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    print(f"captioned {done['n']}, failed {done['failed']}, model {model}")


if __name__ == "__main__":
    main()

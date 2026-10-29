"""The story bible: one read of the whole narration before anything is planned.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/story_bible.py <video_dir> [--free-only] [--no-review]

Reads `sentences.json` (from sentences.py) and the picture library, and asks a long-context model to
read the entire script at once and write down what an editor needs before cutting a frame: who the
people are (and which library photographs really show them), the places and their years, the dated
timeline, the chapters with their purpose, tone and tension, the motifs worth repeating on purpose, and
the sensitivity rules. A second model reviews the bible as a creative director and the first revises
it once. Code then repairs the chapters so they cover every sentence exactly once, in order, and drops
any photograph id that is not in the library. Writes `bible.json`.
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

REVIEW_MODEL = "gpt-6-astra"
LOOKS = ("warm_memory", "cold_dread", "investigation", "revelation", "courtroom", "memorial")
Look = Literal["warm_memory", "cold_dread", "investigation", "revelation", "courtroom", "memorial"]


class Person(BaseModel):
    name: str = Field(description="Full name as the narration gives it.")
    role: str = Field(description="Who they are in this story, three to eight words.")
    aliases: list[str] = Field(default_factory=list, description="Other names the narration uses (nickname, surname only).")
    importance: int = Field(description="3 = central, 2 = recurring, 1 = mentioned briefly.")
    victim: bool = Field(default=False, description="True for a victim of the crimes.")
    photo_ids: list[str] = Field(default_factory=list, description="Library photograph ids that really show this person.")


class Place(BaseModel):
    name: str
    era: str = Field(description="The years this place matters in the story, e.g. '1978'.")
    description: str = Field(description="What it is and why it matters, one sentence.")
    photo_ids: list[str] = Field(default_factory=list, description="Library ids that show this place.")


class Event(BaseModel):
    date: str = Field(description="As exact as the narration states, e.g. 'December 11, 1978' or '1968'.")
    what: str = Field(description="One sentence.")
    sentence: int = Field(description="The number of the sentence that tells it.")


class Chapter(BaseModel):
    title: str = Field(description="A working title for the editor, two to five words.")
    first: int = Field(description="First sentence number of the chapter.")
    last: int = Field(description="Last sentence number of the chapter.")
    purpose: str = Field(description="What this chapter does for the viewer, one sentence.")
    tone: str = Field(description="The feeling, a few words, e.g. 'ordinary, uneasy'.")
    intensity: int = Field(description="0 calm to 10 the film's peak.")
    era: str = Field(description="When the events of this chapter happen, e.g. '1942-1960' or 'December 1978'.")
    look: Look = Field(description="The visual treatment that suits this chapter.")


class Motif(BaseModel):
    name: str
    description: str = Field(description="What it is and when to bring it back.")
    photo_ids: list[str] = Field(default_factory=list)


class Bible(BaseModel):
    logline: str = Field(description="The story in two sentences.")
    people: list[Person]
    places: list[Place]
    timeline: list[Event]
    chapters: list[Chapter]
    motifs: list[Motif] = Field(default_factory=list)
    sensitivities: list[str] = Field(default_factory=list,
                                     description="Rules for showing this story with care, one line each.")


class Review(BaseModel):
    verdict: Literal["approve", "revise"]
    notes: list[str] = Field(description="Specific changes to the bible, most important first.")
    narration_notes: list[str] = Field(default_factory=list, description=(
        "Concerns about the narration itself (facts, wording) for the owner to consider in a future edit. "
        "These do not change the bible."))


SYSTEM = """You are the story editor of a long-form true-crime documentary about a real case, narrated as
an investigation. Before any picture is chosen you read the WHOLE narration and write the story bible
the film will be built from. Everything must come from the narration and the picture library; invent
nothing.

People: everyone the narration names who matters to the story, with their role. Link a person to a
library photograph only when the library entry really shows that person (its people column names
them). Mark victims.

Places and timeline: the places the story happens in, with their years, and the dated events in
order, each with the number of the sentence that tells it.

Chapters: divide the narration into chapters the way an editor would, where the story turns: a cold
open, background, each phase of the crimes and the investigation, the arrest, the trial, the
aftermath, a closing. Chapters run in order, cover every sentence exactly once, and are usually 3 to
10 minutes long (a 90-minute film has 10 to 18 chapters). Give each its purpose, tone, intensity (0-10;
only one or two chapters reach 9-10), the era of its events, and the look that suits it:
- warm_memory: childhood, ordinary life, the time before; faded warm film.
- cold_dread: tension building, suspicion, a disappearance; cold and desaturated.
- investigation: police work, evidence, records, interviews; clear, neutral-cool.
- revelation: the turning point when the truth comes out; high contrast, near monochrome with crimson.
- courtroom: trial, sentencing, legal aftermath; muted and steady.
- memorial: remembering the victims; soft black and white, no effects.

Motifs: images that recur in the story and should come back on purpose (a house, an object, a
photograph), with library ids when the library has them.

Sensitivities: how to show this story with care. Victims are shown with dignity and only while the
narration speaks of them; never human remains; no sensational treatment of violence."""


REVIEW = """You are the creative director of a true-crime documentary channel. The narration is final,
recorded audio: it cannot be changed, and the film is cut to it. The story bible is the plan for the
pictures and structure of that film. Review the bible against the narration and the picture library:
- that the chapters fall where the story really turns;
- that tone, intensity and look fit each chapter;
- that the people and places are right and complete;
- that each photograph is linked only to people its caption shows (the captions are given);
- that the sensitivity rules govern what appears ON SCREEN (pictures, text, effects) with care.
Put changes to the bible in `notes`. Put any concern about the narration itself (a fact, a date, a word
choice) in `narration_notes` for the owner; never ask the bible to rewrite the narration, and never
remove a photograph link that its caption supports."""


def numbered(sentences: list[dict]) -> str:
    """The narration as numbered sentences with a minute mark, for the model to read in one go."""
    lines = []
    for s in sentences:
        m, sec = divmod(int(s["start"]), 60)
        lines.append(f"({s['index']}) [{m}:{sec:02d}] {s['text']}")
    return "\n".join(lines)


def library_index(items: list[dict]) -> str:
    """Who and where the library can show, with the photograph ids and what each picture shows, so links
    from people to photographs can be checked against the captions."""
    by_person: dict[str, list[str]] = {}
    by_place: dict[str, list[str]] = {}
    for item in items:
        for name in item.get("subjects") or []:
            by_person.setdefault(name, []).append(item["id"])
        for name in item.get("places") or []:
            by_place.setdefault(name, []).append(item["id"])
    people = "\n".join(f"  {name}: {', '.join(ids[:12])}" for name, ids in sorted(by_person.items()))
    places = "\n".join(f"  {name}: {', '.join(ids[:12])}" for name, ids in sorted(by_place.items()))
    captions = "\n".join(f"  {i['id']} | {', '.join(i.get('subjects') or []) or '-'} | {(i.get('caption') or '')[:140]}"
                         for i in items)
    return (f"People the library shows (name: photograph ids):\n{people}\n\nPlaces the library shows:\n{places}"
            f"\n\nThe photographs (id | people | what it shows):\n{captions}")


def cover_range(spans: list[tuple[int, int, object]], first: int, last: int) -> list[tuple[int, int, object]]:
    """Make spans cover first..last exactly once, contiguous and in order.

    Overlaps are trimmed from the later span, a gap is given to the span before it, and the last span
    runs to the end. Used for chapters, scenes and beats alike.
    """
    ordered = sorted((s for s in spans if s[1] >= first and s[0] <= last), key=lambda s: (s[0], s[1]))
    out: list[tuple[int, int, object]] = []
    cursor = first
    for a, b, obj in ordered:
        a, b = max(a, cursor), min(b, last)
        if b < a:
            continue
        if a > cursor:
            if out:
                out[-1] = (out[-1][0], a - 1, out[-1][2])
            else:
                a = cursor
        out.append((a, b, obj))
        cursor = b + 1
    if not out:
        return [(first, last, None)]
    if cursor <= last:
        out[-1] = (out[-1][0], last, out[-1][2])
    return out


def tidy(bible: Bible, sentences: list[dict], library_ids: set[str]) -> dict:
    """The bible as plain data, with the chapters made whole and only real photograph ids kept."""
    data = bible.model_dump()
    last = sentences[-1]["index"]
    spans = cover_range([(c["first"], c["last"], c) for c in data["chapters"]], 0, last)
    chapters = []
    for n, (a, b, chapter) in enumerate(spans, start=1):
        chapter = dict(chapter or {"title": "Untitled", "purpose": "", "tone": "", "intensity": 5, "era": "",
                                   "look": "investigation"})
        chapter.update({"id": n, "first": a, "last": b, "start": sentences[a]["start"], "end": sentences[b]["end"],
                        "intensity": max(0, min(10, int(chapter.get("intensity", 5))))})
        chapters.append(chapter)
    data["chapters"] = chapters
    for group in ("people", "places", "motifs"):
        for entry in data[group]:
            entry["photo_ids"] = [p for p in entry.get("photo_ids") or [] if p in library_ids]
    data["timeline"] = [e for e in data["timeline"] if 0 <= e["sentence"] <= last]
    return data


def write_bible(client, model: str, sentences: list[dict], items: list[dict], notes: list[str] | None = None,
                draft: dict | None = None) -> Bible:
    user = library_index(items) + "\n\nThe narration:\n" + numbered(sentences)
    if draft and notes:
        user += ("\n\nYour previous bible:\n" + json.dumps(draft, ensure_ascii=False)
                 + "\n\nThe creative director asks for these changes:\n- " + "\n- ".join(notes)
                 + "\n\nWrite the revised bible. The narration is final audio and is not being changed; keep every "
                   "photograph link that the captions support, and keep the sensitivity rules about what is shown "
                   "on screen.")
    response = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                          {"role": "user", "content": user}], text_format=Bible)
    return response.output_parsed


def review_bible(client, model: str, sentences: list[dict], items: list[dict], bible: Bible) -> Review:
    user = ("The narration:\n" + numbered(sentences) + "\n\n" + library_index(items)
            + "\n\nThe story bible:\n" + json.dumps(bible.model_dump(), ensure_ascii=False))
    response = client.responses.parse(model=model, input=[{"role": "system", "content": REVIEW},
                                                          {"role": "user", "content": user}], text_format=Review)
    return response.output_parsed


def load_library(video_dir: Path, free_only: bool) -> list[dict]:
    from plan_edit import apply_reviews, free_to_use, usable
    items = json.loads((video_dir / "library.json").read_text(encoding="utf-8"))["items"]
    apply_reviews(items, video_dir / "library")
    items = [i for i in items if usable(i)]
    return [i for i in items if free_to_use(i)] if free_only else items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--review-model", default=REVIEW_MODEL)
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    items = load_library(video_dir, args.free_only)
    client = OpenAI(api_key=settings.openai_api_key)
    print(f"{len(sentences)} sentences, {len(items)} library pictures; reading the whole story with {args.model}")
    bible = write_bible(client, args.model, sentences, items)
    review = None
    if not args.no_review:
        review = review_bible(client, args.review_model, sentences, items, bible)
        print(f"  {args.review_model}: {review.verdict}; {len(review.notes)} notes")
        if review.verdict == "revise" and review.notes:
            bible = write_bible(client, args.model, sentences, items, review.notes, bible.model_dump())
    data = tidy(bible, sentences, {i["id"] for i in items})
    data["review"] = review.model_dump() if review else None
    (video_dir / "bible.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote bible.json: {len(data['chapters'])} chapters, {len(data['people'])} people, "
          f"{len(data['places'])} places, {len(data['timeline'])} dated events, {len(data['motifs'])} motifs")
    for c in data["chapters"]:
        m0, s0 = divmod(int(c["start"]), 60)
        print(f"  {c['id']:2d}. {m0:3d}:{s0:02d}  {c['title']:<34} {c['look']:<14} intensity {c['intensity']}")


if __name__ == "__main__":
    main()

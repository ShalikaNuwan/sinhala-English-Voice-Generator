"""The editor's plan: which real picture runs under each line of narration, and what (little) text.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/plan_edit.py <video_dir> [--model gpt-6-sol]

Reads `lines.json` (align stage) and `library.json` (harvest and caption stages), cuts the narration
into shots at the narrator's pauses, and asks a text model to choose, for every shot, pictures from
the library that show what the words are about, ranked. Then it decides in code what a model must
not be trusted with: an image is not repeated within a few shots, a name card appears only the
first time a person is named, and an overlay whose words are not in the narration is dropped. Writes
`shots.json` (the v2 contract in docs/superpowers/specs/2026-10-14-documentary-video-design.md) and
`shots.md` for the owner to read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402
from plan_shots import (  # noqa: E402
    ChartEdge, ChartNode, MapPoint, TimelineEvent, group_into_shots,
)

BATCH = 12
CONTEXT_SHOTS = 3
SENTENCE_BATCH = 30       # sentences planned per model call in beat mode
SENTENCE_CONTEXT = 6      # sentences of story shown before and after a batch
BEAT_MAX_S = 20.0         # a beat longer than this splits at the sentence nearest its middle
BEAT_MIN_S = 3.0          # a beat shorter than this joins the one before it
FREE_LICENCE = re.compile(r"public domain|\bpdm\b|cc0|no restrictions", re.I)
REUSE_WINDOW = 12         # shots before the same picture may come back
MINUTES_PER_USE = 12.0    # a picture may appear once per this many minutes of film (and at least 4 times)
COLOUR_TARGET = 0.60      # the share of photographs in colour the cut leans toward
CLIP_EVERY_S = 40.0       # a fitting clip in a shot's choices runs if the last clip was this long ago
CLIP_MAX_GAP_S = 90.0     # longer than this without footage, a clip is placed on an atmosphere shot
BOARD_EVERY_S = 12 * 60.0 # the evidence board returns about this often, at a scene change
MAP_EVERY_S = 240.0       # an animated map at most this often
BOARD_CLEAR_END_S = 180.0 # no board in the closing minutes: the outro is not part of the case
MAP_SECONDS = (5.0, 14.0) # an animated map needs at least this long, and holds no longer
DRAWN_MAX_S = 6.5         # a chart or map longer than this reads as a slide; use a picture instead
DRAWN_PER_MINUTES = 8.0   # at most one chart or map per this many minutes of film
STAMP_REPEAT_SHOTS = 20   # the same date stamp may not come back inside this many shots


class Overlay(BaseModel):
    style: Literal["stamp", "name"]
    text: str = Field(description="stamp: the date as spoken in the narration. name: the person's full name.")
    subtext: str = Field(default="", description="stamp: the place. name: who they are, three to five words.")


class BoardPin(BaseModel):
    library_id: str = Field(description="A photograph id from the library (not a clip).")
    label: str = Field(description="Who or what it is, two to four words, as the narration names it.")
    sub: str = Field(default="", description="One short fact from the narration, under six words.")


class BoardLink(BaseModel):
    a: int = Field(description="Index into pins, starting at 0.")
    b: int = Field(description="Index into pins, starting at 0.")


class ProfileField(BaseModel):
    label: str = Field(description="NAME, AGE, ROLE, LAST SEEN, or a similar short field name.")
    value: str = Field(description="Exactly as the narration states it.")


class PlannedShot(BaseModel):
    index: int = Field(description="The shot number given in the prompt. Copy it exactly.")
    kind: Literal["archival", "period", "stock", "generated", "chart", "map", "board", "profile", "exhibit", "highlight"]
    choices: list[str] = Field(default_factory=list, description="Library ids (photographs or clip segments), best first, four to six of them.")
    query: str = Field(default="", description="Photo search words, only when no library picture fits.")
    prompt: str = Field(default="", description="Image description, only for kind generated (last resort).")
    motion: Literal["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "still"] = "zoom_in"
    transition_in: Literal["dissolve", "black", "cut"] = "dissolve"
    overlay: Overlay | None = None
    reason: str = Field(default="", description="Half a sentence: why this picture for these words.")
    points: list[MapPoint] = Field(default_factory=list)
    events: list[TimelineEvent] = Field(default_factory=list)
    nodes: list[ChartNode] = Field(default_factory=list)
    edges: list[ChartEdge] = Field(default_factory=list)
    subject_id: str = Field(default="", description="For profile, exhibit and highlight: the library photograph shown.")
    fields: list[ProfileField] = Field(default_factory=list, description="For profile only: two to four fields.")
    tag: str = Field(default="", description="For exhibit only: what the object is and where or when it is from.")
    detail: str = Field(default="", description="For highlight only: the detail inside the picture to circle.")
    label: str = Field(default="", description="For highlight only: two to four words beside the circle, or empty.")
    pins: list[BoardPin] = Field(default_factory=list, description="For kind board only.")
    links: list[BoardLink] = Field(default_factory=list, description="For kind board only.")


class PlannedBatch(BaseModel):
    shots: list[PlannedShot]


class PlannedBeat(PlannedShot):
    index: int = Field(default=0, description="Leave 0.")
    first: int = Field(description="The first sentence number of this beat.")
    last: int = Field(description="The last sentence number of this beat (the same as first for one sentence).")


# A period street of the story's main place, for a shot nothing else covers: case.json's fallback_query,
# set by use_case(video_dir) from each entry point.
FALLBACK_QUERY = "city street"
CASE_KEYWORD = ""             # the culprit's name, at the centre of the evidence board (case.json keyword)


def use_case(video_dir: Path) -> None:
    global FALLBACK_QUERY, CASE_KEYWORD
    import case
    FALLBACK_QUERY = case.fallback_query(video_dir)
    if case.path_of(video_dir).exists():
        CASE_KEYWORD = case.load(video_dir).keyword.lower()


class PlannedBeats(BaseModel):
    beats: list[PlannedBeat]


SYSTEM = """You are the editor of a true-crime documentary about a real case. The narration is cut
into numbered shots. For each shot you choose the real photograph that runs under those words, from
the picture library given below. You are building the film from actual photographs; invent nothing.

Choosing pictures:
- Match the words. When the narration is about a person, a place, a building, a document or an
  event that the library has a picture of, use that picture. Case pictures (the people, the house,
  the pharmacy, the courts, the newspapers, the documents) always beat atmosphere.
- When the narration is general (mood, reflection, time passing), use a period photograph of the
  story's own setting, in its place and era: streets, houses, cars, police, courts, the season.
- Give four to six library ids per shot in `choices`, best first. Prefer quality 3 and above.
- Colour: the film should feel alive, not a wall of grey. When a colour photograph carries the
  moment as well as a black-and-white one, rank the colour one first, and always include at least
  one colour option among your choices. Black-and-white is right for the archive itself: booking
  photographs, newspaper pages, the Tribune's case photographs. Each
  picture can only appear a few times in the whole film, so make the later choices real
  alternatives: other photographs of the same person, place or moment, or period photographs that
  carry it. Spread across the library rather than returning to the same few pictures.
- Never choose a victim's photograph except while the narration is talking about that victim.
- kind: archival for case pictures, period for 1970s setting pictures, stock for other real
  photographs. Use generated only when nothing in the library fits and a search would not either;
  then write `prompt` (cinematic documentary still, period-accurate, no identifiable faces). Aim for
  generated in fewer than one shot in ten.
- When the library has no picture of the exact subject, choose the nearest real period picture in
  the library that carries the moment: a 1978 shop counter for a store scene, a snowy 1978 street
  for a winter evening, a 1970s office for a business. A real photograph of the right time and
  kind of place always beats an empty frame or a search. Only when nothing in the library comes
  close, leave `choices` empty and write `query` as two to four plain search words.

Moving footage: the library also lists short real film clips (ids like v0007#2) with their length in
seconds. Use a clip about once every 45 to 90 seconds of film, where movement adds something: snow
falling, traffic at night, police cars, a city street, a prison yard, a courtroom. A clip must run
at least as long as the shot. Put clip ids in `choices` like photographs, and give a photograph
too, in case the clip is taken.

Tell it as an investigation. The viewer is being walked through a case, so show the evidence as the
narrator talks about it:
- profile: the first time the narration introduces a person whose photograph is in the library
  (Gacy, a victim, an investigator). `subject_id` is that photograph; `fields` are two to four facts
  the narration states right there (NAME, AGE, ROLE, LAST SEEN). Needs six seconds or more.
- exhibit: when the narration talks about a piece of evidence the library holds (the house diagram,
  the business card, a receipt, a newspaper page, a painting). `subject_id` is it; `tag` says what it
  is and where or when it comes from. Needs five seconds or more.
- highlight: when the narration points at a detail inside one of those pictures (an address, a
  name in a headline, a room on the diagram). `subject_id` and `detail` (what to circle, described
  so someone could find it), and `label` only when the picture does not already print that word
  next to the detail. Needs five seconds or more.
- map: when the narration names where places are.
Use these where they help a viewer follow the case: roughly one device every minute or two, not on
every shot. Photographs and clips carry the rest.

Evidence board (kind board): at most one in every ten minutes, where the narration lays out who is
connected to whom or recaps the case. `pins`: three to six library photographs (photograph ids,
never clips) with a short label and a fact from the narration; `links`: pairs of pins joined by red
string. A board needs a shot of at least six seconds.

Text on screen (`overlay`):
- Only facts that the narration of this shot states. Never a title, a tagline, a question or a
  summary. Most shots have no overlay.
- stamp: when the narration moves to a new date or place, the date as said (e.g. "December 11th,
  1978") and the place as `subtext`.
- name: the first time the narration introduces a named person, their name and who they are in three
  to five words ("Lead investigator, Des Plaines police").

Charts and maps: almost never. Only a map when the narration explains where places lie relative to
each other, or a timeline when it lists dates in a row. Fill `points` or `events` when you do.

Motion: zoom_in for tension and arrival, zoom_out for reveal and aftermath, pan_up or pan_down on
documents and newspapers, pans on streets and landscapes.
Transitions: dissolve by default. black where the story jumps in time or place, or after a line that
lands hard. cut only for a deliberate jolt, rarely.

Return exactly one entry per shot you were asked to plan, with the index copied exactly."""


# ---------- the catalogue the model chooses from ----------

BEAT_RULES = """The narration is given as numbered sentences, each with its length in seconds. Group
consecutive sentences into beats, then choose what runs under each beat.

Beats:
- A beat is the run of consecutive sentences about one thing: one person, one place, one object,
  one moment. One picture or clip runs under the whole beat, so the picture changes when the
  narration turns to something else, not on a clock.
- Read the sentences before and after the ones you plan, to see where an idea starts and ends.
- Never split a sentence. A beat is usually 5 to 15 seconds and never longer than 20. A single short
  sentence is its own beat only when it lands hard.
- Cover every sentence you were asked to plan exactly once, in order. `first` and `last` are the
  sentence numbers a beat spans.
"""
BEAT_SYSTEM = SYSTEM.replace(
    "The narration is cut\ninto numbered shots. For each shot you choose the real photograph that runs under those words, from\nthe picture library given below.",
    "You choose the real\nphotographs and footage that run under the narration, from the picture library given below.",
).replace(
    "Return exactly one entry per shot you were asked to plan, with the index copied exactly.",
    "Return the beats covering every sentence you were asked to plan.",
) + "\n\n" + BEAT_RULES


def free_to_use(item: dict) -> bool:
    """Public domain, CC0, or made for this film: nothing that needs a credit or carries a claim."""
    if item.get("rights") in ("generated", "original"):
        return True
    return item.get("rights") == "free" and bool(FREE_LICENCE.search(item.get("licence") or ""))


def usable(item: dict) -> bool:
    """Pictures a viewer may see: captioned, not graphic, not rejected by a verifier, and on disk."""
    return (bool(item.get("caption")) and not item.get("graphic") and not item.get("excluded")
            and bool(item.get("file")))


def apply_reviews(items: list[dict], folder: Path) -> None:
    """Fold in the verifiers' decisions, which live outside library.json because every harvest
    rewrites it: `exclusions.json` ({id: reason}) and any `verify_*.json`
    ({"exclude": {id: reason}, "subjects": {id: [names]}, "caption": {id: text}})."""
    exclude: dict[str, str] = {}
    subjects: dict[str, list[str]] = {}
    captions: dict[str, str] = {}
    plain = folder / "exclusions.json"
    if plain.exists():
        exclude.update(json.loads(plain.read_text(encoding="utf-8")))
    for review in sorted(folder.glob("verify_*.json")):
        data = json.loads(review.read_text(encoding="utf-8"))
        exclude.update(data.get("exclude") or {})
        subjects.update(data.get("subjects") or {})
        captions.update(data.get("caption") or {})
    for item in items:
        if item["id"] in exclude:
            item["excluded"], item["excluded_reason"] = True, exclude[item["id"]]
        if item["id"] in subjects:
            item["subjects"] = subjects[item["id"]]
        if item["id"] in captions:
            item["caption"] = captions[item["id"]]


def catalogue_line(item: dict) -> str:
    who = ", ".join(item.get("subjects") or []) or "-"
    where = ", ".join(item.get("places") or []) or "-"
    return (f"{item['id']} | {item.get('kind', 'photo')} {item.get('colour', '')} "
            f"{item.get('era', '')} q{item.get('quality', '?')} | {who} | {where} | {item['caption']}")


def clip_entries(clips: list[dict]) -> dict[str, dict]:
    """Every usable clip segment as a pickable entry, keyed like "v0007#2"."""
    out: dict[str, dict] = {}
    for clip in clips:
        for n, seg in enumerate(clip.get("segments") or []):
            if seg.get("graphic") or (seg.get("quality") or 0) < 2 or not seg.get("caption"):
                continue
            sid = f"{clip['id']}#{n}"
            out[sid] = {
                "id": sid, "media": "clip", "clip_id": clip["id"], "file": clip["file"],
                "in": float(seg["in"]), "out": float(seg["out"]), "length": float(seg["out"]) - float(seg["in"]),
                "caption": seg["caption"], "era": seg.get("era", ""), "quality": seg.get("quality", 3),
                "kind": "clip", "colour": None, "subjects": [], "places": [], "graphic": False,
                "source": clip.get("source"), "licence": clip.get("licence"), "attribution": clip.get("attribution"),
                "source_url": clip.get("source_url"), "rights": clip.get("rights"),
            }
    return out


def catalogue(items: list[dict], clips: dict[str, dict] | None = None) -> str:
    text = "\n".join(catalogue_line(item) for item in items if usable(item))
    if clips:
        text += "\n\nVideo clips (id | era quality | length | what it shows):\n" + "\n".join(
            f"{c['id']} | {c['era']} q{c['quality']} | {c['length']:.1f}s | {c['caption']}" for c in clips.values())
    return text


# ---------- decisions made in code, not by the model ----------

STOP = {"the", "and", "that", "with", "this", "from", "they", "were", "was", "had", "have", "his", "her",
        "their", "there", "would", "could", "into", "about", "when", "what", "which", "been", "then",
        "than", "them", "said", "says", "also", "just", "even", "only", "like", "some", "very", "more"}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower()) if w not in STOP}


def relevance(item: dict, narration: str) -> float:
    """How much a picture is about what is being said: shared words, with names and places worth more."""
    said = _tokens(narration)
    named = _tokens(" ".join((item.get("subjects") or []) + (item.get("places") or [])))
    score = 3.0 * len(said & named) + len(said & _tokens(item.get("caption") or ""))
    return score + 0.25 * (item.get("quality") or 0)


def max_uses(duration_s: float) -> int:
    return max(4, round(duration_s / 60 / MINUTES_PER_USE))


def pick_assets(shots: list[dict], library: dict[str, dict], duration_s: float | None = None) -> None:
    """Resolve each shot's ranked choices to one picture.

    No picture comes back within REUSE_WINDOW shots or appears more than max_uses times in the film:
    batches are planned separately, so without this a strong picture runs for minutes on end. When a
    shot's own choices are spent, the most relevant picture that is still free runs instead.
    """
    cap = max_uses(duration_s if duration_s else (shots[-1]["end"] if shots else 0))
    last_used: dict[str, int] = {}
    uses: dict[str, int] = {}
    pool = [i for i in library.values() if usable(i)]

    pool = [i for i in pool if i.get("media") != "clip"]  # a fallback never reaches for a random clip

    def free(pid: str, position: int) -> bool:
        if library[pid].get("media") == "clip":
            return uses.get(pid, 0) == 0  # a piece of footage is seen once
        return position - last_used.get(pid, -10**9) > REUSE_WINDOW and uses.get(pid, 0) < cap

    def fits(pid: str, shot: dict) -> bool:
        item = library[pid]
        return item.get("media") != "clip" or item["length"] >= 0.9 * (shot["end"] - shot["start"])

    shown = {"colour": 0, "photos": 0}
    clock = {"last_clip": -10**9}

    def colour_first(candidates: list[str]) -> list[str]:
        """Below the colour target, a colour picture among the first three choices moves to the front."""
        if shown["photos"] and shown["colour"] / shown["photos"] >= COLOUR_TARGET:
            return candidates
        lead = candidates[:3]
        colour = [c for c in lead if library[c].get("colour") == "colour"]
        return colour[:1] + [c for c in candidates if c not in colour[:1]] if colour else candidates

    for position, shot in enumerate(shots):
        if shot.get("kind") in ("board", "map", "chart", "profile", "exhibit", "highlight"):
            shot["library_id"] = None
            continue
        valid = [c for c in shot.get("choices") or [] if c in library and usable(library[c]) and fits(c, shot)]
        clip_due = shot["start"] - clock["last_clip"] >= CLIP_EVERY_S
        free_clips = [c for c in valid if library[c].get("media") == "clip" and free(c, position)]
        if clip_due and free_clips:
            valid = free_clips[:1] + [c for c in valid if c != free_clips[0]]
        else:
            valid = colour_first([c for c in valid if free(c, position)]) + [c for c in valid if not free(c, position)]
        chosen = next((c for c in valid if free(c, position)), None)
        if chosen is None:
            behind = shown["photos"] and shown["colour"] / shown["photos"] < COLOUR_TARGET
            ranked = sorted((i for i in pool if free(i["id"], position)),
                            key=lambda i: relevance(i, shot["narration"]) + (1.5 if behind and i.get("colour") == "colour" else 0),
                            reverse=True)
            if ranked:
                chosen = ranked[0]["id"]
        if chosen is None and pool:
            # Every picture has had its share: spread the overflow evenly, outside the window if possible.
            spaced = [i for i in pool if position - last_used.get(i["id"], -10**9) > REUSE_WINDOW] or pool
            chosen = min(spaced, key=lambda i: (uses.get(i["id"], 0), -relevance(i, shot["narration"]),
                                                last_used.get(i["id"], -10**9)))["id"]
        shot["library_id"] = chosen
        if chosen:
            last_used[chosen] = position
            uses[chosen] = uses.get(chosen, 0) + 1
            if library[chosen].get("media") != "clip":
                shown["photos"] += 1
                shown["colour"] += library[chosen].get("colour") == "colour"
            else:
                clock["last_clip"] = shot["start"]
            if shot["kind"] in ("generated", "stock") and library[chosen].get("source") != "generated":
                shot["kind"] = "archival" if library[chosen].get("subjects") else shot["kind"]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
          "october", "november", "december"}
_ORDINALS = {"first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6",
             "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10", "eleventh": "11",
             "twelfth": "12", "thirteenth": "13", "fourteenth": "14", "fifteenth": "15",
             "sixteenth": "16", "seventeenth": "17", "eighteenth": "18", "nineteenth": "19",
             "twentieth": "20", "thirtieth": "30"}


def spoken_day_numbers(narration: str) -> set[str]:
    """Day numbers the narrator said as words ("the eleventh"), so a stamp of "11" still matches."""
    return {_ORDINALS[w] for w in re.findall(r"[a-z]+", narration.lower()) if w in _ORDINALS}


def overlay_is_stated(overlay: dict, narration: str) -> bool:
    """An overlay may only show what the narration of its shot actually says."""
    said = _words(narration)
    text = overlay.get("text") or ""
    if overlay.get("style") == "name":
        surname = (re.findall(r"[a-z]+", text.lower()) or [""])[-1]
        return bool(surname) and surname in said
    # A stamp: every number in it (day, year) must be spoken, and so must the month if there is one.
    numbers = re.findall(r"\d+", text)
    months = [m for m in re.findall(r"[a-z]+", text.lower()) if m in MONTHS]
    spoken_numbers = {n.lstrip("0") for n in re.findall(r"\d+", narration)} | spoken_day_numbers(narration)
    return (all(n.lstrip("0") in spoken_numbers for n in numbers) and all(m in said for m in months)
            and bool(numbers or months))


def names_match_pictures(shots: list[dict], library: dict[str, dict]) -> None:
    """A name card says "this is that person", so it may only sit on a picture of that person.

    Over a stranger in a period photograph it would put a real name to the wrong face.
    """
    for shot in shots:
        overlay = shot.get("overlay")
        if not overlay or overlay.get("style") != "name":
            continue
        surname = (re.findall(r"[a-z]+", (overlay.get("text") or "").lower()) or [""])[-1]
        item = library.get(shot.get("library_id") or "") or {}
        shown = " ".join(item.get("subjects") or []).lower()
        if not surname or surname not in re.findall(r"[a-z]+", shown):
            shot["overlay"] = None


def thin_overlays(shots: list[dict]) -> None:
    """Keep overlays rare and true: stated in the narration, a name only once, no stamp twice nearby,
    never on two shots in a row."""
    named: set[str] = set()
    last_stamp: dict[str, int] = {}
    previous_had = False
    for position, shot in enumerate(shots):
        overlay = shot.get("overlay")
        keep = bool(overlay) and overlay_is_stated(overlay, shot["narration"])
        if keep and overlay["style"] == "name":
            key = " ".join(re.findall(r"[a-z]+", overlay["text"].lower()))
            keep = key not in named
            if keep:
                named.add(key)
        if keep and overlay["style"] == "stamp":
            key = overlay["text"].strip().lower()
            keep = position - last_stamp.get(key, -10**9) > STAMP_REPEAT_SHOTS
            if keep:
                last_stamp[key] = position
        if keep and previous_had:
            keep = False
        shot["overlay"] = overlay if keep else None
        previous_had = keep


def limit_drawn(shots: list[dict], duration_s: float) -> None:
    """Static charts only when short and rare; animated maps when they have places, time and space.

    A shot that loses its chart or map runs its picture choice instead.
    """
    allowed = max(1, int(duration_s / 60 / DRAWN_PER_MINUTES))
    used = 0
    last_map = -10**9
    for shot in shots:
        # Only maps carry points, so a map an older rule demoted is restored from them.
        if len(shot.get("points") or []) >= 2 and shot["kind"] in ("period", "archival", "stock"):
            shot["kind"] = "map"
        if shot["kind"] not in ("chart", "map"):
            continue
        length = shot["end"] - shot["start"]
        if shot["kind"] == "map":
            if len(shot.get("points") or []) >= 2 and MAP_SECONDS[0] <= length <= MAP_SECONDS[1] \
                    and shot["start"] - last_map >= MAP_EVERY_S:
                last_map = shot["start"]
                continue
        elif (shot.get("events") or shot.get("nodes")) and length <= DRAWN_MAX_S and used < allowed:
            used += 1
            continue
        shot["kind"] = "archival" if shot.get("library_id") else "period"
        if not shot.get("library_id") and not shot.get("query"):
            shot["query"] = FALLBACK_QUERY


DEVICE_MIN_S = {"profile": 6.0, "exhibit": 5.0, "highlight": 5.0}
DEVICE_EVERY_S = {"exhibit": 90.0, "highlight": 120.0}   # at least this long between two of the same
OBJECT_KINDS = {"document", "diagram", "newspaper", "map", "photo"}


def _stated(value: str, narration: str) -> bool:
    """Every number, and most words, of a field must be in the narration it sits under."""
    said = _words(narration) | spoken_day_numbers(narration)
    numbers = re.findall(r"\d+", value)
    words = [w for w in re.findall(r"[a-z]+", value.lower()) if len(w) > 3 and w not in STOP]
    if any(n.lstrip("0") not in {x.lstrip("0") for x in said} for n in numbers):
        return False
    return not words or sum(w in said for w in words) >= max(1, (len(words) + 1) // 2)


def limit_devices(shots: list[dict], library: dict[str, dict]) -> None:
    """Profiles, exhibits and highlights only where they are true, long enough, and not too frequent."""
    profiled: set[str] = set()
    last_at: dict[str, float] = {}
    for position, shot in enumerate(shots):
        kind = shot.get("kind")
        if kind not in DEVICE_MIN_S:
            continue
        item = library.get(shot.get("subject_id") or "")
        keep = bool(item) and usable(item) and item.get("media") != "clip" \
            and shot["end"] - shot["start"] >= DEVICE_MIN_S[kind]
        if keep and kind in DEVICE_EVERY_S:
            keep = shot["start"] - last_at.get(kind, -10**9) >= DEVICE_EVERY_S[kind]
        if keep and kind == "profile":
            context = (shots[position - 1]["narration"] + " " if position else "") + shot["narration"]
            fields = [f for f in shot.get("fields") or [] if f.get("value") and _stated(f["value"], context)]
            names = [f["value"] for f in fields if f.get("label", "").upper() == "NAME"]
            person = (re.findall(r"[a-z]+", names[0].lower()) or [""])[-1] if names else ""
            shown = " ".join(item.get("subjects") or []).lower()
            keep = len(fields) >= 2 and bool(person) and person in shown and person not in profiled
            if keep:
                shot["fields"] = fields
                profiled.add(person)
        if keep and kind == "exhibit":
            keep = item.get("kind") in OBJECT_KINDS and bool((shot.get("tag") or "").strip())
        if keep and kind == "highlight":
            keep = bool((shot.get("detail") or "").strip())
        if keep:
            last_at[kind] = shot["start"]
            shot["library_id"] = None
            continue
        # Not usable as a device: the shot runs its subject as a plain photograph, or its picture choices.
        if item and usable(item) and item.get("media") != "clip":
            shot["choices"] = [shot["subject_id"]] + [c for c in shot.get("choices") or [] if c != shot["subject_id"]]
        shot["kind"] = "archival"


def grow_boards(shots: list[dict], cap: int = 8) -> None:
    """Each return of the board keeps what was pinned before and adds the new, so the case builds."""
    pinned: list[dict] = []
    strings: list[tuple[str, str]] = []
    for shot in shots:
        if shot.get("kind") != "board":
            continue
        new_pins = shot.get("pins") or []
        for link in shot.get("links") or []:
            if link["a"] < len(new_pins) and link["b"] < len(new_pins):
                pair = (new_pins[link["a"]]["library_id"], new_pins[link["b"]]["library_id"])
                if pair not in strings and pair[::-1] not in strings:
                    strings.append(pair)
        for pin in new_pins:
            pinned = [p for p in pinned if p["library_id"] != pin["library_id"]] + [pin]
        pinned = pinned[-cap:]
        ids = [p["library_id"] for p in pinned]
        shot["pins"] = [dict(p) for p in pinned]
        shot["links"] = [{"a": ids.index(a), "b": ids.index(b)} for a, b in strings if a in ids and b in ids]


BOARD_MIN_S = 6.0
BOARD_PER_MINUTES = 10.0


def limit_boards(shots: list[dict], library: dict[str, dict], duration_s: float) -> None:
    """An evidence board needs three real photographs, six seconds, and rarity to land."""
    allowed = max(1, int(duration_s / 60 / BOARD_PER_MINUTES))
    used = 0
    for shot in shots:
        if shot["kind"] != "board":
            continue
        pins = [p for p in shot.get("pins") or []
                if p.get("library_id") in library and usable(library[p["library_id"]])
                and library[p["library_id"]].get("media") != "clip"]
        shot["pins"] = pins
        shot["links"] = [l for l in shot.get("links") or [] if max(l.get("a", 99), l.get("b", 99)) < len(pins)]
        if len(pins) >= 3 and shot["end"] - shot["start"] >= BOARD_MIN_S and used < allowed:
            used += 1
            continue
        shot["kind"] = "archival"


def _atmosphere(shot: dict, library: dict[str, dict]) -> bool:
    """A shot whose picture is mood, not a named person or a piece of evidence."""
    item = library.get(shot.get("library_id") or "") or {}
    return (shot.get("media", "photo") == "photo" and shot["kind"] in ("period", "stock", "archival")
            and not shot.get("overlay") and not item.get("subjects") and item.get("media") != "clip")


def insert_clips(shots: list[dict], library: dict[str, dict]) -> None:
    """Footage at least every CLIP_MAX_GAP_S: the best fitting free clip on an atmosphere shot."""
    used = {s["library_id"] for s in shots if s.get("library_id") in library
            and library[s["library_id"]].get("media") == "clip"}
    clips = [c for c in library.values() if c.get("media") == "clip" and usable(c)]
    last = 0.0
    for shot in shots:
        item = library.get(shot.get("library_id") or "") or {}
        if item.get("media") == "clip":
            last = shot["start"]
            continue
        if shot["start"] - last < CLIP_MAX_GAP_S or not _atmosphere(shot, library):
            continue
        length = shot["end"] - shot["start"]
        era = {"1970s": 1.0, "1960s": 0.5}
        fitting = [c for c in clips if c["id"] not in used and c["length"] >= 0.9 * length]
        if not fitting:
            continue
        best = max(fitting, key=lambda c: relevance(c, shot["narration"]) + era.get(c.get("era", ""), 0))
        shot["library_id"] = best["id"]
        shot["kind"] = "clip"
        used.add(best["id"])
        last = shot["start"]


def insert_boards(shots: list[dict], library: dict[str, dict]) -> None:
    """The evidence board comes back at scene changes and grows with what the case has established.

    Its pins are the people already profiled and the evidence already shown as exhibits, so every
    label is something the film has stated. The first person profiled with the case keyword (case.json,
    e.g. "Gacy") in the name sits at the centre, joined to each of the others.
    """
    established: list[dict] = []
    next_at = BOARD_EVERY_S / 2
    last_pins: list[str] = []
    end = shots[-1]["end"] if shots else 0.0
    for shot in shots:
        kind = shot.get("kind")
        if kind == "profile":
            fields = {f["label"].upper(): f["value"] for f in shot.get("fields") or []}
            sub = fields.get("ROLE") or fields.get("AGE") or fields.get("LAST SEEN") or ""
            name = fields.get("NAME", "")
            # Another murderer is mentioned for comparison; red string to him would claim a link.
            if (CASE_KEYWORD and CASE_KEYWORD in name.lower()) or not re.search(r"killer|murderer", sub, re.I):
                established.append({"library_id": shot["subject_id"], "label": name, "sub": sub[:40]})
        elif kind == "exhibit":
            words = (shot.get("tag") or "").split(",")[0].split()
            label = " ".join(words[:4])
            if not re.search(r"postcard", shot.get("tag") or "", re.I):  # scenery, not evidence
                established.append({"library_id": shot["subject_id"], "label": label, "sub": "Exhibit"})
        if kind in ("board", "profile", "exhibit", "highlight", "map", "clip") or shot["start"] < next_at:
            continue
        if shot["start"] > end - BOARD_CLEAR_END_S:  # the outro is not part of the case
            continue
        if shot.get("transition_in") != "black" or shot["end"] - shot["start"] < BOARD_MIN_S:
            continue
        pins = [p for p in established if p["library_id"] in library and usable(library[p["library_id"]])]
        seen: dict[str, dict] = {}
        for pin in pins:
            seen[pin["library_id"]] = pin
        centre = next((p for p in seen.values() if CASE_KEYWORD and CASE_KEYWORD in p["label"].lower()), None)
        others = [p for p in seen.values() if p is not centre][-6:]
        pins = ([centre] if centre else []) + others
        if len(pins) < 3 or [p["library_id"] for p in pins] == last_pins:
            continue  # too thin, or nothing new since the last board
        last_pins = [p["library_id"] for p in pins]
        shot.update({"kind": "board", "library_id": None, "pins": pins,
                     "links": [{"a": 0, "b": n} for n in range(1, len(pins))], "overlay": None})
        next_at = shot["start"] + BOARD_EVERY_S


def finish_transitions(shots: list[dict]) -> None:
    """The film opens from black; a run of hard cuts is a mistake, so no two in a row."""
    if shots:
        shots[0]["transition_in"] = "black"
    for earlier, later in zip(shots, shots[1:]):
        if earlier.get("transition_in") == "cut" and later.get("transition_in") == "cut":
            later["transition_in"] = "dissolve"


# ---------- the model ----------

def plan_batch(client, model: str, cat: str, batch: list[dict], before: list[dict], after: list[dict]) -> list[PlannedShot]:
    def context(shots, label):
        if not shots:
            return ""
        return f"{label}:\n" + "\n".join(f"  [{s['index']}] {s['narration'][:300]}" for s in shots) + "\n\n"

    user = (
        "Picture library (id | kind colour era quality | people | places | what it shows):\n"
        f"{cat}\n\n"
        + context(before, "Story just before (context only, do not plan these)")
        + "Plan these shots:\n"
        + "\n".join(
            f"  [{s['index']}] ({s['end'] - s['start']:.1f}s)"
            + (f" [picture {s['part'][0]} of {s['part'][1]} over this passage]" if s.get("part", (1, 1))[1] > 1 else "")
            + f" {s['narration']}"
            for s in batch
        )
        + "\n\n"
        + context(after, "Story just after (context only, do not plan these)")
    )
    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        text_format=PlannedBatch,
    )
    return response.output_parsed.shots


def plan_beats(client, model: str, cat: str, batch: list[dict], before: list[dict], after: list[dict]) -> list[PlannedBeat]:
    def lines(items, label):
        if not items:
            return ""
        return f"{label}:\n" + "\n".join(f"  ({s['index']}) {s['text']}" for s in items) + "\n\n"

    user = (
        "Picture library (id | kind colour era quality | people | places | what it shows):\n"
        f"{cat}\n\n"
        + lines(before, "Story just before (context only, do not plan these)")
        + "Plan these sentences:\n"
        + "\n".join(f"  ({s['index']}) [{s['end'] - s['start']:.1f}s] {s['text']}" for s in batch)
        + "\n\n"
        + lines(after, "Story just after (context only, do not plan these)")
    )
    response = client.responses.parse(
        model=model,
        input=[{"role": "system", "content": BEAT_SYSTEM}, {"role": "user", "content": user}],
        text_format=PlannedBeats,
    )
    return response.output_parsed.beats


def repair_beats(beats: list, first: int, last: int) -> list[tuple[int, int, object]]:
    """Make a batch's beats cover first..last exactly once, contiguous and in order."""
    ordered = sorted((b for b in beats if b.last >= first and b.first <= last), key=lambda b: (b.first, b.last))
    spans: list[tuple[int, int, object]] = []
    cursor = first
    for beat in ordered:
        start, end = max(beat.first, cursor), min(beat.last, last)
        if end < start:
            continue
        if start > cursor and spans:  # a gap: the beat before runs on over it
            prev_start, _, prev_beat = spans[-1]
            spans[-1] = (prev_start, start - 1, prev_beat)
        elif start > cursor:
            start = cursor
        spans.append((start, end, beat))
        cursor = end + 1
    if not spans:
        return [(first, last, None)]
    if cursor <= last:
        spans[-1] = (spans[-1][0], last, spans[-1][2])
    return spans


DEVICE_KINDS = {"profile", "exhibit", "highlight", "board", "map"}


def shape_beats(spans: list[tuple[int, int, object]], sentences: list[dict]) -> list[tuple[int, int, object, int]]:
    """Split beats over BEAT_MAX_S at the sentence nearest their middle, and fold short ones back.

    Returns (first, last, plan, part) where part counts the pieces of a split beat from 0: a later
    piece takes the beat's next picture choice rather than repeating the first.
    """
    def length(a, b):
        return sentences[b]["end"] - sentences[a]["start"]

    out: list[tuple[int, int, object, int]] = []
    for first, last, plan in spans:
        stack, pieces = [(first, last)], []
        while stack:
            a, b = stack.pop(0)
            if length(a, b) <= BEAT_MAX_S or a == b:
                pieces.append((a, b))
                continue
            middle = sentences[a]["start"] + length(a, b) / 2
            cut = min(range(a + 1, b + 1), key=lambda k: abs(sentences[k]["start"] - middle))
            stack[:0] = [(a, cut - 1), (cut, b)]
        out += [(a, b, plan, n) for n, (a, b) in enumerate(pieces)]
    merged: list[tuple[int, int, object, int]] = []
    for item in out:
        a, b = item[0], item[1]
        if merged and length(a, b) < BEAT_MIN_S and length(merged[-1][0], b) <= BEAT_MAX_S + BEAT_MIN_S:
            pa, _, pplan, ppart = merged[-1]
            merged[-1] = (pa, b, pplan, ppart)
        else:
            merged.append(item)
    return merged


def beat_entry(index: int, first: int, last: int, plan, part: int, sentences: list[dict]) -> dict:
    """A shot-list entry for a beat; a later piece of a split beat runs the next picture choice."""
    shot = {"index": index, "start": sentences[first]["start"], "end": sentences[last]["end"],
            "narration": " ".join(s["text"] for s in sentences[first:last + 1]), "sentences": [first, last]}
    entry = to_contract(shot, plan)
    if part:
        choices = entry.get("choices") or []
        if entry["kind"] in DEVICE_KINDS and entry.get("subject_id"):
            choices = [entry["subject_id"]] + choices
        entry.update({"kind": "archival" if entry["kind"] in DEVICE_KINDS else entry["kind"],
                      "kind_planned": "archival" if entry["kind"] in DEVICE_KINDS else entry["kind"],
                      "choices": choices[part:] + choices[:part], "overlay": None, "overlay_planned": None})
    return entry


def land_on_effects(entries: list[dict], events: list[dict], whoosh_s: float = 1.2, reach_s: float = 1.5) -> int:
    """The audio already carries a whoosh into each old dip to black: dip to black on the cut nearest it."""
    starts = [e["start"] for e in entries]
    landed = 0
    for event in events:
        if event.get("kind") != "whoosh":
            continue
        cut = float(event["t"]) + whoosh_s
        nearest = min(range(len(starts)), key=lambda k: abs(starts[k] - cut))
        if nearest and abs(starts[nearest] - cut) <= reach_s:
            entries[nearest]["transition_in"] = "black"
            entries[nearest]["dip_reason"] = "whoosh in the audio"
            landed += 1
    return landed


def to_contract(shot: dict, planned: PlannedShot | None) -> dict:
    entry = {
        "index": shot["index"], "start": shot["start"], "end": shot["end"], "narration": shot["narration"],
        "kind": "period", "choices": [], "library_id": None, "query": "", "prompt": "",
        "motion": "zoom_in", "transition_in": "dissolve", "treatment": None, "overlay": None,
        "reason": "", "points": [], "events": [], "nodes": [], "edges": [],
        "asset": None, "colour": None, "licence": None, "attribution": None, "source_url": None, "rights": None,
    }
    if planned is None:
        entry.update({"query": f"{FALLBACK_QUERY} night", "reason": "planner returned nothing for this shot"})
        return entry
    entry.update({
        "kind": planned.kind, "kind_planned": planned.kind, "choices": planned.choices[:6], "query": planned.query, "prompt": planned.prompt,
        "motion": planned.motion, "transition_in": planned.transition_in,
        "overlay": planned.overlay.model_dump() if planned.overlay else None,
        "overlay_planned": planned.overlay.model_dump() if planned.overlay else None,
        "reason": planned.reason,
        "points": [p.model_dump() for p in planned.points],
        "events": [e.model_dump() for e in planned.events],
        "nodes": [n.model_dump() for n in planned.nodes],
        "edges": [[e.source, e.target, e.label] for e in planned.edges],
        "subject_id": planned.subject_id, "fields": [f.model_dump() for f in planned.fields],
        "tag": planned.tag, "detail": planned.detail, "label": planned.label,
        "pins": [p.model_dump() for p in planned.pins],
        "links": [l.model_dump() for l in planned.links],
    })
    return entry


def finalise(entries: list[dict], library: dict[str, dict], duration: float) -> None:
    """Everything decided in code after the model has spoken: pictures, labels, charts, transitions."""
    for entry in entries:
        entry["overlay"] = entry.get("overlay_planned", entry.get("overlay"))
        # The model's own kind, so re-running these decisions starts from the same place every time.
        entry.setdefault("kind_planned", entry["kind"])
        entry["kind"] = entry["kind_planned"]
        for key in ("library_id", "asset", "colour", "kind_hint", "licence", "attribution", "source_url", "rights", "media"):
            entry.pop(key, None)
    limit_boards(entries, library, duration)
    grow_boards(entries)
    limit_devices(entries, library)
    limit_drawn(entries, duration)
    pick_assets(entries, library, duration)
    insert_clips(entries, library)
    insert_boards(entries, library)
    names_match_pictures(entries, library)
    thin_overlays(entries)
    finish_transitions(entries)
    for entry in entries:
        for key in ("media", "clip_id", "clip_in", "clip_out"):
            entry.pop(key, None)
        entry["media"] = "motion" if entry["kind"] in ("board", "map", "profile", "exhibit", "highlight") else "photo"
        subject = library.get(entry.get("subject_id") or "") if entry["media"] == "motion" else None
        if subject:
            entry.update({"licence": subject.get("licence"), "attribution": subject.get("attribution"),
                          "source_url": subject.get("source_url"), "rights": subject.get("rights")})
        item = library.get(entry.get("library_id") or "")
        if not item:
            continue
        entry.update({
            "asset": item["file"], "colour": item.get("colour"), "kind_hint": item.get("kind"),
            "licence": item.get("licence"), "attribution": item.get("attribution"),
            "source_url": item.get("source_url"), "rights": item.get("rights"),
        })
        if item.get("media") == "clip":
            entry.update({"media": "clip", "clip_id": item["clip_id"], "clip_in": item["in"], "clip_out": item["out"],
                          "kind": "clip", "kind_hint": None})


def readable(shots: list[dict], library: dict[str, dict], duration: float) -> str:
    kinds = Counter(s["kind"] for s in shots)
    real = sum(1 for s in shots if s.get("library_id"))
    overlays = sum(1 for s in shots if s.get("overlay"))
    distinct = len({s["library_id"] for s in shots if s.get("library_id")})
    out = [
        "# Edit plan", "",
        f"{len(shots)} shots over {duration / 60:.1f} minutes, average {duration / max(1, len(shots)):.1f}s.",
        f"{real} shots use a real library photograph ({distinct} distinct pictures); "
        f"{overlays} shots carry an on-screen label.", "",
        "Kinds: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()), "",
        "| # | time | picture | label | narration |", "|---|---|---|---|---|",
    ]
    for shot in shots:
        m, s = divmod(int(shot["start"]), 60)
        if shot.get("library_id"):
            shown = f"{shot['library_id']}: {library[shot['library_id']]['caption'][:70]}"
        else:
            shown = f"[{shot['kind']}] " + (shot.get("query") or shot.get("prompt") or "-")[:70]
        label = ""
        if shot.get("overlay"):
            label = " / ".join(x for x in (shot["overlay"]["text"], shot["overlay"]["subtext"]) if x)
        narration = shot["narration"].replace("|", "/")[:100]
        out.append(f"| {shot['index']} | {m}:{s:02d} | {shown.replace('|', '/')} | {label} | {narration} |")
    return "\n".join(out) + "\n"


def plan_by_beats(args, video_dir: Path, library: dict[str, dict], cat: str) -> None:
    """Group sentences into beats with the model, one picture or clip per beat, then the code rules."""
    from openai import OpenAI

    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    duration = sentences[-1]["end"]
    client = OpenAI(api_key=settings.openai_api_key)
    batches = [sentences[i:i + SENTENCE_BATCH] for i in range(0, len(sentences), SENTENCE_BATCH)]
    if args.shots:
        batches = batches[:max(1, -(-args.shots // SENTENCE_BATCH))]
    print(f"{len(sentences)} sentences in {len(batches)} batches; planning beats with {args.model}")

    def one(batch: list[dict]):
        a, b = batch[0]["index"], batch[-1]["index"]
        before = sentences[max(0, a - SENTENCE_CONTEXT):a]
        after = sentences[b + 1:b + 1 + SENTENCE_CONTEXT]
        for attempt in range(3):
            try:
                planned = plan_beats(client, args.model, cat, batch, before, after)
                print(f"  planned sentences {a}-{b}: {len(planned)} beats", flush=True)
                return repair_beats(planned, a, b)
            except Exception as exc:  # noqa: BLE001 - a batch may fail; its sentences still get beats
                if attempt == 2:
                    print(f"  FAILED sentences {a}-{b}: {str(exc)[:160]}", flush=True)
        return [(a, b, None)]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        spans = [span for batch_spans in pool.map(one, batches) for span in batch_spans]
    shaped = shape_beats(spans, sentences)
    entries = [beat_entry(n, a, b, plan, part, sentences) for n, (a, b, plan, part) in enumerate(shaped, start=1)]
    if args.shots:
        entries = entries[:args.shots]
    entries[-1]["end"] = round(entries[-1]["end"], 3)
    if args.events:
        events = json.loads(Path(args.events).read_text(encoding="utf-8"))
        print(f"  {land_on_effects(entries, events)} dips to black land on the whooshes already in the audio")
    finalise(entries, library, duration)
    (video_dir / "shots.json").write_text(json.dumps({
        "audio": args.audio, "duration_s": round(duration, 3),
        "width": 1920, "height": 1080, "fps": 30, "subtitles": "sidecar", "version": 3,
        "shots": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "shots.md").write_text(readable(entries, library, duration), encoding="utf-8")
    lengths = sorted(e["end"] - e["start"] for e in entries)
    print(f"wrote shots.json and shots.md; {len(entries)} beats, median {lengths[len(lengths) // 2]:.1f}s, "
          f"longest {lengths[-1]:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--audio", default="", help="the narration the film is cut to")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--shots", type=int, default=None, help="plan only the first N shots (preview)")
    parser.add_argument("--repick", action="store_true",
                        help="redo pictures and labels from the saved choices, without calling the model")
    parser.add_argument("--beats", action="store_true",
                        help="plan by sentence groups from sentences.json instead of fixed shots")
    parser.add_argument("--free-only", action="store_true",
                        help="use only public-domain, CC0, generated and original material")
    parser.add_argument("--events", default="", help="effects already in the audio (events_mapped.json)")
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    use_case(video_dir)
    lines = json.loads((video_dir / "lines.json").read_text(encoding="utf-8"))
    items = json.loads((video_dir / "library.json").read_text(encoding="utf-8"))["items"]
    apply_reviews(items, video_dir / "library")
    clips_path = video_dir / "clips.json"
    clip_items = json.loads(clips_path.read_text(encoding="utf-8"))["items"] if clips_path.exists() else []
    if args.free_only:
        before = sum(1 for i in items if usable(i))
        items = [i for i in items if free_to_use(i)]
        clip_items = [c for c in clip_items if free_to_use(c)]
        print(f"free-only: {sum(1 for i in items if usable(i))} of {before} pictures kept, {len(clip_items)} clips")
    library = {item["id"]: item for item in items}
    clips = clip_entries(clip_items)
    library.update(clips)
    cat = catalogue(items, clips)
    if args.beats:
        plan_by_beats(args, video_dir, library, cat)
        return
    shots = group_into_shots(lines)
    duration = max(line["end"] for line in lines)
    shots[-1]["end"] = round(duration, 2)
    if args.shots:
        shots = shots[:args.shots]
    if args.repick:
        saved = json.loads((video_dir / "shots.json").read_text(encoding="utf-8"))
        entries = saved["shots"]
        finalise(entries, library, duration)
        (video_dir / "shots.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        (video_dir / "shots.md").write_text(readable(entries, library, duration), encoding="utf-8")
        print(f"re-picked {len(entries)} shots from saved choices")
        return
    print(f"{len(shots)} shots; library {sum(1 for i in items if usable(i))} usable pictures; planning with {args.model}")

    client = OpenAI(api_key=settings.openai_api_key)
    batches = [shots[i:i + BATCH] for i in range(0, len(shots), BATCH)]

    def one(number_and_batch):
        number, batch = number_and_batch
        before = shots[max(0, number * BATCH - CONTEXT_SHOTS):number * BATCH]
        after = shots[number * BATCH + len(batch):number * BATCH + len(batch) + CONTEXT_SHOTS]
        for attempt in range(3):
            try:
                planned = plan_batch(client, args.model, cat, batch, before, after)
                print(f"  planned shots {batch[0]['index']}-{batch[-1]['index']}", flush=True)
                return planned
            except Exception as exc:  # noqa: BLE001 - a batch may fail; the rest still plan
                if attempt == 2:
                    print(f"  FAILED shots {batch[0]['index']}-{batch[-1]['index']}: {str(exc)[:160]}", flush=True)
        return []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        planned_batches = list(pool.map(one, enumerate(batches)))
    by_index = {p.index: p for batch in planned_batches for p in batch}
    entries = [to_contract(shot, by_index.get(shot["index"])) for shot in shots]

    finalise(entries, library, duration)

    (video_dir / "shots.json").write_text(json.dumps({
        "audio": args.audio, "duration_s": round(duration, 2),
        "width": 1920, "height": 1080, "fps": 30, "subtitles": "sidecar", "version": 2,
        "shots": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "shots.md").write_text(readable(entries, library, duration), encoding="utf-8")
    missing = sum(1 for e in entries if e["reason"] == "planner returned nothing for this shot")
    print(f"wrote shots.json and shots.md; {len(entries)} shots, {missing} unplanned")


if __name__ == "__main__":
    main()

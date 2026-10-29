"""The storyboard: chapters into scenes into beats, planned with the whole story in view.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/storyboard.py <video_dir> --free-only \
        [--events events_mapped.json] [--audio mix.wav] [--chapters 1,2]

Reads `sentences.json`, `bible.json` (story_bible.py), `quotes.json` (quotes.py) and the picture library.
For each chapter of the bible a model plans the scenes (setting, story time, purpose, mood, visual idea,
look) and, inside each scene, the beats: runs of consecutive sentences that carry one idea, each with what
the viewer should see and feel, who or what is shown, the shot type, the picture choices, the camera move,
the transition and any effect cues. The model is given the whole bible, the full text of the chapter and
the end of the chapter before, so a sentence is never planned on its own.

Code then does what a model must not be trusted with: scenes and beats cover every sentence exactly once,
beats stay between 3 and 20 seconds, effects keep to each look's budget, quote cards are attached to the
beat they are spoken in, and the existing picture rules (reuse limits, colour lean, clips, devices, labels)
pick the final pictures. Writes `storyboard.json` (the hierarchy, for review) and `shots.json` (the same
beats in the shot-list contract build.py renders).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.config import settings  # noqa: E402
from plan_edit import (  # noqa: E402
    BEAT_SYSTEM, PlannedBeat, apply_reviews, beat_entry, catalogue, clip_entries, finalise, free_to_use,
    land_on_effects, readable, shape_beats, usable,
)
from story_bible import LOOKS, Look, cover_range  # noqa: E402

PART_SENTENCES = 120       # a chapter longer than this is planned in parts
CONTEXT = 6                # sentences of the neighbouring chapters shown for context
Effect = Literal["flash", "shake", "flicker"]
# What each look may use, per scene: dips to black are transitions and are not counted here.
EFFECT_BUDGET = {
    "warm_memory": {},
    "cold_dread": {"flicker": 1},
    "investigation": {"flicker": 1},
    "revelation": {"flash": 1, "shake": 1, "flicker": 1},
    "courtroom": {},
    "memorial": {},
}


Link = Literal["subject", "place_era", "action", "reframe", "hold", "generated"]


class StoryBeat(PlannedBeat):
    intent: str = Field(description="What the viewer should see and feel during this beat, one sentence.")
    link: Link = Field(default="place_era", description="Why the picture belongs to these words (the relevance ladder).")
    link_reason: str = Field(default="", description="Half a sentence: what in the picture connects it to the words.")
    reframe_on: str = Field(default="", description="link reframe: the face or detail of the relevant photograph to frame.")
    detail: str = Field(default="", description="link generated: the object or setting to show, no people, no named place.")
    subjects: list[str] = Field(default_factory=list, description="People, places or objects from the bible shown here.")
    shot_type: Literal["establishing", "portrait", "document", "evidence", "map", "board", "period", "clip", "detail"] = "period"
    continuity: str = Field(default="", description="Only if this beat deliberately repeats or answers an earlier picture.")
    effects: list[Effect] = Field(default_factory=list, description="Rare. Only where the story lands hard.")


class Scene(BaseModel):
    title: str = Field(description="Two to five words for the editor.")
    first: int
    last: int
    setting: str = Field(description="Where, as the narration places it.")
    story_time: str = Field(description="When the events of this scene happen.")
    purpose: str = Field(description="What the viewer must understand by the end of the scene.")
    mood: str
    visual_idea: str = Field(description="How the scene looks, in one sentence: the kind of pictures and how they move.")
    look: Look
    intensity: int = Field(description="0 calm to 10 the film's peak.")
    beats: list[StoryBeat]


class ChapterBoard(BaseModel):
    scenes: list[Scene]


class SceneLook(BaseModel):
    id: int = Field(description="The scene id given, copied exactly.")
    look: Look
    reason: str = Field(default="", description="Half a sentence.")


class SceneLooks(BaseModel):
    scenes: list[SceneLook]


LOOK_SYSTEM = """You are the colourist of a true-crime documentary. Each scene of the storyboard is listed
with when its events happen, its mood and its purpose. Give every scene its look, from what the scene
shows, not from its chapter:
- warm_memory: the time before: childhood, family, school, early jobs, marriage, ordinary community
  life, a public persona shown before its meaning is revealed. Faded warm film.
- cold_dread: unease building: a disappearance, a parent waiting, a suspicion forming, a threat not yet
  understood. Cold and desaturated.
- investigation: police work, evidence, records, statements, interviews, accounts being checked.
  The film's home look.
- revelation: the moments the truth comes out and its immediate aftermath: the confession, the search
  that finds what was hidden. High contrast, colour drained except crimson. Rare.
- courtroom: trial, verdict, sentencing, appeals, prison, the execution.
- memorial: when the narration remembers the victims by name. Soft black and white.
Follow the story: neighbouring scenes share a look unless the story really changes; an investigation
chapter can open on a warm_memory scene of the past it is investigating. Return every scene id once."""


STORYBOARD_RULES = """You are now storyboarding one chapter of the film, with the story bible and the whole
chapter in front of you. Work like an editor: understand what the chapter does, then divide it.

Scenes: a scene is 30 seconds to 3 minutes of narration that happens in one setting or does one job
(establishing a place, introducing a person, a single interview, a search, a revelation). Give each
scene its setting, story time, purpose, mood, a one-sentence visual idea, its intensity and its look:
warm_memory, cold_dread, investigation, revelation, courtroom or memorial (the chapter's look is the
default; change it only where the scene clearly differs, e.g. a memorial passage inside an
investigation chapter).

Beats: inside each scene, group consecutive sentences into beats, one picture or clip per idea, as
before. For every beat write its intent (what the viewer should see and feel), its subjects (names
from the bible), and its shot type. Keep continuity: when a person returns, use the same photographs
the bible links to them, so a face stays the same face; bring motifs back on purpose and say so in
`continuity`.

Quotes: where a listed quote is shown as a card, the beat it is spoken in should show the speaker's
photograph if the library has one (the card is laid over it); for written words, the object.

Effects (flash, shake, flicker): almost never. Only in a revelation scene, at the moment the truth
lands, and never on a victim. Use transition_in black where the story jumps in time or place or a
scene ends on a hard line.

Every sentence of the chapter belongs to exactly one scene and one beat, in order.

THE RELEVANCE LADDER (this overrides any earlier advice about period stand-ins). Every beat states its `link`,
the reason its picture belongs to these words, and picks pictures that truly have that link. Try in order:
1. subject: the person, place, object or document the words name (their photograph in the library).
2. place_era: the same town or area in the same years. A picture of a DIFFERENT city or neighbourhood is
   never acceptable as the setting (a big city's street is not the small suburb the story is in; a
   storefront in one town is not a restaurant in another).
3. action: a real photograph or clip of the object or action the sentence turns on (a cash register, a
   telephone, a car in snow at night, police at a door), of the right era.
4. reframe: no new picture; show another framing of a relevant photograph already in this scene. Put that
   photograph's id first in `choices` and say in `reframe_on` what to frame ("his face", "the sign over the
   door", "the house's front window").
5. hold: stay on the previous beat's picture; the beat simply continues it.
6. generated: an illustrative detail of an OBJECT or SETTING only, for an action no archive shows. Describe it
   in `detail` ("a 1970s cash register on a shop counter at night, snow on the window"). Never a person or a
   face, never a named or identifiable real place, never anything gruesome. Rare.
A picture with none of these links must not be used. Prefer fewer, truer pictures held longer to many
loosely related ones."""

SYSTEM = BEAT_SYSTEM + "\n\n" + STORYBOARD_RULES


def bible_brief(bible: dict) -> str:
    """The bible as the storyboard model reads it: compact, with photograph ids."""
    people = "\n".join(f"  {p['name']} - {p['role']}" + (" (victim)" if p.get("victim") else "")
                       + (f" | photos: {', '.join(p['photo_ids'])}" if p.get("photo_ids") else "") for p in bible["people"])
    places = "\n".join(f"  {p['name']} ({p['era']}) - {p['description']}"
                       + (f" | photos: {', '.join(p['photo_ids'])}" if p.get("photo_ids") else "") for p in bible["places"])
    motifs = "\n".join(f"  {m['name']} - {m['description']}" + (f" | photos: {', '.join(m['photo_ids'])}" if m.get("photo_ids") else "")
                       for m in bible.get("motifs", []))
    chapters = "\n".join(f"  {c['id']}. {c['title']} ({c['era']}; {c['look']}, intensity {c['intensity']})" for c in bible["chapters"])
    rules = "\n".join(f"  - {r}" for r in bible.get("sensitivities", []))
    return (f"Logline: {bible['logline']}\n\nPeople:\n{people}\n\nPlaces:\n{places}\n\nMotifs:\n{motifs}\n\n"
            f"Chapters of the film:\n{chapters}\n\nCare rules:\n{rules}")


def chapter_parts(chapter: dict) -> list[tuple[int, int]]:
    a, b = chapter["first"], chapter["last"]
    n = b - a + 1
    if n <= PART_SENTENCES:
        return [(a, b)]
    pieces = -(-n // PART_SENTENCES)
    size = -(-n // pieces)
    return [(a + k * size, min(b, a + (k + 1) * size - 1)) for k in range(pieces)]


def plan_part(client, model: str, cat: str, brief: str, chapter: dict, part: tuple[int, int],
              sentences: list[dict], quotes: list[dict]) -> ChapterBoard:
    a, b = part
    before = sentences[max(0, a - CONTEXT):a]
    after = sentences[b + 1:b + 1 + CONTEXT]
    here = [q for q in quotes if a <= q["first"] <= b and q.get("show") == "card"]

    def lines(items):
        return "\n".join(f"  ({s['index']}) {s['text']}" for s in items)

    user = (
        "Picture library (id | kind colour era quality | people | places | what it shows):\n" + cat
        + "\n\nThe story bible:\n" + brief
        + f"\n\nThis chapter: {chapter['id']}. {chapter['title']} - {chapter['purpose']} Tone: {chapter['tone']}. "
          f"Era: {chapter['era']}. Look: {chapter['look']}. Intensity: {chapter['intensity']}."
        + (f" (You are planning sentences {a}-{b} of the chapter's {chapter['first']}-{chapter['last']}.)"
           if (a, b) != (chapter["first"], chapter["last"]) else "")
        + ("\n\nJust before (context only):\n" + lines(before) if before else "")
        + "\n\nStoryboard these sentences:\n"
        + "\n".join(f"  ({s['index']}) [{s['end'] - s['start']:.1f}s] {s['text']}" for s in sentences[a:b + 1])
        + ("\n\nJust after (context only):\n" + lines(after) if after else "")
        + ("\n\nQuotes shown as cards here:\n" + "\n".join(
            f"  {q['id']} in sentence {q['first']}: {q['kind']} by {q.get('speaker') or 'unknown'}" for q in here) if here else "")
    )
    response = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                          {"role": "user", "content": user}], text_format=ChapterBoard)
    return response.output_parsed


# ---------------------------------------------------------------------------- decided in code

def scene_spans(board: ChapterBoard | None, a: int, b: int) -> list[tuple[int, int, Scene | None]]:
    return cover_range([(s.first, s.last, s) for s in (board.scenes if board else [])], a, b)


def beat_spans(scene: Scene | None, a: int, b: int) -> list[tuple[int, int, StoryBeat | None]]:
    return cover_range([(bt.first, bt.last, bt) for bt in (scene.beats if scene else [])], a, b)


def assign_looks(client, model: str, scenes: list[dict], bible: dict) -> dict[int, str]:
    """One colourist pass over every scene of the film at once: each scene's look from its own content."""
    lines = "\n".join(f"  [{s['id']}] {s['title']} | when: {s['story_time']} | mood: {s['mood']} | "
                      f"purpose: {s['purpose']} | currently: {s['look']}" for s in scenes)
    user = f"Logline: {bible.get('logline', '')}\n\nScenes in film order:\n{lines}"
    response = client.responses.parse(model=model, input=[{"role": "system", "content": LOOK_SYSTEM},
                                                          {"role": "user", "content": user}], text_format=SceneLooks)
    return {s.id: s.look for s in response.output_parsed.scenes if s.look in LOOKS}


def apply_looks(scenes: list[dict], entries: list[dict], chosen: dict[int, str]) -> None:
    """Scenes take the colourist's looks (a memorial scene stays a memorial); their beats follow."""
    looks_by_scene = {}
    for s in scenes:
        if s["look"] != "memorial" and chosen.get(s["id"]):
            s["look"] = chosen[s["id"]]
        looks_by_scene[s["id"]] = s["look"]
    for e in entries:
        e["look"] = looks_by_scene.get(e.get("scene"), e.get("look"))


SUFFIXES = {"jr", "sr", "ii", "iii", "dr", "mr", "mrs", "detective", "lieutenant", "officer"}


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower().replace("’", "'").replace("'s", "")) if w not in SUFFIXES]


def _first_name(person: dict) -> str | None:
    """A person's given name, or None for an entry described rather than named ("Rob Piest's mother")."""
    name = person.get("name") or ""
    if "’s" in name or "'s" in name:
        return None
    words = _words(name)
    return words[0] if len(words) > 1 else None


def identifying_words(person: dict, others_first_names: set[str]) -> set[str]:
    """The words that, said aloud, name this person: their surname and aliases, and their first name only
    when nobody else in the story shares it (two Michaels make "Michael" ambiguous)."""
    name_words = _words(person.get("name") or "")
    out = {name_words[-1]} if len(name_words) > 1 else set(name_words)
    first = _first_name(person)
    for alias in person.get("aliases") or []:
        out |= {w for w in _words(alias) if len(w) >= 3 and w != first}
    if first and first not in others_first_names:
        out.add(first)
    return {w for w in out if len(w) >= 3}


def guard_victims(entries: list[dict], library: dict[str, dict], bible: dict) -> int:
    """A victim's photograph only while the narration of that beat names that victim.

    A victim is named by their surname or an alias; a first name only counts when nobody else in the
    story has it. A beat that fails takes its next choice that is not a victim's photograph, or the
    photograph of a person it does name, or the most relevant other picture. Returns how many beats
    were changed.
    """
    from plan_edit import relevance
    people = bible.get("people") or []
    firsts = [(_first_name(p), id(p)) for p in people]

    def words_for(person: dict) -> set[str]:
        others = {f for f, pid in firsts if f and pid != id(person)}
        return identifying_words(person, others)

    # Whose photograph needs naming: a victim's always; anyone else's unless they are central to the story
    # (a photograph shared with a central person, like Gacy beside a politician, may run without naming).
    central = {pid for p in people if (p.get("importance") or 0) >= 3 and not p.get("victim") for pid in p.get("photo_ids") or []}
    owner: dict[str, dict] = {}
    for person in people:
        if person.get("victim") or (person.get("importance") or 0) < 3:
            for pid in person.get("photo_ids") or []:
                if pid not in central or person.get("victim"):
                    owner.setdefault(pid, person)
    victim_photos = set(owner)

    def named(person: dict, narration: str) -> bool:
        return bool(words_for(person) & set(_words(narration)))

    changed = 0
    for e in entries:
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        pid = e.get("library_id") or ""
        if pid not in owner or named(owner[pid], e["narration"]):
            continue
        said_people = [p for p in people if p.get("photo_ids") and named(p, e["narration"])]
        # Photographs of people the beat names are fair game, whoever they are.
        allowed = {c for p in said_people for c in p["photo_ids"]}
        # The victim the beat does name comes first; then its other choices; then anyone else it names.
        named_victims = [c for p in said_people if p.get("victim") for c in p["photo_ids"] if c in library]
        options = named_victims + [c for c in e.get("choices") or [] if c in library and c != pid
                                   and library[c].get("media") != "clip"         # a guard swaps in stills only
                                   and (c not in victim_photos or c in allowed)]
        options += [c for p in said_people if not p.get("victim") for c in p["photo_ids"] if c in library]
        if not options:
            pool = [i for i in library.values() if i["id"] not in victim_photos and i.get("media") != "clip"
                    and i.get("caption") and not i.get("excluded") and not i.get("graphic")]
            options = [max(pool, key=lambda i: relevance(i, e["narration"]))["id"]] if pool else []
        if options:
            set_picture(e, library[options[0]])
            e["guarded"] = f"photo {pid} of {owner[pid]['name']} replaced: the beat does not name them"
            changed += 1
    return changed


def replace_excluded(entries: list[dict], library: dict[str, dict]) -> int:
    """A picture excluded after planning (exclusions.json, a verifier's review) leaves every beat it is on."""
    from plan_edit import relevance, usable
    changed = 0
    for e in entries:
        pid = e.get("library_id")
        if not pid or e.get("media") == "motion":
            continue
        item = library.get(pid)
        # Fine if usable and run as what it is (a clip marked as a still cannot render).
        if item and usable(item) and (item.get("media") == "clip") == (e.get("media") == "clip"):
            continue
        options = [c for c in e.get("choices") or [] if c != pid and c in library and usable(library[c])
                   and library[c].get("media") != "clip"]
        if not options:
            pool = [i for i in library.values() if usable(i) and i.get("media") != "clip"]
            options = [max(pool, key=lambda i: relevance(i, e["narration"]))["id"]] if pool else []
        if options:
            set_picture(e, library[options[0]])
            e["guarded"] = f"excluded picture {pid} replaced"
            e.pop("owner", None)                  # no longer the picture the owner chose
            changed += 1
    return changed


MERGE_MAX_S = 26.0         # two beats on the same photograph become one hold, up to this long


def merge_repeats(entries: list[dict], scenes: list[dict]) -> int:
    """Back-to-back beats on the same photograph become one continuous hold: re-cutting to the picture
    already on screen reads as a glitch. Beats are renumbered and the scenes' beat lists rebuilt."""
    merged: list[dict] = []
    count = 0
    for e in entries:
        prev = merged[-1] if merged else None
        same = (prev is not None and e.get("library_id") and e.get("library_id") == prev.get("library_id")
                and e.get("media") == prev.get("media") == "photo" and e.get("scene") == prev.get("scene")
                and e["end"] - prev["start"] <= MERGE_MAX_S and not e.get("owner") and not prev.get("owner"))
        if same:
            prev["end"] = e["end"]
            prev["narration"] = f"{prev['narration']} {e['narration']}".strip()
            if prev.get("sentences") and e.get("sentences"):
                prev["sentences"] = [prev["sentences"][0], e["sentences"][1]]
            prev["effects"] = list(dict.fromkeys((prev.get("effects") or []) + (e.get("effects") or [])))
            count += 1
            continue
        merged.append(e)
    entries[:] = merged
    for n, e in enumerate(entries, start=1):
        e["index"] = n
    for s in scenes:
        s["beats"] = [e["index"] for e in entries if e.get("scene") == s["id"]]
    return count


MOURNING = re.compile(r"funeral|coffin|casket|grave marker|headstone|tombstone|gravesite|memorial service", re.I)


def guard_dates(entries: list[dict], library: dict[str, dict], scenes: dict[int, dict]) -> int:
    """No photograph from after a scene's time: such a beat takes its next choice that is not, or holds."""
    changed = 0
    for e in entries:
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        item = library.get(e.get("library_id") or "")
        scene = scenes.get(e.get("scene"))
        if not item or e.get("media") == "motion" or not from_the_future(item, scene):
            continue
        options = [c for c in e.get("choices") or [] if c in library and library[c].get("media") != "clip"
                   and c != item["id"] and not from_the_future(library[c], scene)]
        if options:
            set_picture(e, library[options[0]])
        else:
            e["link"] = "hold"
        e["guarded"] = f"{item['id']} is dated after this scene ({scene.get('story_time')})"
        changed += 1
    return changed


def guard_mourning(entries: list[dict], library: dict[str, dict], bible: dict) -> int:
    """Funerals, coffins and graves only where the film is remembering the dead: in a memorial scene, or a
    beat that names the person buried. Anywhere else they are someone's grief used as atmosphere."""
    from plan_edit import relevance
    people = bible.get("people") or []
    firsts = [(_first_name(p), id(p)) for p in people]
    victims = [p for p in people if p.get("victim")]
    pool = [i for i in library.values() if i.get("media") != "clip" and i.get("caption") and not i.get("excluded")
            and not i.get("graphic") and not MOURNING.search(i.get("caption") or "")]
    changed = 0
    for e in entries:
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        item = library.get(e.get("library_id") or "")
        if not item or not MOURNING.search(item.get("caption") or "") or e.get("look") == "memorial":
            continue
        said = set(_words(e.get("narration") or ""))
        if any(identifying_words(v, {f for f, i in firsts if f and i != id(v)}) & said for v in victims):
            continue
        options = [c for c in e.get("choices") or [] if c in {i["id"] for i in pool}]   # stills only
        if not options and pool:
            options = [max(pool, key=lambda i: relevance(i, e["narration"]))["id"]]
        if options:
            set_picture(e, library[options[0]])
            e["guarded"] = f"mourning picture {item['id']} replaced: not a memorial passage"
            changed += 1
    return changed


def set_picture(entry: dict, item: dict) -> None:
    entry.update({"library_id": item["id"], "asset": item["file"], "colour": item.get("colour"),
                  "kind_hint": item.get("kind"), "licence": item.get("licence"), "attribution": item.get("attribution"),
                  "source_url": item.get("source_url"), "rights": item.get("rights"), "media": "photo"})
    for key in ("clip_id", "clip_in", "clip_out"):
        entry.pop(key, None)


def speaker_pictures(entries: list[dict], bible: dict, library: dict[str, dict]) -> int:
    """A beat that shows someone's words on a card shows that person, when the bible has their photograph.

    The photograph used least, and not already within a minute of the beat, is chosen. Returns how many
    beats changed picture.
    """
    people = {}
    for p in bible.get("people") or []:
        for name in [p.get("name") or ""] + list(p.get("aliases") or []):
            if name:
                people[name.lower()] = p
    shown_in: dict[str, set[str]] = {}
    for p in bible.get("people") or []:
        for pid in p.get("photo_ids") or []:
            shown_in.setdefault(pid, set()).add(p["name"])
    uses: dict[str, int] = {}
    for e in entries:
        if e.get("library_id"):
            uses[e["library_id"]] = uses.get(e["library_id"], 0) + 1
    changed = 0
    for position, e in enumerate(entries):
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        if e.get("speaker_photo") and len(shown_in.get(e.get("library_id"), ())) > 1:
            e.pop("speaker_photo")                # chosen before this rule: choose again
        cards = [q for q in e.get("quotes") or [] if (q.get("style") or "card") in ("card", "memorial")]
        if not cards:
            continue
        person = people.get((cards[0].get("speaker") or "").lower())
        # Only photographs of the speaker alone: a quote must not sit on someone else who shares the frame.
        photos = [pid for pid in (person or {}).get("photo_ids") or [] if pid in library and len(shown_in.get(pid, ())) == 1]
        if not photos or e.get("library_id") in photos or e.get("media") == "motion":
            continue
        near = {x.get("library_id") for x in entries if abs(x["start"] - e["start"]) < 60 and x is not e}
        pick = min((pid for pid in photos if pid not in near), key=lambda pid: uses.get(pid, 0), default=None)
        if pick:
            set_picture(e, library[pick])
            e["speaker_photo"] = person["name"]
            uses[pick] = uses.get(pick, 0) + 1
            changed += 1
    return changed


def years_of(text: str) -> list[int]:
    """Years named in a caption, era or story time: '1978', 'the 1960s' (-> 1960..1969), '1964–1967'."""
    out = []
    for m in re.finditer(r"\b(1[89]\d\d|20[0-4]\d)(s)?\b", text or ""):
        y = int(m.group(1))
        out += [y, y + 9] if m.group(2) else [y]
    return out


PORTRAIT_SLACK_YEARS = 12


def from_the_future(item: dict, scene: dict | None) -> bool:
    """A photograph dated after the scene's own years (plus two) cannot stand for that time: a 1978 photograph
    under the 1960s presents a later moment as then. A portrait of one person alone may be up to
    PORTRAIT_SLACK_YEARS later (it identifies them, as documentaries do), but not a photograph from a criminal
    record (it belongs strictly to the time it documents), nor one with others in it: Gacy beside Rosalynn Carter in 1978 is not Gacy beside his wife in 1964. Undated pictures and
    undated scenes pass."""
    if not scene:
        return False
    told = years_of(scene.get("story_time") or "")
    taken = years_of(f"{item.get('era') or ''} {item.get('caption') or ''}")
    caption = item.get("caption") or ""
    portrait = (len(item.get("subjects") or []) == 1 and re.search(r"portrait|head-and-shoulders|headshot", caption, re.I)
                and not re.search(r"mugshot|booking|arrest|prison|parole|jail|court|police|record|sheet", caption, re.I))
    record = re.search(r"mugshot|booking|arrest|prison|penitentiary|parole|jail|court|record|sheet", caption, re.I)
    slack = PORTRAIT_SLACK_YEARS if portrait else 0 if record else 2
    return bool(told and taken) and min(taken) > max(told) + slack


def respect_links(entries: list[dict], library: dict[str, dict]) -> int:
    """A beat with a true link runs a picture the planner chose for it, never a stand-in the picker found.

    The shared picker (plan_edit.pick_assets) falls back to "the most relevant free picture" when a beat's
    own choices are used up by the reuse limits: that is where unrelated city views came from. Here a
    subject, place-era or action beat takes its first usable choice even if it repeats (repeats become
    holds), and a beat with no usable choice of its own holds the picture before it. Reframes start from
    their first choice. Returns how many beats changed.
    """
    from plan_edit import usable
    changed = 0
    for e in entries:
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        link = e.get("link")
        if link not in ("subject", "place_era", "action", "reframe") or e.get("media") == "motion":
            continue
        if e.get("crop") or e.get("varied") or e.get("guarded") or (e.get("check") or {}).get("replaced"):
            continue                      # the ladder, a guard or the picture check chose this on purpose
        own = [c for c in e.get("choices") or [] if c in library and usable(library[c])]
        if e.get("library_id") in own:
            continue
        if own:
            item = library[own[0]]
            set_picture(e, item)
            if item.get("media") == "clip":
                if item["length"] >= (e["end"] - e["start"]) + 1.0:
                    e.update({"media": "clip", "kind": "clip", "clip_id": item["clip_id"],
                              "clip_in": item["in"], "clip_out": item["out"]})
                else:
                    stills = [c for c in own if library[c].get("media") != "clip"]
                    if stills:
                        set_picture(e, library[stills[0]])
                    else:
                        e["link"] = "hold"
        else:
            e["link"] = "hold"
        changed += 1
    return changed


def edit_transitions(entries: list[dict]) -> None:
    """Transitions that mean something, as a premium editor uses them: a straight cut by default (illustrations
    included: dissolving into every drawn picture turns a sequence soft); a dissolve into a new scene (a move in
    time or place); a dip to black kept only where the audio already carries one (a whoosh) or a chapter
    begins. Card to card is a cut."""
    for k, e in enumerate(entries):
        if k == 0:
            continue
        prev = entries[k - 1]
        if e.get("transition_in") == "black" and (e.get("dip_reason") or e.get("chapter") != prev.get("chapter")):
            continue
        if e.get("chapter") != prev.get("chapter"):
            e["transition_in"] = "black"
        elif e.get("scene") != prev.get("scene"):
            e["transition_in"] = "dissolve"
        else:
            e["transition_in"] = "cut"


def keep_budget(entries: list[dict]) -> None:
    """Effects within each scene's look budget, in story order; none at all on a memorial or a victim."""
    used: dict[tuple, dict[str, int]] = {}
    for e in entries:
        budget = EFFECT_BUDGET.get(e.get("look"), {})
        spent = used.setdefault((e.get("chapter"), e.get("scene")), {})
        kept = []
        for fx in e.get("effects") or []:
            if spent.get(fx, 0) < budget.get(fx, 0) and not e.get("victim_on_screen"):
                kept.append(fx)
                spent[fx] = spent.get(fx, 0) + 1
        e["effects"] = kept


def attach_quotes(entries: list[dict], quotes: list[dict], sentences: list[dict] | None = None) -> None:
    """A quote card belongs to the beat its first word is spoken in, and carries its quote with it.

    A quote that runs on into the next beat moves the cut to the end of the quote's last sentence, so a
    card is never cut off mid-quote, unless that would leave the next beat under 1.5 seconds.
    """
    for e in entries:
        e["quote_ids"], e["quotes"] = [], []
    for q in quotes:
        if q.get("show") != "card":
            continue
        home = next((e for e in entries if e["start"] <= q["start"] < e["end"]), None)
        if home is None and entries and abs(q["start"] - entries[-1]["end"]) < 0.05:
            home = entries[-1]                    # spoken right on the film's last frame
        if home is None:                          # a quote outside a previewed range stays out
            continue
        home["quote_ids"].append(q["id"])
        home["quotes"].append(q)
        at = entries.index(home)
        if sentences and at + 1 < len(entries) and home.get("sentences") and q["last"] > home["sentences"][1]:
            nxt = entries[at + 1]
            boundary = sentences[q["last"]]["end"]
            if nxt["end"] - boundary >= 1.5:
                home["end"] = nxt["start"] = boundary
                home["sentences"][1] = q["last"]
                if nxt.get("sentences"):
                    nxt["sentences"][0] = q["last"] + 1


def build_entries(boards: dict[int, list[tuple[tuple[int, int], ChapterBoard | None]]], bible: dict,
                  sentences: list[dict]) -> tuple[list[dict], list[dict]]:
    """Beats as shot-list entries, plus the scene records they belong to."""
    entries: list[dict] = []
    scenes_out: list[dict] = []
    victims = {p["name"].lower() for p in bible["people"] if p.get("victim")}
    for chapter in bible["chapters"]:
        for (a, b), board in boards.get(chapter["id"], [((chapter["first"], chapter["last"]), None)]):
            for s_first, s_last, scene in scene_spans(board, a, b):
                scene_id = len(scenes_out) + 1
                look = scene.look if scene else chapter["look"]
                record = {
                    "id": scene_id, "chapter": chapter["id"], "first": s_first, "last": s_last,
                    "start": sentences[s_first]["start"], "end": sentences[s_last]["end"],
                    "title": scene.title if scene else chapter["title"],
                    "setting": scene.setting if scene else "", "story_time": scene.story_time if scene else chapter["era"],
                    "purpose": scene.purpose if scene else chapter["purpose"], "mood": scene.mood if scene else chapter["tone"],
                    "visual_idea": scene.visual_idea if scene else "", "look": look if look in LOOKS else "investigation",
                    "intensity": max(0, min(10, scene.intensity if scene else chapter["intensity"])), "beats": [],
                }
                scenes_out.append(record)
                spans = shape_beats(beat_spans(scene, s_first, s_last), sentences)
                for first, last, beat, part in spans:
                    entry = beat_entry(len(entries) + 1, first, last, beat, part, sentences)
                    entry.update({
                        "link": getattr(beat, "link", "place_era") if part == 0 else "hold",
                        "link_reason": getattr(beat, "link_reason", "") or "",
                        "reframe_on": getattr(beat, "reframe_on", "") or "",
                        "detail": getattr(beat, "detail", "") or "",
                        "sentences": [first, last], "chapter": chapter["id"], "scene": scene_id, "look": record["look"], "intensity": record["intensity"],
                        "intent": getattr(beat, "intent", "") or "", "subjects": list(getattr(beat, "subjects", []) or []),
                        "shot_type": getattr(beat, "shot_type", "period"), "continuity": getattr(beat, "continuity", "") or "",
                        "effects": list(getattr(beat, "effects", []) or []) if part == 0 else [],
                    })
                    entry["victim_on_screen"] = any(s.lower() in victims for s in entry["subjects"])
                    record["beats"].append(entry["index"])
                    entries.append(entry)
    return entries, scenes_out


def load_library(video_dir: Path, free_only: bool) -> tuple[dict[str, dict], str]:
    items = json.loads((video_dir / "library.json").read_text(encoding="utf-8"))["items"]
    apply_reviews(items, video_dir / "library")
    clips_path = video_dir / "clips.json"
    clip_items = json.loads(clips_path.read_text(encoding="utf-8"))["items"] if clips_path.exists() else []
    if free_only:
        items = [i for i in items if free_to_use(i)]
        clip_items = [c for c in clip_items if free_to_use(c)]
    library = {i["id"]: i for i in items}
    clips = clip_entries(clip_items)
    library.update(clips)
    return library, catalogue([i for i in items if usable(i)], clips)


def save(video_dir: Path, sb: dict) -> None:
    (video_dir / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
    shots_path = video_dir / "shots.json"
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    shots["shots"] = sb["beats"]
    shots_path.write_text(json.dumps(shots, ensure_ascii=False, indent=1), encoding="utf-8")


def recheck(video_dir: Path, free_only: bool) -> None:
    """The code rules again over the saved storyboard, with the current quotes.json: no model calls."""
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    quotes_path = video_dir / "quotes.json"
    if quotes_path.exists():
        sb["quotes"] = json.loads(quotes_path.read_text(encoding="utf-8"))
    library, _ = load_library(video_dir, free_only)
    relinked = respect_links(sb["beats"], library)
    excluded = replace_excluded(sb["beats"], library)
    dated = guard_dates(sb["beats"], library, {s["id"]: s for s in sb["scenes"]})
    changed = guard_victims(sb["beats"], library, sb["bible"])
    mourning = guard_mourning(sb["beats"], library, sb["bible"])
    keep_budget(sb["beats"])
    joined = merge_repeats(sb["beats"], sb["scenes"])
    edit_transitions(sb["beats"])
    attach_quotes(sb["beats"], sb["quotes"], sentences)
    speakers = speaker_pictures(sb["beats"], sb["bible"], library)
    save(video_dir, sb)
    print(f"rechecked: {dated} photographs dated after their scene replaced; "
          f"{relinked} beats back on their own chosen picture; {excluded} excluded pictures replaced; {speakers} quote beats now show their speaker; "
          f"{changed} photographs of people moved off beats that do not name them; "
          f"{mourning} funeral or grave pictures moved out of non-memorial passages; "
          f"{joined} back-to-back repeats joined into one hold; "
          f"{sum(len(b['quote_ids']) for b in sb['beats'])} quote cards")


def apply_scenarios(video_dir: Path, client, model: str, free_only: bool, fresh: bool = False) -> None:
    """Label every scene's scenario, then rebuild each scene that has a recipe (scenarios.HANDLERS)."""
    import scenarios
    from quotes import script_tokens, token_times
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    words = json.loads((video_dir / "words.json").read_text(encoding="utf-8"))
    library, _ = load_library(video_dir, free_only)
    tokens = script_tokens(sentences)
    times = token_times(tokens, words)
    # Saved labels and entries are reused (a design change should not re-ask the model); --fresh redoes them.
    if fresh or not all(s.get("scenario") for s in sb["scenes"]):
        labels = scenarios.label_scenes(client, model, sb["scenes"], sentences)
    else:
        labels = {s["id"]: s["scenario"] for s in sb["scenes"]}
    by_scene: dict[int, list[dict]] = {}
    for b in sb["beats"]:
        by_scene.setdefault(b["scene"], []).append(b)
    rebuilt = []
    for s in sb["scenes"]:
        s["scenario"] = labels.get(s["id"], s.get("scenario") or "atmosphere")
        handler = scenarios.HANDLERS.get(s["scenario"])
        old = by_scene.get(s["id"], [])
        if handler:
            plan, build = handler
            entries = s.get("entries") if s.get("entries") and not fresh else \
                plan(client, model, s, sentences, tokens, times, sb["bible"], library, video_dir)
            s["entries"] = entries
            beats = build(s, old, entries, sentences, video_dir, library)
            print(f"  scene {s['id']} '{s['title']}': {s['scenario']}, {len(entries)} entries, {len(beats)} beats")
        else:
            beats = old
        rebuilt += beats
    sb["beats"] = rebuilt
    for n, b in enumerate(sb["beats"], start=1):
        b["index"] = n
    for s in sb["scenes"]:
        s["beats"] = [b["index"] for b in sb["beats"] if b.get("scene") == s["id"]]
    attach_quotes(sb["beats"], sb["quotes"], sentences)
    save(video_dir, sb)
    from collections import Counter
    print("scenarios: " + ", ".join(f"{k} {v}" for k, v in Counter(s["scenario"] for s in sb["scenes"]).most_common()))


def run_ladder(video_dir: Path, free_only: bool) -> None:
    import ladder
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    library, _ = load_library(video_dir, free_only)
    guard_dates(sb["beats"], library, {s["id"]: s for s in sb["scenes"]})
    done = ladder.apply_ladder(sb["beats"], library, video_dir, sb["bible"], {s["id"]: s for s in sb["scenes"]},
                               sentences=sentences)
    edit_transitions(sb["beats"])
    attach_quotes(sb["beats"], sb["quotes"], sentences)
    save(video_dir, sb)
    print("ladder: " + ", ".join(f"{k} {v}" for k, v in done.items()) + f"; {len(sb['beats'])} beats")


def run_illustrate(video_dir: Path) -> None:
    import ladder
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    library, _ = load_library(video_dir, free_only=True)
    guard_dates(sb["beats"], library, {s["id"]: s for s in sb["scenes"]})
    made = ladder.illustrate_holds(sb["beats"], video_dir, sb["bible"], {s["id"]: s for s in sb["scenes"]},
                                   sentences=sentences, library=library)
    loops = ladder.break_loops(sb["beats"], library, video_dir, None, {s["id"]: s for s in sb["scenes"]}, sb["bible"])
    edit_transitions(sb["beats"])
    attach_quotes(sb["beats"], sb["quotes"], sentences)
    save(video_dir, sb)
    for b in sb["beats"]:
        if b.get("kind") == "generated":
            m, s = divmod(int(b["start"]), 60)
            print(f"  {m}:{s:02d} illustration: {b.get('detail')}")
        elif b.get("ladder_note"):
            print(f"  note #{b['index']}: {b['ladder_note']}")
    print(f"illustrations: {made}; {loops} A-B-A-B loops broken; {len(sb['beats'])} beats")


def strip_key_facts(entries: list[dict]) -> None:
    """Undo earlier key-fact labels: each label and the pieces of the shot it was cut from are replaced by that
    shot as it was (the label keeps a copy), so placing labels twice never stacks or eats shots."""
    out: list[dict] = []
    original = None             # the shot whose second piece may follow
    for e in entries:
        if original is not None:
            o, original = original, None
            if (e.get("kind") != "keyfact" and e.get("asset") == o.get("asset")
                    and e["start"] >= o["start"] - 0.01 and e["end"] <= o["end"] + 0.01):
                continue                                   # the piece after the label: already restored
        if e.get("kind") == "keyfact":
            o = e.get("key_fact_original")
            if not o:
                continue                                   # nothing to restore it from: the label just goes
            prev = out[-1] if out else None
            if (prev and prev.get("asset") == o.get("asset") and prev["start"] >= o["start"] - 0.01
                    and prev["end"] <= o["end"] + 0.01):
                out.pop()                                  # the piece before the label
            out.append(dict(o))
            original = o
            continue
        out.append(e)
    entries[:] = out


def run_key_facts(video_dir: Path, client, model: str, fresh: bool = False) -> None:
    """Find the physical clues the narration stresses and label them on screen (key_facts.py)."""
    import key_facts
    from quotes import script_tokens, token_times
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    words = json.loads((video_dir / "words.json").read_text(encoding="utf-8"))
    tokens = script_tokens(sentences)
    times = token_times(tokens, words)
    if fresh or "key_facts_found" not in sb:
        sb["key_facts_found"] = [f.model_dump() for f in key_facts.find_facts(client, model, sentences)]
    facts = [key_facts.KeyFact(**f) for f in sb["key_facts_found"]]
    strip_key_facts(sb["beats"])
    sb["key_facts"] = key_facts.place(sb["beats"], facts, sentences, tokens, times, sb["bible"], video_dir)
    for n, b in enumerate(sb["beats"], start=1):
        b["index"] = n
    for s in sb["scenes"]:
        s["beats"] = [b["index"] for b in sb["beats"] if b.get("scene") == s["id"]]
    edit_transitions(sb["beats"])
    save(video_dir, sb)
    for f in sb["key_facts"]:
        m, s = divmod(int(f["at"]), 60)
        print(f"  {m}:{s:02d} {f['label'].upper()} (sentence {f['sentence']})")
    print(f"key facts: {len(facts)} found, {len(sb['key_facts'])} labels placed")


def relook(video_dir: Path, client, model: str) -> None:
    """Re-run only the colourist pass on a saved storyboard and rewrite storyboard.json and shots.json."""
    sb = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    apply_looks(sb["scenes"], sb["beats"], assign_looks(client, model, sb["scenes"], sb["bible"]))
    keep_budget(sb["beats"])
    (video_dir / "storyboard.json").write_text(json.dumps(sb, ensure_ascii=False, indent=1), encoding="utf-8")
    shots_path = video_dir / "shots.json"
    shots = json.loads(shots_path.read_text(encoding="utf-8"))
    shots["shots"] = sb["beats"]
    shots_path.write_text(json.dumps(shots, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    minutes = Counter()
    for b in sb["beats"]:
        minutes[b["look"]] += (b["end"] - b["start"]) / 60
    print("looks: " + ", ".join(f"{k} {v:.1f} min" for k, v in minutes.most_common()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--free-only", action="store_true")
    parser.add_argument("--events", default="", help="effects already in the audio (events_mapped.json)")
    parser.add_argument("--audio", default="", help="the narration or final mix the film is cut to")
    parser.add_argument("--chapters", default="", help="plan only these chapter ids, e.g. 1,2 (a preview)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--recheck", action="store_true",
                        help="re-apply the code rules (victim guard, quotes, effect budgets) to the saved storyboard")
    parser.add_argument("--scenarios", action="store_true",
                        help="label every scene's scenario and rebuild the scenes that have a recipe (saved storyboard)")
    parser.add_argument("--fresh", action="store_true", help="with --scenarios: re-label and re-extract")
    parser.add_argument("--illustrate", action="store_true",
                        help="replace long holds on places and third repeats with object/setting illustrations")
    parser.add_argument("--ladder", action="store_true",
                        help="carry out the relevance ladder (reframe, hold, illustrate) on the saved storyboard")
    parser.add_argument("--key-facts", action="store_true",
                        help="label the physical clues the narration stresses (run last; --fresh re-asks the model)")
    parser.add_argument("--relook", action="store_true",
                        help="only re-run the colourist pass on the saved storyboard (no re-planning)")
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    import plan_edit
    plan_edit.use_case(video_dir)
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    bible = json.loads((video_dir / "bible.json").read_text(encoding="utf-8"))
    quotes_path = video_dir / "quotes.json"
    quotes = json.loads(quotes_path.read_text(encoding="utf-8")) if quotes_path.exists() else []
    if args.recheck:
        recheck(video_dir, args.free_only)
        return
    if args.ladder:
        run_ladder(video_dir, args.free_only)
        return
    if args.illustrate:
        run_illustrate(video_dir)
        return
    client = OpenAI(api_key=settings.openai_api_key)
    if args.scenarios:
        apply_scenarios(video_dir, client, args.model, args.free_only, args.fresh)
        return
    if args.relook:
        relook(video_dir, client, args.model)
        return
    if args.key_facts:
        run_key_facts(video_dir, client, args.model, args.fresh)
        return
    library, cat = load_library(video_dir, args.free_only)
    wanted = {int(x) for x in args.chapters.split(",") if x.strip()}
    chapters = [c for c in bible["chapters"] if not wanted or c["id"] in wanted]
    jobs = [(c, part) for c in chapters for part in chapter_parts(c)]
    brief = bible_brief(bible)
    print(f"{len(chapters)} chapters in {len(jobs)} parts; storyboarding with {args.model}")

    def one(job):
        chapter, part = job
        for attempt in range(3):
            try:
                board = plan_part(client, args.model, cat, brief, chapter, part, sentences, quotes)
                print(f"  chapter {chapter['id']} sentences {part[0]}-{part[1]}: {len(board.scenes)} scenes, "
                      f"{sum(len(s.beats) for s in board.scenes)} beats", flush=True)
                return chapter["id"], part, board
            except Exception as exc:  # noqa: BLE001 - one part may fail; its sentences still get beats
                if attempt == 2:
                    print(f"  FAILED chapter {chapter['id']} {part}: {str(exc)[:160]}", flush=True)
        return chapter["id"], part, None

    boards: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for chapter_id, part, board in pool.map(one, jobs):
            boards.setdefault(chapter_id, []).append((part, board))
    if wanted:
        bible = dict(bible, chapters=chapters)
    entries, scenes = build_entries(boards, bible, sentences)
    duration = sentences[-1]["end"] if not wanted else entries[-1]["end"]
    if args.events:
        events = json.loads(Path(args.events).read_text(encoding="utf-8"))
        print(f"  {land_on_effects(entries, events)} dips to black land on the whooshes already in the audio")
    finalise(entries, library, duration)
    print(f"  {respect_links(entries, library)} beats put back on a picture their own link chose")
    print(f"  {guard_dates(entries, library, {s['id']: s for s in scenes})} photographs dated after their scene replaced")
    print(f"  {guard_victims(entries, library, bible)} beats kept a person's photograph off a beat that does not name them")
    apply_looks(scenes, entries, assign_looks(client, args.model, scenes, bible))
    print(f"  {guard_mourning(entries, library, bible)} funeral or grave pictures moved out of non-memorial passages")
    keep_budget(entries)
    print(f"  {merge_repeats(entries, scenes)} back-to-back repeats joined into one hold")
    edit_transitions(entries)
    attach_quotes(entries, quotes, sentences)
    print(f"  {speaker_pictures(entries, bible, library)} quote beats now show their speaker")
    (video_dir / "storyboard.json").write_text(json.dumps({
        "audio": args.audio, "duration_s": round(duration, 3), "bible": bible, "scenes": scenes,
        "quotes": quotes, "beats": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "shots.json").write_text(json.dumps({
        "audio": args.audio, "duration_s": round(duration, 3), "width": 1920, "height": 1080, "fps": 30,
        "subtitles": "sidecar", "version": 4, "shots": entries,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "shots.md").write_text(readable(entries, library, duration), encoding="utf-8")
    lengths = sorted(e["end"] - e["start"] for e in entries)
    print(f"wrote storyboard.json and shots.json: {len(scenes)} scenes, {len(entries)} beats, "
          f"median {lengths[len(lengths) // 2]:.1f}s, {sum(len(e['quote_ids']) for e in entries)} quote cards")


if __name__ == "__main__":
    main()

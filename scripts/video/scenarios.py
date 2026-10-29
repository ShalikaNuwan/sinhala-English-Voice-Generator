"""Scenarios: passages of a recognisable kind get their own visual recipe instead of generic stand-ins.

The storyboard labels each scene with a scenario from SCENARIOS. A scenario with a handler rebuilds that
scene's beats with its recipe; the rest keep the beats the storyboard planned. Handlers emit ordinary beats,
so rendering, looks, checks and critics are unchanged.

roll_call: a list of people with ages and dates (the victims, in order). The model lists the entries; code
keeps only those whose words are in the narration, times each to the second its name is spoken, spells
names as the story bible and the picture library do, and draws one card per person: their portrait when the
library has one, otherwise a typographic memorial card; an unidentified victim gets a plain card; a year
heading ("In 1977, he killed ten people") gets its own card. Text on these cards is only what the narration
states.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field

import graphics

SCENARIOS = ("roll_call", "timeline", "statistic", "quote", "conversation", "introduction", "evidence",
             "geography", "legal", "atmosphere")
Scenario = Literal["roll_call", "timeline", "statistic", "quote", "conversation", "introduction", "evidence",
                   "geography", "legal", "atmosphere"]


# ------------------------------------------------------------------------------- labelling scenes

class SceneScenario(BaseModel):
    id: int
    scenario: Scenario


class SceneScenarios(BaseModel):
    scenes: list[SceneScenario]


LABEL_SYSTEM = """You label the scenes of a true-crime documentary by the kind of passage they are, so each
kind can be shown with its own recipe. Use the narration excerpt, not just the title.
- roll_call: the narration lists people one after another with ages and/or dates (e.g. the victims in order).
- timeline: a run of dated events told in sequence.
- statistic: the passage turns on numbers (how many, how long, how far).
- quote: it mainly reports someone's own words.
- conversation: an exchange between named people, line by line.
- introduction: it introduces a person.
- evidence: it walks through an object, document or clue.
- geography: it explains where places are.
- legal: charges, trial, verdict, sentence, appeals.
- atmosphere: anything else (story, mood, events without one of the above shapes).
Return every scene id once."""


def label_scenes(client, model: str, scenes: list[dict], sentences: list[dict]) -> dict[int, str]:
    lines = []
    for s in scenes:
        text = " ".join(x["text"] for x in sentences[s["first"]:s["last"] + 1])
        lines.append(f"[{s['id']}] {s['title']}: {text[:600]}")
    response = client.responses.parse(model=model, input=[{"role": "system", "content": LABEL_SYSTEM},
                                                          {"role": "user", "content": "\n\n".join(lines)}],
                                      text_format=SceneScenarios)
    return {x.id: x.scenario for x in response.output_parsed.scenes}


# --------------------------------------------------------------------------------------- roll call

class RollEntry(BaseModel):
    kind: Literal["person", "unidentified", "heading"]
    name: str = Field(default="", description="person: the full name exactly as the narration says it.")
    label: str = Field(default="", description="unidentified: the narration's words, e.g. 'Unidentified male, body number 5'.")
    age: str = Field(default="", description="The age as a number, only if stated, e.g. '18'.")
    date: str = Field(default="", description="The date exactly as stated, e.g. 'April 6th' or 'between June and August'.")
    year: str = Field(default="", description="heading: the year stated, e.g. '1977'. person: the year of the heading above it.")
    count: str = Field(default="", description="heading: the count exactly as stated, e.g. 'ten people'.")
    sentence: int = Field(description="The number of the sentence the entry is in.")


class RollCall(BaseModel):
    entries: list[RollEntry]


ROLL_SYSTEM = """List the entries of this roll call in the order they are spoken: each person with their age
and date, each unidentified victim with the narration's own words, and each year heading (a sentence like
"In 1977, he killed ten people") as a heading. A summary of the years before the list (e.g. "After he started
killing in 1972, he had killed only three people by 1975") is a heading too, its year the span it states
("1972–1975") and its count as stated ("three people"). Copy names, dates and counts exactly as the
narration has them. Do not add anything the narration does not say."""


def extract_roll(client, model: str, sentences: list[dict]) -> list[RollEntry]:
    text = "\n".join(f"({s['index']}) {s['text']}" for s in sentences)
    response = client.responses.parse(model=model, input=[{"role": "system", "content": ROLL_SYSTEM},
                                                          {"role": "user", "content": text}], text_format=RollCall)
    return response.output_parsed.entries


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().replace("’", "'")).strip()


def stated(entry: RollEntry, sentences: dict[int, dict]) -> bool:
    """Every field the card would show is in the narration of the entry's sentence (or the sentences either
    side, for a name the splitter cut at an initial: "Michael M." / "Marino, 14 years old")."""
    s = sentences.get(entry.sentence)
    if not s:
        return False
    around = [sentences[k]["text"] for k in (entry.sentence - 1, entry.sentence, entry.sentence + 1) if k in sentences]
    said = _norm(" ".join(around))
    fields = [entry.name if entry.kind == "person" else "", entry.label, entry.date, entry.count]
    if entry.kind == "heading":
        # A span ("1972–1975") is stated when each of its years is.
        fields += re.findall(r"\b\d{4}\b", entry.year) or [entry.year]
    if entry.age and not re.search(rf"\b{re.escape(entry.age)}\b", " ".join(around)):
        return False
    return all(_norm(f) in said for f in fields if f)


SUFFIX = re.compile(r"\b(jr|sr|ii|iii)\.?$", re.I)


def _parts(name: str) -> tuple[str, str]:
    words = SUFFIX.sub("", name.strip()).replace(",", " ").split()
    return (words[0].lower(), words[-1].lower()) if len(words) >= 2 else ("", "")


def canonical(name: str, known: list[str]) -> str:
    """The spelling a trusted source uses for this person (the story bible, the picture library, the case's
    reference article), if one matches closely: a near-identical given name and a close surname. Otherwise
    the narration's own spelling."""
    given, surname = _parts(name)
    if not given:
        return name
    best, score = name, 0.0
    for other in known:
        g, s_ = _parts(other)
        if not g or difflib.SequenceMatcher(None, given, g).ratio() < 0.8:
            continue
        s = difflib.SequenceMatcher(None, surname, s_).ratio()
        if s > score or (s == score and len(other) > len(best)):
            best, score = other, s
    if score < 0.6:
        return name
    return corrected(name, best)


def corrected(narrated: str, trusted: str) -> str:
    """The narrated full name with only its misspelled parts replaced: the trusted source's given name and
    surname, the narrator's middle names and suffix kept ("John Steven Prestige" + "Jon Prestidge" ->
    "Jon Steven Prestidge"; "Randall Wayne Reffett" stays whole)."""
    suffix = SUFFIX.search(narrated.strip())
    words = SUFFIX.sub("", narrated.strip()).replace(",", " ").split()
    good = SUFFIX.sub("", trusted.strip()).replace(",", " ").split()
    if len(words) < 2 or len(good) < 2:
        return trusted
    out = [good[0]] + words[1:-1] + [good[-1]]
    return " ".join(out) + (" " + suffix.group(0) if suffix else "")


REFERENCE_UA = "SinhalaEnglishVoice/1.0 (+https://github.com/ShalikaNuwan/sinhala-English-Voice-Generator)"


def reference_names(video_dir: Path, bible: dict) -> list[str]:
    """Personal names from the case's English Wikipedia article (the central person's), fetched once and
    cached in reference.json: a checkable source for how the people of the case spell their names."""
    path = video_dir / "reference.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("names", [])
    central = [p["name"] for p in bible.get("people") or [] if (p.get("importance") or 0) >= 3 and not p.get("victim")]
    case_path = video_dir / "case.json"
    if case_path.exists():                        # the case's own article, as case.py verified it
        central = [json.loads(case_path.read_text(encoding="utf-8"))["subject"]] + central
    if not central:
        return []
    import httpx
    try:
        r = httpx.get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "format": "json",
            "titles": central[0]}, headers={"User-Agent": REFERENCE_UA}, timeout=30)
        pages = r.json()["query"]["pages"]
        text = next(iter(pages.values())).get("extract") or ""
    except Exception:  # noqa: BLE001 - no reference is not an error; names stay as narrated
        return []
    names = sorted(set(re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+| [A-Z]\.){1,3}(?: Jr\.)?", text)))
    path.write_text(json.dumps({"title": central[0], "names": names}, ensure_ascii=False, indent=1), encoding="utf-8")
    return names


def spoken_at(entry: RollEntry, sentence: dict, tokens: list[dict], times: list) -> float | None:
    return spoken(entry, sentence, tokens, times)[0]


def spoken(entry: RollEntry, sentence: dict, tokens: list[dict], times: list) -> tuple[float | None, bool]:
    """When the entry begins, and whether that is exact (its own first word heard) or approximate."""
    at, exact = _spoken(entry, sentence, tokens, times)
    return at, exact


def _spoken(entry: RollEntry, sentence: dict, tokens: list[dict], times: list) -> tuple[float | None, bool]:
    """When the entry's first word is said: the name, the unidentified label, or the heading's year."""
    target = entry.name or entry.label or (f"In {entry.year}" if entry.year else "")
    first = _norm(target).split()[:1]
    if not first:
        return None, False
    heard = [(k, t) for k, t in enumerate(tokens) if t["sentence"] == sentence["index"] and times[k]]
    for k, t in heard:
        if t["norm"] == first[0]:
            return times[k][0], True
    # A name whisper misheard ("Darrel" as "Daryl"): the first word it did hear in the sentence, else the
    # sentence's own start; approximate, and spread out later between exact neighbours.
    if heard:
        return times[heard[0][0]][0], False
    return sentence.get("start"), False


def spread(entries: list[dict], end: float) -> None:
    """Entries timed only approximately are spaced evenly between the exactly timed entries around them,
    in spoken order, so no card is squeezed to nothing or two land on the same moment."""
    n = len(entries)
    k = 0
    while k < n:
        if entries[k].get("exact", True):
            k += 1
            continue
        j = k
        while j < n and not entries[j].get("exact", True):
            j += 1
        before = entries[k - 1]["at"] if k > 0 else entries[k]["at"]
        after = entries[j]["at"] if j < n else end
        slots = j - k + (1 if k > 0 else 0)
        for m, idx in enumerate(range(k, j)):
            entries[idx]["at"] = round(before + (after - before) * (m + (1 if k > 0 else 0)) / max(1, slots), 3)
        k = j


# --------------------------------------------------------------------------------------- the cards

NUMBERS = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
           "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
           "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20"}


def tidy_date(date: str) -> str:
    """One way of writing dates on every card: 'April 6th' -> 'April 6'; ranges keep their words."""
    return re.sub(r"\b(\d{1,2})(st|nd|rd|th)\b", r"\1", date or "").strip()


def tidy_count(count: str) -> str:
    """Counts as numerals on every card: 'ten people' -> '10 people'."""
    return " ".join(NUMBERS.get(w.lower(), w) for w in (count or "").split())


def split_unidentified(label: str) -> tuple[str, str]:
    """'Unidentified male, body number 5' -> ('Unidentified victim', 'Body No. 5'): a person first, the record second."""
    m = re.search(r"body (?:number|no\.?)\s*(\d+)", label or "", re.I)
    return "Unidentified victim", (f"Body No. {m.group(1)}" if m else "")


def card_image(entry: dict, portrait: Path | None, width: int = 1920, height: int = 1080) -> Image.Image:
    """A memorial card: soft black, the portrait as a print on the left when there is one, the name in a
    serif with the stated age and date beneath. A heading card shows the year and the stated count."""
    k = width / 1920
    img = Image.new("RGB", (width, height), (12, 12, 13))
    vignette = Image.new("L", (width, height), 0)
    ImageDraw.Draw(vignette).ellipse((-width * 0.2, -height * 0.3, width * 1.2, height * 1.3), fill=255)
    img = Image.composite(img, Image.new("RGB", (width, height), (4, 4, 5)), vignette.filter(ImageFilter.GaussianBlur(220)))
    d = ImageDraw.Draw(img)
    soft, muted = (236, 233, 226), (178, 174, 166)
    if entry["kind"] == "heading":
        big = graphics.font(graphics.SERIF, int(150 * k))
        small = graphics.font(graphics.LABEL, int(40 * k))
        year = entry.get("year") or ""
        d.text(((width - big.getlength(year)) / 2, height * 0.36), year, font=big, fill=soft)
        line = tidy_count(entry.get("count") or "")
        d.text(((width - small.getlength(line)) / 2, height * 0.36 + 190 * k), line, font=small, fill=muted)
        return img
    text_left = None                      # None: no portrait, so the text block is centred
    if portrait is not None and portrait.exists():
        photo = ImageOps.grayscale(Image.open(portrait)).convert("RGB")
        # Fit to the frame's portrait box; a small archival scan is enlarged, but never past 2.2 times its
        # own size (the same limit the photo prints use), so it does not turn to mush.
        box_w, box_h = int(560 * k), int(700 * k)
        scale = min(box_w / photo.width, box_h / photo.height, 2.2)
        photo = photo.resize((max(1, int(photo.width * scale)), max(1, int(photo.height * scale))), Image.LANCZOS)
        border = max(4, int(10 * k))
        framed = ImageOps.expand(photo, border=border, fill=(232, 229, 222))
        px, py = int(width * 0.14), (height - framed.height) // 2
        shadow = Image.new("L", (framed.width + 80, framed.height + 80), 0)
        ImageDraw.Draw(shadow).rectangle((40, 40, 40 + framed.width, 40 + framed.height), fill=150)
        img.paste((0, 0, 0), (px - 30, py - 22), shadow.filter(ImageFilter.GaussianBlur(18)))
        img.paste(framed, (px, py))
        text_left = px + framed.width + int(110 * k)
    name_font = graphics.font(graphics.SERIF, int(78 * k))
    detail_font = graphics.font(graphics.LABEL, int(52 * k))
    muted = (198, 194, 186)                     # secondary text, readable on a phone
    name = entry.get("display") or entry.get("name") or entry.get("label") or ""
    record = ""
    if entry["kind"] == "unidentified":
        name, record = split_unidentified(entry.get("label") or name)
    centred = text_left is None
    max_w = width * 0.8 if centred else width - text_left - width * 0.08
    lines = graphics.wrap(name, name_font, max_w)[:2]
    x_of = (lambda w: (width - w) / 2) if centred else (lambda w: text_left)
    y = height * 0.44 - len(lines) * 50 * k
    for line in lines:
        d.text((x_of(name_font.getlength(line)), y), line, font=name_font, fill=soft)
        y += 96 * k
    details = [record] if record else []
    if entry.get("age"):
        details.append(f"{entry['age']} years old")
    date = tidy_date(entry.get("date") or "")
    if date:
        details.append(date + (f", {entry['year']}" if entry.get("year") and not date.startswith("between") else
                               (f" {entry['year']}" if entry.get("year") else "")))
    elif entry.get("year"):
        details.append(entry["year"])
    detail = "  ·  ".join(details)
    d.text((x_of(detail_font.getlength(detail)), y + 20 * k), detail, font=detail_font, fill=muted)
    # The register's year stays on every card, so a new year is never missed between its headings.
    if entry.get("year"):
        label = graphics.font(graphics.LABEL, int(34 * k))
        year = str(entry["year"])
        d.text(((width - label.getlength(year)) / 2, height * 0.09), year, font=label, fill=(150, 146, 140))
    return img


# ------------------------------------------------------------------------------- the scene rebuild

def roll_call_beats(scene: dict, old: list[dict], entries: list[dict], sentences: list[dict], video_dir: Path,
                    library: dict[str, dict]) -> list[dict]:
    """The scene's beats with one card per entry, cut when it is spoken. The narration before the first
    entry keeps the beats the storyboard planned (trimmed to end where the first card begins)."""
    if not entries:
        return old
    out = []
    first_at = entries[0]["at"]
    for b in old:
        if b["start"] >= first_at:
            break
        b = dict(b)
        b["end"] = min(b["end"], first_at)
        if b["end"] - b["start"] >= 1.0:
            out.append(b)
    if out:
        out[-1]["end"] = first_at
    else:
        first_at = scene["start"]
        entries[0]["at"] = first_at
    folder = video_dir / "assets" / "cards"
    folder.mkdir(parents=True, exist_ok=True)
    # One portrait policy for the whole register: portraits only if every named person in it has one, so no
    # victim is given more weight than another by having a face where the others have none.
    named = [e for e in entries if e["kind"] == "person"]
    portraits = bool(named) and all(e.get("photo") in library for e in named)
    for n, e in enumerate(entries):
        start = e["at"]
        end = entries[n + 1]["at"] if n + 1 < len(entries) else scene["end"]
        if end - start < 0.5:
            continue
        portrait = video_dir / library[e["photo"]]["file"] if portraits and e.get("photo") in library else None
        # The card design is part of the key: a change to card_image redraws every card.
        design = hashlib.sha1(card_image.__code__.co_code + repr(card_image.__code__.co_consts).encode()).hexdigest()[:8]
        key = hashlib.sha1(json.dumps([e, str(portrait), design], sort_keys=True, default=str).encode()).hexdigest()[:10]
        path = folder / f"card_{key}.png"
        if not path.exists():
            card_image(e, portrait).save(path)
        # The card's words are its own entry's sentence (and any up to the next entry's), not whatever the
        # approximate sentence times overlap: a misheard stretch must not label a card with the wrong name.
        own = e.get("sentence")
        nxt = entries[n + 1].get("sentence") if n + 1 < len(entries) else None
        spoken = [s for s in sentences if own is not None and own <= s["index"] < (nxt if nxt is not None else own + 1)]
        if not spoken:
            spoken = [s for s in sentences if s["end"] > start and s["start"] < end]
        out.append({
            "start": start, "end": end, "narration": " ".join(s["text"] for s in spoken),
            "sentences": [spoken[0]["index"], spoken[-1]["index"]] if spoken else None,
            "kind": "card", "media": "photo", "asset": str(path.relative_to(video_dir)), "colour": "bw",
            # Card to card is a cut: dissolving one name into the next smears the text. The first card
            # dissolves in from the passage before it.
            "library_id": None, "choices": [], "motion": "still",
            "transition_in": "dissolve" if not any(b.get("scenario") == "roll_call" for b in out) else "cut",
            "overlay": None,
            "look": "memorial", "effects": [], "quote_ids": [], "quotes": [], "scenario": "roll_call",
            "intent": f"Remember {e.get('display') or e.get('label') or e.get('year')}.",
            "subjects": [e["display"]] if e.get("display") else [], "shot_type": "memorial",
            "rights": "original", "licence": "Original artwork", "attribution": "Drawn for this film",
            "chapter": scene["chapter"], "scene": scene["id"], "intensity": scene["intensity"],
        })
    return out


def plan_roll_call(client, model: str, scene: dict, sentences: list[dict], tokens: list[dict], times: list,
                   bible: dict, library: dict[str, dict], video_dir: Path) -> list[dict]:
    """The scene's roll-call entries, checked against the narration, timed, spelled and matched to portraits."""
    here = sentences[scene["first"]:scene["last"] + 1]
    by_index = {s["index"]: s for s in here}
    raw = extract_roll(client, model, here)
    people = bible.get("people") or []
    # Only trusted spellings can correct a name: the picture library's captions and the case's reference
    # article. The bible's names came from the narration itself, misspellings included.
    known = sorted({n for i in library.values() for n in (i.get("subjects") or [])}) + reference_names(video_dir, bible)
    photos: dict[str, str] = {}
    for p in people:
        if p.get("photo_ids"):
            photos.setdefault(p["name"], p["photo_ids"][0])
    for i in library.values():
        if i.get("media") != "clip" and len(i.get("subjects") or []) == 1 and not i.get("excluded"):
            photos.setdefault(i["subjects"][0], i["id"])
    entries, year = [], ""
    for r in raw:
        if not stated(r, by_index):
            continue
        at, exact = spoken(r, by_index[r.sentence], tokens, times)
        if at is None:
            continue
        if r.kind == "heading":
            year = r.year or year
        e = r.model_dump()
        e["year"] = e["year"] or year
        e["at"] = round(at, 3)
        e["exact"] = exact
        if r.kind == "person":
            e["display"] = canonical(r.name, known)
            if e["display"] != r.name:
                e["narrated_as"] = r.name           # shown on the storyboard page for the owner to confirm
            e["photo"] = photos.get(e["display"])
        elif r.kind == "unidentified":
            e["display"] = r.label[:1].upper() + r.label[1:]
        entries.append(e)
    entries.sort(key=lambda x: (x["sentence"], x["at"]))      # spoken order is sentence order
    spread(entries, scene["end"])
    return entries


HANDLERS = {"roll_call": (plan_roll_call, roll_call_beats)}

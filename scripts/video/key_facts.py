"""Key-fact labels: a physical clue the narration stresses gets a short label on screen, in its own words.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/storyboard.py <video_dir> --key-facts [--fresh]

When the narration makes an object matter ("That day, Rob was wearing a blue jacket ... That jacket is
important to this story") and the archive has no picture of it, the clue is said but never shown. A key-fact
label shows it: the words, in small capitals, over a blurred and darkened copy of the picture on screen at
that moment, for about KEY_FACT_S. A model lists the clues; code keeps only labels that are

- the narration's own words (every word of the label is spoken in the sentence it cites),
- an object, not a person, place or date, and nothing gruesome,
- on a sentence that mentions the object, at least KEY_FACT_GAP_S after the label before.

The label cuts in when the object's word is spoken and cuts back to the same picture afterwards, so the
shot it interrupts carries on. Beats with a quote card or inside a roll call are never interrupted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel, Field

import graphics
from align_script import normalise

KEY_FACT_S = 3.0
KEY_FACT_GAP_S = 60.0        # the same clue, or another, no sooner than this after a label
MIN_PART_S = 3.0             # a piece of shot shorter than this either side of a label is absorbed into it
KEY_FACT_MAX_S = 6.0         # but a label is never held longer than this
AFTER_WORD_S = 2.0           # and stays at least this long after its word is said
MAX_WORDS = 3
CRIMSON = (155, 17, 30)
GRUESOME = re.compile(r"\b(body|bodies|remains|blood|corpse|grave|rope|knife|strangl\w*|skull|bone\w*|crawl)\b", re.I)


class KeyFact(BaseModel):
    label: str = Field(description="One to three words copied exactly from the narration, e.g. 'blue jacket'.")
    stated_in: int = Field(description="The number of the sentence where all the label's words are said.")
    at: list[int] = Field(description="The sentences where the label should appear: where the narration makes the "
                                      "object matter (often where it says it is important), and later returns to "
                                      "it as a clue. Each must mention the object.")


class KeyFacts(BaseModel):
    facts: list[KeyFact]


SYSTEM = """You mark the physical clues in a true-crime narration that the viewer must remember: an object the
narration explicitly makes matter (it says the object is important, or the case later turns on it), such as
a jacket, a receipt or a ring. Never a person, a place, a date, a vehicle the narration does not stress, or
anything violent or gruesome. Usually there are very few in a whole film; list none rather than weak ones.
Copy the label from the narration exactly (one to three words)."""


def find_facts(client, model: str, sentences: list[dict], chunk: int = 400) -> list[KeyFact]:
    out: list[KeyFact] = []
    for k in range(0, len(sentences), chunk):
        part = sentences[k:k + chunk]
        text = "\n".join(f"({s['index']}) {s['text']}" for s in part)
        r = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                       {"role": "user", "content": text}], text_format=KeyFacts)
        out += r.output_parsed.facts
    return out


def words_of(text: str) -> list[str]:
    return [w for w in (normalise(x) for x in re.split(r"[\s\-–—]+", text or "")) if w]


def valid(fact: KeyFact, by_index: dict[int, dict], bible: dict) -> bool:
    """The label is the narration's words, short, about an object, and not about a person or gruesome."""
    label = words_of(fact.label)
    said = by_index.get(fact.stated_in)
    if not label or len(label) > MAX_WORDS or not said or GRUESOME.search(fact.label):
        return False
    if " ".join(label) not in " ".join(words_of(said["text"])):
        return False
    names = {w for p in (bible.get("people") or []) + (bible.get("places") or [])
             for w in words_of(p.get("name") or "")}
    return not (set(label) & names)


def anchors(fact: KeyFact, by_index: dict[int, dict]) -> list[int]:
    """The sentences the label may appear on: those that mention the object (the label's last word)."""
    noun = words_of(fact.label)[-1]
    return sorted({k for k in fact.at if k in by_index and noun in words_of(by_index[k]["text"])})


def word_time(sentence: dict, word: str, tokens: list[dict], times: list) -> float:
    """When `word` is said in this sentence; the sentence's start when whisper did not hear it."""
    for k, t in enumerate(tokens):
        if t["sentence"] == sentence["index"] and t["norm"] == word and times[k]:
            return times[k][0]
    return sentence["start"]


def label_time(sentence: dict, label: list[str], tokens: list[dict], times: list) -> float:
    """When the label starts being said: its first word where the sentence says the whole label ("tan" of
    "tan Volkswagen Beetle"), else its object word ("jacket" in "That jacket is important")."""
    said = words_of(sentence["text"])
    if " ".join(label) in " ".join(said):
        return word_time(sentence, label[0], tokens, times)
    return word_time(sentence, label[-1], tokens, times)


def card(label: str, background: Path | None, width: int = 1920, height: int = 1080) -> Image.Image:
    """The label in letter-spaced small capitals with a short crimson rule, over the picture on screen blurred
    and darkened (or plain near-black when there is none)."""
    k = width / 1920
    if background is not None and background.exists():
        img = Image.open(background).convert("RGB")
        scale = max(width / img.width, height / img.height)
        img = img.resize((int(img.width * scale) + 1, int(img.height * scale) + 1), Image.LANCZOS)
        left, top = (img.width - width) // 2, (img.height - height) // 2
        img = img.crop((left, top, left + width, top + height)).filter(ImageFilter.GaussianBlur(28 * k))
        img = Image.blend(img, Image.new("RGB", (width, height), (8, 8, 9)), 0.72)
    else:
        img = Image.new("RGB", (width, height), (12, 12, 13))
    d = ImageDraw.Draw(img)
    fnt = graphics.font(graphics.LABEL, int(66 * k))
    text = label.upper()
    tracking = 0.22 * fnt.size
    widths = [fnt.getlength(ch) for ch in text]
    total = sum(widths) + tracking * (len(text) - 1)
    x, y = (width - total) / 2, height * 0.46
    for ch, w in zip(text, widths):
        d.text((x, y), ch, font=fnt, fill=(238, 235, 228))
        x += w + tracking
    rule_w, rule_y = 110 * k, y + fnt.size * 1.5
    d.line(((width - rule_w) / 2, rule_y, (width + rule_w) / 2, rule_y), fill=CRIMSON, width=max(3, int(5 * k)))
    return img


def insert(entries: list[dict], at: float, label: str, video_dir: Path) -> bool:
    """Split the beat on screen at `at` around a KEY_FACT_S label beat. False when it may not be interrupted."""
    k = next((n for n, e in enumerate(entries) if e["start"] <= at < e["end"]), None)
    if k is None:
        return False
    e = entries[k]
    if e.get("quote_ids") or e.get("scenario") == "roll_call" or e.get("kind") in ("card", "keyfact"):
        return False
    start = e["start"] if at - e["start"] < MIN_PART_S else at
    if e["end"] - start < KEY_FACT_S:
        # Too near the shot's end to be read: the label starts earlier in the shot (while the sentence is
        # being said), or takes the whole shot.
        start = max(e["start"], e["end"] - KEY_FACT_S)
        if start - e["start"] < MIN_PART_S:
            start = e["start"]
    end = min(e["end"], start + max(KEY_FACT_S, at - start + AFTER_WORD_S))
    if e["end"] - end < MIN_PART_S and e["end"] - start <= KEY_FACT_MAX_S:
        end = e["end"]
    background = video_dir / e["asset"] if e.get("media") == "photo" and e.get("asset") else None
    design = hashlib.sha1(card.__code__.co_code + repr(card.__code__.co_consts).encode()).hexdigest()[:8]
    key = hashlib.sha1(json.dumps([label, str(background), design]).encode()).hexdigest()[:10]
    path = video_dir / "assets" / "cards" / f"keyfact_{key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        card(label, background).save(path)
    fact = dict(e)
    fact.update({"start": start, "end": end, "kind": "keyfact", "media": "photo", "motion": "still", "colour": "colour",
                 "asset": str(path.relative_to(video_dir)), "library_id": None, "choices": [], "crop": None,
                 "effects": [], "overlay": None, "transition_in": "cut", "key_fact": label,
                 "key_fact_original": {k: v for k, v in e.items() if k != "key_fact_original"},  # for undoing it
                 "rights": "original", "licence": "Original artwork", "attribution": "Drawn for this film"})
    for key_ in ("clip_id", "clip_in", "clip_out"):
        fact.pop(key_, None)
    parts = []
    if start > e["start"]:
        parts.append(dict(e, end=start))
    parts.append(fact)
    if end < e["end"]:
        after = dict(e, start=end, transition_in="cut")
        if e.get("media") == "clip":
            after["clip_in"] = float(e.get("clip_in") or 0) + (end - e["start"])
        parts.append(after)
    entries[k:k + 1] = parts
    return True


def place(entries: list[dict], facts: list[KeyFact], sentences: list[dict], tokens: list[dict], times: list,
          bible: dict, video_dir: Path) -> list[dict]:
    """Insert every valid label at its anchors, KEY_FACT_GAP_S apart; returns what was placed."""
    by_index = {s["index"]: s for s in sentences}
    wanted = []
    for f in facts:
        if not valid(f, by_index, bible):
            continue
        for a in anchors(f, by_index):
            wanted.append((label_time(by_index[a], words_of(f.label), tokens, times), f.label, a))
    placed, last = [], -1e9
    for at, label, a in sorted(wanted):
        if at - last < KEY_FACT_GAP_S:
            continue
        if insert(entries, at, label, video_dir):
            placed.append({"label": label, "sentence": a, "at": round(at, 2)})
            last = at
    return placed

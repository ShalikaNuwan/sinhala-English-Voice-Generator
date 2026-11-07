"""Regenerate one shot from the owner's review: a text model chooses its new picture, a second model checks it.

The owner writes what is wrong with a shot ("show the house, not a street"). The text model reads that review
with the narration, the scene and the film's usable free pictures and footage, and answers with a library
picture (optionally a closer framing of a photograph), searches when nothing fits (the free archives and the
web; then it chooses again among the new pictures only), or, only after a search found nothing fitting, an
illustration of an object or setting. Pictures from the web are copyrighted and credited as fair use. The choice is applied through
video_review.choose(), as Choose another… does; a vision model checks the new picture against the review, and
one rejection earns one more choice with the checker's reason. Then build.py renders just that shot (480p, with
its own narration) for the owner to play and approve.

State per shot lives in <film>/review/regen.json (video_review.regen_view reads it for the page). Every model
call and external step is injectable. Regenerator runs one regeneration at a time on a worker thread.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from . import config
from . import video_review as review
from .video_jobs import load_env

ROOT = review.ROOT
SCRIPTS = review.VIDEO_SCRIPTS
MAX_OPTIONS = 400              # pictures offered to the text model
CAPTION_CHARS = 160
MAX_QUERIES = 3
MAX_WEB_PAGES = 6
LIBRARY_ID = re.compile(r"\b[a-z]\d{1,5}(?:#\d+)?\b")   # a library id as a review may name it ("c0292", "v0021#0")
# Stock libraries (watermarked, sold), social networks and AI-image sites are never searched for pictures.
WEB_SKIP = ("gettyimages.", "alamy.", "shutterstock.", "istockphoto.", "dreamstime.", "123rf.", "depositphotos.",
            "pinterest.", "facebook.", "instagram.", "tiktok.", "x.com", "twitter.", "reddit.", "lexica.",
            "midjourney.", "openart.", "nightcafe.")
NEARBY_SHOTS = 3              # shots either side whose pictures a new choice must not repeat
RENDER_TIMEOUT_S = 30 * 60
SEARCH_TIMEOUT_S = 45 * 60
STEPS = {"queued": "Queued", "choose": "Choosing a picture…", "search": "Searching the archives and the web…",
         "check": "Checking…", "render": "Rendering the shot…", "done": "Done"}
_NOTE_LOCK = threading.Lock()
# Regenerations may run side by side (Regenerator workers); these steps write the film's shared files, so they
# take turns: a search rewrites library.json and library/needs.json, a render shots.filled.json and credits.md.
SEARCH_LOCK = threading.Lock()
RENDER_LOCK = threading.Lock()


class RegenError(Exception):
    """A regeneration that cannot go on (nothing fitting, an id not offered, a failed render)."""


class RegenBusy(Exception):
    """This shot is already queued or being regenerated."""


class Decision(BaseModel):
    action: Literal["library", "illustration", "search", "keep"]
    library_id: str = Field(default="", description="library: the id exactly as listed in the options.")
    focus: str = Field(default="", description=(
        "library, photographs only: a few plain words for a closer framing ('his face', 'the front door'); "
        "empty for the whole picture."))
    detail: str = Field(default="", description=(
        "illustration: one short concrete line, an object or a setting in the scene's period; never a person "
        "or a named place."))
    queries: list[str] = Field(default_factory=list, description="search: 1-3 short archive search phrases.")
    name_card: str = Field(default="", description=(
        "When the review asks for a name card or caption naming someone or something: the name, spelled as the "
        "narration spells it. The film draws it over the picture as clean text. Empty otherwise."))
    name_role: str = Field(default="", description=(
        "With name_card: a few words from the narration on who or what it is (\"Ted's grandfather\"); "
        "never an invented fact."))
    name_hold: bool = Field(default=False, description=(
        "With name_card: true when the review asks for the name to stay up for the whole shot; otherwise the "
        "card shows for its first few seconds."))
    remove_card: bool = Field(default=False, description=(
        "True when the review asks to take a name or date card off the shot (\"remove that name\"); the picture "
        "stays unless the review also asks for a new one."))
    explanation: str = Field(description="One line for the owner: what the shot now shows and why.")


class WebPage(BaseModel):
    page_url: str = Field(description="The web page the photograph is on.")
    image_url: str = Field(default="", description="The photograph's own file address when seen; else empty.")
    subject: str = Field(description="A few words on what the photograph shows.")


class WebFinds(BaseModel):
    pages: list[WebPage] = Field(default_factory=list)


class Verdict(BaseModel):
    satisfied: bool = Field(description="Does the new picture do what the review asks, within the film's rules?")
    problem: str = Field(default="", description="When not satisfied: one short sentence on what is still wrong.")


CHOOSE = """You fix one shot of a true-crime documentary built from real archival photographs and footage.
The owner watched the shot and wrote a review of what is wrong with it. Read the review, the narration over
the shot and the scene, and make a new version of this one shot that does what the review asks.

Answer with one action:
- library: the id of one picture or clip from the options, copied exactly. To frame a photograph closer (a
  face, a doorway, an object in it), also give focus: a few plain words for what to frame ("his face", "the
  front door"). Focus works on photographs only, not on footage (ids with '#'). The current picture may be
  chosen again with a focus when the review asks for a closer view of it.
- illustration (only once a search has found nothing that fits): detail = one short concrete line for a drawn picture of an OBJECT or a SETTING in the scene's
  period, e.g. "a 1970s kitchen table with a radio and a coffee cup, morning light". Never a person, a face,
  hands or a body; never a named or recognisable real place, landmark, sign or shop name. Never writing of any
  kind: no name card, label, letter, document, newspaper or sign with words (the image model invents names).
  Use it only when no option fits and an object or setting can carry the words.
- keep: leave the current picture as it is, e.g. when the review only asks for a name card, or when no other
  picture would be better.
- search: 1-3 short search phrases when nothing offered fits. The search covers the free archives and the
  web (news sites, Wikipedia, official pages), so it finds real photographs; always search before drawing. Archives match the words of file titles, so write what a title would say: two to four plain words,
  the thing and its name or place ("Chi Omega Tallahassee", "Volkswagen Beetle 1972", "Des Plaines police
  station"). Leave out colours, moods and words like "1970s" or "photo"; make the phrases differ (one
  specific, one broader) rather than repeat each other.
A name card: when the review asks to show a name, give name_card (and name_role) with any action; the film
draws the name over the picture itself. It comes up by itself as the name is first said in the shot (from the
start if it isn't said there) and shows for a few seconds; set name_hold when the review asks for it to stay
through the rest of the clip. A review asking to take a card or name off the picture: set remove_card (with
"keep" unless it also asks for another picture). Never put a name into an illustration's detail.
A person's name card goes only over a picture of that person (or one they are clearly part of). When no
picture of the person is found, do not put their name over a drawing, an empty room or an unrelated place:
the viewer would take it for them or for their own room. Use a place or date card that fits the picture
instead ("Puyallup, Washington" with "May 27, 1974"), or no card, and say so in the explanation.
A present-day photo of a place gets a card that says so ("the building today", "today ...").
Always give explanation: one line for the owner naming the picture and why it answers the review, e.g. "Used
c0022, Bundy's former house (case photo): the review asked for the house".

The film's rules:
- Archive options are public domain or CC0. Options found on the web ("news: ..." in where found) are real
  photographs from news and other sites, copyrighted and credited as fair use: use one when it shows what the
  review asks for and no archive picture shows it as well. Use only a listed id; never invent one.
- Never remains, bodies, gore or a crime scene's horror. Victims are shown with dignity: their own portraits
  and their lives, never their deaths.
- Options read: id | category | where found | era | caption. actual = a real picture of this case itself
  (the person, place or thing named); stand-in = a real picture of something like it, or of the period.
- A stand-in must not pretend to be the actual person or place: never another man for the person named,
  another house as "his house", or another town's street for the town named. A period scene, a road or an
  object of the era is an honest stand-in.
- When the review asks for the real person, place or thing, choose an actual picture that shows it; if none
  does, search for it rather than pass a stand-in off as it.
- Keep the film's era: the picture must belong to the scene's time (no modern cars, clothes, signs or
  buildings in a 1970s scene), unless the narration speaks of a later time.
- Do what the review asks. Where it would break these rules, keep the rules and say so in the explanation."""

CHECK = """You check one regenerated shot of a true-crime documentary. The owner reviewed the shot and wrote
what was wrong; an editor then chose the new picture you see. Judge only what is visible, against the review,
the narration and the scene's time and place.
- satisfied: true if the picture does what the review asks and still fits the narration. False if it ignores
  or contradicts the review, shows something else, belongs to another era, presents a stand-in as the actual
  person or place, or shows remains, a body, or a victim without dignity. A drawn illustration of an object
  or setting is fine where the review allows it. A name card the film draws over the picture (named in the
  description; it is added in the edit, so it is not in the image you see) counts as showing that name. Any
  name or writing inside the picture itself that is not the name asked for is a problem.
- problem: when not satisfied, one short sentence on what is still wrong, concrete enough to choose again
  (e.g. "this is a street; the review asked for the house itself"). Empty when satisfied."""


# ---------- state ----------

def note(d: Path, n: int, **fields) -> dict:
    """Update a shot's entry in review/regen.json (written whole, atomically)."""
    with _NOTE_LOCK:
        data = review.load_regen(d)
        entry = data.get(str(n)) or {}
        entry.update(fields, at=time.time(), worker=review.WORKER)
        data[str(n)] = entry
        path = review.regen_path(d)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return dict(entry)


# ---------- choosing ----------

def web_finds(d: Path) -> frozenset:
    """Ids of the pictures regenerations found on the web (copyrighted, credited as fair use)."""
    return frozenset(k for k, v in review.load_library(d).items() if v.get("rights") == "fair_use")


def options(d: Path, n: int, fair_use: frozenset = frozenset(), named=()) -> list[dict]:
    """The film's usable free pictures and long-enough clips: the ids the review names first, then the shot's
    own picture and choices, then its scene's, then the web finds, then the rest of the library. Excluded and
    graphic pictures are never offered. Copyrighted pictures are offered only when named in `fair_use` (the web
    finds)."""
    storyboard = review.engine()
    sb = review.load_storyboard(d)
    beat = review.find_beat(sb, n)
    library = review.load_library(d, free_only=True)
    if fair_use:
        library.update({k: v for k, v in review.load_library(d).items() if k in fair_use})
    scene_choices = [c for b in sb["beats"] if b.get("scene") == beat.get("scene") for c in b.get("choices") or []]
    duration = beat["end"] - beat["start"]
    out = []
    for c in dict.fromkeys(list(named) + [beat.get("library_id") or ""] + (beat.get("choices") or [])
                           + scene_choices + sorted(fair_use) + list(library)):
        item = library.get(c)
        if not item or not storyboard.usable(item):
            continue
        if item.get("media") == "clip" and item.get("length", 0) < duration + 1.0:
            continue
        out.append({"id": c, "media": item.get("media") or "photo", "caption": item.get("caption", ""),
                    "found": review.found_words(item), "era": item.get("era") or "",
                    "category": review.picture_category(item)})
        if len(out) >= MAX_OPTIONS:
            break
    return out


def option_line(o: dict) -> str:
    caption = " ".join((o.get("caption") or "").split())[:CAPTION_CHARS]
    return f"{o['id']} | {o['category']} | {o.get('found') or '-'} | {o.get('era') or '-'} | {caption}"


def current_words(beat: dict, current: dict | None) -> str:
    if beat.get("kind") == "generated" or beat.get("rights") == "generated":
        return f"an illustration: {beat.get('detail') or '-'}"
    if current:
        closer = "; framed closer on a detail" if beat.get("crop") else ""
        return f"{option_line(review_option(current))}{closer}"
    return review.kind_words(beat)


def review_option(item: dict) -> dict:
    return {"id": item["id"], "caption": item.get("caption", ""), "found": review.found_words(item),
            "era": item.get("era") or "", "category": review.picture_category(item)}


def nearby_pictures(sb: dict, n: int, library: dict[str, dict], reach: int = NEARBY_SHOTS) -> list[str]:
    """What the shots around shot n show (a drawing's detail or a picture's caption), nearest first."""
    around = sorted((b for b in sb["beats"] if b["index"] != n and abs(b["index"] - n) <= reach),
                    key=lambda b: (abs(b["index"] - n), b["index"]))
    lines = []
    for b in around:
        what = b.get("detail") or (library.get(b.get("library_id") or "") or {}).get("caption") or ""
        if what:
            gap = abs(b["index"] - n)
            lines.append(f"{gap} shot{'s' if gap > 1 else ''} {'before' if b['index'] < n else 'after'}: {what[:160]}")
    return lines


def decide(review_text: str, beat: dict, scene: dict, current: dict | None, offered: list[dict],
           allow_search: bool, feedback: str, ask, rules=(), nearby=(), allow_drawing: bool = True,
           drawings: bool = True) -> Decision:
    """The text model's choice for this shot, checked against what was offered."""
    parts = [f"The owner's review of this shot (verbatim):\n{review_text}\n",
             f"Narration over the shot: {beat.get('narration') or '-'}",
             f"Scene: {scene.get('title') or '-'} - {scene.get('setting') or '-'}, {scene.get('story_time') or '-'}. "
             f"Mood: {scene.get('mood') or '-'}.",
             f"Current picture: {current_words(beat, current)}"]
    if rules:
        parts.append("This film's care rules:\n" + "\n".join(f"- {r}" for r in list(rules)[:8]))
    if nearby:
        parts.append("Shots around this one show (choose something clearly different: not another of the same "
                     "kind of room, object or view):\n" + "\n".join(f"- {line}" for line in nearby))
    if feedback:
        parts.append(f"\n{feedback}")
    if not drawings:
        parts.append("\nDrawings are switched off for this film: never answer illustration. "
                     + ("The options below are what the search just found: choose one of them, or keep the current "
                        "picture if none fits (a card may still be added or changed)." if not allow_search else
                        "Choose a real picture from the options, search for one, or keep the current picture."))
    elif not allow_search:
        parts.append("\nSearch is not available now: the options below are what the search just found. "
                     "Choose one of them, or an illustration if none fits.")
    elif not allow_drawing:
        parts.append("\nAn illustration is not available yet: choose a real picture from the options, or search "
                     "for one. A drawing is made only when a search has found nothing that fits.")
    listing = "\n".join(option_line(o) for o in offered) or "(none)"
    parts.append(f"\nOptions (id | category | where found | era | caption):\n{listing}")
    decision = ask(CHOOSE, "\n".join(parts))
    if decision.action == "library":
        decision.library_id = decision.library_id.strip()
        if decision.library_id not in {o["id"] for o in offered}:
            raise RegenError("nothing fitting was found" if not offered
                             else f"the model chose {decision.library_id or 'nothing'}, which is not on offer")
    elif decision.action == "search":
        if not allow_search:
            raise RegenError("nothing fitting was found in the archives")
        decision.queries = [q.strip() for q in decision.queries if q.strip()][:MAX_QUERIES]
        if not decision.queries:
            raise RegenError("the model asked for a search without saying what to search for")
    elif decision.action == "illustration" and not decision.detail.strip():
        raise RegenError("the model asked for an illustration without saying what to draw")
    return decision


def client():
    from openai import OpenAI
    # Several regenerations run at once and share the account's tokens-per-minute limit: a 429 waits and retries
    # (the client backs off, honouring the server's retry-after) instead of failing the shot.
    return OpenAI(api_key=config.settings.openai_api_key or load_env().get("OPENAI_API_KEY"), max_retries=10)


def ask_model(system: str, user: str) -> Decision:
    r = client().responses.parse(model=config.settings.text_model, text_format=Decision,
                                 input=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if not r.output_parsed:
        raise RegenError("the model gave no answer")
    return r.output_parsed


def verify(image_bytes: bytes, review_text: str, beat: dict, scene: dict, look: str) -> Verdict:
    """The checking model (settings.qa_model) looks at the new picture against the review."""
    text = (f"The owner's review (verbatim):\n{review_text}\n\nNarration: {beat.get('narration') or '-'}\n"
            f"Scene: {scene.get('title') or '-'} - {scene.get('setting') or '-'}, {scene.get('story_time') or '-'}.\n"
            f"What the editor chose: {look}")
    image = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
    r = client().responses.parse(model=config.settings.qa_model, text_format=Verdict, input=[
        {"role": "system", "content": CHECK},
        {"role": "user", "content": [{"type": "input_text", "text": text},
                                     {"type": "input_image", "image_url": image, "detail": "high"}]}])
    if not r.output_parsed:
        raise RegenError("the checker gave no answer")
    return r.output_parsed


# ---------- applying ----------

def default_locate(photo: Path, what: str):
    review.engine()
    from investigation import locate
    return locate(photo, what)


def apply(d: Path, n: int, decision: Decision, locate, generate=None, fair_use: frozenset = frozenset()) -> dict:
    """Put the decision on the shot, then its name card (if any)."""
    shot = apply_picture(d, n, decision, locate, generate, fair_use)
    if decision.action == "keep" and not decision.name_card.strip() and not decision.remove_card:
        return shot
    with review.LOCK:
        storyboard = review.engine()
        sb = review.load_storyboard(d)
        beat = review.find_beat(sb, n)
        if decision.remove_card:
            beat["overlay"] = None
        elif decision.name_card.strip():
            old = beat.get("overlay") or {}
            beat["overlay"] = {"style": "name", "text": decision.name_card.strip(), "subtext": decision.name_role.strip()}
            # Held stays held for the same name: asking to change its words is not asking for it to go sooner.
            if decision.name_hold or (old.get("hold") and old.get("text") == beat["overlay"]["text"]):
                beat["overlay"]["hold"] = True
        beat["owner"] = dict(beat.get("owner") or {"choice": beat.get("library_id") or beat.get("asset")}, at=time.time())
        storyboard.save(d, sb)
        return review.shot_view(beat, review.load_library(d), review.load_marks(d), review.built_from(d),
                                review.load_regen(d))


def apply_picture(d: Path, n: int, decision: Decision, locate, generate=None, fair_use: frozenset = frozenset()) -> dict:
    """Put the decision's picture on the shot through video_review.choose(); a focus crops a photograph (not
    footage). "keep" leaves the picture as it is."""
    if decision.action == "keep":
        sb = review.load_storyboard(d)
        return review.shot_view(review.find_beat(sb, n), review.load_library(d), review.load_marks(d),
                                review.built_from(d), review.load_regen(d))
    if decision.action == "illustration":
        # choose() still runs the detail past ladder.safe_detail, which refuses people and named places.
        return review.choose(d, n, "illustration", describe=lambda *_: decision.detail.strip(), generate=generate)
    shot = review.choose(d, n, decision.library_id, fair_use=fair_use)
    if not decision.focus.strip() or not shot["picture"] or shot["picture"]["media"] == "clip":
        return shot
    review.engine()
    import ladder
    box = ladder.located(locate, d / shot["picture"]["file"], decision.focus.strip())
    if not box:
        return shot
    with review.LOCK:
        storyboard = review.engine()
        sb = review.load_storyboard(d)
        beat = review.find_beat(sb, n)
        beat["crop"] = ladder.frame_box(box)
        storyboard.save(d, sb)
        return review.shot_view(beat, review.load_library(d), review.load_marks(d), review.built_from(d),
                                review.load_regen(d))


def restore(d: Path, n: int, beat: dict) -> None:
    """Put the shot back exactly as it was."""
    with review.LOCK:
        storyboard = review.engine()
        sb = review.load_storyboard(d)
        sb["beats"] = [beat if b["index"] == n else b for b in sb["beats"]]
        storyboard.save(d, sb)


def picture_bytes(d: Path, beat: dict) -> bytes | None:
    """A JPEG of what the shot now shows: the photograph as framed, or a frame from the middle of its clip."""
    from PIL import Image
    path = d / (beat.get("asset") or "")
    if not beat.get("asset") or not path.exists():
        return None
    try:
        if beat.get("media") == "clip":
            at = (float(beat.get("clip_in") or 0) + float(beat.get("clip_out") or beat.get("clip_in") or 0)) / 2
            done = subprocess.run([review.VIDEO_FFMPEG, "-v", "error", "-ss", f"{at:.2f}", "-i", str(path),
                                   "-frames:v", "1", "-vf", "scale=768:-2", "-f", "image2", "-c:v", "mjpeg", "-"],
                                  capture_output=True, stdin=subprocess.DEVNULL, timeout=60)
            return done.stdout or None
        with Image.open(path) as img:
            img = img.convert("RGB")
            if beat.get("crop"):
                x0, y0, x1, y1 = beat["crop"]
                img = img.crop((round(x0 * img.width), round(y0 * img.height),
                                round(x1 * img.width), round(y1 * img.height)))
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            return buf.getvalue()
    except Exception:  # noqa: BLE001 - an unreadable picture is not checked; the render will say more
        return None


def look_words(beat: dict, decision: Decision, library: dict[str, dict]) -> str:
    """What the checker is told it is looking at."""
    if beat.get("kind") == "generated":
        return (f"an illustration drawn for the film (an object or setting only): {beat.get('detail')}"
                f"{card_words(beat)}")
    item = library.get(beat.get("library_id") or "")
    what = option_line(review_option(item)) if item else review.kind_words(beat)
    framed = f"; framed closer on: {decision.focus.strip()}" if beat.get("crop") and decision.focus.strip() else ""
    return f"{what}{framed}{card_words(beat)}. The editor's note: {decision.explanation}"


def card_words(beat: dict) -> str:
    card = beat.get("overlay") or {}
    if card.get("style") != "name" or not card.get("text"):
        return ""
    return f"; with the film's name card drawn over it as clean text: {card['text']}" + (
        f" ({card['subtext']})" if card.get("subtext") else "")


# ---------- the engine's steps ----------

def run_step(cmd: list[str], d: Path, timeout: float) -> None:
    """One engine script, as the review's own jobs run it (project .env), logged to logs/review_regen.log."""
    log = d / "logs" / "review_regen.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    name = Path(cmd[2]).name
    with open(log, "a", encoding="utf-8") as out:
        out.write(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        out.flush()
        try:
            done = subprocess.run(cmd, cwd=ROOT, env=load_env(), stdin=subprocess.DEVNULL, stdout=out,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RegenError(f"{name} took longer than {timeout / 60:.0f} minutes")
    if done.returncode:
        lines = [x for x in log.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
        raise RegenError(f"{name} failed: {' / '.join(lines[-3:])[:300]}")


def library_ids(d: Path) -> list[str]:
    p = d / "library.json"
    return [i["id"] for i in json.loads(p.read_text(encoding="utf-8"))["items"]] if p.exists() else []


def find_web_pages(queries: list[str], about: str = "") -> list[dict]:
    """Web pages with a real photograph of what the phrases name, found by the text model's web search tool;
    as harvest seeds ({"page_url", "image_url", "subject"})."""
    text = ("Find web pages that show a real photograph of: " + "; ".join(queries) + "\n"
            + (f"It is for this documentary shot: {about}\n" if about else "")
            + f"Give up to {MAX_WEB_PAGES} pages, the best first: news articles, Wikipedia, museum, library or "
              "official pages. Never stock-photo sites, social networks, AI-made images, drawings or posters of "
              "other things. Give the photograph's own file address as image_url when you see it.")
    r = client().responses.parse(model=config.settings.text_model, tools=[{"type": "web_search"}],
                                 text_format=WebFinds, input=text)
    found = r.output_parsed.pages if r.output_parsed else []
    seeds = []
    for page in found:
        url = page.page_url.strip()
        if not url.startswith("http") or any(s in url.lower() for s in WEB_SKIP):
            continue
        image = page.image_url.strip()
        # The search tool appends its citations to the text: "… taken in 1973. ([site](url))".
        seed = {"page_url": url, "subject": page.subject.split(" ([")[0].strip()[:120]}
        if image.startswith("http") and not any(s in image.lower() for s in WEB_SKIP):
            seed["image_url"] = image
        seeds.append(seed)
    return seeds[:MAX_WEB_PAGES]


def search_online(d: Path, queries: list[str], about: str = "", web=None) -> list[str]:
    """Search the archives and the web for these phrases with the engine's own steps; returns the new
    library ids.

    The phrases go to the front of library/needs.json ({"query", "beats"}, as check_pictures.py writes it);
    during the harvest the file holds only them, so only this shot's searches run. Then
    `harvest.py --stages needs` (Commons, Openverse, Internet Archive; it only adds to the library) and
    `caption.py --only <new ids>` (captions, and flags graphic pictures). The web search's pages go to
    library/pages.json for the harvest's `pages` stage (their pictures are copyrighted: fair use)."""
    fresh = {q.strip().lower(): q.strip() for q in queries if q.strip()}
    try:  # the web search writes nothing, so it runs outside the lock, side by side with other shots'
        pages = (web or find_web_pages)(list(fresh.values()), about)
    except Exception:  # noqa: BLE001 - a failed web search still leaves the archives' finds
        pages = []
    with SEARCH_LOCK:
        return _harvest_finds(d, fresh, pages)


def _harvest_finds(d: Path, fresh: dict[str, str], pages: list[dict]) -> list[str]:
    """search_online's file work (needs.json, pages.json, the harvest and the captions), one shot at a time."""
    path = d / "library" / "needs.json"
    needs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    known = {x["query"].strip().lower(): x for x in needs}
    front = [known.get(key) or {"query": q, "beats": []} for key, q in fresh.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    full = front + [x for x in needs if x["query"].strip().lower() not in fresh]
    before = set(library_ids(d))
    # The harvest runs every search in the file: while it runs the file holds only this shot's, so one shot's
    # search is not the film's whole list of searches (minutes, not most of an hour). The full list goes back.
    pages_path = path.parent / "pages.json"
    path.write_text(json.dumps(front, ensure_ascii=False, indent=1), encoding="utf-8")
    stages = "needs"
    if pages:
        pages_path.write_text(json.dumps(pages, ensure_ascii=False, indent=1), encoding="utf-8")
        stages = "needs,pages"
    try:
        run_step([sys.executable, "-u", str(SCRIPTS / "harvest.py"), str(d), "--stages", stages], d, SEARCH_TIMEOUT_S)
    finally:
        path.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
        pages_path.unlink(missing_ok=True)
    new = [i for i in library_ids(d) if i not in before]
    if new:
        run_step([sys.executable, "-u", str(SCRIPTS / "caption.py"), str(d), "--only", ",".join(new)], d,
                 SEARCH_TIMEOUT_S)
    return new


def render_preview(d: Path, n: int) -> str:
    """Render exactly shot n (build.py's --from is a shot index) as a 480p clip; returns its film-relative path.
    The shot's earlier clips are removed."""
    relative = Path("review") / "clips" / f"{n}_{int(time.time())}.mp4"
    out = d / relative
    out.parent.mkdir(parents=True, exist_ok=True)
    run_step([sys.executable, "-u", str(SCRIPTS / "build.py"), str(d), "--animatic", "--from", str(n),
              "--shots", "1", "--out", str(out)], d, RENDER_TIMEOUT_S)
    if not out.exists():
        raise RegenError("the shot was not rendered")
    for old in out.parent.glob(f"{n}_*.mp4"):
        if old != out:
            old.unlink(missing_ok=True)
    return relative.as_posix()


# ---------- one regeneration ----------

def scene_of(sb: dict, beat: dict) -> dict:
    return next((s for s in sb.get("scenes", []) if s["id"] == beat.get("scene")), {})


def regenerate(d: Path, n: int, review_text: str, *, ask=None, check=None, search=None, render=None, locate=None,
               generate=None, drawings: bool | None = None) -> dict:
    """Choose, (search, choose again), apply, check, (choose once more), render; regen.json follows each step.
    On any error the shot is put back as it was and the entry is failed with the reason. With drawings off
    (REGEN_DRAWINGS=0) no new illustration is made: when no real picture fits, the current picture is kept."""
    if drawings is None:
        drawings = os.getenv("REGEN_DRAWINGS", "1") != "0"
    ask, check = ask or ask_model, check or verify
    search = search or (lambda d, queries: search_online(d, queries, about=review_text))
    render, locate = render or render_preview, locate or default_locate
    d = Path(d)
    original = copy.deepcopy(review.find_beat(review.load_storyboard(d), n))
    touched = searched = False
    found: frozenset = web_finds(d)         # web pictures (fair use): earlier regenerations' and this one's
    named = LIBRARY_ID.findall(review_text)  # pictures the review asks for by id are always offered
    note(d, n, review=review_text, state="running", step=STEPS["choose"], explanation="", check=None, clip=None,
         error="")

    def attempt(feedback: str) -> Decision:
        nonlocal touched, searched, found
        sb = review.load_storyboard(d)
        beat = review.find_beat(sb, n)
        scene, rules = scene_of(sb, beat), (sb.get("bible") or {}).get("sensitivities") or []
        library = review.load_library(d)
        current = library.get(beat.get("library_id") or "")
        nearby = nearby_pictures(sb, n, library)
        decision = decide(review_text, beat, scene, current, options(d, n, found, named), not searched, feedback, ask,
                          rules, nearby, allow_drawing=searched, drawings=drawings)
        if decision.action == "illustration" and not searched:
            # A drawing comes only after a search for a real picture: asked once more, then searched anyway.
            decision = decide(review_text, beat, scene, current, options(d, n, found, named), True,
                              (f"{feedback}\nYou chose a drawing ({decision.detail!r}), but a real picture must be "
                               "searched for first: choose an option, or give search phrases.").strip(), ask,
                              rules, nearby, allow_drawing=False, drawings=drawings)
            if decision.action == "illustration":
                words = (decision.name_card or decision.detail).split()
                decision = Decision(action="search", queries=[" ".join(words[:6])],
                                    explanation=decision.explanation)
        if decision.action == "search":
            note(d, n, step=STEPS["search"])
            new = set(search(d, decision.queries))
            searched, found = True, found | frozenset(new)
            note(d, n, step=STEPS["choose"])
            offered = [o for o in options(d, n, found, named) if o["id"] in new]
            decision = decide(review_text, beat, scene, current, offered, False, feedback, ask, rules, nearby,
                              drawings=drawings)
        if decision.action == "illustration" and not drawings:
            # Drawings are off: no real picture fits, so the shot keeps its picture (its card may still change).
            decision = Decision(action="keep", name_card=decision.name_card, name_role=decision.name_role,
                                name_hold=decision.name_hold, remove_card=decision.remove_card,
                                explanation="No real picture fits; the current picture is kept (drawings are off).")
        touched = True
        try:
            apply(d, n, decision, locate, generate, found)
        except review.FilmError as exc:
            if decision.action != "illustration" or "refused" not in str(exc):
                raise
            # A refused drawing gets one more choice, told why, before the shot fails.
            note(d, n, step=STEPS["choose"])
            decision = decide(review_text, beat, scene, current, options(d, n), False,
                              (f"{feedback}\nYour drawing was refused by the illustration rules: {decision.detail!r}. "
                               "A drawing must be an object or a setting with no person and no named or recognisable "
                               "place (no city, state, campus or building names). Describe a generic one, or choose "
                               "an option.").strip(), ask, rules, nearby)
            apply(d, n, decision, locate, generate, found)
        return decision

    def checked(decision: Decision) -> Verdict | None:
        note(d, n, step=STEPS["check"])
        sb = review.load_storyboard(d)
        beat = review.find_beat(sb, n)
        image = picture_bytes(d, beat)
        if image is None:
            return None
        try:
            return check(image, review_text, beat, scene_of(sb, beat), look_words(beat, decision, review.load_library(d)))
        except Exception:  # noqa: BLE001 - a checker that cannot answer leaves the choice unchecked
            return None

    try:
        decision = attempt("")
        verdict = checked(decision)
        if verdict is not None and not verdict.satisfied:
            note(d, n, step=STEPS["choose"])
            decision = attempt(f"A first choice was checked and rejected. It was: {decision.explanation or '-'}\n"
                               f"The checker said: {verdict.problem or 'it does not do what the review asks'}\n"
                               "Choose again so that the review is satisfied.")
            verdict = checked(decision)
        explanation = decision.explanation.strip()
        beat = review.find_beat(review.load_storyboard(d), n)
        if decision.action == "library" and decision.focus.strip() and beat.get("media") != "clip" \
                and not beat.get("crop"):
            explanation += f" (could not find {decision.focus.strip()!r} to frame closer: the whole picture is shown)"
        note(d, n, step=STEPS["render"])
        with RENDER_LOCK:
            clip = render(d, n)
        review.set_mark(d, n, None)                 # the picture changed: any approval was of the old one
        return note(d, n, state="done", step=STEPS["done"], explanation=explanation,
                    check=verdict.model_dump() if verdict else None, clip=clip, error="")
    except Exception as exc:  # noqa: BLE001 - every failure is reported on the shot, which is left as it was
        if touched:
            restore(d, n, original)
        return note(d, n, state="failed", step="", error=str(exc)[:400] or type(exc).__name__)


# ---------- the worker ----------

class Regenerator:
    """Worker threads (`workers` of them) regenerating queued shots, first come first served. Shots run side by
    side; their searches and renders take turns (SEARCH_LOCK, RENDER_LOCK)."""

    def __init__(self, run=None, workers: int = 1):
        self.run = run or regenerate
        self.workers = max(1, workers)
        self.queue: deque = deque()
        self.current: list = []
        self.threads = 0
        self.cond = threading.Condition()

    @staticmethod
    def _key(d: Path) -> str:
        return str(Path(d).resolve())

    def _waiting(self) -> list:
        return list(self.queue) + list(self.current)

    def submit(self, d: Path, n: int, review_text: str) -> dict:
        review_text = (review_text or "").strip()
        if not review_text:
            raise review.FilmError("write what is wrong with the shot first")
        sb = review.load_storyboard(d)
        allowed, why = review.illustration_allowed(sb, review.find_beat(sb, n))
        if not allowed:
            raise review.FilmError(why)
        with self.cond:
            if any(self._key(qd) == self._key(d) and qn == n for qd, qn, _ in self._waiting()):
                raise RegenBusy(f"shot {n} is already being regenerated")
            entry = note(d, n, review=review_text, state="queued", step=STEPS["queued"], explanation="",
                         check=None, clip=None, error="")
            self.queue.append((Path(d), n, review_text))
            if self.threads < self.workers:
                self.threads += 1
                threading.Thread(target=self._work, name="regenerate", daemon=True).start()
        return review.regen_view(entry)

    def busy(self, d: Path) -> bool:
        with self.cond:
            return any(self._key(qd) == self._key(d) for qd, _, _ in self._waiting())

    def join(self, timeout: float | None = None) -> bool:
        """Wait until nothing is queued or running; False on timeout."""
        with self.cond:
            return self.cond.wait_for(lambda: not self.queue and not self.current, timeout)

    def _work(self) -> None:
        while True:
            with self.cond:
                if not self.queue:
                    self.threads -= 1
                    self.cond.notify_all()
                    return
                job = self.queue.popleft()
                self.current.append(job)
            d, n, text = job
            try:
                self.run(d, n, text)
            except Exception as exc:  # noqa: BLE001 - the worker must outlive any one shot
                note(d, n, state="failed", step="", error=str(exc)[:400] or type(exc).__name__)
            finally:
                with self.cond:
                    self.current.remove(job)
                    self.cond.notify_all()

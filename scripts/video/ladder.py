"""The relevance ladder in code: beats without a true picture of their own reframe, hold or illustrate.

The storyboard gives every beat a `link` (storyboard.STORYBOARD_RULES): subject, place_era, action, reframe,
hold or generated; check_pictures.py turns a picture whose declared link does not hold into a ladder step.
This module carries those steps out:

- reframe: another framing of a relevant photograph already in the scene. The vision model locates what to
  frame (investigation.locate, which confirms the crop with a second look); the beat runs that crop.
- hold: the beat continues the picture before it (the two become one longer shot), up to HOLD_MAX_S.
- generated: an illustrative detail of an object or setting, never a person or a named place, drawn by the
  image model in the film's period style and tagged ILLUSTRATION in the corner. At most one per
  GENERATED_EVERY_S of film.

A step that cannot be carried out falls to the next one (reframe -> hold -> generated -> as planned).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from PIL import Image, ImageDraw

import graphics

HOLD_MAX_S = 25.0
SPLIT_PERSON_S = 10.0         # a portrait held longer than this cuts to another framing or photograph
SPLIT_PLACE_S = 8.0           # an establishing or place picture held longer than this gives way
EXPOSURE_S = 16.0             # a picture's total screen time inside REPEAT_WINDOW_S before it counts as overused
SPLIT_OVER_S = SPLIT_PERSON_S
WHOLE = 0.85                  # a "detail" covering this much of the picture is not a reframe
GENERATED_EVERY_S = 60.0     # at most one illustration in this much film (a misleading picture is exempt)
GENERATED_REPEAT_S = 20.0     # an overused stand-in with nothing else to show may be illustrated this soon after one
GENERATED_AFTER_S = 6.0       # and a hold far too long (over STALE_HOLD_S) this soon after one ends
STALE_HOLD_S = 15.0
REPEAT_WINDOW_S = 120.0       # a picture's third showing inside this window is replaced by an illustration
RELEVANT = {"subject", "place_era", "action"}
MARGIN = 0.12                 # a located detail is framed with this much room around it
MIN_CROP = 0.28               # a crop narrower than this fraction of the photograph is too soft to hold
MIN_CROP_PX = 360             # nor shorter than this many source pixels (1080 / the prints' 3x enlargement limit)
MIN_ILLUSTRATION_S = 4.0      # an illustration shorter than this is a flash, not a picture
ILLUSTRATION_MAX_S = 14.0     # one illustration held longer than this splits into two of the beat's words
ILLUSTRATION_RUN = 2          # at most this many illustrations in a row
ILLUSTRATION_SHARE = 0.4      # nor more than this share of a scene's time
ILLUSTRATION_RUN_ON_S = 18.0  # an illustration may run on this long rather than a third follow it
PERSON_RUN_S = 24.0           # one person's photographs this long in a row, and the next beat cuts away


def relevant_photo(entries: list[dict], at: int) -> dict | None:
    """The nearest earlier beat in the same scene whose photograph has a true link of its own."""
    scene = entries[at].get("scene")
    for k in range(at - 1, -1, -1):
        b = entries[k]
        if b.get("scene") != scene:
            break
        if b.get("media") == "photo" and b.get("library_id") and b.get("link") in RELEVANT:
            return b
    return None


def frame_box(box: tuple[float, float, float, float]) -> list[float]:
    """A located box widened by MARGIN, each side at least MIN_CROP of the photograph (centred on the
    detail, kept inside the picture), so the crop is neither cramped nor too soft to hold."""
    x0, y0, x1, y1 = box

    def side(a: float, b: float) -> tuple[float, float]:
        pad = (b - a) * MARGIN
        a, b = a - pad, b + pad
        if b - a < MIN_CROP:
            c = (a + b) / 2
            a, b = c - MIN_CROP / 2, c + MIN_CROP / 2
        shift = max(0.0, -a) - max(0.0, b - 1.0)
        return max(0.0, a + shift), min(1.0, b + shift)

    (x0, x1), (y0, y1) = side(x0, x1), side(y0, y1)
    return [round(v, 4) for v in (x0, y0, x1, y1)]


def located(locate, photo: Path, what: str):
    """A located box that is a real detail (not the whole picture) and big enough to show, else None."""
    box = locate(photo, what)
    if not box:
        return None
    x0, y0, x1, y1 = box
    if (x1 - x0) * (y1 - y0) >= WHOLE:
        return None
    try:
        with Image.open(photo) as img:
            height = img.height
    except Exception:  # noqa: BLE001 - unreadable here means the render will say so; do not frame it
        return None
    fx0, fy0, fx1, fy1 = frame_box(box)
    return box if (fy1 - fy0) * height >= MIN_CROP_PX else None


def can_hold(entries: list[dict], at: int) -> bool:
    if at == 0:
        return False
    prev, here = entries[at - 1], entries[at]
    if prev.get("media") == "clip":
        # Footage can only run on as long as the clip itself lasts (plus a dissolve's tail).
        footage = float(prev.get("clip_out") or 0) - float(prev.get("clip_in") or 0)
        if footage < here["end"] - prev["start"] + 1.0:
            return False
    return (prev.get("scene") == here.get("scene") and prev.get("media") in ("photo", "clip")
            and here["end"] - prev["start"] <= HOLD_MAX_S and not here.get("quote_ids"))


GENERIC = {"the", "house", "home", "street", "avenue", "road", "west", "east", "north", "south", "county", "city",
           "high", "school", "police", "department", "station", "jail", "court", "courthouse", "church", "hospital",
           "pharmacy", "store", "shop", "restaurant", "office", "company", "garage", "park", "river", "lake", "college",
           "university", "prison", "penitentiary", "building", "center", "centre", "hotel", "bar", "club", "farm"}


def safe_detail(detail: str, bible: dict) -> bool:
    """A generated detail may show an object or a setting, never a person or a named place."""
    import sources
    text = detail.lower()
    if not text.strip() or sources.names_a_real_person(detail):
        return False
    names = [p["name"] for p in bible.get("people") or []] + [p["name"] for p in bible.get("places") or []]
    for name in names:
        # Only the distinctive proper words of a name count ("Des Plaines", "Summerdale", "Nisson"); the
        # ordinary words in it ("house", "street", "police") are what illustrations are made of.
        proper = [w.lower() for w in re.findall(r"\b[A-Z][a-z]{2,}", name) if w.lower() not in GENERIC]
        if any(re.search(rf"\b{re.escape(w)}\b", text) for w in proper):
            return False
    # "a boy's jacket" is an object; "a boy" is a person.
    return not re.search(r"\b(person|people|man|woman|boy|girl|face|body|victim|blood|corpse)\b(?!['’]s)", text)


def tag_illustration(source: Path, path: Path | None = None) -> None:
    """An ILLUSTRATION tag in the lower right, large enough to read on a phone and inset so a slow push-in
    never crops it. Drawn from the untagged `source` into `path` (in place when no path is given)."""
    img = Image.open(source).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    fnt = graphics.font(graphics.LABEL, max(20, img.width // 46))
    text = "ILLUSTRATION"
    w = fnt.getlength(text)
    pad = fnt.size * 0.45
    x, y = img.width * 0.90 - w, img.height * 0.86
    d.rectangle((x - pad, y - pad * 0.6, x + w + pad, y + fnt.size + pad * 0.9), fill=(0, 0, 0, 165))
    d.text((x, y), text, font=fnt, fill=(240, 237, 230, 245))
    img.save(path or source)


def generate_detail(detail: str, era: str, video_dir: Path) -> Path:
    import sources
    prompt = (f"Documentary photograph, {era or '1970s'}, a quiet close detail: {detail}. Real-world, period-accurate, "
              "35 mm film grain, muted natural colour, soft available light, well exposed with the subject clearly "
              "visible and detail held in the shadows. No people, no faces, no hands, no text, no signs, no logos, "
              "no brand names, nothing gruesome.")
    name = f"detail_{hashlib.sha1(prompt.encode()).hexdigest()[:10]}"
    folder = video_dir / "assets" / "generated"
    out, raw = folder / f"{name}.png", folder / "untagged" / f"{name}.png"
    if not raw.exists():
        # The untagged image is kept, so a change to the tag redraws it without paying for the image again.
        sources.generate_image(prompt, raw)
    if not out.exists() or out.stat().st_mtime < raw.stat().st_mtime:
        tag_illustration(raw, out)
    return out


def vary_subjects(entries: list[dict], bible: dict, library: dict[str, dict],
                  scenes: dict[int, dict] | None = None) -> int:
    """Back-to-back beats on the same photograph of a person who has others take the next one in turn, so a
    passage about Rob shows Rob throughout without one picture standing for a minute. Returns changes."""
    from storyboard import set_picture
    owner = {}
    for p in bible.get("people") or []:
        photos = [pid for pid in p.get("photo_ids") or [] if pid in library and library[pid].get("media") != "clip"]
        for pid in photos:
            owner.setdefault(pid, photos)
    changed = 0
    for k in range(1, len(entries)):
        prev, e = entries[k - 1], entries[k]
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        pid = e.get("library_id")
        if not pid or pid != prev.get("library_id") or e.get("crop") or pid not in owner or len(owner[pid]) < 2:
            continue
        scene = (scenes or {}).get(e.get("scene"))
        near = {b.get("library_id") for b in entries[max(0, k - 3):k + 2]}
        others = [x for x in owner[pid] if x not in near and not from_the_future(library[x], scene)]
        if others:
            set_picture(e, library[others[0]])
            e["varied"] = pid
            changed += 1
    return changed


def split_point(e: dict, sentences: list[dict] | None, at: float | None = None) -> float:
    """Where to cut a long beat: the sentence start nearest its middle (or `at`), so the picture changes when
    the words do; the target itself when no sentence starts near it. Both parts keep MIN_ILLUSTRATION_S."""
    length = e["end"] - e["start"]
    mid = (e["start"] + e["end"]) / 2 if at is None else at
    lo, hi = (e["start"] + 0.3 * length, e["end"] - 0.3 * length) if at is None else \
        (max(e["start"] + MIN_ILLUSTRATION_S, at - 3.0), min(e["end"] - MIN_ILLUSTRATION_S, at + 3.0))
    starts = [s["start"] for s in sentences or [] if lo <= s["start"] <= hi]
    return round(min(starts, key=lambda x: abs(x - mid)) if starts else mid, 3)


def split_long_holds(entries: list[dict], library: dict[str, dict], video_dir: Path, locate,
                     bible: dict | None = None, sentences: list[dict] | None = None,
                     scenes: dict[int, dict] | None = None) -> list[dict]:
    """A photograph held past SPLIT_OVER_S cuts halfway to a closer framing of its face or key detail, or, when
    the photograph is too small to frame closer, to another photograph of the same person; so a long hold
    breathes instead of sitting still."""
    from storyboard import set_picture
    people = person_photos(bible or {})
    others: dict[str, list[str]] = {}
    for p in (bible or {}).get("people") or []:
        photos = [pid for pid in p.get("photo_ids") or [] if pid in library and library[pid].get("media") != "clip"]
        for pid in photos:
            others.setdefault(pid, [x for x in photos if x != pid])
    out = []
    for e in entries:
        item = library.get(e.get("library_id") or "")
        if (e["end"] - e["start"] <= SPLIT_OVER_S or e.get("media") != "photo" or not item or e.get("crop")
                or e.get("quote_ids") or e.get("owner")):
            out.append(e)
            continue
        box = located(locate, video_dir / item["file"], "the face, or the most telling detail")
        scene = (scenes or {}).get(e.get("scene"))
        alternative = next((x for x in others.get(item["id"]) or [] if not from_the_future(library[x], scene)), None)
        if not box and not alternative:
            # A small scan cannot be framed closer and there is no other photograph of them from then: the
            # second half shows a picture of the scene's place or setting from the beat's own choices.
            near = {b.get("library_id") for b in out[-3:]}
            alternative = next((c for c in e.get("choices") or [] if c in library and c not in people and c != item["id"]
                                and c not in near and library[c].get("media") != "clip"
                                and not from_the_future(library[c], scene)), None)
        if not box and not alternative:
            out.append(e)
            continue
        half = split_point(e, sentences)
        first, second = dict(e), dict(e)
        first["end"], second["start"] = half, half
        second.update({"transition_in": "cut", "effects": []})
        if box:
            second.update({"crop": frame_box(box), "link": "reframe"})
        else:
            set_picture(second, library[alternative])
            second["varied"] = item["id"]
        out += [first, second]
    return out


DESCRIBE = """You write the subject of one illustrative photograph for a true-crime documentary. It will show
an OBJECT or a SETTING that fits these words, in the period given: never a person, a face, hands or a body; never
a named or recognisable real place (no city skylines, landmarks, shop names, signs); nothing gruesome. Write one
short line, concrete and visual, e.g. "a 1960s kitchen table with a radio and a coffee cup, morning light" or
"a 1970s pharmacy counter with a cash register at closing time"."""


def describe_detail(narration: str, scene: dict) -> str:
    """The model's one-line object or setting for an illustration of these words."""
    from openai import OpenAI
    from pydantic import BaseModel

    from app.config import settings

    class Detail(BaseModel):
        detail: str

    user = (f"Period and place of the scene: {scene.get('story_time') or ''}, {scene.get('setting') or ''}. "
            f"Mood: {scene.get('mood') or ''}.\nWords: {narration}")
    if scene.get("already"):
        user += ("\nAlready illustrated in this scene (choose a clearly different object or setting): "
                 + "; ".join(scene["already"]))
    r = OpenAI(api_key=settings.openai_api_key).responses.parse(
        model=settings.text_model, input=[{"role": "system", "content": DESCRIBE}, {"role": "user", "content": user}],
        text_format=Detail)
    return r.output_parsed.detail.strip()


def from_the_future(item: dict, scene: dict | None) -> bool:
    from storyboard import from_the_future as later
    return later(item, scene)


def closer(target: dict, library: dict[str, dict], video_dir: Path, locate) -> bool:
    """Give the second half of a split hold a closer framing of its own picture; False if none is found."""
    item = library.get(target.get("library_id") or "")
    if not item or item.get("media") == "clip":
        return False
    if locate is None:
        from investigation import locate
    box = located(locate, video_dir / item["file"], "the most telling detail")
    if not box:
        return False
    target.update({"crop": frame_box(box), "link": "reframe", "transition_in": "cut", "effects": []})
    return True


def framing(e: dict) -> tuple:
    return (e.get("library_id") or e.get("asset"), tuple(e.get("crop") or ()))


def break_loops(entries: list[dict], library: dict[str, dict], video_dir: Path, locate,
                scenes: dict[int, dict] | None = None, bible: dict | None = None) -> int:
    """No A-B-A-B (a scene boundary between them does not hide it): when a beat would bring back the framing
    two beats before it, and the one between also repeats, it takes a picture of its choices, or another
    photograph of the same person, not seen in the last three beats (and not dated after the scene), or
    else a new framing of its own picture. Returns beats changed."""
    from storyboard import set_picture
    same_person: dict[str, list[str]] = {}
    for p in (bible or {}).get("people") or []:
        for pid in p.get("photo_ids") or []:
            same_person.setdefault(pid, []).extend(x for x in p["photo_ids"] if x != pid)
    tainted = tainted_pictures(entries, bible)
    changed = 0
    for k in range(3, len(entries)):
        e = entries[k]
        if e.get("owner"):
            continue                      # the owner chose this picture in the review page
        if (e.get("media") != "photo" or e.get("quote_ids")
                or framing(e) != framing(entries[k - 2]) or framing(entries[k - 1]) != framing(entries[k - 3])):
            continue
        scene = (scenes or {}).get(e.get("scene"))
        seen = {b.get("library_id") for b in entries[k - 3:k + 2]}      # nor the next beat's: they would merge
        options = list(e.get("choices") or []) + same_person.get(e.get("library_id") or "", [])
        alt = next((c for c in options if c in library and c not in seen and c not in tainted
                    and library[c].get("media") != "clip" and not from_the_future(library[c], scene)), None)
        if alt:
            set_picture(e, library[alt])
            e["crop"] = None
        else:
            item = library.get(e.get("library_id") or "")
            if locate is None:
                from investigation import locate
            box = located(locate, video_dir / item["file"], "a different part of the picture from its centre") if item else None
            if not box or frame_box(box) == list(entries[k - 2].get("crop") or []):
                continue
            e["crop"], e["link"] = frame_box(box), "reframe"
        e["loop_broken"] = True
        changed += 1
    return changed


def person_photos(bible: dict) -> set[str]:
    return {pid for p in bible.get("people") or [] for pid in p.get("photo_ids") or []}


def misled(e: dict) -> bool:
    """The check found this beat's picture misleading (a sign naming another business, the wrong era or
    time of day) and could not swap it."""
    check = e.get("check") or {}
    return bool(check.get("misleads")) and not check.get("replaced")


TWIN_DISTANCE = 10            # two photographs this close (of 64 hash bits) are one picture to the eye


def twins(entries: list[dict], library: dict[str, dict], video_dir: Path) -> dict[str, str]:
    """Each photograph used, mapped to the first used one it is a near-duplicate of (two frames of the same
    view under different ids): for the repetition rules they are the same picture."""
    from harvest import average_hash, hamming
    hashes: dict[str, int] = {}
    for e in entries:
        pid = e.get("library_id")
        item = library.get(pid or "")
        if not item or pid in hashes or item.get("media") == "clip" or e.get("media") != "photo":
            continue
        try:
            with Image.open(video_dir / item["file"]) as img:
                hashes[pid] = average_hash(img)
        except Exception:  # noqa: BLE001 - an unreadable file is only itself
            continue
    out: dict[str, str] = {}
    seen: list[str] = []
    for pid, h in hashes.items():
        out[pid] = next((q for q in seen if hamming(h, hashes[q]) <= TWIN_DISTANCE), pid)
        if out[pid] == pid:
            seen.append(pid)
    return out


def run_on(prev: dict, beat: dict) -> None:
    """The picture before continues over this beat: one shot, its narration and sentences joined."""
    prev["end"] = beat["end"]
    prev["narration"] = f"{prev['narration']} {beat['narration']}".strip()
    if prev.get("sentences") and beat.get("sentences"):
        prev["sentences"] = [prev["sentences"][0], beat["sentences"][1]]


def tainted_pictures(entries: list[dict], bible: dict | None) -> set[str]:
    """Pictures the check found misleading on some beat, other than people's photographs (a CVS sign is
    wrong for Nisson Pharmacy everywhere; Rob's portrait under Kim's words is wrong only there)."""
    people = person_photos(bible or {})
    found = {e["library_id"] for e in entries if misled(e) and e.get("library_id")}
    found |= {e["misleading_id"] for e in entries if e.get("misleading_id")}     # already replaced
    return found - people


def illustrate_holds(entries: list[dict], video_dir: Path, bible: dict, scenes: dict[int, dict],
                     describe=describe_detail, generate=generate_detail, sentences: list[dict] | None = None,
                     library: dict[str, dict] | None = None, locate=None) -> int:
    """Long holds on a place or stand-in (over SPLIT_PLACE_S), and a picture's third showing or more than
    EXPOSURE_S of screen time in two minutes, give way: to an illustration of an object or setting matched to
    what is being said, or (when illustrations are rationed or refused) to another of the beat's own fitting
    choices. A person's photograph gives way less readily: only at its third showing within REPEAT_WINDOW_S,
    when held far too long, or after PERSON_RUN_S of that person's photographs without a cutaway.

    A picture the check found misleading is wrong wherever it appears (a CVS sign is not Nisson Pharmacy on
    any beat), and so is a photograph dated after its scene: every such showing is replaced, ahead of the
    ration. (A person's photograph found misleading on one beat is wrong only there.)

    An illustration just before a beat that gives way runs on over it (up to ILLUSTRATION_MAX_S) rather than
    another being drawn. Returns illustrations placed."""
    from storyboard import set_picture
    people = person_photos(bible)
    tainted = tainted_pictures(entries, bible)
    owner = {pid: p["name"] for p in bible.get("people") or [] for pid in p.get("photo_ids") or []}
    same = twins(entries, library or {}, video_dir)

    def unusable(pid, scene_id) -> bool:
        return pid in tainted or (pid in library and from_the_future(library[pid], scenes.get(scene_id)))
    library = library or {}
    last = last_end = -1e9
    shown: dict[str, list[tuple[float, float]]] = {}
    out: list[dict] = []
    made = 0
    for e in entries:
        if e.get("owner"):
            out.append(e)                 # the owner chose this picture in the review page
            continue
        pid = e.get("library_id")
        # A near-duplicate of the picture just before is the same shot cut twice: it runs on as one.
        prev = out[-1] if out else None
        if (pid and prev and prev.get("library_id") and prev["library_id"] != pid
                and same.get(pid, pid) == same.get(prev["library_id"], prev["library_id"])
                and prev.get("scene") == e.get("scene") and not e.get("quote_ids") and not prev.get("crop")
                and not prev.get("owner") and e["end"] - prev["start"] <= HOLD_MAX_S):
            run_on(prev, e)
            shown.setdefault(same.get(pid, pid), []).append((e["start"], e["end"]))
            continue
        # A person's photograph is exempt only where it is the subject: on a "hold" it is standing in.
        person = pid in people and e.get("link") != "hold"
        is_photo = e.get("media") == "photo" and pid and not person and not e.get("quote_ids")
        portrait_too_long = (e.get("media") == "photo" and person and not e.get("quote_ids") and not e.get("crop")
                             and e["end"] - e["start"] > SPLIT_PERSON_S + 4)
        key = same.get(pid, pid)
        recent = [(a, b) for a, b in shown.get(key, []) if e["start"] - a < REPEAT_WINDOW_S] if pid else []
        exposure = sum(b - a for a, b in recent) + (e["end"] - e["start"])
        wrong = bool(pid) and (unusable(pid, e.get("scene")) or (misled(e) and not person))
        long_hold = (is_photo and e["end"] - e["start"] > SPLIT_PLACE_S and not e.get("crop")) or portrait_too_long
        # How long this person's photographs have been on screen without a cutaway, up to this beat.
        run_s = 0.0
        for x in reversed(out):
            if not pid or owner.get(x.get("library_id") or "") != owner.get(pid):
                break
            run_s += x["end"] - x["start"]
        cutaway = person and run_s > 0 and run_s + e["end"] - e["start"] > PERSON_RUN_S
        repeat = (is_photo and (len(recent) >= 2 or exposure > EXPOSURE_S)) or (
            person and e.get("media") == "photo" and not e.get("quote_ids") and (len(recent) >= 2 or cutaway))
        if pid:
            shown.setdefault(key, []).append((e["start"], e["end"]))
        if not (long_hold or repeat or wrong):
            out.append(e)
            continue
        if wrong and misled(e):
            e["misleading_id"] = pid             # remembered once the beat shows something else
        if long_hold and not wrong:
            # The picture keeps the opening of the beat (at most about SPLIT_PLACE_S); the rest gives way.
            half = split_point(e, sentences, e["start"] + min(SPLIT_PLACE_S, (e["end"] - e["start"]) / 2))
            first, target = dict(e), dict(e)
            first["end"], target["start"] = half, half
        else:
            first, target = None, dict(e)
        prev = out[-1] if out and not first else None
        adjacent = (prev is not None and prev.get("kind") == "generated" and prev.get("scene") == e.get("scene")
                    and not prev.get("owner")
                    and abs(prev["end"] - target["start"]) < 0.05)
        scene = dict(scenes.get(e.get("scene"), {}))
        # A run of drawn pictures reads as a slideshow, not as an archive film.
        run = 0
        for x in reversed(out):
            if x.get("kind") != "generated" or x.get("scene") != e.get("scene"):
                break
            run += 1
        span = max(1.0, float(scene.get("end") or 0) - float(scene.get("start") or 0)) if scene.get("end") else None
        share = sum(x["end"] - x["start"] for x in out if x.get("kind") == "generated" and x.get("scene") == e.get("scene"))
        crowded = run >= ILLUSTRATION_RUN or (span and share + target["end"] - target["start"] > ILLUSTRATION_SHARE * span)
        if crowded:
            # Too many drawn pictures here: a photograph of someone the scene is about, not shown in the last
            # three beats and true to the scene's time, if there is one; else the illustration before runs on,
            # else one more illustration (a misleading picture is worse).
            near = {x.get("library_id") for x in out[-3:]}
            subject = next((c for c in e.get("choices") or [] if c in people and c in library and c not in near
                            and library[c].get("media") != "clip" and not unusable(c, e.get("scene"))
                            and not (cutaway and owner.get(c) == owner.get(pid))), None)
            if subject:
                set_picture(target, library[subject])
                target.update({"transition_in": "cut", "crop": None, "link": "subject"})
                out += ([first] if first else []) + [target]
                continue
        if adjacent and target["end"] - prev["start"] <= (ILLUSTRATION_RUN_ON_S if crowded else ILLUSTRATION_MAX_S):
            run_on(prev, target)                 # adds no illustration, so the ration does not apply
            continue
        if target["end"] - target["start"] < MIN_ILLUSTRATION_S:
            out.append(e)
            continue
        stale = long_hold and e["end"] - e["start"] > STALE_HOLD_S
        ration = 0.0 if wrong else GENERATED_REPEAT_S if repeat else GENERATED_EVERY_S
        rationed = e["start"] - last < ration and not (stale and target["start"] - last_end >= GENERATED_AFTER_S)
        if rationed:
            # Illustrations are rationed: another of the beat's own choices that is not overused, if any.
            near = {x.get("library_id") for x in out[-4:]}
            alt = next((c for c in e.get("choices") or [] if c in library and c not in people and c != pid
                        and c not in near and library[c].get("media") != "clip"
                        and not unusable(c, e.get("scene"))), None)
            if alt:
                set_picture(target, library[alt])
                target.update({"transition_in": "cut", "crop": None})
                out += ([first] if first else []) + [target]
            elif first and closer(target, library, video_dir, locate):
                out += [first, target]
            else:
                out.append(e)
            continue
        # Illustrations of the last two minutes, so the next is of something clearly different.
        scene["already"] = [x["detail"] for x in out if x.get("kind") == "generated" and x.get("detail")
                            and target["start"] - x["start"] < REPEAT_WINDOW_S]
        parts = [target]
        if target["end"] - target["start"] > ILLUSTRATION_MAX_S:
            cut = split_point(target, sentences)
            if min(cut - target["start"], target["end"] - cut) >= MIN_ILLUSTRATION_S:
                second = dict(target)
                target["end"], second["start"] = cut, cut
                parts = [target, second]
        drawn = []
        for n, part in enumerate(parts):
            words = part["narration"]
            if sentences and part.get("sentences"):
                words = " ".join(s["text"] for s in sentences if s["end"] > part["start"] and s["start"] < part["end"]) or words
            try:
                detail = describe(words, scene)
                if not safe_detail(detail, bible):
                    raise ValueError(f"refused: {detail!r}")
                if drawn and detail == drawn[-1]["detail"]:
                    break                                  # the same picture again would be a jump cut
                scene["already"] = scene["already"] + [detail]
                path = generate(detail, scene.get("story_time") or "", video_dir)
            except Exception as exc:  # noqa: BLE001 - no illustration for this part
                e["ladder_note"] = f"illustration skipped: {str(exc)[:90]}"
                break
            part.update({"asset": str(path.relative_to(video_dir)), "media": "photo", "kind": "generated",
                         "library_id": None, "colour": "colour", "rights": "generated", "detail": detail,
                         "licence": "Generated for this film (illustration)", "attribution": "Illustration",
                         "link": "generated", "transition_in": "dissolve" if n == 0 else "cut",
                         "effects": [], "crop": None})
            drawn.append(part)
        if not drawn:
            # No illustration: a closer framing, else the hold stays as it was.
            if first and closer(target, library, video_dir, locate):
                out += [first, target]
            else:
                out.append(e)
            continue
        if len(drawn) < len(parts):
            drawn[-1]["end"] = parts[-1]["end"]          # the second could not be drawn: the first runs on
        out += ([first] if first else []) + drawn
        last, last_end = drawn[-1]["start"], drawn[-1]["end"]
        made += len(drawn)
    entries[:] = out
    for n, e in enumerate(entries, start=1):
        e["index"] = n
    for s in scenes.values():
        s["beats"] = [e["index"] for e in entries if e.get("scene") == s["id"]]
    return made


def apply_ladder(entries: list[dict], library: dict[str, dict], video_dir: Path, bible: dict,
                 scenes: dict[int, dict], locate=None, generate=generate_detail,
                 sentences: list[dict] | None = None) -> dict[str, int]:
    """Carry out every reframe, hold and generated beat; returns how many of each were done."""
    if locate is None:
        from investigation import locate
    from storyboard import set_picture
    done = {"reframe": 0, "hold": 0, "generated": 0, "varied": 0, "loops": 0}
    last_generated = -1e9
    keep: list[dict] = []
    for at, e in enumerate(entries):
        if e.get("owner"):
            keep.append(e)                # the owner chose this picture in the review page
            continue
        step = e.get("link")
        if step == "reframe":
            source = library.get((e.get("choices") or [None])[0] or "")
            if not source or source.get("media") == "clip":
                prev = relevant_photo(entries, at)
                source = library.get(prev["library_id"]) if prev else None
            box = located(locate, video_dir / source["file"], e.get("reframe_on") or "the main subject") if source else None
            if box:
                set_picture(e, source)
                e["crop"] = frame_box(box)
                done["reframe"] += 1
                keep.append(e)
                continue
            step = "hold"
        if step == "hold":
            if keep and not keep[-1].get("owner") and can_hold(keep + [e], len(keep)):
                prev = keep[-1]
                prev["end"] = e["end"]
                prev["narration"] = f"{prev['narration']} {e['narration']}".strip()
                if prev.get("sentences") and e.get("sentences"):
                    prev["sentences"] = [prev["sentences"][0], e["sentences"][1]]
                done["hold"] += 1
                continue
            # Too long to continue the picture before: a closer framing of it instead, so the beat still shows
            # something true rather than whatever the picker had left on it.
            prev = keep[-1] if keep and keep[-1].get("scene") == e.get("scene") else None
            source = library.get(prev.get("library_id") or "") if prev else None
            if source and source.get("media") != "clip":
                box = located(locate, video_dir / source["file"], "the most telling detail, such as a face")
                if box:
                    set_picture(e, source)
                    e["crop"], e["link"] = frame_box(box), "reframe"
                    done["reframe"] += 1
                    keep.append(e)
                    continue
            if e.get("detail"):
                step = "generated"
            else:
                # A plain repeat beats a stale stand-in: the picture before if it is a still, else the last
                # relevant still photograph in the scene (footage cannot run on past its own length).
                still = prev if prev and prev.get("media") == "photo" and prev.get("library_id") in library else \
                    relevant_photo(keep + [e], len(keep))
                if still and (misled(still) or from_the_future(library[still["library_id"]], scenes.get(e.get("scene")))):
                    still = None
                if still:
                    set_picture(e, library[still["library_id"]])
                    keep.append(e)
                    continue
        if step == "generated":
            scene = scenes.get(e.get("scene"), {})
            # A picture the check found misleading may always be replaced: accuracy comes before the ration.
            wrong = misled(e) or (e.get("check") or {}).get("link_ok") is False
            if safe_detail(e.get("detail") or "", bible) and (wrong or e["start"] - last_generated >= GENERATED_EVERY_S):
                try:
                    if misled(e) and e.get("library_id"):
                        e["misleading_id"] = e["library_id"]
                    path = generate(e["detail"], scene.get("story_time") or "", video_dir)
                    e.update({"asset": str(path.relative_to(video_dir)), "media": "photo", "kind": "generated",
                              "library_id": None, "colour": "colour", "rights": "generated",
                              "licence": "Generated for this film (illustration)", "attribution": "Illustration"})
                    last_generated = e["start"]
                    done["generated"] += 1
                except Exception as exc:  # noqa: BLE001 - an image that cannot be drawn leaves the beat as planned
                    e["ladder_note"] = f"generation failed: {str(exc)[:80]}"
        keep.append(e)
    entries[:] = split_long_holds(keep, library, video_dir, locate, bible, sentences, scenes)
    done["varied"] = vary_subjects(entries, bible, library, scenes)
    done["loops"] = break_loops(entries, library, video_dir, locate, scenes, bible)
    for n, e in enumerate(entries, start=1):
        e["index"] = n
    for s in scenes.values():
        s["beats"] = [e["index"] for e in entries if e.get("scene") == s["id"]]
    return done

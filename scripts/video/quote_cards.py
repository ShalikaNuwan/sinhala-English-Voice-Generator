"""Quote cards: people's own words on screen, revealed as they are spoken.

A beat whose shot carries `quotes` (copied from quotes.json by storyboard.py) gets a transparent overlay
the length of the shot. The card fades in when the quote's first word is spoken, each line appears as it
is said (long quotes turn pages), and the card fades out a moment after the last word, or before the shot
ends. Three styles:

- card: the speaker's picture stays visible on the left; a dark panel on the right holds a large crimson
  quotation mark, the words in a serif, and an attribution line made only of what the narration states.
- memorial: the same layout, softer, with no crimson: for victims and their families.
- exhibit: written words (an engraving, an inscription, a signature) typed on a paper slip in the lower
  middle of the frame, with a small label.

Written as a qtrle .mov with straight alpha (motion.FrameWriter), laid over the shot by assembly.
"""

from __future__ import annotations

import re

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

import graphics
import motion

CRIMSON = (200, 16, 46)
TEXT = (242, 238, 230)
MUTED = (182, 176, 166)
FADE_IN_S, FADE_OUT_S, LINE_IN_S = 0.35, 0.45, 0.28
HOLD_S = 1.5
LEAD_S = 0.12            # the card starts arriving just before the first word


def _scale(width: int) -> float:
    return width / 1920


RELATIVE = re.compile(r",?\s+\b(who|which|that|while|as|when)\b.*$", re.I)


def attribution(q: dict) -> str:
    """'— Speaker, setting, date', with any trailing clause dropped from the setting ("to his mother, who was
    waiting in her car" -> "to his mother"): an attribution names, it does not narrate."""
    setting = RELATIVE.sub("", q.get("setting") or "").strip(" ,")
    parts = [q.get("speaker") or "", setting, q.get("date") or ""]
    parts = [p for p in parts if p]
    return ("— " + ", ".join(parts)) if parts else ""


def window(q: dict, shot_start: float, shot_s: float) -> tuple[float, float]:
    """When, in shot time, the card is up: from just before the first word to a little after the last."""
    begin = max(0.0, q["start"] - shot_start - LEAD_S)
    end = min(shot_s - 0.1, q["end"] - shot_start + HOLD_S)
    return begin, max(begin + 0.6, end)


def _ramp(t: float, begin: float, end: float) -> float:
    """0 before begin, rising to 1 over FADE_IN_S, 1, falling to 0 over FADE_OUT_S at end."""
    if t < begin or t > end:
        return 0.0
    return max(0.0, min(1.0, (t - begin) / FADE_IN_S, (end - t) / FADE_OUT_S))


def line_alphas(q: dict, t_film: float) -> list[float]:
    """How visible each line is at film time t: a line fades in as it is said; only the current page shows."""
    lines, pages = q.get("lines") or [{"text": q["text"], "at": q["start"]}], q.get("pages") or [[0]]
    said = [n for n, line in enumerate(lines) if t_film >= line["at"] - 0.05]
    current = max(said) if said else 0
    page = next((p for p in pages if current in p), pages[0])
    out = []
    for n, line in enumerate(lines):
        if n not in page:
            out.append(0.0)
            continue
        out.append(max(0.0, min(1.0, (t_film - line["at"] + 0.05) / LINE_IN_S)) if n > page[0] else 1.0)
    return out


def _panel(width: int, height: int, strength: float) -> Image.Image:
    """A dark ground over the right of the frame, clear on the left where the speaker's picture shows:
    nothing up to 36% of the width, rising to full by 58%."""
    row = Image.new("L", (width, 1))
    row.putdata([int(255 * strength * max(0.0, min(1.0, (x / width - 0.36) / 0.22))) for x in range(width)])
    ground = Image.new("RGBA", (width, height), (8, 8, 10, 0))
    ground.putalpha(row.resize((width, height)))
    return ground


def card_frame(q: dict, width: int, height: int, card_alpha: float, alphas: list[float], memorial: bool) -> Image.Image:
    k = _scale(width)
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if card_alpha <= 0:
        return frame
    frame.alpha_composite(_panel(width, height, 0.92 if not memorial else 0.8))
    lines = q.get("lines") or [{"text": q["text"]}]
    left, max_w = int(width * 0.52), int(width * 0.42)
    size = int(56 * k)
    font = graphics.font(graphics.SERIF, size)
    while size > 30 * k and max(font.getlength(l["text"]) for l in lines) > max_w:
        size -= 2
        font = graphics.font(graphics.SERIF, size)
    pages = q.get("pages") or [[0]]
    page_len = max(len(p) for p in pages)
    lh = int(size * 1.38)
    top = int(height * 0.5 - (page_len * lh) / 2)
    d = ImageDraw.Draw(frame)
    mark = graphics.font(graphics.SERIF, int(150 * k))
    d.text((left - int(10 * k), top - int(140 * k)), "“", font=mark,
           fill=(MUTED if memorial else CRIMSON) + (int(255 * card_alpha),))
    visible = [n for n, a in enumerate(alphas) if a > 0]
    first = visible[0] if visible else 0
    for n in visible:
        y = top + (n - first) * lh
        d.text((left, y), lines[n]["text"], font=font, fill=TEXT + (int(255 * card_alpha * alphas[n]),))
    who = attribution(q)
    if who:
        small = graphics.font(graphics.LABEL, int(40 * k))
        y = top + page_len * lh + int(22 * k)
        for line in attribution_lines(who, small, max_w):
            d.text((left, y), line, font=small, fill=(206, 201, 192) + (int(240 * card_alpha),))
            y += int(44 * k)
    return frame


def attribution_lines(text: str, fnt, max_width: float, max_lines: int = 2) -> list[str]:
    """The attribution wrapped to the card's width, two lines at most, ending in an ellipsis if cut."""
    lines = graphics.wrap(text, fnt, max_width)
    if len(lines) <= max_lines:
        return lines
    last = lines[max_lines - 1]
    while last and fnt.getlength(last + "…") > max_width:
        last = last.rsplit(" ", 1)[0] if " " in last else last[:-1]
    return lines[:max_lines - 1] + [last.rstrip(",;: ") + "…"]


def exhibit_frame(q: dict, width: int, height: int, card_alpha: float) -> Image.Image:
    """Written words on a paper slip: typed, a little askew, with a soft shadow and a small label."""
    k = _scale(width)
    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if card_alpha <= 0:
        return frame
    type_font = graphics.font(graphics.TYPEWRITER, int(54 * k))
    text = q["text"]
    tw = type_font.getlength(text)
    w, h = int(max(tw + 120 * k, 420 * k)), int(170 * k)
    slip = Image.new("RGBA", (w + 80, h + 80), (0, 0, 0, 0))
    sd = ImageDraw.Draw(slip)
    sd.rectangle((40, 40, 40 + w, 40 + h), fill=(234, 228, 214, 255))
    sd.text((40 + (w - tw) / 2, 40 + h * 0.42), text, font=type_font, fill=(32, 28, 26, 255))
    label = graphics.font(graphics.LABEL, int(22 * k))
    tag = "AS WRITTEN" + (f"  ·  {q['speaker'].upper()}" if q.get("speaker") else "")
    sd.text((40 + 26 * k, 40 + 22 * k), tag, font=label, fill=CRIMSON + (255,))
    slip = slip.rotate(-1.6, resample=Image.BICUBIC, expand=True)
    shadow = Image.new("RGBA", slip.size, (0, 0, 0, 0))
    shadow.putalpha(slip.getchannel("A").point(lambda v: int(v * 0.6)).filter(ImageFilter.GaussianBlur(14)))
    x, y = (width - slip.width) // 2, int(height * 0.66) - slip.height // 2
    frame.alpha_composite(shadow, (x + int(12 * k), y + int(16 * k)))
    frame.alpha_composite(slip, (x, y))
    if card_alpha < 1:
        frame.putalpha(frame.getchannel("A").point(lambda v: int(v * card_alpha)))
    return frame


def frame_at(quotes: list[dict], t: float, shot_start: float, shot_s: float, width: int, height: int) -> tuple:
    """The visual state at shot time t (for caching) and a function that draws it."""
    for q in quotes:
        begin, end = window(q, shot_start, shot_s)
        a = _ramp(t, begin, end)
        if a > 0:
            style = q.get("style") or "card"
            if style == "exhibit":
                return ("x", q["id"], round(a, 2)), lambda: exhibit_frame(q, width, height, a)
            alphas = [round(x, 2) for x in line_alphas(q, shot_start + t)]
            return (("c", q["id"], round(a, 2), tuple(alphas)),
                    lambda: card_frame(q, width, height, a, alphas, style == "memorial"))
    return ("empty",), lambda: Image.new("RGBA", (width, height), (0, 0, 0, 0))


def make(shot: dict, total_frames: int, folder: Path, width: int, height: int, fps: int) -> Path:
    """The shot's quote overlay, drawn once and reused."""
    quotes = shot.get("quotes") or []
    key = json.dumps([quotes, shot["start"], total_frames, width, height, fps,
                      Path(__file__).stat().st_mtime_ns], sort_keys=True, default=str)
    path = Path(folder) / f"quote_{hashlib.sha1(key.encode()).hexdigest()[:10]}.mov"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    shot_s = total_frames / fps
    cache: dict = {}
    with motion.FrameWriter(path, width, height, fps, alpha=True, frames=total_frames) as out:
        for n in range(total_frames):
            state, draw = frame_at(quotes, n / fps, float(shot["start"]), shot_s, width, height)
            if state not in cache:
                cache[state] = draw().tobytes()
            out.write_bytes(cache[state])
    return path

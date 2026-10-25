"""Draw what a camera never took: lower-left overlays, and the rare chart, map or legacy card.

Every full-frame renderer returns a PIL image at the frame size, so assembly treats a drawn shot
exactly like a sourced still. `overlay_image` is the exception: a transparent RGBA frame holding a
date stamp or a name caption in the lower left, which assembly lays over a photograph for a few
seconds. There is no full-screen text in a Revision 2 cut; the title, date and quote cards remain
only so a v1 shot list still renders.

The look is a crime documentary: a near-black ground, off-white text, one deep crimson for rules
and highlights, a neutral grey for secondary labels. Pure and offline - the only thing read from
disk is a font file, and a machine without those fonts falls back to Pillow's default instead of
failing the render.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 1920, 1080

BACKGROUND = (0x0A, 0x0A, 0x0A)  # #0A0A0A, near black but not black
FOREGROUND = (0xF2, 0xF2, 0xF2)  # #F2F2F2, off-white text
ACCENT = (0x9B, 0x11, 0x1E)      # #9B111E, crimson: rules, markers, the one highlighted thing
MUTED = (0x8C, 0x8C, 0x8C)       # #8C8C8C, secondary grey: places, roles, small labels
PANEL = (0x16, 0x16, 0x16)       # map and node fill, a shade off the ground
RULE = (0x3A, 0x3A, 0x3A)        # neutral structure lines; crimson is kept for what matters

FONTS = "/System/Library/Fonts"
SERIF = (f"{FONTS}/Supplemental/Georgia.ttf", f"{FONTS}/Supplemental/Times New Roman.ttf",
         f"{FONTS}/Times.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf")
SERIF_ITALIC = (f"{FONTS}/Supplemental/Georgia Italic.ttf",
                f"{FONTS}/Supplemental/Times New Roman Italic.ttf", f"{FONTS}/Times.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf")
SANS = (f"{FONTS}/Supplemental/Arial.ttf", f"{FONTS}/Helvetica.ttc", f"{FONTS}/SFNS.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
SANS_BOLD = (f"{FONTS}/Supplemental/Arial Bold.ttf", f"{FONTS}/HelveticaNeue.ttc",
             f"{FONTS}/SFNS.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

# Overlay faces. A candidate is a path, or (path, face index) for a .ttc collection.
TYPEWRITER = ((f"{FONTS}/Supplemental/American Typewriter.ttc", 0),
              (f"{FONTS}/Supplemental/AmericanTypewriter.ttc", 0),
              f"{FONTS}/Supplemental/Courier New.ttf", (f"{FONTS}/Courier.ttc", 0),
              "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
NAME_BOLD = ((f"{FONTS}/HelveticaNeue.ttc", 1), f"{FONTS}/Supplemental/Arial Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
LABEL = ((f"{FONTS}/HelveticaNeue.ttc", 10), (f"{FONTS}/HelveticaNeue.ttc", 0),
         f"{FONTS}/Supplemental/Arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

OVERLAY_STYLES = ("stamp", "name")
OVERLAY_LEFT = 0.07     # of the width: the overlay's left edge, inside the title-safe margin
OVERLAY_BOTTOM = 0.82   # of the height: the overlay's lowest ink, clear of bottom-centre subtitles
OVERLAY_WIDTH = 0.30    # of the width: text wraps before this, so it stays clear of a print

MAX_EVENTS = 8   # beyond this a timeline is mush; the planner splits the shot instead
MAX_NODES = 8    # likewise for a relationship chart
DATE_CARD_CHARS = 34  # a card this short with a subtext is a scene stamp, not a title


# --- primitives ------------------------------------------------------------

def font(candidates, size: int):
    """The first candidate that loads at this size, else Pillow's own default.

    A candidate is a font path, or (path, index) to pick one face out of a .ttc collection.
    """
    for candidate in candidates:
        path, index = candidate if isinstance(candidate, tuple) else (candidate, 0)
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow without a sizeable default
        return ImageFont.load_default()


def line_height(fnt, leading: float = 1.25) -> int:
    """Baseline-to-baseline step for this font, with room for descenders."""
    return max(1, int(fnt.getbbox("Agjy")[3] * leading))


def wrap(text: str, fnt, max_width: float) -> list[str]:
    """Greedy word wrap; a word wider than the line gets a line of its own."""
    lines: list[str] = []
    current = ""
    for word in str(text).split():
        trial = f"{current} {word}".strip()
        if current and fnt.getlength(trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def fit_lines(text: str, candidates, size: float, max_width: float, max_lines: int = 3,
              min_size: int = 20):
    """Largest font at or below `size` whose wrap of `text` fits `max_lines` of `max_width`.

    Returns (font, lines). Text too long even at `min_size` loses its tail rather than overflowing.
    """
    size = int(size)
    while size > min_size:
        fnt = font(candidates, size)
        lines = wrap(text, fnt, max_width)
        if len(lines) <= max_lines:
            return fnt, lines
        size = int(size * 0.94)
    fnt = font(candidates, min_size)
    return fnt, wrap(text, fnt, max_width)[:max_lines]


def frame(width: int, height: int) -> Image.Image:
    """An empty frame on the documentary ground."""
    return Image.new("RGB", (width, height), BACKGROUND)


def draw_lines(draw, lines, fnt, top: float, colour, centre: float | None = None,
               left: float | None = None, leading: float = 1.25) -> float:
    """Stack lines centred on `centre` or flush to `left`; returns the y below the block."""
    step = line_height(fnt, leading)
    for i, line in enumerate(lines):
        x = centre - fnt.getlength(line) / 2 if centre is not None else left
        draw.text((x, top + i * step), line, font=fnt, fill=colour)
    return top + len(lines) * step


def tracked_width(text: str, fnt, tracking: float) -> float:
    return sum(fnt.getlength(c) for c in text) + tracking * max(0, len(text) - 1)


def draw_tracked(draw, xy, text: str, fnt, colour, tracking: float) -> None:
    """Letter-spaced label: Pillow has no tracking, so step glyph by glyph."""
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=fnt, fill=colour)
        x += fnt.getlength(char) + tracking


def rule(draw, x: float, y: float, length: float, colour=ACCENT, thickness: int = 2) -> None:
    draw.rectangle((x, y, x + length, y + thickness - 1), fill=colour)


# --- cards -----------------------------------------------------------------

def title_card(text: str, subtext: str | None = None, width: int = WIDTH, height: int = HEIGHT,
               serif=SERIF, sans=SANS) -> Image.Image:
    """The episode or chapter title: large centred serif over an accent rule, optional subtext."""
    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.11)
    max_width = width - 2 * margin

    title_font, lines = fit_lines(text, serif, height * 0.105, max_width, max_lines=3,
                                  min_size=int(height * 0.030))
    sub_font = font(sans, int(height * 0.026))
    tracking = height * 0.006
    sub_lines = wrap(subtext, sub_font, max_width) if subtext else []

    gap = height * 0.045
    block = len(lines) * line_height(title_font, 1.2)
    sub_block = len(sub_lines) * line_height(sub_font, 1.5) if sub_lines else 0
    total = block + gap + 2 + (gap + sub_block if sub_block else 0)
    top = (height - total) / 2

    bottom = draw_lines(draw, lines, title_font, top, FOREGROUND, centre=width / 2, leading=1.2)
    rule_length = width * 0.07
    rule(draw, (width - rule_length) / 2, bottom + gap, rule_length)
    y = bottom + gap + 2 + gap
    for line in sub_lines:
        upper = line.upper()
        draw_tracked(draw, ((width - tracked_width(upper, sub_font, tracking)) / 2, y),
                     upper, sub_font, MUTED, tracking)
        y += line_height(sub_font, 1.5)
    return image


def date_card(text: str, subtext: str | None = None, width: int = WIDTH, height: int = HEIGHT,
              serif=SERIF, sans=SANS) -> Image.Image:
    """The scene stamp: a date lower-left under a short accent rule, place name beneath."""
    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    left = int(width * 0.085)
    max_width = width * 0.50

    date_font, lines = fit_lines(text, serif, height * 0.055, max_width, max_lines=2,
                                 min_size=int(height * 0.028))
    sub_font = font(sans, int(height * 0.024))
    tracking = height * 0.005
    sub_lines = wrap(subtext, sub_font, max_width) if subtext else []

    gap = height * 0.022
    block = len(lines) * line_height(date_font, 1.2)
    sub_block = len(sub_lines) * line_height(sub_font, 1.5) if sub_lines else 0
    total = 2 + gap + block + (gap + sub_block if sub_block else 0)
    top = height * 0.87 - total

    rule(draw, left, top, width * 0.045)
    y = draw_lines(draw, lines, date_font, top + 2 + gap, FOREGROUND, left=left, leading=1.2)
    y += gap if sub_lines else 0
    for line in sub_lines:
        upper = line.upper()
        draw_tracked(draw, (left, y), upper, sub_font, MUTED, tracking)
        y += line_height(sub_font, 1.5)
    return image


def quote_card(text: str, attribution: str | None = None, width: int = WIDTH, height: int = HEIGHT,
               serif=SERIF_ITALIC, sans=SANS) -> Image.Image:
    """A pulled quotation in serif italic, hanging accent quotation mark, small attribution."""
    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.14)
    max_width = width - 2 * margin

    quote_font, lines = fit_lines(text, serif, height * 0.068, max_width, max_lines=5,
                                  min_size=int(height * 0.028))
    attr_font = font(sans, int(height * 0.024))
    tracking = height * 0.005

    gap = height * 0.055
    block = len(lines) * line_height(quote_font, 1.34)
    total = block + (gap + line_height(attr_font, 1.4) if attribution else 0)
    top = (height - total) / 2

    mark_font = font(SERIF, int(height * 0.17))
    draw.text((margin - mark_font.getlength("“") * 0.8, top - height * 0.045),
              "“", font=mark_font, fill=ACCENT)
    y = draw_lines(draw, lines, quote_font, top, FOREGROUND, left=margin, leading=1.34)
    if attribution:
        label = f"— {attribution}".upper()
        draw_tracked(draw, (margin, y + gap), label, attr_font, MUTED, tracking)
    return image


# --- timeline --------------------------------------------------------------

def timeline(events: list[dict], highlight: int | None = None, width: int = WIDTH,
             height: int = HEIGHT) -> Image.Image:
    """A horizontal timeline: baseline rule, a tick and dot per event, label above, text below.

    Events sit in equal slots so the first and last keep their margin. When a label or a line of
    text is wider than its slot the blocks are staggered, odd events standing further from the
    baseline than even ones, so neighbours cannot collide. The highlighted event is drawn in the
    accent colour and slightly larger, its text in off-white. Beyond MAX_EVENTS it raises, so the planner splits the shot.
    """
    if not events:
        raise ValueError("a timeline needs at least one event")
    if len(events) > MAX_EVENTS:
        raise ValueError(f"{len(events)} events will not read; split at {MAX_EVENTS}")
    if highlight is not None and not 0 <= highlight < len(events):
        raise ValueError(f"highlight {highlight} is outside 0..{len(events) - 1}")

    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    margin = int(width * 0.09)
    usable = width - 2 * margin
    slot = usable / len(events)
    baseline = height * 0.52
    rule(draw, margin, baseline, usable, RULE)

    label_size, text_size = height * 0.024, height * 0.027
    blocks = []
    for i, event in enumerate(events):
        hot = i == highlight
        scale = 1.18 if hot else 1.0
        label_font, label_lines = fit_lines(event.get("label", ""), SANS_BOLD, label_size * scale,
                                            slot * 0.92, max_lines=2, min_size=14)
        text_font, text_lines = fit_lines(event.get("text", ""), SERIF, text_size * scale,
                                          slot * 0.92, max_lines=3, min_size=14)
        blocks.append((hot, label_font, label_lines, text_font, text_lines))

    def widest(fnt, lines) -> float:
        return max(fnt.getlength(line) for line in lines)

    crowded = any(widest(lf, ll) > slot * 0.9 or widest(tf, tl) > slot * 0.9
                  for _, lf, ll, tf, tl in blocks)
    step = height * 0.085 if crowded else 0
    tick, gap = height * 0.017, height * 0.022

    for i, (hot, label_font, label_lines, text_font, text_lines) in enumerate(blocks):
        centre = margin + slot * (i + 0.5)
        colour = ACCENT if hot else FOREGROUND
        level = (i % 2) if crowded else 0
        radius = height * 0.011 if hot else height * 0.0075
        reach = tick + level * step
        draw.line((centre, baseline - reach, centre, baseline + reach), fill=RULE, width=2)
        draw.ellipse((centre - radius, baseline - radius + 1, centre + radius, baseline + radius + 1),
                     fill=colour)

        label_block = len(label_lines) * line_height(label_font, 1.2)
        draw_lines(draw, label_lines, label_font, baseline - reach - gap - label_block, colour,
                   centre=centre, leading=1.2)
        draw_lines(draw, text_lines, text_font, baseline + reach + gap, FOREGROUND if hot else MUTED,
                   centre=centre, leading=1.25)
    return image


# --- relationship chart ----------------------------------------------------

def node_boxes(count: int, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Box per node: a circle up to six, two rows beyond that. Never overlapping, always inside."""
    if count < 1:
        raise ValueError("a relationship chart needs at least one node")
    if count > MAX_NODES:
        raise ValueError(f"{count} nodes will not read; split at {MAX_NODES}")

    box_w, box_h = int(width * 0.205), int(height * 0.115)
    centres: list[tuple[float, float]] = []
    if count <= 6:
        rx = (width - box_w) / 2 * 0.76
        ry = (height - box_h) / 2 * 0.74
        for i in range(count):
            angle = math.radians(-90 + i * 360 / count)
            centres.append((width / 2 + rx * math.cos(angle), height / 2 + ry * math.sin(angle)))
    else:
        rows = ((count + 1) // 2, count // 2)
        box_w = min(box_w, int(width / max(rows)) - int(width * 0.015))
        for n, y in zip(rows, (height * 0.30, height * 0.70)):
            for j in range(n):
                centres.append((width / n * (j + 0.5), y))
    return [(int(cx - box_w / 2), int(cy - box_h / 2), int(cx + box_w / 2), int(cy + box_h / 2))
            for cx, cy in centres]


def clear_of(rect, boxes) -> bool:
    """True when a rectangle touches none of the node boxes."""
    return all(rect[2] <= b[0] or b[2] <= rect[0] or rect[3] <= b[1] or b[3] <= rect[1] for b in boxes)


def relationship_chart(nodes: list[dict], edges: list[tuple[int, int, str]] | None = None,
                       width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """Who connects to whom: rounded boxes laid on a circle (two rows past six), labelled edges."""
    boxes = node_boxes(len(nodes), width, height)
    edges = list(edges or [])
    for a, b, _label in edges:
        if not (0 <= a < len(nodes) and 0 <= b < len(nodes)):
            raise ValueError(f"edge ({a}, {b}) points outside 0..{len(nodes) - 1}")

    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    edge_font = font(SANS, int(height * 0.019))
    box_h = boxes[0][3] - boxes[0][1]
    name_size, role_size = height * 0.030, int(height * 0.021)

    for a, b, label in edges:
        ax, ay = (boxes[a][0] + boxes[a][2]) / 2, (boxes[a][1] + boxes[a][3]) / 2
        bx, by = (boxes[b][0] + boxes[b][2]) / 2, (boxes[b][1] + boxes[b][3]) / 2
        draw.line((ax, ay, bx, by), fill=RULE, width=2)
        if not label:
            continue
        # The midpoint of a short edge lands inside a box, which is drawn over it, so step the
        # label off the line perpendicular until it clears every box.
        w, h, pad = edge_font.getlength(label), line_height(edge_font, 1.1), height * 0.009
        span = math.hypot(bx - ax, by - ay) or 1
        nx, ny = -(by - ay) / span, (bx - ax) / span
        for offset in (0, box_h / 2 + h, -(box_h / 2 + h)):
            mx, my = (ax + bx) / 2 + nx * offset, (ay + by) / 2 + ny * offset
            rect = (mx - w / 2 - pad, my - h / 2 - pad, mx + w / 2 + pad, my + h / 2 + pad)
            if clear_of(rect, boxes):
                break
        draw.rectangle(rect, fill=BACKGROUND)
        draw.text((mx - w / 2, my - h / 2), label, font=edge_font, fill=MUTED)

    for node, box in zip(nodes, boxes):
        draw.rounded_rectangle(box, radius=int(height * 0.014), fill=PANEL, outline=RULE, width=2)
        centre_x, inner = (box[0] + box[2]) / 2, (box[2] - box[0]) - height * 0.028
        name_font, name_lines = fit_lines(node.get("name", ""), SANS_BOLD, name_size, inner,
                                          max_lines=1, min_size=14)
        role_font = font(SANS, role_size)
        role_lines = wrap(node.get("role", ""), role_font, inner)[:1] if node.get("role") else []
        block = line_height(name_font, 1.2) + (line_height(role_font, 1.4) if role_lines else 0)
        y = (box[1] + box[3]) / 2 - block / 2
        y = draw_lines(draw, name_lines, name_font, y, FOREGROUND, centre=centre_x, leading=1.2)
        draw_lines(draw, role_lines, role_font, y, MUTED, centre=centre_x, leading=1.4)
    return image


# --- map -------------------------------------------------------------------

def map_card(points: list[dict], title: str | None = None, width: int = WIDTH,
             height: int = HEIGHT, connect: bool = True) -> Image.Image:
    """A schematic map, not a geographic one: points placed by fractional panel coordinates."""
    if not points:
        raise ValueError("a map needs at least one point")

    image = frame(width, height)
    draw = ImageDraw.Draw(image)
    left, right = int(width * 0.10), int(width * 0.90)
    top, bottom = int(height * 0.20), int(height * 0.86)

    if title:
        title_font = font(SANS, int(height * 0.026))
        tracking = height * 0.006
        draw_tracked(draw, (left, top - height * 0.072), str(title).upper(), title_font, MUTED, tracking)

    draw.rounded_rectangle((left, top, right, bottom), radius=int(height * 0.006),
                           fill=PANEL, outline=RULE, width=2)

    def place(point):
        x = min(max(float(point.get("x", 0.5)), 0.04), 0.96)
        y = min(max(float(point.get("y", 0.5)), 0.04), 0.96)
        return left + x * (right - left), top + y * (bottom - top)

    spots = [place(p) for p in points]
    if connect and len(spots) > 1:
        draw.line([(x, y) for x, y in spots], fill=RULE, width=1)

    label_font = font(SANS, int(height * 0.024))
    radius = height * 0.008
    for (x, y), point in zip(spots, points):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=ACCENT)
        name = str(point.get("name", ""))
        offset, text_w = height * 0.018, label_font.getlength(name)
        text_x = x + offset if x + offset + text_w < right - height * 0.012 else x - offset - text_w
        draw.text((text_x, y - line_height(label_font, 1.0) / 2), name, font=label_font, fill=FOREGROUND)
    return image


# --- overlays ---------------------------------------------------------------
#
# The only text a Revision 2 cut puts on screen: a fact from the narration, small, lower left, for
# a few seconds. Each overlay is drawn as coverage masks (one per colour) and composed as straight
# alpha, so glyph edges keep their colour instead of fringing dark. Behind the ink sits a soft,
# translucent dark cloud and a tight blurred shadow, which hold the text off a bright photograph
# without ever reading as a box.

def _mask(size) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    mask = Image.new("L", size, 0)
    return mask, ImageDraw.Draw(mask)


def _layer(size, colour, mask: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", size, colour + (0,))
    layer.putalpha(mask)
    return layer


def _reveal(lines: list[str], count: int | None) -> list[str]:
    """The first `count` characters of a wrapped block, line by line; None shows everything.

    The space a wrap consumed between two lines is not counted, so typing never pauses on it.
    """
    if count is None:
        return list(lines)
    shown, left = [], max(0, int(count))
    for line in lines:
        if left <= 0:
            break
        shown.append(line[:left])
        left -= len(line)
    return shown


def stamp_text_length(text: str, subtext: str | None, width: int = WIDTH,
                      height: int = HEIGHT) -> tuple[int, int]:
    """How many characters a stamp types: the date's, then the place's (as laid out, in caps)."""
    date_font, lines, place_font, places = _stamp_layout(text, subtext, width, height)
    return sum(len(line) for line in lines), sum(len(line) for line in places)


def _stamp_layout(text, subtext, width, height):
    max_width = width * OVERLAY_WIDTH
    date_font, lines = fit_lines(text, TYPEWRITER, height * 0.050, max_width, max_lines=2,
                                 min_size=max(8, int(height * 0.030)))
    place_font = font(LABEL, max(6, int(height * 0.0205)))
    places = wrap(subtext.upper(), place_font, max_width * 0.9)[:2] if subtext else []
    return date_font, lines, place_font, places


def _stamp_masks(text, subtext, width, height, size, date_chars=None, place_chars=None, caret=None):
    """A date in a typewriter face under a short crimson rule, the place in tracked grey caps.

    The layout always comes from the whole text, so a stamp typing itself out never shifts: only
    the first `date_chars` of the date and `place_chars` of the place are inked, and `caret`
    ("date" or "place") puts a typewriter block just after the last character typed on that line.
    """
    left, bottom = width * OVERLAY_LEFT, height * OVERLAY_BOTTOM
    max_width = width * OVERLAY_WIDTH
    date_font, lines, place_font, places = _stamp_layout(text, subtext, width, height)
    tracking = height * 0.0042
    thickness = max(2, round(height * 0.0037))
    gap = height * 0.016

    date_step, place_step = line_height(date_font, 1.18), line_height(place_font, 1.55)
    place_block = len(places) * place_step
    date_top = bottom - place_block - (gap if places else 0) - len(lines) * date_step
    rule_top = date_top - gap - thickness
    place_top = date_top + len(lines) * date_step + (gap if places else 0)

    white, white_draw = _mask(size)
    grey, grey_draw = _mask(size)
    red, red_draw = _mask(size)
    red_draw.rectangle((left, rule_top, left + width * 0.034, rule_top + thickness - 1), fill=255)
    shown_dates = _reveal(lines, date_chars)
    draw_lines(white_draw, shown_dates, date_font, date_top, 255, left=left, leading=1.18)
    shown_places = _reveal(places, place_chars)
    y = place_top
    for line in shown_places:
        draw_tracked(grey_draw, (left, y), line, place_font, 255, tracking)
        y += place_step

    if caret in ("date", "place"):
        if caret == "date" or not places:
            row = max(0, len(shown_dates) - 1)
            typed = shown_dates[-1] if shown_dates else ""
            x = left + date_font.getlength(typed) + (date_font.getlength(" ") * 0.25 if typed else 0)
            top_y = date_top + row * date_step
            glyph = date_font
        else:
            row = max(0, len(shown_places) - 1)
            typed = shown_places[-1] if shown_places else ""
            x = left + tracked_width(typed, place_font, tracking) + (tracking * 1.5 if typed else 0)
            top_y = place_top + row * place_step
            glyph = place_font
        _, cap_top, _, base = glyph.getbbox("H")
        block_w = max(2, round(glyph.getlength("n") * 0.55))
        white_draw.rectangle((x, top_y + cap_top, x + block_w - 1, top_y + base), fill=255)
    return white, grey, red, (left, rule_top, left + max_width, bottom)


def _name_masks(text, subtext, width, height, size):
    """A name in bold sans, the role in grey beneath, a thin crimson bar to their left."""
    left, bottom = width * OVERLAY_LEFT, height * OVERLAY_BOTTOM
    bar = max(2, round(height * 0.0046))
    indent = bar + height * 0.020
    max_width = width * OVERLAY_WIDTH - indent
    name_font, lines = fit_lines(text, NAME_BOLD, height * 0.046, max_width, max_lines=2,
                                 min_size=max(8, int(height * 0.028)))
    role_font = font(LABEL, max(6, int(height * 0.025)))
    roles = wrap(subtext, role_font, max_width)[:2] if subtext else []
    gap = height * 0.008

    name_step, role_step = line_height(name_font, 1.15), line_height(role_font, 1.35)
    block = len(lines) * name_step + (gap + len(roles) * role_step if roles else 0)
    top = bottom - block

    white, white_draw = _mask(size)
    grey, grey_draw = _mask(size)
    red, red_draw = _mask(size)
    # The bar runs from the cap height of the name to the baseline of the last role line.
    cap_top = top + name_font.getbbox("H")[1]
    red_draw.rectangle((left, cap_top, left + bar - 1, bottom - height * 0.004), fill=255)
    y = draw_lines(white_draw, lines, name_font, top, 255, left=left + indent, leading=1.15)
    if roles:
        draw_lines(grey_draw, roles, role_font, y + gap, 255, left=left + indent, leading=1.35)
    return white, grey, red, (left, cap_top, left + indent + max_width, bottom)


def overlay_image(style: str, text: str, subtext: str | None = None, width: int = WIDTH,
                  height: int = HEIGHT) -> Image.Image:
    """A transparent frame with a date stamp or a name caption in the lower left.

    `stamp`: the date in a typewriter face, the place beneath in small letter-spaced grey caps, a
    short crimson rule above. `name`: the name in bold sans, the role beneath in grey, a thin
    crimson bar to the left. Nothing is drawn in the top half or the right third of the frame, and
    no ink falls below OVERLAY_BOTTOM, which keeps bottom-centre subtitles clear.
    """
    if style not in OVERLAY_STYLES:
        raise ValueError(f"overlay style {style!r} is not one of {OVERLAY_STYLES}")
    text, subtext = _overlay_text(text, subtext)
    size = (width, height)
    build = _stamp_masks if style == "stamp" else _name_masks
    white, grey, red, box = build(text, subtext, width, height, size)
    return _compose(white, grey, red, box, width, height)


def stamp_frame(text: str, subtext: str | None, width: int = WIDTH, height: int = HEIGHT, *,
                date_chars: int | None = None, place_chars: int | None = None,
                caret: str | None = None, cloud: Image.Image | None = None) -> Image.Image:
    """One frame of a date stamp typing itself out: the stamp as `overlay_image` draws it, with
    only part of the text inked and an optional caret. The soft backing is the whole stamp's from
    the first frame (pass `cloud` from `stamp_cloud` to reuse it), so it never grows as text types.
    """
    text, subtext = _overlay_text(text, subtext)
    size = (width, height)
    white, grey, red, box = _stamp_masks(text, subtext, width, height, size, date_chars=date_chars,
                                         place_chars=place_chars, caret=caret)
    return _compose(white, grey, red, box, width, height, cloud=cloud)


def stamp_cloud(text: str, subtext: str | None, width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """The soft dark backing of a whole stamp, computed once for a typing animation."""
    text, subtext = _overlay_text(text, subtext)
    _, _, _, box = _stamp_masks(text, subtext, width, height, (width, height))
    return _cloud(box, width, height)


def _overlay_text(text, subtext):
    text = (text or "").strip()
    if not text:
        raise ValueError("an overlay needs text")
    return text, (subtext or "").strip() or None


def _cloud(box, width, height) -> Image.Image:
    left, top, right, bottom = box
    cloud, cloud_draw = _mask((width, height))
    pad_x, pad_y = width * 0.03, height * 0.04
    cloud_draw.rectangle((max(0, left - pad_x), top - pad_y, right, min(height, bottom + pad_y)),
                         fill=150)
    return cloud.filter(ImageFilter.GaussianBlur(height * 0.035))


def _compose(white, grey, red, box, width, height, cloud=None) -> Image.Image:
    """Ink over legibility: a broad soft cloud behind the block, a tight shadow under every stroke."""
    size = (width, height)
    ink = _lighter(white, grey, red)
    cloud = _cloud(box, width, height) if cloud is None else cloud
    shadow = Image.new("L", size, 0)
    shadow.paste(ink, (max(1, round(height * 0.0012)), max(1, round(height * 0.0018))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(1.0, height * 0.0045))).point(lambda a: a * 0.9)
    dark = _lighter(cloud, shadow)
    # Hard limits, well outside where the cloud has already faded: never the top half, never the
    # right third.
    keep, keep_draw = _mask(size)
    keep_draw.rectangle((0, height // 2, int(width * 0.68), height), fill=255)
    dark = Image.composite(dark, Image.new("L", size, 0), keep)

    out = _layer(size, (0, 0, 0), dark)
    for colour, mask in ((MUTED, grey), (FOREGROUND, white), (ACCENT, red)):
        out = Image.alpha_composite(out, _layer(size, colour, mask))
    return out


def _lighter(*masks: Image.Image) -> Image.Image:
    """The per-pixel maximum of several masks."""
    out = masks[0]
    for mask in masks[1:]:
        out = ImageChops.lighter(out, mask)
    return out


def save_overlay(image: Image.Image, path) -> Path:
    """Write an overlay as an RGBA PNG, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGBA").save(path, "PNG")
    return path


# --- dispatch and output ---------------------------------------------------

def render_shot(shot: dict, width: int = WIDTH, height: int = HEIGHT) -> Image.Image:
    """Draw one shots.json shot of a drawn kind.

    `card` is a date card when it carries a subtext and its text is short enough to be a stamp
    (up to DATE_CARD_CHARS, the width of "December 11th, 1978"); anything longer, or with no
    subtext, is a title card. `chart` is a timeline when it carries `events` and a relationship
    chart when it carries `nodes`. `map` draws `points`. Any other kind is not ours to draw.
    """
    kind = shot.get("kind")
    text = shot.get("text") or ""
    if kind == "card":
        if not text:
            raise ValueError("a card shot needs text")
        subtext = shot.get("subtext")
        if subtext and len(text) <= DATE_CARD_CHARS:
            return date_card(text, subtext, width, height)
        return title_card(text, subtext, width, height)
    if kind == "chart":
        if shot.get("events"):
            return timeline(shot["events"], shot.get("highlight"), width, height)
        if shot.get("nodes"):
            return relationship_chart(shot["nodes"], shot.get("edges"), width, height)
        raise ValueError("a chart shot needs events or nodes")
    if kind == "map":
        if not shot.get("points"):
            raise ValueError("a map shot needs points")
        return map_card(shot["points"], text or None, width, height)
    raise ValueError(f"kind {kind!r} is not drawn")


def save(image: Image.Image, path) -> Path:
    """Write the frame as a JPEG at quality 92, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "JPEG", quality=92)
    return path


def main() -> None:
    """Render one of each kind with Gacy-case content, for a human to look at."""
    out = Path(__file__).resolve().parents[2] / "data" / "video" / "samples"
    samples = {
        "title_card.jpg": title_card("Thirty Three", "The man who lived above a crawl space"),
        "date_card.jpg": date_card("December 11th, 1978", "Des Plaines, Illinois"),
        "timeline.jpg": timeline([
            {"label": "Dec 11, 1978", "text": "Rob Piest disappears after his shift"},
            {"label": "Dec 12, 1978", "text": "Gacy named by the pharmacy owner"},
            {"label": "Dec 13, 1978", "text": "First search warrant on Summerdale"},
            {"label": "Dec 21, 1978", "text": "Gacy confesses; the crawl space is opened"},
            {"label": "Mar 13, 1980", "text": "Convicted of thirty three murders"},
        ], highlight=3),
        "relationship_chart.jpg": relationship_chart([
            {"name": "John Wayne Gacy", "role": "contractor, PDM Contractors"},
            {"name": "Rob Piest", "role": "victim, aged 15"},
            {"name": "Nisson Pharmacy", "role": "Piest's workplace"},
            {"name": "Des Plaines Police", "role": "Lt. Joseph Kozenczak"},
            {"name": "Michael Rossi", "role": "employee, drove the Oldsmobile"},
            {"name": "Cook County", "role": "state's attorney"},
        ], [
            (0, 1, "offered work"),
            (1, 2, "employed at"),
            (2, 3, "reported"),
            (3, 0, "surveillance"),
            (0, 4, "employed"),
            (3, 5, "charged by"),
        ]),
        "quote_card.jpg": quote_card(
            "There are thirty three of them. I only remember a handful of the names.",
            "John Wayne Gacy, December 1978"),
        "map_card.jpg": map_card([
            {"name": "Des Plaines", "x": 0.33, "y": 0.18},
            {"name": "Nisson Pharmacy", "x": 0.52, "y": 0.30},
            {"name": "8213 W Summerdale Ave", "x": 0.44, "y": 0.62},
            {"name": "Des Plaines River", "x": 0.74, "y": 0.80},
        ], "Cook County, Illinois"),
    }
    for name, image in samples.items():
        print(save(image, out / name))


if __name__ == "__main__":
    main()

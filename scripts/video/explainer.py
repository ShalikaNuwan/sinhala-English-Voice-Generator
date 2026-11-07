"""Explainers: a map or a floor plan that tells a passage of narration as it is spoken.

    .venv/bin/python scripts/video/explainer.py <spec.json> <out.mp4> <video_dir>

Used where no real photograph of the moment exists: instead of a stand-in picture, the film shows
what the narration says, drawn honestly. A spec (JSON) names a base and a timeline:

- base: {"kind": "map", "bbox": [lat_min, lon_min, lat_max, lon_max], "zoom": 13} - OpenStreetMap
  tiles (cached in <video_dir>/assets/maps), restyled dark: land near black, water blue-black,
  streets and names faint. Or {"kind": "plan", "rooms": [...], "walls": [...], "notes": [...]}:
  a line-drawn floor plan on dark drafting paper, every box and line in plan units 0-100 x 0-56.
- elements, each appearing at its `at` (seconds into the explainer) and gone after `until`:
  pin, tag (a typed index-card label on a leader line), route (a polyline drawing itself, dashed
  or solid), pulse (rings spreading from a point: an alarm, a telephone), icon (bed, alarm,
  phone, bicycle, closet, door, stairs, tv, bag, key), photo (a real library picture as a pinned
  print), mark (a crimson box drawn round something), dim (the whole board darkens).
- hud: a typed date stamp, a clock whose hands move between two times, an hours counter.
- theme: "dark" (the default) or "light": a warm linen board and a light sepia map, charcoal ink.
- camera: [[t, x, y, zoom], ...] in board fractions, joined and softened (motion.camera_path).

Positions are board fractions [x, y]; on a map an element may give "ll": [lat, lon] instead, and
on a plan "pt": [u, v] in plan units. The board is drawn at three times the frame size and filmed
through a moving box, so close-ups stay sharp. Every frame is a pure function of the spec and t.
"""

from __future__ import annotations

import json
import math
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphics  # noqa: E402
import motion  # noqa: E402
from graphics import ACCENT, FOREGROUND, MUTED  # noqa: E402

FPS = 30
SS = 3                                   # board pixels per frame pixel at zoom 1
PAPER = (13, 16, 21)                     # the plan's drafting paper
INK = (226, 228, 232)                    # plan lines
RED = (206, 44, 58)                      # crimson that reads on a dark board
LAND, WATER = (22, 22, 25), (9, 13, 21)
CARD, CARD_EDGE = motion.PAPER, None     # index cards and tags; an edge only where the board is light
PANEL = (6, 7, 9)                        # the box behind big words on a map
NOTE = MUTED                             # small print: notes, room names, map credits
TEXT = FOREGROUND                        # HUD words drawn straight on the frame
MAP_STYLE = "dark"

THEMES = {
    "dark": {"PAPER": (13, 16, 21), "INK": (226, 228, 232), "RED": (206, 44, 58), "CARD": motion.PAPER,
             "CARD_EDGE": None, "PANEL": (6, 7, 9), "NOTE": MUTED, "TEXT": FOREGROUND, "MAP_STYLE": "dark",
             "GRID": (22, 27, 35)},
    # a warm linen drafting board: charcoal ink, deeper crimson, white cards with a fine edge, a sepia map
    "light": {"PAPER": (222, 215, 201), "INK": (40, 40, 44), "RED": (178, 28, 42), "CARD": (251, 249, 244),
              "CARD_EDGE": (176, 168, 152), "PANEL": (247, 244, 237), "NOTE": (98, 92, 84), "TEXT": (34, 34, 38),
              "MAP_STYLE": "light", "GRID": (206, 198, 182)},
}
# mid-tone boards, between the dark and the light: cream cards, light ink (dark ink on the paler stone), and a
# map tinted to the board colour
_MID = {"CARD": (248, 245, 238), "CARD_EDGE": None, "MAP_STYLE": "tint"}
THEMES.update({
    "slate": dict(_MID, PAPER=(70, 80, 92), GRID=(80, 91, 104), INK=(236, 232, 224), RED=(222, 70, 76),
                  PANEL=(38, 44, 52), NOTE=(196, 200, 206), TEXT=(240, 238, 234), MAP_WATER=(50, 62, 80)),
    "stone": dict(_MID, PAPER=(142, 130, 115), GRID=(152, 140, 125), INK=(30, 28, 26), RED=(150, 22, 34),
                  PANEL=(250, 247, 240), NOTE=(58, 52, 46), TEXT=(28, 26, 24), MAP_WATER=(112, 124, 132),
                  CARD_EDGE=(110, 100, 88)),
    "tobacco": dict(_MID, PAPER=(86, 70, 56), GRID=(97, 80, 65), INK=(240, 230, 214), RED=(226, 82, 72),
                    PANEL=(46, 37, 29), NOTE=(205, 190, 170), TEXT=(244, 236, 222), MAP_WATER=(62, 70, 74)),
    "petrol": dict(_MID, PAPER=(44, 70, 76), GRID=(54, 82, 88), INK=(232, 236, 232), RED=(232, 86, 80),
                   PANEL=(24, 40, 44), NOTE=(184, 204, 204), TEXT=(240, 242, 238), MAP_WATER=(28, 48, 62)),
    "graphite": dict(_MID, PAPER=(98, 96, 92), GRID=(108, 106, 101), INK=(240, 238, 232), RED=(224, 70, 72),
                     PANEL=(52, 51, 49), NOTE=(200, 198, 190), TEXT=(244, 242, 236), MAP_WATER=(74, 82, 92)),
})
MAP_WATER = (50, 62, 80)
GRID = THEMES["dark"]["GRID"]


def use_theme(name: str | None) -> None:
    """Set the palette every drawing function reads: "dark" (the default) or "light"."""
    globals().update(THEMES[name or "dark"])
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
USER_AGENT = "doc-video-research/1.0"
PLAN_W, PLAN_H = 100.0, 56.0


# --- geometry -------------------------------------------------------------------

def lonlat_to_px(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Web Mercator pixel coordinates at a tile zoom (256 px tiles)."""
    n = 256 * 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n
    return x, y


def map_frame(bbox, zoom: int, aspect: float = 16 / 9) -> tuple[float, float, float, float]:
    """The pixel box (at `zoom`) round a lat/lon box, widened to the frame's aspect."""
    lat0, lon0, lat1, lon1 = bbox
    x0, y1 = lonlat_to_px(lat0, lon0, zoom)
    x1, y0 = lonlat_to_px(lat1, lon1, zoom)
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:
        grow = h * aspect - w
        x0, x1 = x0 - grow / 2, x1 + grow / 2
    else:
        grow = w / aspect - h
        y0, y1 = y0 - grow / 2, y1 + grow / 2
    return x0, y0, x1, y1


def ease(p: float) -> float:
    return motion.ease_in_out_sine(motion.clamp(p))


def window(t: float, at: float, until: float | None, fade: float = 0.35) -> float:
    """How visible an element is at t: rises over `fade` from `at`, falls over `fade` before `until`."""
    if t < at:
        return 0.0
    up = motion.clamp((t - at) / fade) if fade else 1.0
    down = 1.0 if until is None else motion.clamp((until - t) / fade) if fade else float(t < until)
    return ease(min(up, down))


def polyline_part(points, fraction: float):
    """The first `fraction` of a polyline, by length."""
    if fraction >= 1:
        return list(points)
    lengths = [math.dist(a, b) for a, b in zip(points, points[1:])]
    goal = sum(lengths) * max(0.0, fraction)
    out = [points[0]]
    for (a, b), seg in zip(zip(points, points[1:]), lengths):
        if goal <= seg:
            u = goal / seg if seg else 0
            out.append((a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u))
            return out
        out.append(b)
        goal -= seg
    return out


def dashed(points, dash: float, gap: float):
    """A polyline cut into dashes."""
    out, carry, on = [], 0.0, True
    for a, b in zip(points, points[1:]):
        seg = math.dist(a, b)
        pos = 0.0
        while pos < seg:
            step = min((dash if on else gap) - carry, seg - pos)
            u0, u1 = pos / seg, (pos + step) / seg
            if on:
                out.append(((a[0] + (b[0] - a[0]) * u0, a[1] + (b[1] - a[1]) * u0),
                            (a[0] + (b[0] - a[0]) * u1, a[1] + (b[1] - a[1]) * u1)))
            pos += step
            carry += step
            if carry >= (dash if on else gap) - 1e-9:
                carry, on = 0.0, not on
    return out


# --- bases ----------------------------------------------------------------------

def fetch_tile(z: int, x: int, y: int, cache: Path) -> Image.Image:
    path = cache / str(z) / str(x) / f"{y}.png"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(TILE_URL.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
        data = urllib.request.urlopen(req, timeout=30).read()
        path.write_bytes(data)
        time.sleep(0.15)
    return Image.open(path).convert("RGB")


def dark_map(img: Image.Image) -> Image.Image:
    """An OpenStreetMap rendering made dark: land near black, water blue-black, streets and names faint."""
    r, g, b = img.split()
    water = ImageChops.subtract(b, r).point(lambda v: 255 if v > 18 else 0).filter(ImageFilter.MaxFilter(3))
    grey = ImageOps.invert(ImageOps.grayscale(img)).point(lambda v: int(14 + v * 0.55))
    out = Image.merge("RGB", (grey.point(lambda v: int(v * 0.93)), grey.point(lambda v: int(v * 0.96)), grey))
    out.paste(WATER, (0, 0), water.filter(ImageFilter.GaussianBlur(1)))
    return out


def light_map(img: Image.Image) -> Image.Image:
    """An OpenStreetMap rendering made light and quiet: warm paper land, grey-blue water, streets and names soft."""
    r, g, b = img.split()
    water = ImageChops.subtract(b, r).point(lambda v: 255 if v > 18 else 0).filter(ImageFilter.MaxFilter(3))
    grey = ImageOps.grayscale(img).point(lambda v: int(150 + v * 0.36))
    out = Image.merge("RGB", (grey.point(lambda v: min(255, int(v * 1.0))), grey.point(lambda v: int(v * 0.965)),
                              grey.point(lambda v: int(v * 0.90))))
    out.paste((168, 184, 196), (0, 0), water.filter(ImageFilter.GaussianBlur(1)))
    return out


def tinted_map(img: Image.Image) -> Image.Image:
    """An OpenStreetMap rendering tinted to the board: land the board colour, streets a shade lighter, names darker."""
    r, g, b = img.split()
    water = ImageChops.subtract(b, r).point(lambda v: 255 if v > 18 else 0).filter(ImageFilter.MaxFilter(3))
    grey = ImageOps.grayscale(img)
    out = Image.merge("RGB", [grey.point(lambda v, c=c: min(255, int(c * (0.55 + 0.45 * v / 235)))) for c in PAPER])
    out.paste(MAP_WATER, (0, 0), water.filter(ImageFilter.GaussianBlur(1)))
    return out


def map_base(base: dict, size: tuple[int, int], cache: Path):
    """The dark map filling `size`, and a function from (lat, lon) to board pixels."""
    zoom = int(base["zoom"])
    x0, y0, x1, y1 = map_frame(base["bbox"], zoom, size[0] / size[1])
    tx0, ty0, tx1, ty1 = int(x0 // 256), int(y0 // 256), int(x1 // 256), int(y1 // 256)
    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            mosaic.paste(fetch_tile(zoom, tx, ty, cache), ((tx - tx0) * 256, (ty - ty0) * 256))
    box = (x0 - tx0 * 256, y0 - ty0 * 256, x1 - tx0 * 256, y1 - ty0 * 256)
    restyle = {"light": light_map, "tint": tinted_map}.get(MAP_STYLE, dark_map)
    picture = restyle(mosaic.resize(size, Image.Resampling.LANCZOS, box=box))
    vignette = Image.new("L", size, 0)
    ImageDraw.Draw(vignette).ellipse((-size[0] * 0.15, -size[1] * 0.2, size[0] * 1.15, size[1] * 1.2), fill=255)
    vignette = vignette.filter(ImageFilter.GaussianBlur(size[0] * 0.08))
    edge = {"light": (188, 180, 164), "tint": tuple(round(c * 0.72) for c in PAPER)}.get(MAP_STYLE, (4, 5, 7))
    picture = Image.composite(picture, Image.new("RGB", size, edge), vignette)
    sx, sy = size[0] / (x1 - x0), size[1] / (y1 - y0)

    def place(lat, lon):
        px, py = lonlat_to_px(lat, lon, zoom)
        return (px - x0) * sx, (py - y0) * sy
    return picture, place


def plan_base(base: dict, size: tuple[int, int]):
    """A floor plan on dark drafting paper, and a function from plan units to board pixels."""
    w, h = size
    s = min(w / PLAN_W, h / PLAN_H)
    ox, oy = (w - PLAN_W * s) / 2, (h - PLAN_H * s) / 2

    def place(u, v):
        return ox + u * s, oy + v * s
    paper = Image.new("RGB", size, PAPER)
    grain = Image.effect_noise(size, 10).point(lambda v: 128 + (v - 128) * 0.25)
    paper = ImageChops.add(paper, Image.merge("RGB", [grain] * 3), 1.0, -128)
    d = ImageDraw.Draw(paper)
    step = s * 2
    grid = GRID
    x = ox % step
    while x < w:
        d.line([(x, 0), (x, h)], fill=grid, width=max(1, round(s * 0.04)))
        x += step
    y = oy % step
    while y < h:
        d.line([(0, y), (w, y)], fill=grid, width=max(1, round(s * 0.04)))
        y += step
    wall = max(2, round(s * 0.30))
    thin = max(1, round(s * 0.12))
    for room in base.get("rooms") or []:
        u0, v0, u1, v1 = room["box"]
        d.rectangle((*place(u0, v0), *place(u1, v1)), outline=INK, width=wall if room.get("outer") else thin)
        if room.get("name"):
            fnt = graphics.font(graphics.SANS, round(s * (1.25 if room.get("outer") else 0.95)))
            lx, ly = place(u0 + 0.8, v0 + 0.6)
            _tracked(d, (lx, ly), room["name"].upper(), fnt, NOTE if not room.get("outer") else INK, s * 0.12)
    for line in base.get("walls") or []:
        pts = [place(*p) for p in line["pts"]]
        d.line(pts, fill=INK, width=wall if line.get("outer") else thin)
    for note in base.get("notes") or []:
        fnt = graphics.font(graphics.SANS, round(s * note.get("size", 0.8)))
        _tracked(d, place(*note["pt"]), note["text"], fnt, NOTE, s * 0.08)
    return paper, place, s


def _tracked(d, xy, text, fnt, colour, tracking):
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=fnt, fill=colour)
        x += fnt.getlength(ch) + tracking


# --- icons ---------------------------------------------------------------------

def icon(name: str, size: int, colour=INK) -> Image.Image:
    """A line icon drawn at four times its size and reduced: crisp at any zoom the camera reaches."""
    k = 4
    S = size * k
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = colour + (255,)
    w = max(2, S // 22)
    if name == "bed":
        d.rectangle((S * .18, S * .08, S * .82, S * .92), outline=c, width=w)
        d.rectangle((S * .26, S * .14, S * .74, S * .30), outline=c, width=w)
        d.line([(S * .18, S * .42), (S * .82, S * .42)], fill=c, width=w)
    elif name == "alarm":
        d.ellipse((S * .22, S * .26, S * .78, S * .82), outline=c, width=w)
        d.ellipse((S * .14, S * .14, S * .34, S * .32), outline=c, width=w)
        d.ellipse((S * .66, S * .14, S * .86, S * .32), outline=c, width=w)
        d.line([(S * .5, S * .54), (S * .5, S * .38)], fill=c, width=w)
        d.line([(S * .5, S * .54), (S * .62, S * .60)], fill=c, width=w)
    elif name == "phone":
        d.rounded_rectangle((S * .16, S * .38, S * .84, S * .84), radius=S * .08, outline=c, width=w)
        d.arc((S * .12, S * .10, S * .88, S * .56), 190, 350, fill=c, width=w * 2)
        d.ellipse((S * .36, S * .48, S * .64, S * .76), outline=c, width=w)
    elif name == "bicycle":
        d.ellipse((S * .04, S * .48, S * .40, S * .84), outline=c, width=w)
        d.ellipse((S * .60, S * .48, S * .96, S * .84), outline=c, width=w)
        d.line([(S * .22, S * .66), (S * .42, S * .40), (S * .66, S * .40), (S * .78, S * .66),
                (S * .50, S * .66), (S * .42, S * .40)], fill=c, width=w)
        d.line([(S * .66, S * .40), (S * .62, S * .28), (S * .72, S * .28)], fill=c, width=w)
        d.line([(S * .40, S * .36), (S * .48, S * .36)], fill=c, width=w)
    elif name == "closet":
        d.rectangle((S * .10, S * .10, S * .90, S * .92), outline=c, width=w)
        d.line([(S * .14, S * .24), (S * .86, S * .24)], fill=c, width=w)
        for x0 in (.18, .42, .64):
            d.line([(S * (x0 + .09), S * .24), (S * (x0 + .09), S * .30)], fill=c, width=w)
            d.polygon([(S * (x0 + .02), S * .32), (S * (x0 + .16), S * .32), (S * (x0 + .19), S * .82),
                       (S * (x0 - .01), S * .82)], outline=c, width=w)
    elif name == "door":
        d.line([(S * .15, S * .85), (S * .85, S * .85)], fill=c, width=w * 2)
        d.line([(S * .15, S * .85), (S * .15, S * .15)], fill=c, width=w)
        d.arc((S * -.55, S * .15, S * .85, S * 1.55), 270, 360, fill=c, width=w)
    elif name == "stairs":
        for i in range(6):
            y = S * (.12 + i * .14)
            d.line([(S * .2, y), (S * .8, y)], fill=c, width=w)
        d.rectangle((S * .2, S * .12, S * .8, S * .82), outline=c, width=w)
        d.line([(S * .5, S * .80), (S * .5, S * .22)], fill=c, width=w)
        d.polygon([(S * .42, S * .30), (S * .58, S * .30), (S * .5, S * .18)], fill=c)
    elif name == "tv":
        d.rounded_rectangle((S * .10, S * .24, S * .90, S * .78), radius=S * .06, outline=c, width=w)
        d.line([(S * .38, S * .10), (S * .5, S * .24), (S * .62, S * .10)], fill=c, width=w)
    elif name == "bag":
        d.polygon([(S * .2, S * .32), (S * .8, S * .32), (S * .74, S * .9), (S * .26, S * .9)], outline=c, width=w)
        d.line([(S * .3, S * .32), (S * .38, S * .14), (S * .62, S * .14), (S * .7, S * .32)], fill=c, width=w)
    elif name == "key":
        d.ellipse((S * .08, S * .34, S * .40, S * .66), outline=c, width=w)
        d.line([(S * .40, S * .5), (S * .92, S * .5)], fill=c, width=w)
        d.line([(S * .78, S * .5), (S * .78, S * .66)], fill=c, width=w)
        d.line([(S * .66, S * .5), (S * .66, S * .62)], fill=c, width=w)
    elif name == "person":       # a plain pictogram; never a likeness
        d.ellipse((S * .38, S * .06, S * .62, S * .30), outline=c, width=w)
        d.line([(S * .5, S * .30), (S * .5, S * .64)], fill=c, width=w)
        d.line([(S * .5, S * .64), (S * .36, S * .94)], fill=c, width=w)
        d.line([(S * .5, S * .64), (S * .64, S * .94)], fill=c, width=w)
        d.line([(S * .5, S * .40), (S * .32, S * .58)], fill=c, width=w)
        d.line([(S * .5, S * .40), (S * .68, S * .58)], fill=c, width=w)
    elif name == "sling":        # the pictogram with one arm in a sling, carrying books
        d.ellipse((S * .38, S * .06, S * .62, S * .30), outline=c, width=w)
        d.line([(S * .5, S * .30), (S * .5, S * .64)], fill=c, width=w)
        d.line([(S * .5, S * .64), (S * .36, S * .94)], fill=c, width=w)
        d.line([(S * .5, S * .64), (S * .64, S * .94)], fill=c, width=w)
        d.line([(S * .5, S * .40), (S * .62, S * .50), (S * .44, S * .52)], fill=c, width=w)
        d.line([(S * .40, S * .32), (S * .62, S * .54)], fill=RED + (255,), width=w * 2)
        d.line([(S * .5, S * .40), (S * .30, S * .56)], fill=c, width=w)
        d.rectangle((S * .14, S * .54, S * .34, S * .70), outline=c, width=w)
        d.line([(S * .14, S * .62), (S * .34, S * .62)], fill=c, width=w)
    elif name == "car":
        d.rounded_rectangle((S * .06, S * .46, S * .94, S * .74), radius=S * .1, outline=c, width=w)
        d.arc((S * .22, S * .20, S * .78, S * .80), 180, 360, fill=c, width=w)
        d.ellipse((S * .18, S * .64, S * .34, S * .80), outline=c, width=w)
        d.ellipse((S * .66, S * .64, S * .82, S * .80), outline=c, width=w)
    return im.resize((size, size), Image.Resampling.LANCZOS)


# --- labels and prints -------------------------------------------------------------

def tag_image(text: str, sub: str | None, scale: float, accent: bool = False, chars: int | None = None) -> Image.Image:
    """A small typed index card, its first `chars` characters showing (text, then sub)."""
    fnt = graphics.font(graphics.TYPEWRITER, max(8, round(34 * scale)))
    sfnt = graphics.font(graphics.TYPEWRITER, max(7, round(25 * scale)))
    total = len(text) + len(sub or "")
    n = total if chars is None else max(0, min(total, chars))
    shown, sub_shown = text[:n], (sub or "")[:max(0, n - len(text))]
    pad = 16 * scale
    w = max(fnt.getlength(text), sfnt.getlength(sub) if sub else 0) + 2 * pad
    h = graphics.line_height(fnt, 1.15) + (graphics.line_height(sfnt, 1.3) if sub else 0) + 1.6 * pad
    card = Image.new("RGBA", (math.ceil(w), math.ceil(h)), CARD + (255,))
    d = ImageDraw.Draw(card)
    if CARD_EDGE:
        d.rectangle((0, 0, card.width - 1, card.height - 1), outline=CARD_EDGE + (255,), width=max(1, round(2 * scale)))
    if accent:
        d.rectangle((0, 0, max(2, round(5 * scale)), h), fill=RED + (255,))
    d.text((pad, pad * 0.8), shown, font=fnt, fill=(RED if accent else motion.PAPER_INK) + (255,))
    if sub:
        d.text((pad, pad * 0.8 + graphics.line_height(fnt, 1.15)), sub_shown, font=sfnt, fill=motion.PAPER_GREY + (255,))
    return card


def print_image(path: Path, height: int, crop=None) -> Image.Image:
    with Image.open(path) as src:
        pic = ImageOps.exif_transpose(src).convert("RGB")
    if crop:
        pic = pic.crop((round(pic.width * crop[0]), round(pic.height * crop[1]),
                        round(pic.width * crop[2]), round(pic.height * crop[3])))
    w = round(height * pic.width / pic.height)
    return motion._as_print(pic, w, height).convert("RGBA")


# --- the renderer -------------------------------------------------------------------

class Board:
    """A spec resolved onto a board: the static base, and every element in board pixels."""

    def __init__(self, spec: dict, video_dir: Path, width: int = 1920, height: int = 1080):
        use_theme(spec.get("theme"))
        self.spec, self.size = spec, (width * SS, height * SS)
        self.frame = (width, height)
        base = spec["base"]
        cache = Path(video_dir) / "assets" / "maps"
        if base["kind"] == "map":
            self.canvas, place_ll = map_base(base, self.size, cache)
            self.unit = self.size[1] / 56.0
            self.place = lambda e: self._xy(e, place_ll=place_ll)
        else:
            self.canvas, place_pt, self.unit = plan_base(base, self.size)
            self.place = lambda e: self._xy(e, place_pt=place_pt)
        self.video_dir = Path(video_dir)
        self.library = {i["id"]: i for i in json.loads((self.video_dir / "library.json").read_text())["items"]} \
            if (self.video_dir / "library.json").exists() else {}
        self.cache: dict = {}

    def _xy(self, e, place_ll=None, place_pt=None, key="xy"):
        if "ll" in e and place_ll:
            return place_ll(*e["ll"])
        if "pt" in e and place_pt:
            return place_pt(*e["pt"])
        x, y = e[key]
        return x * self.size[0], y * self.size[1]

    def point(self, p):
        """A route or box corner: a dict ({"ll": ...} or {"pt": ...}), plan units on a plan, else board fractions."""
        if isinstance(p, dict):
            return self.place(p)
        if self.spec["base"]["kind"] == "plan":
            return self.place({"pt": p})
        return p[0] * self.size[0], p[1] * self.size[1]

    def cached(self, key, make):
        if key not in self.cache:
            self.cache[key] = make()
        return self.cache[key]

    # each draw_* lays one element on `img`, a crop of the board whose top-left is `o`
    def draw(self, img: Image.Image, o, e: dict, t: float) -> None:
        kind = e["kind"]
        a = window(t, e.get("at", 0.0), e.get("until"), e.get("fade", 0.35))
        if a <= 0 and kind not in ("route",):
            return
        getattr(self, "draw_" + kind)(img, o, e, t, a)

    def _paste(self, img, o, patch, centre, alpha=1.0, scale=1.0):
        if scale != 1.0:
            patch = patch.resize((max(1, round(patch.width * scale)), max(1, round(patch.height * scale))),
                                 Image.Resampling.BICUBIC)
        if alpha < 1:
            patch = patch.copy()
            patch.putalpha(patch.getchannel("A").point(lambda v: round(v * alpha)))
        img.paste(patch, (round(centre[0] - patch.width / 2 - o[0]), round(centre[1] - patch.height / 2 - o[1])), patch)

    def draw_pin(self, img, o, e, t, a):
        x, y = self.place(e)
        r = self.unit * e.get("r", 0.55)
        pin = self.cached(("pin", round(r)), lambda: motion._pin_patch(r))
        pop = motion.ease_out_back(motion.clamp((t - e.get("at", 0)) / 0.3), 1.6)
        self._paste(img, o, pin, (x, y), a, max(0.05, pop))

    def draw_icon(self, img, o, e, t, a):
        x, y = self.place(e)
        size = round(self.unit * e.get("size", 3.0))
        colour = tuple(RED) if e.get("accent") else INK
        patch = self.cached(("icon", e["icon"], size, colour), lambda: icon(e["icon"], size, colour))
        self._paste(img, o, patch, (x, y), a)

    def draw_pulse(self, img, o, e, t, a):
        x, y = self.place(e)
        period = e.get("period", 0.9)
        rmax = self.unit * e.get("radius", 3.0)
        d = ImageDraw.Draw(img, "RGBA")
        colour = RED if e.get("accent", True) else INK
        for k in range(3):
            p = ((t - e.get("at", 0)) / period + k / 3) % 1.0
            rr = rmax * (0.25 + 0.75 * p)
            alpha = round(255 * a * (1 - p) * 0.9)
            wdt = max(2, round(self.unit * 0.18))
            d.ellipse((x - rr - o[0], y - rr - o[1], x + rr - o[0], y + rr - o[1]), outline=colour + (alpha,), width=wdt)

    def draw_mark(self, img, o, e, t, a):
        u0, v0 = self.point(e["box"][:2])
        u1, v1 = self.point(e["box"][2:])
        grow = motion.ease_out_cubic(motion.clamp((t - e.get("at", 0)) / 0.5))
        pts = polyline_part([(u0, v0), (u1, v0), (u1, v1), (u0, v1), (u0, v0)], grow)
        d = ImageDraw.Draw(img, "RGBA")
        d.line([(p[0] - o[0], p[1] - o[1]) for p in pts], fill=RED + (round(255 * a),),
               width=max(2, round(self.unit * 0.22)), joint="curve")

    def draw_route(self, img, o, e, t, a_in):
        start, end = e["from"], e["to"]
        if t < start:
            return
        a = window(t, start, e.get("until"), 0.3)
        pts = [self.point(p) for p in e["points"]]
        part = polyline_part(pts, ease((t - start) / max(1e-6, end - start)))
        if len(part) < 2:
            return
        d = ImageDraw.Draw(img, "RGBA")
        colour = (RED if e.get("accent", True) else INK) + (round(255 * a),)
        width = max(2, round(self.unit * e.get("width", 0.22)))
        shifted = [(p[0] - o[0], p[1] - o[1]) for p in part]
        if e.get("dash", True):
            for seg in dashed(shifted, self.unit * 0.9, self.unit * 0.6):
                d.line(seg, fill=colour, width=width)
        else:
            d.line(shifted, fill=colour, width=width, joint="curve")
        if e.get("head", True) and t < end + 0.2:
            hx, hy = shifted[-1]
            r = width * 1.6
            d.ellipse((hx - r, hy - r, hx + r, hy + r), fill=colour)

    def draw_tag(self, img, o, e, t, a):
        x, y = self.place(e)
        scale = e.get("scale", 1.0) * self.unit * 0.9 / 34
        text, sub = e["text"], e.get("sub")
        total = len(text) + len(sub or "")
        chars = min(total, int((t - e.get("at", 0)) / e.get("char_s", 0.035)) + 1)
        card = self.cached(("tag", text, sub, chars, e.get("accent", False), round(scale, 3)),
                           lambda: tag_image(text, sub, scale, e.get("accent", False), chars))
        full = self.cached(("tagsize", text, sub, round(scale, 3)), lambda: tag_image(text, sub, scale).size)
        dx, dy = e.get("offset", [3.0, -3.0])
        cx, cy = x + dx * self.unit, y + dy * self.unit
        if e.get("leader", True):
            d = ImageDraw.Draw(img, "RGBA")
            d.line([(x - o[0], y - o[1]), (cx - o[0], cy - o[1])], fill=INK + (round(200 * a),),
                   width=max(1, round(self.unit * 0.08)))
        if a < 1:
            card = card.copy()
            card.putalpha(card.getchannel("A").point(lambda v: round(v * a)))
        img.paste(card, (round(cx - full[0] / 2 - o[0]), round(cy - full[1] / 2 - o[1])), card)

    def draw_photo(self, img, o, e, t, a):
        x, y = self.place(e)
        h = round(self.unit * e.get("h", 12.0))
        path = self.video_dir / self.library[e["library_id"]]["file"] if "library_id" in e else Path(e["image"])
        pic = self.cached(("photo", str(path), h, tuple(e.get("crop") or ())),
                          lambda: print_image(path, h, e.get("crop")).rotate(e.get("angle", -2.0), expand=True,
                                                                               resample=Image.Resampling.BICUBIC))
        land = motion.clamp((t - e.get("at", 0)) / 0.5)
        scale = 1 + 0.15 * (1 - motion.ease_out_back(land, 0.9))
        shadow = self.cached(("shadow", id(pic)), lambda: pic.getchannel("A").filter(ImageFilter.GaussianBlur(self.unit * 0.5)))
        sh = shadow if scale == 1 else shadow.resize((round(shadow.width * scale), round(shadow.height * scale)))
        sh = sh.point(lambda v: round(v * 0.8 * a))
        img.paste((0, 0, 0), (round(x - sh.width / 2 - o[0] + self.unit * 0.3), round(y - sh.height / 2 - o[1] + self.unit * 0.6)), sh)
        self._paste(img, o, pic, (x, y), a, scale)
        if e.get("pin", True):
            r = self.unit * 0.5
            pin = self.cached(("pin", round(r)), lambda: motion._pin_patch(r))
            self._paste(img, o, pin, (x, y - pic.height * scale / 2 + r * 1.4), a)

    # a statement card: a pinned paper card whose title types itself and whose lines come in one by one
    def card_layout(self, e) -> dict:
        def make():
            u = self.unit
            w = round(e.get("w", 0.22) * self.size[0])
            tf = graphics.font(graphics.TYPEWRITER, round(u * 1.15))
            sf = graphics.font(graphics.SANS, round(u * 0.72))
            lf = graphics.font(graphics.TYPEWRITER, round(u * 0.86))
            pad = u * 0.9
            y = pad
            title_y = y
            y += graphics.line_height(tf, 1.15)
            sub_y = y
            if e.get("sub"):
                y += graphics.line_height(sf, 1.35) * len(graphics.wrap(e["sub"], sf, w - 2 * pad))
            y += u * 0.7
            rows = []
            for line in e["lines"]:
                wrapped = graphics.wrap(line["text"], lf, w - 2 * pad - u * 1.2)
                rows.append({"y": y, "lines": wrapped, "h": graphics.line_height(lf, 1.3) * len(wrapped)})
                y += rows[-1]["h"] + u * 0.45
            return {"w": w, "h": round(y + pad * 0.6), "tf": tf, "sf": sf, "lf": lf, "pad": pad,
                    "title_y": title_y, "sub_y": sub_y, "rows": rows}
        return self.cached(("cardlayout", e["id"]), make)

    def card_anchor(self, e, row: int, side: str):
        lay = self.card_layout(e)
        x, y = self.place(e)
        r = lay["rows"][row]
        yy = y + r["y"] + graphics.line_height(lay["lf"], 1.3) * 0.5
        return (x - self.unit * 0.3 if side == "left" else x + lay["w"] + self.unit * 0.3), yy

    def draw_card(self, img, o, e, t, a):
        lay = self.card_layout(e)
        x, y = self.place(e)
        u = self.unit
        shown = tuple(i for i, line in enumerate(e["lines"]) if t >= line["at"])
        chars = int((t - e.get("at", 0)) / 0.04) + 1
        sub_chars = int((t - e.get("sub_at", e.get("at", 0) + 0.6)) / 0.03) + 1 if t >= e.get("sub_at", 0) else 0
        key = ("card", e["id"], shown, min(chars, len(e["title"])), min(max(0, sub_chars), len(e.get("sub") or "")))

        def make():
            card = Image.new("RGBA", (lay["w"], lay["h"]), CARD + (255,))
            d = ImageDraw.Draw(card)
            if CARD_EDGE:
                d.rectangle((0, 0, lay["w"] - 1, lay["h"] - 1), outline=CARD_EDGE + (255,), width=max(1, round(u * 0.05)))
            d.text((lay["pad"], lay["title_y"]), e["title"][:key[3]], font=lay["tf"], fill=motion.PAPER_INK)
            if e.get("sub") and key[4]:
                yy = lay["sub_y"]
                left = key[4]
                for line in graphics.wrap(e["sub"], lay["sf"], lay["w"] - 2 * lay["pad"]):
                    d.text((lay["pad"], yy), line[:max(0, left)], font=lay["sf"], fill=motion.PAPER_GREY)
                    left -= len(line) + 1
                    yy += graphics.line_height(lay["sf"], 1.35)
            for i in shown:
                row, line = lay["rows"][i], e["lines"][i]
                colour = RED if line.get("accent") else motion.PAPER_INK
                d.ellipse((lay["pad"], row["y"] + u * 0.32, lay["pad"] + u * 0.36, row["y"] + u * 0.68), fill=colour)
                yy = row["y"]
                for text in row["lines"]:
                    d.text((lay["pad"] + u * 1.0, yy), text, font=lay["lf"], fill=colour)
                    yy += graphics.line_height(lay["lf"], 1.3)
            d.line([(lay["pad"], lay["sub_y"] - u * 0.15), (lay["w"] - lay["pad"], lay["sub_y"] - u * 0.15)],
                   fill=RED, width=max(2, round(u * 0.06)))
            return card.rotate(e.get("angle", 0.0), expand=True, resample=Image.Resampling.BICUBIC)
        card = self.cached(key, make)
        cx, cy = x + lay["w"] / 2, y + lay["h"] / 2
        shadow = self.cached(("cardshadow", e["id"], card.size),
                             lambda: card.getchannel("A").filter(ImageFilter.GaussianBlur(u * 0.5)))
        img.paste((0, 0, 0), (round(cx - shadow.width / 2 - o[0] + u * 0.3), round(cy - shadow.height / 2 - o[1] + u * 0.6)),
                  shadow.point(lambda v: round(v * 0.8 * a)))
        self._paste(img, o, card, (cx, cy), a)
        r = u * 0.55
        pin = self.cached(("pin", round(r)), lambda: motion._pin_patch(r))
        self._paste(img, o, pin, (cx, y + r * 0.6), a)

    def draw_link(self, img, o, e, t, a):
        cards = {c["id"]: c for c in self.spec["elements"] if c["kind"] == "card"}
        (ca, ra), (cb, rb) = e["a"], e["b"]
        pa = self.card_anchor(cards[ca], ra, "right")
        pb = self.card_anchor(cards[cb], rb, "left")
        part = polyline_part([pa, pb], ease((t - e["at"]) / 0.6))
        d = ImageDraw.Draw(img, "RGBA")
        d.line([(p[0] - o[0], p[1] - o[1]) for p in part], fill=RED + (round(255 * a),), width=max(3, round(self.unit * 0.16)))
        for p in (pa, pb) if t > e["at"] + 0.6 else (pa,):
            rr = self.unit * 0.25
            d.ellipse((p[0] - rr - o[0], p[1] - rr - o[1], p[0] + rr - o[0], p[1] + rr - o[1]), fill=RED + (round(255 * a),))

    def draw_panel(self, img, o, e, t, a):
        x0, y0 = self.point(e["box"][:2])
        x1, y1 = self.point(e["box"][2:])
        layer = Image.new("L", img.size, 0)
        ImageDraw.Draw(layer).rectangle((x0 - o[0], y0 - o[1], x1 - o[0], y1 - o[1]), fill=round(255 * e.get("amount", 0.8) * a))
        img.paste(PANEL, (0, 0), layer)

    def draw_dim(self, img, o, e, t, a):
        img.paste(Image.blend(img, Image.new("RGB", img.size, (0, 0, 0)), e.get("amount", 0.55) * a))

    def draw_text(self, img, o, e, t, a):
        x, y = self.place(e)
        fnt = graphics.font(graphics.SANS_BOLD if e.get("bold") else graphics.SANS, round(self.unit * e.get("size", 1.2)))
        d = ImageDraw.Draw(img, "RGBA")
        colour = (RED if e.get("accent") else INK) + (round(255 * a),)
        d.text((x - o[0], y - o[1]), e["text"], font=fnt, fill=colour, anchor=e.get("anchor", "mm"))


# --- the screen-space overlays ---------------------------------------------------------

def clock_face(hours: float, size: int, alpha: float) -> Image.Image:
    k = 4
    S = size * k
    im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((S * .04, S * .04, S * .96, S * .96), fill=(8, 9, 12, 190), outline=FOREGROUND + (230,), width=S // 28)
    for i in range(12):
        ang = math.radians(i * 30)
        r0, r1 = S * .38, S * .44
        d.line([(S / 2 + r0 * math.sin(ang), S / 2 - r0 * math.cos(ang)),
                (S / 2 + r1 * math.sin(ang), S / 2 - r1 * math.cos(ang))], fill=FOREGROUND + (200,), width=S // 40)
    ha = math.radians((hours % 12) * 30)
    ma = math.radians((hours % 1) * 360)
    d.line([(S / 2, S / 2), (S / 2 + S * .22 * math.sin(ha), S / 2 - S * .22 * math.cos(ha))], fill=FOREGROUND + (255,), width=S // 18)
    d.line([(S / 2, S / 2), (S / 2 + S * .34 * math.sin(ma), S / 2 - S * .34 * math.cos(ma))], fill=RED + (255,), width=S // 30)
    d.ellipse((S * .46, S * .46, S * .54, S * .54), fill=FOREGROUND + (255,))
    im = im.resize((size, size), Image.Resampling.LANCZOS)
    im.putalpha(im.getchannel("A").point(lambda v: round(v * alpha)))
    return im


def parse_hours(text: str) -> float:
    """'5:30' or '17:45' as hours."""
    h, m = text.split(":")
    return int(h) + int(m) / 60


def clock_label(hours: float) -> str:
    h = int(hours) % 24
    m = int(round((hours - int(hours)) * 60)) % 60
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def counter_layout(item: dict, h: int) -> tuple:
    """Where a counter's number and unit sit so that both are centred inside its ring: the ring keeps its size if
    they fit, grows by up to a third if not, and only then does the type shrink. A long unit takes two lines.
    Returns the radius, the two fonts, the number's baseline and the unit's (line, baseline) pairs, from the centre."""
    size = item.get("size", 0.16)
    start, end = int(item["start"]), int(item["end"])
    digits = [f"{item.get('prefix', '')}{v}" for v in (start, end, item.get("final", end))]
    unit = (item.get("unit") or "").split()
    ring = h * size * 1.05
    gap, pad = h * 0.012, h * 0.022
    for scale in (1, 0.94, 0.88, 0.82, 0.76, 0.7, 0.64, 0.58, 0.52, 0.46):
        fnt = graphics.font(graphics.NAME_BOLD, round(h * size * scale))
        ufnt = graphics.font(graphics.SANS, round(h * 0.03 * max(scale, 0.72)))
        number_w = max(fnt.getlength(t) for t in digits)
        number_h = -fnt.getbbox("0", anchor="ls")[1]
        cap = -ufnt.getbbox("H", anchor="ls")[1]
        rows = [" ".join(unit)] if unit else []
        if len(unit) > 1 and ufnt.getlength(rows[0]) > number_w * 1.15:
            cut = min(range(1, len(unit)), key=lambda i: abs(ufnt.getlength(" ".join(unit[:i]))
                                                             - ufnt.getlength(" ".join(unit[i:]))))
            rows = [" ".join(unit[:cut]), " ".join(unit[cut:])]
        step = cap * 1.55
        total = number_h + (gap + cap + step * (len(rows) - 1) if rows else 0)
        top = -total / 2
        baselines = [top + number_h + gap + cap + step * n for n in range(len(rows))]
        need = math.hypot(number_w / 2, max(abs(top), abs(top + number_h)))
        for row, y in zip(rows, baselines):
            need = max(need, math.hypot(ufnt.getlength(row) / 2, abs(y) + cap * 0.3))
        need += pad
        if need <= ring * 1.33:
            return max(ring, need), fnt, ufnt, top + number_h, list(zip(rows, baselines))
    return need, fnt, ufnt, top + number_h, list(zip(rows, baselines))


def draw_hud(frame: Image.Image, spec: dict, t: float) -> None:
    w, h = frame.size
    d = ImageDraw.Draw(frame, "RGBA")
    for item in spec.get("hud") or []:
        a = window(t, item.get("at", 0.0), item.get("until"), 0.4)
        if a <= 0:
            continue
        kind = item["kind"]
        if kind == "stamp":
            fnt = graphics.font(graphics.TYPEWRITER, round(h * 0.034))
            sfnt = graphics.font(graphics.SANS, round(h * 0.022))
            n = int((t - item.get("at", 0)) / 0.04) + 1
            text = item["text"][:n]
            x, y = w * 0.055, h * 0.075
            pw = max(fnt.getlength(item["text"]), sfnt.getlength(item.get("sub") or "")) + w * 0.03
            d.rectangle((x - w * 0.02, y - h * 0.018, x + pw, y + h * (0.095 if item.get("sub") else 0.06)),
                        fill=(6, 7, 9, round(200 * a)))
            d.rectangle((x - w * 0.012, y + h * 0.004, x - w * 0.009, y + h * 0.07 if item.get("sub") else y + h * 0.04),
                        fill=RED + (round(255 * a),))
            d.text((x, y), text, font=fnt, fill=FOREGROUND + (round(245 * a),))  # on its own dark chip
            if item.get("sub") and n > len(item["text"]):
                d.text((x, y + h * 0.046), item["sub"][:n - len(item["text"])], font=sfnt, fill=MUTED + (round(240 * a),))
        elif kind == "clock":
            p = ease((t - item["move_from"]) / max(1e-6, item["move_to"] - item["move_from"]))
            h0, h1 = parse_hours(item["start"]), parse_hours(item["end"])
            hours = h0 + (h1 - h0) * p
            size = round(h * 0.12)
            face = clock_face(hours, size, a)
            frame.paste(face, (round(w * 0.935 - size), round(h * 0.07)), face)
            if item.get("digits", True):
                fnt = graphics.font(graphics.TYPEWRITER, round(h * 0.03))
                label = item.get("label") or clock_label(hours)
                tw = fnt.getlength(label)
                d.text((w * 0.935 - size / 2 - tw / 2, h * 0.07 + size + h * 0.012), label, font=fnt,
                       fill=TEXT + (round(240 * a),))
        elif kind == "counter":
            p = ease((t - item["move_from"]) / max(1e-6, item["move_to"] - item["move_from"]))
            value = item["start"] + (item["end"] - item["start"]) * p
            text = f"{item.get('prefix', '')}{int(round(value)) if p < 1 else item.get('final', int(item['end']))}"
            cx, cy = w * item.get("x", 0.72), h * item.get("y", 0.46)
            r, fnt, ufnt, base, lines = counter_layout(item, h)
            d.text((cx, cy + base), text, font=fnt, fill=TEXT + (round(250 * a),), anchor="ms")
            for line, y in lines:
                d.text((cx, cy + y), line, font=ufnt, fill=NOTE + (round(240 * a),), anchor="ms")
            # a ring filling round the number
            k = 4
            ring = Image.new("RGBA", (round(r * 2 * k), round(r * 2 * k)), (0, 0, 0, 0))
            rd = ImageDraw.Draw(ring)
            rw = max(2, round(h * 0.006 * k))
            rd.ellipse((rw, rw, ring.width - rw, ring.height - rw), outline=(60, 60, 66, round(255 * a)), width=rw)
            rd.arc((rw, rw, ring.width - rw, ring.height - rw), -90, -90 + 360 * (value - item["start"]) /
                   max(1e-6, item.get("ring_of", item["end"]) - item["start"]), fill=RED + (round(255 * a),), width=rw)
            ring = ring.resize((round(r * 2), round(r * 2)), Image.Resampling.LANCZOS)
            frame.paste(ring, (round(cx - r), round(cy - r)), ring)
        elif kind == "note":
            fnt = graphics.font(graphics.SANS, round(h * 0.018))
            d.text((w * 0.97, h * 0.965), item["text"], font=fnt, fill=NOTE + (round(200 * a),), anchor="rs")


# --- the whole clip ------------------------------------------------------------------

def frame_at(board: Board, spec: dict, camera, t: float) -> Image.Image:
    """One frame: the board through the camera at t, its moving elements, the overlays and any fade."""
    width, height = board.frame
    bw, bh = board.size
    cx, cy, zoom = camera
    box = motion.view_box(cx, cy, zoom, board.size)
    left, top = max(0, math.floor(box[0]) - 2), max(0, math.floor(box[1]) - 2)
    right, bottom = min(bw, math.ceil(box[2]) + 2), min(bh, math.ceil(box[3]) + 2)
    img = board.canvas.crop((left, top, right, bottom))
    for e in spec.get("elements") or []:
        board.draw(img, (left, top), e, t)
    frame = img.resize((width, height), Image.Resampling.BICUBIC,
                       box=(box[0] - left, box[1] - top, box[2] - left, box[3] - top))
    draw_hud(frame, spec, t)
    for fade in spec.get("fades") or []:          # [from, to, from_alpha, to_alpha] to black
        if fade[0] <= t <= fade[1]:
            k = fade[2] + (fade[3] - fade[2]) * (t - fade[0]) / max(1e-6, fade[1] - fade[0])
            frame = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), k)
    return frame


def camera_frames(board: Board, spec: dict, total: int, fps: int = FPS):
    """The camera at every frame. Keys are [t, x, y, zoom] in board fractions, or [t, lat, lon, zoom] on a
    map whose spec says "camera_ll": true."""
    bw, bh = board.size
    if spec.get("camera_ll"):
        keys = [(k[0], *board.place({"ll": [k[1], k[2]]}), k[3]) for k in spec["camera"]]
    else:
        keys = [(k[0], k[1] * bw, k[2] * bh, k[3]) for k in spec["camera"]]
    return motion.camera_path(keys, total, fps, spec.get("smooth", 0.45))


def render(spec: dict, video_dir, out_path, seconds: float | None = None, *, width: int = 1920, height: int = 1080,
           fps: int = FPS) -> Path:
    seconds = seconds or spec["end"] - spec["start"]
    board = Board(spec, Path(video_dir), width, height)
    total = motion.frame_count(seconds, fps)
    path = camera_frames(board, spec, total, fps)
    with motion.FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            out.write(frame_at(board, spec, path[n], n / fps))
    return Path(out_path)


def stills(spec: dict, video_dir, times: list[float], out_dir, *, fps: int = FPS) -> list[Path]:
    """Frames at chosen moments, for checking a spec before rendering it."""
    board = Board(spec, Path(video_dir))
    total = motion.frame_count(spec["end"] - spec["start"], fps)
    path = camera_frames(board, spec, total, fps)
    out = []
    for t in times:
        target = Path(out_dir) / f"still_{t:07.2f}.jpg"
        frame_at(board, spec, path[min(total - 1, round(t * fps))], t).save(target, quality=88)
        out.append(target)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    spec = json.loads(Path(sys.argv[1]).read_text())
    began = time.monotonic()
    print(render(spec, sys.argv[3], sys.argv[2]), f"{time.monotonic() - began:.1f}s")

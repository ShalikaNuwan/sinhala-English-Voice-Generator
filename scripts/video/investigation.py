"""Investigation devices: a case-file profile, a tagged exhibit, a highlight drawn onto a picture.

    .venv/bin/python scripts/video/investigation.py demo [out_dir]

Each device renders a full-frame 1920x1080, 30 fps H.264 mp4 of exactly round(seconds * 30)
frames, tagged BT.709 limited range like assembly's intermediates, so a shot list runs it as
`media: "motion"`. Assembly gives motion shots no grade and no vignette, so the devices grade
their own photographs (black-and-white stays black-and-white with a faint cool tone, colour is
muted and cooled, blacks are crushed) and darken their own edges. Grain is added later, over
everything, when assembly joins the shots.

- `profile_card`: the photograph as a pinned print on the left, a dark case-file panel on the
  right whose values type themselves on, line after line, over the first half of the clip.
- `exhibit`: an evidence object on a dark desk with an off-white exhibit tag tied to it, the tag
  text typing on, the camera pushing slowly in.
- `highlight`: the photograph as a print (the same treatment as assembly's `print`), the camera
  pushing in until a detail fills about 45% of the frame height, a crimson pen stroke drawing
  itself around it, an optional label typing beside it.
- `locate`: asks a vision model where a described detail is, then checks the crop with a second
  call. A box comes back only when the second call agrees; results are cached by file and text.

Frames are drawn with Pillow and piped raw into the isolated ffmpeg 9 build (`VIDEO_FFMPEG`,
default `/opt/miniconda3/envs/video/bin/ffmpeg`); nothing is written to disk but the mp4. Layout,
timing, the camera paths, the stroke and the cache are pure functions, tested without rendering.

The palette is the film's: ground #0A0A0A, text #F2F2F2, crimson #9B111E, grey #8C8C8C. Apart
from structural labels ("CASE FILE", "EXHIBIT", field names) every word on screen is passed in by
the caller, who takes it from the narration.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graphics import (  # noqa: E402
    ACCENT, BACKGROUND, FOREGROUND, LABEL, MUTED, NAME_BOLD, TYPEWRITER, fit_lines, font,
    line_height, tracked_width, wrap,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
WIDTH, HEIGHT, FPS = 1920, 1080, 30
SUPERSAMPLE = 2  # the profile and exhibit canvases are drawn at twice the frame size

PRINT_BORDER = (0xE9, 0xE6, 0xDF)  # the same off-white as assembly's prints
PAPER = (0xE4, 0xE0, 0xD5)         # the exhibit tag
PAPER_INK = (0x1A, 0x1A, 0x1A)
PAPER_GREY = (0x6A, 0x67, 0x62)
CARD = (0x12, 0x12, 0x12)          # the case-file panel
CARD_EDGE = (0x2C, 0x2C, 0x2C)
HAIRLINE = (0x26, 0x26, 0x26)

MAX_UPSCALE = 2.2        # a photograph is never shown at more than this many times its own pixels
PRINT_FIT = 0.84         # highlight: the print fits this much of the frame, like assembly's print
PRINT_DARKEN = 0.30      # the blurred ground behind a print, as a fraction of its own brightness
CARET_BLINK_S = 0.5


# --- small helpers -----------------------------------------------------------------

def ffmpeg_path(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("VIDEO_FFMPEG") or DEFAULT_FFMPEG


def frame_count(seconds: float, fps: int = FPS) -> int:
    return max(1, round(seconds * fps))


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def ease_in_out_sine(p: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * clamp(p))


def ease_in_out_cubic(p: float) -> float:
    p = clamp(p)
    return 4 * p ** 3 if p < 0.5 else 1 - (-2 * p + 2) ** 3 / 2


def ease_out_quad(p: float) -> float:
    p = clamp(p)
    return 1 - (1 - p) ** 2


def progress(t: float, start: float, end: float) -> float:
    return 1.0 if end <= start else clamp((t - start) / (end - start))


def rotate_point(x: float, y: float, degrees: float) -> tuple[float, float]:
    """A point turned about the origin the way Image.rotate turns a picture (counter-clockwise)."""
    a = math.radians(degrees)
    return x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)


def rotated_extent(w: float, h: float, degrees: float) -> tuple[float, float]:
    a = math.radians(abs(degrees))
    return w * math.cos(a) + h * math.sin(a), w * math.sin(a) + h * math.cos(a)


def overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def inside(rect, width: float, height: float) -> bool:
    return rect[0] >= 0 and rect[1] >= 0 and rect[2] <= width and rect[3] <= height


class FrameWriter:
    """Raw RGB frames on ffmpeg's stdin, out as an H.264 mp4 tagged BT.709 limited range."""

    def __init__(self, out_path, width: int, height: int, fps: int = FPS, *, frames: int,
                 crf: int = 15, preset: str = "medium"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.size, self.count, self.frames = (width, height), 0, frames
        self.partial = self.out_path.with_name(self.out_path.stem + ".part" + self.out_path.suffix)
        argv = [ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-framerate", str(fps), "-i", "-",
                "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "film",
                "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart",
                "-frames:v", str(frames), "-r", str(fps), str(self.partial)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, data) -> None:
        if isinstance(data, Image.Image):
            if data.size != self.size:
                raise ValueError(f"frame is {data.size}, expected {self.size}")
            data = (data if data.mode == "RGB" else data.convert("RGB")).tobytes()
        self.proc.stdin.write(data)
        self.count += 1

    def close(self) -> Path:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed writing {self.out_path.name}: {err.strip()[-1500:]}")
        if self.count != self.frames:
            self.partial.unlink(missing_ok=True)
            raise RuntimeError(f"wrote {self.count} frames, expected {self.frames}")
        self.partial.replace(self.out_path)
        return self.out_path

    def __enter__(self):
        return self

    def __exit__(self, kind, value, trace):
        if kind is None:
            self.close()
        else:
            self.proc.kill()
            self.proc.wait()
            self.partial.unlink(missing_ok=True)


# --- the look: grading, prints, grounds ----------------------------------------------

def _curve(points) -> list[int]:
    """A 256-entry lookup table through (in, out) pairs given as fractions, linear between."""
    lut = []
    for i in range(256):
        x = i / 255
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x <= x1:
                y = y0 + (y1 - y0) * (x - x0) / (x1 - x0 or 1)
                break
        else:
            y = points[-1][1]
        lut.append(round(clamp(y) * 255))
    return lut


COLOUR_CURVE = _curve([(0, 0), (0.06, 0.015), (0.25, 0.22), (0.5, 0.50), (0.8, 0.81), (1, 0.96)])
BW_CURVE = _curve([(0, 0), (0.06, 0.015), (0.25, 0.22), (0.5, 0.50), (1, 0.97)])


def is_monochrome(image: Image.Image) -> bool:
    """True when a picture carries no real colour: mean channel spread under 6 levels (as assembly)."""
    thumb = image.convert("RGB").resize((48, 48))
    data = thumb.tobytes()
    spread = sum(max(data[i:i + 3]) - min(data[i:i + 3]) for i in range(0, len(data), 3))
    return spread / (48 * 48) < 6


def load_picture(path) -> Image.Image:
    """The file as RGB, any transparency flattened onto light grey."""
    with Image.open(path) as source:
        source.load()
        if source.mode in ("RGBA", "LA", "P", "PA"):
            source = source.convert("RGBA")
            flat = Image.new("RGB", source.size, (235, 235, 235))
            flat.paste(source, mask=source.getchannel("A"))
            return flat
        return source.convert("RGB")


def grade(image: Image.Image) -> Image.Image:
    """The film's grade in Pillow, since assembly leaves motion shots ungraded.

    Black-and-white pictures (sepia scans included) become true greyscale with a faint cool tone;
    colour is muted to 85% saturation and cooled. Both get the shadow-crushing curve.
    """
    if is_monochrome(image) or _is_toned(image):
        grey = image.convert("L").point(BW_CURVE)
        return Image.merge("RGB", (grey.point(lambda v: v * 0.985), grey.point(lambda v: v * 0.995),
                                   grey.point(lambda v: min(255, v * 1.015 + 1))))
    image = ImageEnhance.Color(image).enhance(0.85)
    image = ImageEnhance.Contrast(image).enhance(1.06)
    r, g, b = image.split()
    image = Image.merge("RGB", (r.point(lambda v: v * 0.97), g.point(lambda v: v * 0.99),
                                b.point(lambda v: min(255, v * 1.03 + 2))))
    return image.point(COLOUR_CURVE * 3)


def _is_toned(image: Image.Image) -> bool:
    """A sepia or cyanotype scan: some colour, but one hue throughout and only weakly saturated."""
    h, sat, _ = image.convert("RGB").resize((48, 48)).convert("HSV").split()
    hues = [hv for hv, sv in zip(h.tobytes(), sat.tobytes()) if sv > 12]
    if len(hues) < 50:  # hardly a coloured pixel: black and white
        return True
    strong = sum(1 for sv in sat.tobytes() if sv > 100) / (48 * 48)
    if strong > 0.005 or sum(sat.tobytes()) / (48 * 48) > 80:  # real colour somewhere (red ink, say)
        return False
    mean = sum(hues) / len(hues)
    return (sum((v - mean) ** 2 for v in hues) / len(hues)) ** 0.5 < 10


def vignette_mask(size, strength: float = 0.42, seed_shape: float = 0.80) -> Image.Image:
    """An L mask, 255 in the middle, falling to (1 - strength) at the corners, smooth."""
    sw, sh = 64, 36
    small = Image.new("L", (sw, sh))
    for y in range(sh):
        for x in range(sw):
            d = math.hypot((x / (sw - 1) - 0.5) * 2, (y / (sh - 1) - 0.5) * 2) / math.sqrt(2)
            fall = clamp((d - 0.35) / (seed_shape + 0.2 - 0.35))
            small.putpixel((x, y), round(255 * (1 - strength * fall ** 1.6)))
    return small.resize(size, Image.Resampling.BICUBIC)


def apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    return ImageChops.multiply(image, Image.merge("RGB", (mask, mask, mask)))


def blurred_ground(picture: Image.Image, size, darken: float = PRINT_DARKEN, blur: float | None = None) -> Image.Image:
    """The picture covering `size`, heavily blurred and darkened: assembly's print ground."""
    w, h = size
    small = ImageOps.fit(picture, (max(2, w // 4), max(2, h // 4)), Image.Resampling.BILINEAR)
    small = small.filter(ImageFilter.GaussianBlur((blur or h * 0.036) / 4))
    small = small.point(lambda v: v * darken)
    return small.resize(size, Image.Resampling.BICUBIC)


def desk_ground(size, seed: int = 8213) -> Image.Image:
    """A dark desk: a soft pool of light from upper left, a faint long grain, a little mottle."""
    w, h = size
    sw, sh = 96, 54
    small = Image.new("L", (sw, sh))
    for y in range(sh):
        for x in range(sw):
            d = math.hypot((x / (sw - 1) - 0.42) * 1.1, (y / (sh - 1) - 0.38))
            small.putpixel((x, y), round(31 - 21 * clamp(d / 0.80) ** 1.3))
    light = small.resize(size, Image.Resampling.BICUBIC)
    random.seed(seed)
    grain = Image.effect_noise((max(1, w // 40), h), 40).resize(size, Image.Resampling.BICUBIC)
    grain = grain.point(lambda v: round(128 + (v - 128) * 0.05))
    fibre = Image.effect_noise(size, 16).point(lambda v: round(128 + (v - 128) * 0.04))
    mottle = Image.effect_noise((max(1, w // 30), max(1, h // 30)), 40).resize(size, Image.Resampling.BICUBIC)
    mottle = mottle.point(lambda v: round(128 + (v - 128) * 0.04))
    value = ImageChops.add(ImageChops.add(ImageChops.add(light, grain, 1.0, -128), fibre, 1.0, -128),
                           mottle, 1.0, -128)
    return Image.merge("RGB", (value, value.point(lambda v: v * 0.985), value.point(lambda v: v * 0.97)))


def bordered(picture: Image.Image, w: int, h: int, border: int) -> Image.Image:
    """The picture fitted to (w, h) inside an off-white print border."""
    inner = picture.resize((max(1, w - 2 * border), max(1, h - 2 * border)), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (w, h), PRINT_BORDER)
    out.paste(inner, (border, border))
    return out


def paste_with_shadow(canvas: Image.Image, image: Image.Image, centre, angle: float, *,
                      blur: float, offset: tuple[float, float], opacity: float) -> tuple[int, int, Image.Image]:
    """Turn an object, then lay it on the canvas over a soft shadow falling down and right."""
    turned = image.convert("RGBA").rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    alpha = turned.getchannel("A")
    pad = round(blur * 3)
    shadow = Image.new("L", (turned.width + 2 * pad, turned.height + 2 * pad), 0)
    shadow.paste(alpha.point(lambda a: a * opacity), (pad, pad))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))
    x, y = round(centre[0] - turned.width / 2), round(centre[1] - turned.height / 2)
    canvas.paste((0, 0, 0), (x - pad + round(offset[0]), y - pad + round(offset[1])), shadow)
    canvas.paste(turned, (x, y), turned)
    return x, y, turned


def pin_patch(radius: float) -> Image.Image:
    """A crimson push pin from the front: a domed head, a darker rim, a small highlight, a shadow."""
    scale = 4
    side = math.ceil(radius * 2.8)
    patch = Image.new("RGBA", (side * scale, side * scale), (0, 0, 0, 0))
    r, c = radius * scale, side * scale / 2
    shade = Image.new("L", patch.size, 0)
    ImageDraw.Draw(shade).ellipse((c - r * 0.9 + r * 0.35, c - r * 0.9 + r * 0.5,
                                   c + r * 0.9 + r * 0.35, c + r * 0.9 + r * 0.5), fill=170)
    shade = shade.filter(ImageFilter.GaussianBlur(r * 0.28))
    patch.paste((0, 0, 0, 255), (0, 0), shade)
    draw = ImageDraw.Draw(patch)
    rim = tuple(round(v * 0.5) for v in ACCENT)
    draw.ellipse((c - r, c - r, c + r, c + r), fill=rim + (255,))
    draw.ellipse((c - r * 0.88, c - r * 0.92, c + r * 0.80, c + r * 0.74), fill=ACCENT + (255,))
    lit = Image.new("L", patch.size, 0)
    ImageDraw.Draw(lit).ellipse((c - r * 0.70, c - r * 0.75, c + r * 0.10, c + r * 0.02), fill=90)
    lit = lit.filter(ImageFilter.GaussianBlur(r * 0.22))
    patch.paste(tuple(min(255, round(v * 1.5 + 25)) for v in ACCENT) + (255,), (0, 0), lit)
    glint = Image.new("L", patch.size, 0)
    ImageDraw.Draw(glint).ellipse((c - r * 0.46, c - r * 0.52, c - r * 0.24, c - r * 0.32), fill=110)
    glint = glint.filter(ImageFilter.GaussianBlur(r * 0.08))
    patch.paste((235, 215, 215, 255), (0, 0), glint)
    patch = patch.resize((side, side), Image.Resampling.LANCZOS)
    # Pasting this RGBA patch with itself as mask premultiplies; the shadow stays soft and dark.
    return patch


def draw_caret(draw: ImageDraw.ImageDraw, x: float, top: float, fnt, colour) -> None:
    _, cap_top, _, base = fnt.getbbox("H")
    w = max(2, round(fnt.getlength("n") * 0.42))
    draw.rectangle((x, top + cap_top, x + w - 1, top + base), fill=colour)


def caret_visible(t: float, typing: bool) -> bool:
    """The caret is solid while keys are struck and blinks while the typist waits."""
    return typing or (t % (2 * CARET_BLINK_S)) < CARET_BLINK_S


# --- typing ----------------------------------------------------------------------

def typing_plan(lengths: list[int], start: float, end: float, pauses: list[float] | None = None,
                min_cps: float = 13.0) -> list[tuple[float, float]]:
    """When each line types: [(t0, t1)] per line, in seconds, sequential, finishing by `end`.

    `pauses[i]` is the beat before line i (a carriage return, or a new field). The pace is even
    across all characters: as slow as the window allows, but never under `min_cps`, so a short
    value in a long clip finishes early instead of crawling. If the window is too short for the
    pauses, they shrink in proportion.
    """
    pauses = list(pauses) if pauses is not None else [0.0] + [0.22] * (len(lengths) - 1)
    window = max(1e-3, end - start)
    total_pause = sum(pauses)
    if total_pause > window * 0.5:
        pauses = [p * window * 0.5 / total_pause for p in pauses]
        total_pause = sum(pauses)
    chars = sum(lengths)
    cps = max(min_cps, chars / max(1e-3, window - total_pause)) if chars else min_cps
    plan, t = [], start
    for length, pause in zip(lengths, pauses):
        t += pause
        plan.append((t, t + length / cps))
        t += length / cps
    return plan


def typed_counts(t: float, plan, lengths) -> list[int]:
    """Characters showing on each line at time t."""
    out = []
    for (t0, t1), n in zip(plan, lengths):
        out.append(n if t >= t1 else 0 if t < t0 else min(n, int((t - t0) / (t1 - t0) * n) + 1))
    return out


def typing_line(t: float, plan, lengths) -> int | None:
    """The line the caret sits on at time t: the one being typed, or the next to come; None when done."""
    for i, (t0, t1) in enumerate(plan):
        if t < t1:
            return i
    return None


# --- profile card --------------------------------------------------------------------

PROFILE_DRIFT = 0.035      # the camera's slow push over the whole profile clip
PROFILE_PHOTO_BOX = (0.07, 0.12, 0.40, 0.88)  # the left third the print stays inside
PROFILE_PHOTO_MAX_H = 0.66
PROFILE_PHOTO_MAX_W = 0.28
PROFILE_TILT = -2.2
PROFILE_PANEL_X = (0.44, 0.895)
PROFILE_TYPE_FROM = 0.45   # seconds before the first key
PROFILE_TYPE_BY = 0.50     # of the clip: everything typed by half way


def photo_scale(src_size, max_w: float, max_h: float, max_upscale: float = MAX_UPSCALE,
                zoom: float = 1.0) -> float:
    """Frame pixels per source pixel: fits the box, never past `max_upscale` on screen at `zoom`."""
    sw, sh = src_size
    return min(max_w / sw, max_h / sh, max_upscale / max(1.0, zoom))


def _profile_fonts(height: int, value_size: float):
    return {
        "header": font(LABEL, max(6, round(height * 0.0185))),
        "label": font(LABEL, max(6, round(height * 0.0170))),
        "value_size": value_size,
    }


def profile_layout(photo_size, fields, width: int = WIDTH, height: int = HEIGHT,
                   header: str = "CASE FILE") -> dict:
    """Where everything on a profile card goes, in frame pixels.

    Returns {"photo": {cx, cy, w, h, angle, border, footprint, pin}, "panel": (x0, y0, x1, y1),
    "header": (x, y, text, tracking), "rule": (x0, y, x1, thickness), "fields": [{label, label_xy,
    lines, value_xy, step}], "fonts": {...}}. The print stays inside the left third and never
    touches the panel; a small photograph is held to MAX_UPSCALE on screen, drift included.
    """
    fields = [(str(k).upper(), str(v)) for k, v in fields]
    if not fields:
        raise ValueError("a profile needs at least one field")
    # Photograph.
    box = PROFILE_PHOTO_BOX
    scale = photo_scale(photo_size, width * PROFILE_PHOTO_MAX_W, height * PROFILE_PHOTO_MAX_H,
                        zoom=1 + PROFILE_DRIFT)
    border = max(3, round(height * 0.0055))
    pw = round(photo_size[0] * scale) + 2 * border
    ph = round(photo_size[1] * scale) + 2 * border
    cx = width * (box[0] + box[2]) / 2
    cy = height * 0.49
    ew, eh = rotated_extent(pw, ph, PROFILE_TILT)
    footprint = (cx - ew / 2, cy - eh / 2, cx + ew / 2, cy + eh / 2)
    # The pin goes through the top border, a little left of centre, never into the subject's face.
    px, py = rotate_point(-pw * 0.18, -ph / 2 + border * 0.6, PROFILE_TILT)
    pin = (cx + px, cy + py)
    photo = {"cx": cx, "cy": cy, "w": pw, "h": ph, "angle": PROFILE_TILT, "border": border,
             "scale": scale, "footprint": footprint, "pin": pin, "pin_r": height * 0.0125}

    # Panel and its text, shrunk until it fits the frame height.
    x0, x1 = width * PROFILE_PANEL_X[0], width * PROFILE_PANEL_X[1]
    pad_x, pad_y = height * 0.052, height * 0.050
    inner = x1 - x0 - 2 * pad_x
    value_size = height * 0.041
    for _ in range(12):
        fonts = _profile_fonts(height, value_size)
        value_font, laid = None, []
        for label, value in fields:
            value_font, lines = fit_lines(value, TYPEWRITER, value_size, inner, max_lines=2,
                                          min_size=max(8, round(value_size * 0.80)))
            laid.append((label, lines, value_font))
        header_h = line_height(fonts["header"], 1.0)
        label_h = line_height(fonts["label"], 1.0)
        gap_rule, gap_label, gap_field = height * 0.020, height * 0.010, height * 0.034
        total = header_h + gap_rule + 2 + gap_field * 0.9
        for i, (_, lines, vf) in enumerate(laid):
            total += label_h + gap_label + len(lines) * line_height(vf, 1.18)
            total += gap_field if i < len(laid) - 1 else 0
        if total + 2 * pad_y <= height * 0.80:
            break
        value_size *= 0.92
    panel_h = total + 2 * pad_y
    y0 = height * 0.49 - panel_h / 2
    panel = (x0, y0, x1, y0 + panel_h)

    tracking = height * 0.0062
    y = y0 + pad_y
    head = (x0 + pad_x, y, header.upper(), tracking)
    y += header_h + gap_rule
    rule_line = (x0 + pad_x, y, x1 - pad_x, max(2, round(height * 0.0026)))
    y += 2 + gap_field * 0.9
    placed = []
    for i, (label, lines, vf) in enumerate(laid):
        label_xy = (x0 + pad_x, y)
        y += label_h + gap_label
        step = line_height(vf, 1.18)
        placed.append({"label": label, "label_xy": label_xy, "lines": lines, "font": vf,
                       "value_xy": (x0 + pad_x, y), "step": step,
                       "separator": None})
        y += len(lines) * step
        if i < len(laid) - 1:
            placed[-1]["separator"] = (x0 + pad_x, y + gap_field / 2 - height * 0.004, x1 - pad_x)
            y += gap_field
    return {"photo": photo, "panel": panel, "header": head, "rule": rule_line, "fields": placed,
            "fonts": fonts, "label_tracking": height * 0.0042}


def profile_lines(layout: dict) -> tuple[list[int], list[float]]:
    """Line lengths in typing order, and the pause before each (longer at a new field)."""
    lengths, pauses = [], []
    for i, field in enumerate(layout["fields"]):
        for j, line in enumerate(field["lines"]):
            lengths.append(len(line))
            pauses.append(0.0 if (i, j) == (0, 0) else 0.42 if j == 0 else 0.18)
    return lengths, pauses


def drift_camera(t: float, seconds: float, width: int, height: int, push: float,
                 start=(0.5, 0.5), end=(0.5, 0.5)) -> tuple[float, float, float]:
    """A slow eased push from `start` to `end` (frame fractions): (cx, cy, zoom) in frame pixels."""
    e = ease_in_out_sine(t / seconds if seconds > 0 else 1)
    cx = width * (start[0] + (end[0] - start[0]) * e)
    cy = height * (start[1] + (end[1] - start[1]) * e)
    return cx, cy, 1.0 + push * e


def view_box(cx: float, cy: float, zoom: float, size) -> tuple[float, float, float, float]:
    """The canvas rectangle a camera at (cx, cy) and `zoom` sees, kept inside the canvas."""
    cw, ch = size
    zoom = max(1.0, zoom)
    w, h = cw / zoom, ch / zoom
    x0 = clamp(cx - w / 2, 0, cw - w)
    y0 = clamp(cy - h / 2, 0, ch - h)
    return (x0, y0, x0 + w, y0 + h)


def _shoot(canvas: Image.Image, box, size) -> Image.Image:
    return canvas.resize(size, Image.Resampling.BICUBIC, box=box)


def profile_card(photo, fields, out_path, seconds: float, header: str = "CASE FILE", *,
                 width: int = WIDTH, height: int = HEIGHT, fps: int = FPS) -> Path:
    """A case-file record: the photograph pinned on the left, the fields typing on at the right.

    `fields`: ordered [(label, value)], e.g. [("NAME", "Robert Piest"), ("AGE", "15")]. Labels are
    shown as small grey caps; values type on, line by line, across the first half of the clip,
    then the card holds while the camera drifts slowly in. Exactly round(seconds * fps) frames.
    """
    if seconds <= 0:
        raise ValueError("a profile needs a positive length")
    picture = grade(load_picture(photo))
    layout = profile_layout(picture.size, fields, width, height, header)
    S = SUPERSAMPLE
    size = (width * S, height * S)

    canvas = blurred_ground(picture, size, darken=0.16)
    canvas = apply_mask(canvas, vignette_mask(size, 0.55))
    # The panel: a dark card with a fine edge and a soft shadow.
    px0, py0, px1, py1 = (v * S for v in layout["panel"])
    shadow = Image.new("L", size, 0)
    ImageDraw.Draw(shadow).rectangle((px0 + 6 * S, py0 + 10 * S, px1 + 6 * S, py1 + 10 * S), fill=170)
    shadow = shadow.filter(ImageFilter.GaussianBlur(14 * S))
    canvas.paste((0, 0, 0), (0, 0), shadow)
    card = Image.new("L", size, 0)
    ImageDraw.Draw(card).rectangle((px0, py0, px1, py1), fill=round(255 * 0.93))
    canvas.paste(CARD, (0, 0), card)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((px0, py0, px1, py1), outline=CARD_EDGE, width=max(1, S))

    # The print, pinned.
    p = layout["photo"]
    print_image = bordered(picture, round(p["w"] * S), round(p["h"] * S), round(p["border"] * S))
    paste_with_shadow(canvas, print_image, (p["cx"] * S, p["cy"] * S), p["angle"],
                      blur=height * S * 0.012, offset=(height * S * 0.005, height * S * 0.011), opacity=0.85)
    pin = pin_patch(p["pin_r"] * S)
    canvas.paste(pin, (round(p["pin"][0] * S - pin.width / 2), round(p["pin"][1] * S - pin.height / 2)), pin)

    # Static text: the header and its rule.
    hx, hy, htext, tracking = layout["header"]
    hfont = font(LABEL, round(layout["fonts"]["header"].size * S))
    x = hx * S
    for ch in htext:
        draw.text((x, hy * S), ch, font=hfont, fill=MUTED)
        x += hfont.getlength(ch) + tracking * S
    rx0, ry, rx1, thick = layout["rule"]
    draw.rectangle((rx0 * S, ry * S, rx1 * S, ry * S + thick * S - 1), fill=ACCENT)

    label_font = font(LABEL, round(layout["fonts"]["label"].size * S))
    fields_px = []
    for field in layout["fields"]:
        fields_px.append(dict(field, big=_font_at(field["font"], S)))
    lengths, pauses = profile_lines(layout)
    plan = typing_plan(lengths, PROFILE_TYPE_FROM, max(PROFILE_TYPE_FROM + 0.5, seconds * PROFILE_TYPE_BY), pauses)
    line_index = [(fi, li) for fi, f in enumerate(fields_px) for li in range(len(f["lines"]))]
    caret_end = plan[-1][1] + 0.9

    def state_at(t):
        counts = typed_counts(t, plan, lengths)
        current = typing_line(t, plan, lengths)
        typing = current is not None and t >= plan[current][0]
        caret = None
        if t < caret_end and caret_visible(t, typing):
            caret = current if current is not None else len(lengths) - 1
            if current is not None and not typing and counts[current] == 0:
                caret = max(0, current - 1) if current and t < plan[current][0] else current
        return tuple(counts), caret

    def compose(state):
        counts, caret = state
        image = canvas.copy()
        d = ImageDraw.Draw(image)
        for n, (fi, li) in enumerate(line_index):
            field = fields_px[fi]
            if li == 0 and (counts[n] > 0 or caret == n):
                _draw_tracked(d, (field["label_xy"][0] * S, field["label_xy"][1] * S), field["label"],
                              label_font, MUTED, layout["label_tracking"] * S)
            text = field["lines"][li][:counts[n]]
            vx, vy = field["value_xy"][0] * S, (field["value_xy"][1] + li * field["step"]) * S
            if text:
                d.text((vx, vy), text, font=field["big"], fill=FOREGROUND)
            if caret == n:
                gap = field["big"].getlength(" ") * 0.2 if text else 0
                draw_caret(d, vx + field["big"].getlength(text) + gap, vy, field["big"], ACCENT)
        # Separators appear with the field that follows them.
        for n, (fi, li) in enumerate(line_index):
            if li == 0 and fi > 0 and (counts[n] > 0 or caret == n):
                sx0, sy, sx1 = fields_px[fi - 1]["separator"]
                d.rectangle((sx0 * S, sy * S, sx1 * S, sy * S + max(1, S // 2)), fill=HAIRLINE)
        return image

    total = frame_count(seconds, fps)
    current_state, current_image = None, None
    with FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            t = n / fps
            state = state_at(t)
            if state != current_state:
                current_state, current_image = state, compose(state)
            cx, cy, zoom = drift_camera(t, seconds, size[0], size[1], PROFILE_DRIFT,
                                        start=(0.505, 0.50), end=(0.52, 0.495))
            out.write(_shoot(current_image, view_box(cx, cy, zoom, size), (width, height)))
    return Path(out_path)


def _font_at(fnt, scale: float):
    """The same face at `scale` times the size (canvases are drawn at SUPERSAMPLE)."""
    path = getattr(fnt, "path", None)
    if path is None:
        return fnt
    from PIL import ImageFont
    return ImageFont.truetype(path, round(fnt.size * scale), index=getattr(fnt, "index", 0))


def _draw_tracked(draw, xy, text, fnt, colour, tracking) -> None:
    x, y = xy
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=colour)
        x += fnt.getlength(ch) + tracking


# --- exhibit -----------------------------------------------------------------------

EXHIBIT_PUSH = 0.10
EXHIBIT_MAX_W = 0.60
EXHIBIT_MAX_H = 0.64
EXHIBIT_TILT = -1.8
TAG_TILT = 2.4
TAG_W = 0.225              # of the frame width
EXHIBIT_TYPE_FROM = 0.55
EXHIBIT_TYPE_BY = 0.55     # of the clip


def _tag_fonts(height: int):
    return (font(NAME_BOLD, max(6, round(height * 0.0200))),
            font(TYPEWRITER, max(6, round(height * 0.0235))))


def tag_layout(tag: str, number, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """The exhibit tag, unrotated, in frame pixels: its size, header, rule and typed lines."""
    head_font, text_font = _tag_fonts(height)
    w = width * TAG_W
    hole_zone = height * 0.050
    pad = height * 0.020
    inner = w - hole_zone - pad * 1.2
    text_font, lines = fit_lines(tag, TYPEWRITER, text_font.size, inner, max_lines=5,
                                 min_size=max(6, round(height * 0.018)))
    head = "EXHIBIT"
    num = f"No. {number}" if number not in (None, "") else None
    step = line_height(text_font, 1.24)
    head_h = line_height(head_font, 1.0)
    y = pad * 1.05
    head_y = y
    y += head_h + height * 0.010
    rule_y = y
    y += 2 + height * 0.014
    text_y = y
    h = y + len(lines) * step + pad * 0.75
    h = max(h, height * 0.13)
    return {"w": w, "h": h, "hole": (hole_zone * 0.48, h / 2), "hole_r": height * 0.0085,
            "text_x": hole_zone, "head": head, "number": num, "head_y": head_y, "rule_y": rule_y,
            "rule_x1": w - pad, "text_y": text_y, "lines": lines, "step": step,
            "fonts": (head_font, text_font), "tracking": height * 0.0055}


def exhibit_layout(photo_size, tag_size, width: int = WIDTH, height: int = HEIGHT) -> dict:
    """The object and its tag on the desk, in frame pixels.

    The object is fitted inside EXHIBIT_MAX_W x EXHIBIT_MAX_H (never past MAX_UPSCALE on screen at
    the end of the push) and turned a little; the tag lies against its lower right corner, turned
    the other way. The pair is centred in the frame, and both stay inside it.
    """
    scale = photo_scale(photo_size, width * EXHIBIT_MAX_W, height * EXHIBIT_MAX_H,
                        zoom=1 + EXHIBIT_PUSH)
    ow, oh = photo_size[0] * scale, photo_size[1] * scale
    tw, th = tag_size
    oew, oeh = rotated_extent(ow, oh, EXHIBIT_TILT)
    tew, teh = rotated_extent(tw, th, TAG_TILT)
    # Tag centre relative to the object's centre. A wide object has the tag below it, against its
    # lower right corner; anything else has it beside its lower right. Either way it overlaps the
    # object's edge only slightly, so it never hides what the exhibit shows.
    if ow / oh >= 1.3:
        tx = oew / 2 - tew / 2 - oew * 0.02
        ty = oeh / 2 + teh / 2 - teh * 0.10
    else:
        tx = oew / 2 + tew / 2 - tew * 0.08
        ty = oeh / 2 - teh / 2 - oeh * 0.04
    group = (min(-oew / 2, tx - tew / 2), min(-oeh / 2, ty - teh / 2),
             max(oew / 2, tx + tew / 2), max(oeh / 2, ty + teh / 2))
    gw, gh = group[2] - group[0], group[3] - group[1]
    fit = min(1.0, width * 0.90 / gw, height * 0.86 / gh)
    if fit < 1.0:  # too big with its tag: shrink the object, keep the tag's size
        return _exhibit_shrunk(photo_size, scale * fit * 0.98, tag_size, width, height)
    ox = width / 2 - (group[0] + group[2]) / 2
    oy = height * 0.49 - (group[1] + group[3]) / 2
    obj = {"cx": ox, "cy": oy, "w": ow, "h": oh, "angle": EXHIBIT_TILT, "scale": scale,
           "footprint": (ox - oew / 2, oy - oeh / 2, ox + oew / 2, oy + oeh / 2)}
    tag = {"cx": ox + tx, "cy": oy + ty, "w": tw, "h": th, "angle": TAG_TILT,
           "footprint": (ox + tx - tew / 2, oy + ty - teh / 2, ox + tx + tew / 2, oy + ty + teh / 2)}
    return {"object": obj, "tag": tag}


def _exhibit_shrunk(photo_size, scale, tag_size, width, height):
    fake = (photo_size[0] * scale, photo_size[1] * scale)
    # Re-run with a source that fits exactly, then put the true scale back.
    out = exhibit_layout(fake, tag_size, width, height)
    out["object"]["scale"] = scale
    return out


def exhibit_camera(layout: dict, width: int = WIDTH, height: int = HEIGHT,
                   push: float = EXHIBIT_PUSH, margin: float = 0.03) -> tuple[tuple[float, float], tuple[float, float]]:
    """Start and end centres (frame fractions) of the push: in toward the object, never losing the tag.

    The end view is (1/(1+push)) of the frame; its centre moves from the frame centre toward the
    object's centre, and back just enough that the tag's footprint (and the object's, where it
    fits) stays inside the view with `margin` of the frame to spare.
    """
    vw, vh = width / (1 + push), height / (1 + push)
    o, tg = layout["object"], layout["tag"]
    cx = width / 2 + (o["cx"] - width / 2) * 0.8
    cy = height / 2 + (o["cy"] - height / 2) * 0.8
    mx, my = width * margin, height * margin
    keep = [tg["footprint"], o["footprint"]]
    for x0, y0, x1, y1 in keep:
        if x1 - x0 + 2 * mx <= vw:
            cx = min(max(cx, x1 + mx - vw / 2), x0 - mx + vw / 2)
        if y1 - y0 + 2 * my <= vh:
            cy = min(max(cy, y1 + my - vh / 2), y0 - my + vh / 2)
    cx = min(max(cx, vw / 2), width - vw / 2)
    cy = min(max(cy, vh / 2), height - vh / 2)
    return (0.5, 0.5), (cx / width, cy / height)


def is_paper(picture: Image.Image) -> bool:
    """A scan of printed paper (light, even edges) is laid down as a clipping, not a bordered print."""
    w, h = picture.size
    grey = picture.convert("L")
    band = max(2, round(min(w, h) * 0.03))
    strips = [grey.crop((0, 0, w, band)), grey.crop((0, h - band, w, h)),
              grey.crop((0, 0, band, h)), grey.crop((w - band, 0, w, h))]
    values = [v for s in strips for v in s.tobytes()]
    mean = sum(values) / len(values)
    return mean > 200


def _tag_image(tl: dict, scale: float, chars: int | None, caret: bool, seed: int = 11) -> Image.Image:
    """The off-white tag at `scale`, with `chars` of its text typed (None: all), on transparency."""
    w, h = round(tl["w"] * scale), round(tl["h"] * scale)
    ss = 2
    big = (w * ss, h * ss)
    mask = Image.new("L", big, 0)
    md = ImageDraw.Draw(mask)
    notch = tl["h"] * scale * ss * 0.20
    md.polygon([(notch, 0), (big[0], 0), (big[0], big[1]), (notch, big[1]), (0, big[1] - notch),
                (0, notch)], fill=255)
    hx, hy = tl["hole"][0] * scale * ss, tl["hole"][1] * scale * ss
    r = tl["hole_r"] * scale * ss
    md.ellipse((hx - r, hy - r, hx + r, hy + r), fill=0)
    mask = mask.resize((w, h), Image.Resampling.LANCZOS)

    random.seed(seed)
    paper = Image.new("RGB", (w, h), PAPER)
    grain = Image.effect_noise((w, h), 14).point(lambda v: 128 + (v - 128) * 0.30)
    paper = ImageChops.multiply(paper, Image.merge("RGB", [grain.point(lambda v: min(255, v + 127))] * 3))
    # A faint darkening toward the edges, as card stock has.
    paper = apply_mask(paper, vignette_mask((w, h), 0.10, 0.9))
    draw = ImageDraw.Draw(paper)
    # Reinforcement ring round the hole.
    ring = r / ss * 1.85
    draw.ellipse((hx / ss - ring, hy / ss - ring, hx / ss + ring, hy / ss + ring),
                 outline=(0xC9, 0xC3, 0xB6), width=max(1, round(scale * 1.5)))

    head_font, text_font = (_font_at(f, scale) for f in tl["fonts"])
    x = tl["text_x"] * scale
    _draw_tracked(draw, (x, tl["head_y"] * scale), tl["head"], head_font, PAPER_INK, tl["tracking"] * scale)
    if tl["number"]:
        nw = tracked_width(tl["number"], head_font, tl["tracking"] * scale)
        _draw_tracked(draw, (tl["rule_x1"] * scale - nw, tl["head_y"] * scale), tl["number"], head_font,
                      ACCENT, tl["tracking"] * scale)
    draw.rectangle((x, tl["rule_y"] * scale, tl["rule_x1"] * scale, tl["rule_y"] * scale + max(1, round(scale * 1.6))),
                   fill=ACCENT)
    lengths = [len(line) for line in tl["lines"]]
    shown_lines = [line[:n] for line, n in zip(tl["lines"], _spread(sum(lengths) if chars is None else chars, lengths))]
    for i, shown in enumerate(shown_lines):
        if shown:
            draw.text((x, (tl["text_y"] + i * tl["step"]) * scale), shown, font=text_font, fill=PAPER_INK)
    if caret:
        # After the last character typed; at the start of the next line once a line is full.
        row = max([i for i, sh in enumerate(shown_lines) if sh] or [0])
        if shown_lines[row] == tl["lines"][row] and row + 1 < len(shown_lines) and chars is not None \
                and 0 < chars < sum(lengths):
            row += 1
        typed = shown_lines[row]
        y = (tl["text_y"] + row * tl["step"]) * scale
        draw_caret(draw, x + text_font.getlength(typed) + (text_font.getlength(" ") * 0.2 if typed else 0),
                   y, text_font, PAPER_INK)
    out = paper.convert("RGBA")
    out.putalpha(mask)
    return out


def _spread(count: int, lengths: list[int]) -> list[int]:
    """`count` characters poured line by line into lines of these lengths."""
    out, left = [], max(0, int(count))
    for n in lengths:
        out.append(min(n, left))
        left -= min(n, left)
    return out


def exhibit(photo, tag, out_path, seconds: float, number=None, *, width: int = WIDTH,
            height: int = HEIGHT, fps: int = FPS) -> Path:
    """An evidence object on a dark desk with an exhibit tag laid against it; slow push in.

    `tag` types on beneath "EXHIBIT" (and `number`, if given) over the first half of the clip.
    Exactly round(seconds * fps) frames.
    """
    if seconds <= 0:
        raise ValueError("an exhibit needs a positive length")
    if not str(tag or "").strip():
        raise ValueError("an exhibit needs tag text")
    picture = grade(load_picture(photo))
    paper = is_paper(picture)
    tl = tag_layout(str(tag).strip(), number, width, height)
    layout = exhibit_layout(picture.size, (tl["w"], tl["h"]), width, height)
    start, end = exhibit_camera(layout, width, height)
    S = SUPERSAMPLE
    size = (width * S, height * S)
    o, tg = layout["object"], layout["tag"]

    desk = desk_ground(size)
    desk = apply_mask(desk, vignette_mask(size, 0.45))
    ow, oh = round(o["w"] * S), round(o["h"] * S)
    if paper:  # newsprint lies flat: no border, whites knocked back to a paper tone
        obj = picture.resize((ow, oh), Image.Resampling.LANCZOS)
        obj = obj.point(lambda v: round(10 + v * 0.86))
        r, g, b = obj.split()
        obj = Image.merge("RGB", (r, g.point(lambda v: v * 0.995), b.point(lambda v: v * 0.985)))
    else:
        obj = bordered(picture, ow, oh, max(3, round(min(ow, oh) * 0.022)))

    paste_with_shadow(desk, obj, (o["cx"] * S, o["cy"] * S), o["angle"],
                      blur=height * S * 0.011, offset=(height * S * 0.005, height * S * 0.012), opacity=0.88)
    base = desk

    total_chars = sum(len(line) for line in tl["lines"])
    plan = typing_plan([total_chars], EXHIBIT_TYPE_FROM, max(EXHIBIT_TYPE_FROM + 0.5, seconds * EXHIBIT_TYPE_BY))
    t0, t1 = plan[0]
    caret_end = t1 + 0.9

    def state_at(t):
        chars = typed_counts(t, plan, [total_chars])[0]
        caret = t < caret_end and caret_visible(t, t0 <= t < t1)
        return chars, caret

    def compose(state):
        chars, caret = state
        image = base.copy()
        tag_img = _tag_image(tl, S, chars, caret)
        paste_with_shadow(image, tag_img, (tg["cx"] * S, tg["cy"] * S), tg["angle"],
                          blur=height * S * 0.006, offset=(height * S * 0.003, height * S * 0.007), opacity=0.75)
        return image

    total = frame_count(seconds, fps)
    current_state, current_image = None, None
    with FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            t = n / fps
            state = state_at(t)
            if state != current_state:
                current_state, current_image = state, compose(state)
            cx, cy, zoom = drift_camera(t, seconds, size[0], size[1], EXHIBIT_PUSH, start, end)
            out.write(_shoot(current_image, view_box(cx, cy, zoom, size), (width, height)))
    return Path(out_path)


# --- highlight -------------------------------------------------------------------

HIGHLIGHT_TARGET_H = 0.45    # the box's height at the end, as a fraction of the frame height
HIGHLIGHT_MAX_W = 0.62       # unless that would make it wider than this fraction of the width
HIGHLIGHT_PUSH = (0.05, 0.60)  # of the clip: the push runs between these
HIGHLIGHT_DRAW_AT = 0.40     # of the clip: the pen starts
HIGHLIGHT_DRAW_S = 0.80      # seconds the pen takes
HIGHLIGHT_MIN_BOX = 0.03     # a box side under this fraction of the photograph is widened to it


def clamp_box(box, min_side: float = HIGHLIGHT_MIN_BOX) -> tuple[float, float, float, float]:
    """A box as (x0, y0, x1, y1) fractions: ordered, inside [0, 1], each side at least `min_side`."""
    if box is None or len(box) != 4:
        raise ValueError(f"a box is four fractions (x0, y0, x1, y1), not {box!r}")
    x0, y0, x1, y1 = (float(v) for v in box)
    if any(math.isnan(v) for v in (x0, y0, x1, y1)):
        raise ValueError("a box has no NaN corners")
    x0, x1 = sorted((clamp(x0), clamp(x1)))
    y0, y1 = sorted((clamp(y0), clamp(y1)))

    def widen(a, b):
        if b - a >= min_side:
            return a, b
        mid = clamp((a + b) / 2, min_side / 2, 1 - min_side / 2)
        return mid - min_side / 2, mid + min_side / 2

    x0, x1 = widen(x0, x1)
    y0, y1 = widen(y0, y1)
    return x0, y0, x1, y1


def print_rect(src_size, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float, float, float]:
    """Where assembly's print treatment puts a photograph (border excluded), in frame pixels."""
    scale = photo_scale(src_size, width * PRINT_FIT, height * PRINT_FIT)
    w, h = src_size[0] * scale, src_size[1] * scale
    return ((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)


def highlight_target(box_px, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float, float]:
    """The final camera: centred on the box, zoomed so it fills HIGHLIGHT_TARGET_H of the height
    (or HIGHLIGHT_MAX_W of the width, whichever binds first). Returns (cx, cy, zoom)."""
    bx0, by0, bx1, by1 = box_px
    bw, bh = max(1e-6, bx1 - bx0), max(1e-6, by1 - by0)
    zoom = min(HIGHLIGHT_TARGET_H * height / bh, HIGHLIGHT_MAX_W * width / bw)
    return (bx0 + bx1) / 2, (by0 + by1) / 2, max(1.0, zoom)


def highlight_camera(p: float, box_px, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float, float]:
    """The camera at push progress p in [0, 1]: (cx, cy, zoom) in frame pixels.

    Zoom grows geometrically (an even push to the eye), and the box's centre slides toward the
    middle of the screen in step with the eased progress, so it arrives centred as the zoom lands.
    """
    e = ease_in_out_cubic(p)
    tx, ty, z1 = highlight_target(box_px, width, height)
    z0 = 1.0
    zoom = z0 * (z1 / z0) ** e
    # The target's offset from screen centre, in screen pixels, shrinks linearly with e.
    dx0, dy0 = (tx - width / 2) * z0, (ty - height / 2) * z0
    cx = tx - dx0 * (1 - e) / zoom
    cy = ty - dy0 * (1 - e) / zoom
    return cx, cy, zoom


def to_screen(point, camera, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float]:
    cx, cy, zoom = camera
    return (point[0] - cx) * zoom + width / 2, (point[1] - cy) * zoom + height / 2


def stroke_progress(t: float, start: float, duration: float = HIGHLIGHT_DRAW_S) -> float:
    """How much of the pen stroke is drawn at t: 0 before `start`, 1 after; quick in the middle."""
    return ease_in_out_sine(progress(t, start, start + duration))


def stroke_points(shape: str = "ellipse", seed: int = 7, count: int = 160) -> list[tuple[float, float, float]]:
    """A hand-drawn closed stroke around a unit box (-1..1 each way): [(x, y, width_factor)].

    `ellipse`: a slightly wobbling loop that starts upper left, runs clockwise a little past full
    circle and lands just outside where it began, as a pen would. `box`: a rounded rectangle drawn
    the same way. The width factor tapers at both ends, like pen pressure.
    """
    rnd = random.Random(seed)
    phases = [rnd.uniform(0, 2 * math.pi) for _ in range(3)]
    sweep = 2 * math.pi * 1.09
    start = math.radians(rnd.uniform(-160, -130))
    tilt = math.radians(rnd.uniform(-3.5, -1.5))
    out = []
    for i in range(count + 1):
        s = i / count
        a = start + sweep * s
        wobble = 1 + 0.028 * math.sin(2 * a + phases[0]) + 0.016 * math.sin(3 * a + phases[1]) \
            + 0.010 * math.sin(5 * a + phases[2])
        spiral = 1 + 0.055 * s
        if shape == "box":
            # A superellipse reads as a rounded rectangle drawn freehand.
            c, sn = math.cos(a), math.sin(a)
            n = 8.0
            x = math.copysign(abs(c) ** (2 / n), c)
            y = math.copysign(abs(sn) ** (2 / n), sn)
            rx, ry = 1.16 * wobble * spiral, 1.20 * wobble * spiral
        else:
            x, y = math.cos(a), math.sin(a)
            rx, ry = 1.30 * wobble * spiral, 1.36 * wobble * spiral
        px, py = x * rx, y * ry
        px, py = px * math.cos(tilt) - py * math.sin(tilt), px * math.sin(tilt) + py * math.cos(tilt)
        taper = min(1.0, s / 0.06, (1 - s) / 0.10)
        out.append((px, py, 0.45 + 0.55 * math.sqrt(clamp(taper))))
    return out


def partial_stroke(points, fraction: float):
    """The first `fraction` of a stroke by length, ending on an interpolated point."""
    fraction = clamp(fraction)
    if fraction <= 0 or len(points) < 2:
        return []
    lengths = [0.0]
    for (x0, y0, _), (x1, y1, _) in zip(points, points[1:]):
        lengths.append(lengths[-1] + math.hypot(x1 - x0, y1 - y0))
    goal = lengths[-1] * fraction
    out = [points[0]]
    for i in range(1, len(points)):
        if lengths[i] <= goal:
            out.append(points[i])
            continue
        span = lengths[i] - lengths[i - 1] or 1
        k = (goal - lengths[i - 1]) / span
        (x0, y0, w0), (x1, y1, w1) = points[i - 1], points[i]
        out.append((x0 + (x1 - x0) * k, y0 + (y1 - y0) * k, w0 + (w1 - w0) * k))
        break
    return out


def _stroke_layer(screen_pts, width_px: float, size) -> tuple[Image.Image, tuple[int, int]] | None:
    """The stroke drawn antialiased: (RGBA patch, top-left) in screen pixels, or None if empty."""
    if len(screen_pts) < 2:
        return None
    pad = width_px * 3
    xs = [p[0] for p in screen_pts]
    ys = [p[1] for p in screen_pts]
    x0, y0 = max(0, math.floor(min(xs) - pad)), max(0, math.floor(min(ys) - pad))
    x1, y1 = min(size[0], math.ceil(max(xs) + pad)), min(size[1], math.ceil(max(ys) + pad))
    if x1 <= x0 or y1 <= y0:
        return None
    k = 3
    mask = Image.new("L", ((x1 - x0) * k, (y1 - y0) * k), 0)
    d = ImageDraw.Draw(mask)
    local = [((x - x0) * k, (y - y0) * k, w) for x, y, w in screen_pts]
    for (ax, ay, aw), (bx, by, bw) in zip(local, local[1:]):
        w = width_px * k * (aw + bw) / 2
        d.line((ax, ay, bx, by), fill=255, width=max(1, round(w)))
        r = w / 2
        d.ellipse((bx - r, by - r, bx + r, by + r), fill=255)
    ax, ay, aw = local[0]
    r = width_px * k * aw / 2
    d.ellipse((ax - r, ay - r, ax + r, ay + r), fill=255)
    mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    shadow = Image.new("L", mask.size, 0)
    shadow.paste(mask.point(lambda v: v * 0.55), (max(1, round(width_px * 0.25)), max(1, round(width_px * 0.4))))
    shadow = shadow.filter(ImageFilter.GaussianBlur(width_px * 0.6))
    patch = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    patch.paste((0, 0, 0, 255), (0, 0), shadow)
    ink = Image.new("RGBA", mask.size, ACCENT + (0,))
    ink.putalpha(mask.point(lambda v: round(v * 0.96)))
    patch = Image.alpha_composite(patch, ink)
    return patch, (x0, y0)


class _PrintLayers:
    """The bordered print at a few resolutions, so each frame resamples from the nearest one.

    Each level holds the print at `scale` layer pixels per frame pixel, with its off-white border
    and a transparent PAD round it, so a resample box that touches the edge still lands inside.
    """

    PAD = 2

    def __init__(self, picture: Image.Image, rect, border_px: float, top_scale: float):
        self.rect = rect  # frame pixels, border excluded
        self.border = border_px
        w, h = rect[2] - rect[0], rect[3] - rect[1]
        scales, target = [], max(1.0, top_scale)
        while True:
            scales.append(target)
            if target <= 1.0:
                break
            target = max(1.0, target / 2)
        self.levels = []
        for scale in sorted(scales):
            pw, ph = max(2, round(w * scale)), max(2, round(h * scale))
            b = max(1, round(border_px * scale))
            inner = picture.resize((pw, ph), Image.Resampling.LANCZOS)
            if pw / picture.width > 2.5:  # strong enlargement of a small scan: restore some edge
                inner = inner.filter(ImageFilter.UnsharpMask(radius=pw / picture.width * 0.6, percent=70, threshold=2))
            layer = Image.new("RGBA", (pw + 2 * (b + self.PAD), ph + 2 * (b + self.PAD)), PRINT_BORDER + (0,))
            solid = Image.new("RGBA", (pw + 2 * b, ph + 2 * b), PRINT_BORDER + (255,))
            solid.paste(inner, (b, b))
            layer.paste(solid, (self.PAD, self.PAD))
            # The true layer-pixel scale on each axis, after rounding the print size.
            self.levels.append({"scale": scale, "sx": pw / w, "sy": ph / h, "b": b, "layer": layer})

    def draw(self, frame: Image.Image, camera, width: int, height: int) -> None:
        zoom = camera[2]
        level = next((lv for lv in self.levels if lv["scale"] >= zoom * 0.97), self.levels[-1])
        layer, sx, sy, off = level["layer"], level["sx"], level["sy"], level["b"] + self.PAD
        lw, lh = layer.size
        # Screen position of the layer's top-left corner, and screen pixels per layer pixel.
        sx0, sy0 = to_screen((self.rect[0] - off / sx, self.rect[1] - off / sy), camera, width, height)
        kx, ky = zoom / sx, zoom / sy
        ox0, oy0 = max(0, math.floor(sx0)), max(0, math.floor(sy0))
        ox1, oy1 = min(width, math.ceil(sx0 + lw * kx)), min(height, math.ceil(sy0 + lh * ky))
        if ox1 <= ox0 or oy1 <= oy0:
            return
        box = (clamp((ox0 - sx0) / kx, 0, lw), clamp((oy0 - sy0) / ky, 0, lh),
               clamp((ox1 - sx0) / kx, 0, lw), clamp((oy1 - sy0) / ky, 0, lh))
        piece = layer.resize((ox1 - ox0, oy1 - oy0), Image.Resampling.BICUBIC, box=box)
        frame.paste(piece, (ox0, oy0), piece)


def _label_plate(text: str, chars: int, caret: bool, height: int) -> tuple[Image.Image, float]:
    """A short typed label: off-white typewriter text on a dark plate with a crimson bar."""
    fnt = font(TYPEWRITER, max(6, round(height * 0.034)))
    full_w = fnt.getlength(text)
    pad_x, pad_y = height * 0.020, height * 0.014
    bar = max(2, round(height * 0.0045))
    w = round(bar + pad_x * 2 + full_w + fnt.getlength("n"))
    h = round(line_height(fnt, 1.0) + pad_y * 2)
    plate = Image.new("RGBA", (w, h), BACKGROUND + (round(255 * 0.84),))
    d = ImageDraw.Draw(plate)
    d.rectangle((0, 0, bar - 1, h), fill=ACCENT + (255,))
    shown = text[:chars]
    x = bar + pad_x
    d.text((x, pad_y), shown, font=fnt, fill=FOREGROUND + (255,))
    if caret:
        draw_caret(d, x + fnt.getlength(shown) + (fnt.getlength(" ") * 0.2 if shown else 0), pad_y, fnt,
                   FOREGROUND + (255,))
    return plate, full_w


def label_spot(ellipse_screen, plate_size, width: int = WIDTH, height: int = HEIGHT) -> tuple[float, float]:
    """Where a label plate goes beside the drawn ring (screen pixels, top-left): right, else left, else below."""
    x0, y0, x1, y1 = ellipse_screen
    pw, ph = plate_size
    gap, margin = width * 0.018, width * 0.04
    cy = (y0 + y1) / 2 - ph / 2
    cy = clamp(cy, height * 0.06, height * 0.80 - ph)
    if x1 + gap + pw <= width - margin:
        return x1 + gap, cy
    if x0 - gap - pw >= margin:
        return x0 - gap - pw, cy
    x = clamp((x0 + x1) / 2 - pw / 2, margin, width - margin - pw)
    y = y1 + gap if y1 + gap + ph <= height * 0.84 else max(height * 0.06, y0 - gap - ph)
    return x, y


def highlight(photo, box, out_path, seconds: float, label: str | None = None, shape: str = "ellipse", *,
              width: int = WIDTH, height: int = HEIGHT, fps: int = FPS, seed: int = 7) -> Path:
    """Push in on a detail of a photograph and ring it in crimson, like a pen on a print.

    `box`: (x0, y0, x1, y1) as fractions of the photograph. The photograph starts as a print on
    its own blurred dark ground (assembly's print look), the camera pushes in until the box fills
    about 45% of the frame height, centred; at 40% of the clip a crimson ring (`shape="ellipse"`)
    or rounded box (`shape="box"`) draws itself round the detail; `label`, if given, types on
    beside it; then it holds. Exactly round(seconds * fps) frames.
    """
    if seconds <= 0:
        raise ValueError("a highlight needs a positive length")
    if shape not in ("ellipse", "box"):
        raise ValueError(f"shape {shape!r} is not 'ellipse' or 'box'")
    fx0, fy0, fx1, fy1 = clamp_box(box)
    picture = grade(load_picture(photo))
    rect = print_rect(picture.size, width, height)
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    box_px = (rect[0] + fx0 * rw, rect[1] + fy0 * rh, rect[0] + fx1 * rw, rect[1] + fy1 * rh)
    final = highlight_target(box_px, width, height)
    border = max(2.0, height * 0.0028)

    # Ground: the blurred, darkened picture, mirrored outward so a camera near an edge never runs off.
    tile = blurred_ground(picture, (width, height), darken=PRINT_DARKEN)
    ground = Image.new("RGB", (width * 3, height * 3))
    for i in range(3):
        for j in range(3):
            piece = tile
            if i != 1:
                piece = ImageOps.mirror(piece)
            if j != 1:
                piece = ImageOps.flip(piece)
            ground.paste(piece, (i * width, j * height))
    soft = height * 0.012
    shadow = Image.new("L", ground.size, 0)
    ImageDraw.Draw(shadow).rectangle((width + rect[0] - border + soft * 0.3, height + rect[1] - border + soft * 0.6,
                                      width + rect[2] + border + soft * 0.3, height + rect[3] + border + soft * 0.6),
                                     fill=round(255 * 0.78))
    shadow = shadow.filter(ImageFilter.GaussianBlur(soft))
    ground.paste((0, 0, 0), (0, 0), shadow)
    layers = _PrintLayers(picture, rect, border, top_scale=min(final[2], 7000 / max(rw, rh)))
    vignette = vignette_mask((width, height), 0.38)
    vignette_rgb = Image.merge("RGB", (vignette,) * 3)

    points = stroke_points(shape, seed)
    bcx, bcy = (box_px[0] + box_px[2]) / 2, (box_px[1] + box_px[3]) / 2
    half_w, half_h = (box_px[2] - box_px[0]) / 2, (box_px[3] - box_px[1]) / 2
    stroke_w = height * 0.0076
    push0, push1 = (v * seconds for v in HIGHLIGHT_PUSH)
    draw_at = seconds * HIGHLIGHT_DRAW_AT
    draw_s = min(HIGHLIGHT_DRAW_S, max(0.3, seconds * 0.2))
    text = (label or "").strip()
    label_plan = None
    if text:
        lt0 = draw_at + draw_s + 0.25
        label_plan = typing_plan([len(text)], lt0, max(lt0 + 0.4, min(seconds - 0.6, lt0 + len(text) / 16)))
    caret_end = (label_plan[0][1] + 1.0) if label_plan else 0

    def camera_at(t):
        return highlight_camera(progress(t, push0, push1), box_px, width, height)

    total = frame_count(seconds, fps)
    last_key, last_bytes = None, None
    with FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            t = n / fps
            camera = camera_at(t)
            fraction = stroke_progress(t, draw_at, draw_s)
            chars, caret = 0, False
            if label_plan:
                chars = typed_counts(t, label_plan, [len(text)])[0]
                caret = t >= label_plan[0][0] and t < caret_end and caret_visible(
                    t, label_plan[0][0] <= t < label_plan[0][1])
            key = (tuple(round(v, 4) for v in camera), round(fraction, 4), chars, caret)
            if key == last_key:
                out.write(last_bytes)
                continue
            cx, cy, zoom = camera
            vw, vh = width / zoom, height / zoom
            gx0, gy0 = cx - vw / 2 + width, cy - vh / 2 + height
            frame = ground.resize((width, height), Image.Resampling.BILINEAR,
                                  box=(gx0, gy0, gx0 + vw, gy0 + vh))
            layers.draw(frame, camera, width, height)
            frame = ImageChops.multiply(frame, vignette_rgb)
            if fraction > 0:
                part = partial_stroke(points, fraction)
                screen = [(*to_screen((bcx + x * half_w, bcy + y * half_h), camera, width, height), w)
                          for x, y, w in part]
                drawn = _stroke_layer(screen, stroke_w, (width, height))
                if drawn:
                    patch, xy = drawn
                    frame.paste(patch, xy, patch)
            if label_plan and (chars or caret):
                plate, _ = _label_plate(text, chars, caret, height)
                ring = [to_screen((bcx + x * half_w, bcy + y * half_h), camera, width, height) for x, y, _ in points]
                extent = (min(p[0] for p in ring), min(p[1] for p in ring), max(p[0] for p in ring), max(p[1] for p in ring))
                lx, ly = label_spot(extent, plate.size, width, height)
                frame.paste(plate, (round(lx), round(ly)), plate)
            last_key, last_bytes = key, frame.tobytes()
            out.write(last_bytes)
    return Path(out_path)


# --- locate ------------------------------------------------------------------------

LOCATE_CACHE = ROOT / "data" / "video" / "locate_cache.json"   # keyed by file hash: shared by every film
LOCATE_MAX_SIDE = 1536
LOCATE_MIN_SIDE = 1024   # a small scan is enlarged for the model, so its lettering can be read
VERIFY_MARGIN = 0.20     # of the box's own size, added round the crop the second call sees
MIN_BOX_AREA = 0.0004

LOCATE_PROMPT = (
    "You are marking up an archival picture for a documentary. Find this detail in the image: "
    "\"{description}\". If it is a labelled area of a diagram, the box covers the area itself and its "
    "label. Return the tightest bounding box around it as fractions of the image width and height "
    "(x0, y0 = top-left, x1, y1 = bottom-right, each between 0 and 1). The image is {w}x{h} pixels. "
    "Set found to false, and the box to zeros, if the detail is not clearly visible."
)
VERIFY_PROMPT = (
    "This is a crop from a larger archival picture. Does this crop clearly show \"{description}\"? "
    "Answer shows=true only if the described detail is visibly inside this crop (for a diagram, the "
    "labelled area or its label is visible). Give a one-sentence reason."
)


def _models():
    from pydantic import BaseModel

    class FoundBox(BaseModel):
        found: bool
        x0: float
        y0: float
        x1: float
        y1: float
        reason: str = ""

    class Verdict(BaseModel):
        shows: bool
        reason: str

    return FoundBox, Verdict


def normalise_box(raw, image_size=None) -> tuple[float, float, float, float] | None:
    """A model's box as ordered fractions in [0, 1], or None when it is unusable.

    Accepts a 4-sequence or a dict with x0/y0/x1/y1. Values over 1 are read as pixels when the
    image size is known. Corners are ordered; anything degenerate, NaN or tiny is rejected.
    """
    if raw is None:
        return None
    try:
        if isinstance(raw, dict):
            values = [float(raw[k]) for k in ("x0", "y0", "x1", "y1")]
        else:
            values = [float(v) for v in raw]
    except (KeyError, TypeError, ValueError):
        return None
    if len(values) != 4 or any(math.isnan(v) or math.isinf(v) for v in values):
        return None
    if any(v > 1.02 for v in values):  # a hair over 1 is rounding; well over is pixels
        if not image_size or any(v > s * 1.02 for v, s in zip(values, image_size * 2)):
            return None
        w, h = image_size
        values = [values[0] / w, values[1] / h, values[2] / w, values[3] / h]
    if any(v < -0.02 for v in values):
        return None
    x0, x1 = sorted((clamp(values[0]), clamp(values[2])))
    y0, y1 = sorted((clamp(values[1]), clamp(values[3])))
    if (x1 - x0) * (y1 - y0) < MIN_BOX_AREA or x1 - x0 < 0.005 or y1 - y0 < 0.005:
        return None
    return round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)


def file_hash(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:20]


def cache_key(path, description: str) -> str:
    return f"{file_hash(path)}|{' '.join(str(description).split()).lower()}"


def read_cache(path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_cache(path, key: str, entry: dict) -> None:
    """Add one entry, re-reading first so a concurrent writer's entries survive; atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = read_cache(path)
    data[key] = entry
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _data_url(image: Image.Image, max_side: int = LOCATE_MAX_SIDE, min_side: int = LOCATE_MIN_SIDE) -> str:
    image = image.convert("RGB")
    long = max(image.size)
    if long < min_side:
        k = min_side / long
        image = image.resize((round(image.width * k), round(image.height * k)), Image.Resampling.LANCZOS)
    elif long > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def verify_crop(image: Image.Image, box, margin: float = VERIFY_MARGIN) -> Image.Image:
    """The box with `margin` of its own size added round it, clipped to the picture."""
    w, h = image.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    crop = (max(0, math.floor((x0 - mx) * w)), max(0, math.floor((y0 - my) * h)),
            min(w, math.ceil((x1 + mx) * w)), min(h, math.ceil((y1 + my) * h)))
    return image.crop(crop)


def _ask(client, model: str, prompt: str, image: Image.Image, schema):
    response = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _data_url(image), "detail": "high"},
        ]}],
        text_format=schema,
    )
    return getattr(response, "output_parsed", None)


def locate(photo, description: str, *, client=None, model: str | None = None,
           cache_path=LOCATE_CACHE, refresh: bool = False) -> tuple[float, float, float, float] | None:
    """The box (x0, y0, x1, y1, as fractions) of a described detail, confirmed by a second look.

    First call: the vision model returns the detail's bounding box (structured output). Second
    call: the model sees only that crop (plus a margin) and says whether it shows the detail.
    The box is returned only on yes; a refusal, a malformed answer or an API error gives None.
    Every answer, yes or no, is cached by (file hash, description) so a rerun costs nothing;
    API errors are not cached, so they are retried next time.
    """
    description = str(description or "").strip()
    if not description:
        raise ValueError("locate needs a description")
    key = cache_key(photo, description)
    if not refresh:
        hit = read_cache(cache_path).get(key)
        if isinstance(hit, dict) and "box" in hit:
            return tuple(hit["box"]) if hit["box"] else None

    if model is None:
        from app.config import settings
        model = settings.text_model or "gpt-6-sol"
    if client is None:
        from openai import OpenAI
        from app.config import settings
        client = OpenAI(api_key=settings.openai_api_key)
    FoundBox, Verdict = _models()
    image = load_picture(photo)
    entry = {"file": str(photo), "description": description, "model": model,
             "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    try:
        found = _ask(client, model, LOCATE_PROMPT.format(description=description, w=image.width, h=image.height),
                     image, FoundBox)
    except Exception as exc:  # noqa: BLE001 - an API failure is a "no", not a crash; not cached
        print(f"  locate: first call failed: {str(exc)[:160]}", file=sys.stderr)
        return None
    raw = found.model_dump() if hasattr(found, "model_dump") else found
    box = normalise_box(raw, image.size) if raw and raw.get("found") else None
    entry["raw"] = raw
    if box is None:
        entry.update(box=None, verified=False, reason="no usable box from the first call")
        write_cache(cache_path, key, entry)
        return None
    entry["candidate"] = list(box)
    try:
        verdict = _ask(client, model, VERIFY_PROMPT.format(description=description),
                       verify_crop(image, box), Verdict)
    except Exception as exc:  # noqa: BLE001
        print(f"  locate: verification call failed: {str(exc)[:160]}", file=sys.stderr)
        return None
    verdict = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict
    shows = bool(isinstance(verdict, dict) and verdict.get("shows") is True)
    entry.update(box=list(box) if shows else None, verified=shows,
                 reason=(verdict or {}).get("reason", "malformed verification") if isinstance(verdict, dict)
                 else "malformed verification")
    write_cache(cache_path, key, entry)
    return box if shows else None


def cached_entry(photo, description: str, cache_path=LOCATE_CACHE) -> dict | None:
    """The full cached record for a locate call (box, candidate, verdict reason), if any."""
    return read_cache(cache_path).get(cache_key(photo, description))


# --- demo --------------------------------------------------------------------------

def demo(out_dir) -> None:
    out = Path(out_dir)
    lib = ROOT / "data" / "video" / "gacy" / "library"
    timings = {}

    def timed(name, fn, *args, **kwargs):
        t0 = time.time()
        path = fn(*args, **kwargs)
        timings[name] = time.time() - t0
        print(f"{name}: {path} in {timings[name]:.1f}s", flush=True)

    timed("profile_piest", profile_card, lib / "c0035.png",
          [("NAME", "Robert Piest"), ("AGE", "15"),
           ("LAST SEEN", "December 11, 1978, Nisson Pharmacy, Des Plaines")],
          out / "profile_piest.mp4", 9.0)
    timed("profile_gacy", profile_card, lib / "c0011.jpg",
          [("NAME", "John Wayne Gacy"), ("ROLE", "Contractor, P.D.M. Contractors")],
          out / "profile_gacy.mp4", 8.0)
    timed("exhibit_diagram", exhibit, lib / "c0007.jpg",
          "Diagram of the house, 8213 W. Summerdale Avenue. Kokomo Tribune, December 1978",
          out / "exhibit_diagram.mp4", 8.0, number=1)
    timed("exhibit_card", exhibit, lib / "n0039.jpg",
          "Business card, P.D.M. Contractors, John Gacy, 8213 W. Summerdale, Norwood Park",
          out / "exhibit_card.mp4", 8.0, number=2)
    description = "the area labelled \"Crawl space\" under the house"
    box = locate(lib / "c0007.jpg", description)
    entry = cached_entry(lib / "c0007.jpg", description)
    print(f"locate: {box} | {json.dumps(entry)[:600]}", flush=True)
    if box:
        timed("highlight_crawl_space", highlight, lib / "c0007.jpg", box, out / "highlight_crawl_space.mp4",
              8.0, label="Crawl space")
    print(json.dumps({k: round(v, 1) for k, v in timings.items()}))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        demo(sys.argv[2] if len(sys.argv) > 2 else ROOT / "data" / "video" / "samples" / "investigation")
    else:
        print(__doc__)

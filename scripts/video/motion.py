"""Motion graphics for the cut: an evidence board, an animated map, a date stamp that types itself.

    .venv/bin/python scripts/video/motion.py demo [out_dir]

Each renderer draws its frames with Pillow and pipes them, raw, into the isolated ffmpeg 9 build
(`VIDEO_FFMPEG`, default `/opt/miniconda3/envs/video/bin/ffmpeg`), which encodes them. The board
and the map leave as H.264 mp4 files at the frame size, tagged BT.709 limited range exactly like
assembly's own intermediates, so a shot list can run them as `media: "motion"`. The typed stamp
leaves as a QuickTime Animation (qtrle) .mov with straight alpha, which assembly lays over a shot
in place of the static stamp PNG when the overlay says `"animate": true`.

The camera moves are done in Pillow too, not with ffmpeg's zoompan: each frame is resampled out of
a canvas at twice the frame size through a sub-pixel box, so a slow drift never steps from one
whole pixel to the next.

Everything that decides where or when something happens (the board layout, the camera path, the
order the strings draw, when each map pin drops and each label types, how much of a stamp shows
at a given moment) is a pure function, tested without rendering.

The palette is the film's: ground #0A0A0A, text #F2F2F2, crimson #9B111E, grey #8C8C8C.
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphics  # noqa: E402
from graphics import ACCENT, BACKGROUND, FOREGROUND, MUTED  # noqa: E402

DEFAULT_FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
FPS = 30
SUPERSAMPLE = 2  # every canvas is drawn at twice the frame size and resampled down per frame

PAPER = (0xE6, 0xE2, 0xD8)        # label tags: an off-white index card
PAPER_INK = (0x17, 0x17, 0x17)    # typed label text
PAPER_GREY = (0x5A, 0x58, 0x55)   # typed label subtext
PRINT_BORDER = (0xE9, 0xE6, 0xDF)  # the same off-white as assembly's prints
STRING_CORE = (0xC8, 0x2A, 0x38)  # the lit side of the crimson string


def ffmpeg_path(env=None) -> str:
    env = os.environ if env is None else env
    return env.get("VIDEO_FFMPEG") or DEFAULT_FFMPEG


# --- easing -----------------------------------------------------------------

def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def smoothstep(p: float) -> float:
    p = clamp(p)
    return p * p * (3 - 2 * p)


def ease_out_cubic(p: float) -> float:
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in_out_sine(p: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * clamp(p))


def ease_out_back(p: float, overshoot: float = 1.2) -> float:
    """Arrives, passes the mark by a few percent, settles back: a pin landing."""
    p = clamp(p) - 1
    return 1 + (overshoot + 1) * p ** 3 + overshoot * p ** 2


def progress(t: float, start: float, end: float) -> float:
    return 1.0 if end <= start else clamp((t - start) / (end - start))


# --- frames into ffmpeg -----------------------------------------------------

class FrameWriter:
    """Raw frames on ffmpeg's stdin: RGB to H.264 mp4, or RGBA to a qtrle .mov with alpha."""

    def __init__(self, out_path, width: int, height: int, fps: int = FPS, *, alpha: bool = False,
                 frames: int | None = None, crf: int = 15, preset: str = "medium"):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.size, self.alpha, self.count = (width, height), alpha, 0
        self.partial = self.out_path.with_name(self.out_path.stem + ".part" + self.out_path.suffix)
        argv = [ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo",
                "-pix_fmt", "rgba" if alpha else "rgb24", "-s", f"{width}x{height}",
                "-framerate", str(fps), "-i", "-"]
        if alpha:
            argv += ["-c:v", "qtrle", "-pix_fmt", "argb"]
        else:
            argv += ["-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
                     "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "film",
                     "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709",
                     "-color_trc", "bt709", "-color_range", "tv", "-movflags", "+faststart"]
        if frames:
            argv += ["-frames:v", str(frames)]
        argv += ["-r", str(fps), str(self.partial)]
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, image: Image.Image) -> None:
        mode = "RGBA" if self.alpha else "RGB"
        if image.size != self.size:
            raise ValueError(f"frame is {image.size}, expected {self.size}")
        self.proc.stdin.write((image if image.mode == mode else image.convert(mode)).tobytes())
        self.count += 1

    def write_bytes(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.count += 1

    def close(self) -> Path:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.out_path.name}: {err.strip()[-1500:]}")
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


def frame_count(seconds: float, fps: int = FPS) -> int:
    return max(1, round(seconds * fps))


def view_box(cx: float, cy: float, zoom: float, canvas: tuple[int, int]) -> tuple[float, float, float, float]:
    """The canvas rectangle a camera at (cx, cy) and `zoom` sees, kept inside the canvas.

    zoom 1.0 sees the whole canvas; 2.0 sees half of it in each direction.
    """
    cw, ch = canvas
    zoom = max(1.0, zoom)
    w, h = cw / zoom, ch / zoom
    x0 = clamp(cx - w / 2, 0, cw - w)
    y0 = clamp(cy - h / 2, 0, ch - h)
    return (x0, y0, x0 + w, y0 + h)


def shoot(canvas: Image.Image, box, size) -> Image.Image:
    """One frame of the camera: the box resampled to the frame size, with sub-pixel accuracy."""
    return canvas.resize(size, Image.Resampling.BICUBIC, box=box)


def _aa_patch(size, scale: int = 4):
    """A transparent patch `scale` times larger, for drawing antialiased shapes Pillow cannot."""
    return Image.new("RGBA", (size[0] * scale, size[1] * scale), (0, 0, 0, 0))


# --- evidence board ------------------------------------------------------------

BOARD_MARGIN = 0.035       # of the board: nothing is laid out closer to its edge than this
BOARD_HUB_H = 0.40         # the first pin, the subject, is the largest print, centred
BOARD_PRINT_H = 0.27       # every other print's height, as a fraction of the board height
BOARD_PRINT_MAX_W = 0.27   # and no print wider than this fraction of the board width
BOARD_TILT = (1.2, 4.0)    # degrees of rotation either way, never square, never drunk
BOARD_OVERLAP = 0.03       # two footprints may share at most this much of the smaller one
LABEL_SIZE = 0.024         # label type, as a fraction of the board height
SUB_SIZE = 0.0175


def _label_box(label: str, sub: str | None, board_h: int) -> tuple[float, float]:
    fnt = graphics.font(graphics.TYPEWRITER, max(6, round(board_h * LABEL_SIZE)))
    sub_fnt = graphics.font(graphics.TYPEWRITER, max(6, round(board_h * SUB_SIZE)))
    pad_x, pad_y = board_h * LABEL_SIZE * 0.6, board_h * LABEL_SIZE * 0.42
    width = max(fnt.getlength(label or " "), sub_fnt.getlength(sub) if sub else 0) + 2 * pad_x
    height = graphics.line_height(fnt, 1.15) + (graphics.line_height(sub_fnt, 1.3) if sub else 0) + 2 * pad_y
    return width, height


def _rotated_extent(w: float, h: float, degrees: float) -> tuple[float, float]:
    a = math.radians(degrees)
    return abs(w * math.cos(a)) + abs(h * math.sin(a)), abs(w * math.sin(a)) + abs(h * math.cos(a))


def rotate_point(x: float, y: float, degrees: float) -> tuple[float, float]:
    """A point about the origin, turned as Pillow's `rotate(degrees)` turns an image (anticlockwise
    on screen, where y points down)."""
    a = math.radians(degrees)
    return x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)


def board_slots(count: int) -> list[tuple[float, float]]:
    """Where each print's centre sits, as board fractions: the subject centred, the rest around it."""
    if count < 1:
        raise ValueError("an evidence board needs at least one pin")
    if count == 1:
        return [(0.5, 0.45)]
    if count == 2:
        return [(0.32, 0.45), (0.68, 0.45)]
    if count == 3:
        return [(0.5, 0.33), (0.26, 0.60), (0.74, 0.60)]
    others = count - 1
    slots = [(0.5, 0.50)]
    for k in range(others):
        angle = math.radians(-90 + 360 * (k + 0.5) / others)
        slots.append((0.5 + 0.335 * math.cos(angle), 0.50 + 0.315 * math.sin(angle)))
    return slots


def footprint(item: dict) -> tuple[float, float, float, float]:
    """The board area a pin occupies: its tilted print and the label tag under it."""
    px, py, pw, ph = item["cx"], item["cy"], item["ext_w"], item["ext_h"]
    lx, ly, lw, lh = item["label_cx"], item["label_cy"], item["label_w"], item["label_h"]
    return (min(px - pw / 2, lx - lw / 2), py - ph / 2, max(px + pw / 2, lx + lw / 2), ly + lh / 2)


def overlap_fraction(a, b) -> float:
    """How much of the smaller rectangle the two share."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return w * h / smaller


def board_layout(aspects: list[float], labels: list[tuple[str, str | None]], board: tuple[int, int],
                 seed: int = 1978) -> list[dict]:
    """Place prints on the board: sizes, centres, tilts, pin points and label tags.

    The first pin is the subject: largest, centred. The others ring it. Each print gets a small
    random tilt and nudge (seeded, so a board renders the same twice). If any two footprints
    overlap by more than BOARD_OVERLAP, or one leaves the board's margin, every print shrinks a
    little and the layout is tried again.
    """
    bw, bh = board
    rng = random.Random(seed)
    jitter = [(rng.uniform(-0.018, 0.018), rng.uniform(-0.016, 0.016),
               rng.choice((-1, 1)) * rng.uniform(*BOARD_TILT), rng.uniform(-1.4, 1.4))
              for _ in aspects]
    slots = board_slots(len(aspects))
    tags = [_label_box(label, sub, bh) for label, sub in labels]
    scale = 1.0
    for _attempt in range(40):
        items = []
        for i, (aspect, (sx, sy), (jx, jy, tilt, tag_tilt), (tw, th)) in enumerate(
                zip(aspects, slots, jitter, tags)):
            h = bh * (BOARD_HUB_H if i == 0 and len(aspects) > 3 else BOARD_PRINT_H) * scale
            w = h * aspect
            max_w = bw * BOARD_PRINT_MAX_W * (1.2 if i == 0 else 1.0) * scale
            if w > max_w:
                w, h = max_w, max_w / aspect
            ext_w, ext_h = _rotated_extent(w, h, tilt)
            cx, cy = bw * (sx + jx), bh * (sy + jy)
            label_cy = cy + ext_h / 2 + bh * 0.012 + th / 2
            # the stack (print over tag) is centred where the slot is
            stack_shift = (bh * 0.012 + th) / 2
            cy -= stack_shift
            label_cy -= stack_shift
            items.append({"cx": cx, "cy": cy, "w": w, "h": h, "angle": tilt, "ext_w": ext_w,
                          "ext_h": ext_h, "label_cx": cx, "label_cy": label_cy, "label_w": tw,
                          "label_h": th, "label_angle": tag_tilt})
        # pull anything over the margin back inside
        margin_x, margin_y = bw * BOARD_MARGIN, bh * BOARD_MARGIN
        for item in items:
            x0, y0, x1, y1 = footprint(item)
            dx = max(0, margin_x - x0) - max(0, x1 - (bw - margin_x))
            dy = max(0, margin_y - y0) - max(0, y1 - (bh - margin_y))
            for key in ("cx", "label_cx"):
                item[key] += dx
            for key in ("cy", "label_cy"):
                item[key] += dy
        prints = [footprint(item) for item in items]
        inside = all(p[0] >= margin_x - 1 and p[1] >= margin_y - 1 and p[2] <= bw - margin_x + 1
                     and p[3] <= bh - margin_y + 1 for p in prints)
        apart = all(overlap_fraction(a, b) <= BOARD_OVERLAP
                    for i, a in enumerate(prints) for b in prints[i + 1:])
        if inside and apart:
            break
        scale *= 0.95
    for item in items:
        # the pin goes through the top border, a little in from the edge
        r = pin_radius(bh)
        px, py = rotate_point(0, -item["h"] / 2 + r * 1.25, item["angle"])
        item["pin"] = (item["cx"] + px, item["cy"] + py)
    return items


def pin_radius(board_h: int) -> float:
    return board_h * 0.0105


def resolve_links(links, pins: list[dict]) -> list[tuple[int, int]]:
    """Links as index pairs; a link may name its pins by index or by label."""
    by_label = {str(p.get("label", "")).lower(): i for i, p in enumerate(pins)}
    out = []
    for a, b in links:
        pair = []
        for end in (a, b):
            if isinstance(end, int):
                index = end
            elif str(end).lower() in by_label:
                index = by_label[str(end).lower()]
            else:
                raise ValueError(f"link end {end!r} is not a pin index or label")
            if not 0 <= index < len(pins):
                raise ValueError(f"link end {end!r} is outside 0..{len(pins) - 1}")
            pair.append(index)
        if pair[0] == pair[1]:
            raise ValueError(f"a link joins a pin to itself: {a!r}")
        out.append((pair[0], pair[1]))
    return out


def string_schedule(count: int, seconds: float, lead: float = 0.9, tail: float = 1.4,
                    draw_s: float = 1.1) -> list[tuple[float, float]]:
    """When each string draws on: one at a time, evenly spread, all done well before the end."""
    if count == 0:
        return []
    usable = max(0.5, seconds - lead - tail)
    slot = usable / count
    length = min(draw_s, slot * 0.7)
    return [(lead + k * slot, lead + k * slot + length) for k in range(count)]


BOARD_PUSH_MAX = 1.25  # the camera ends at most this far in: about a fifth of the board cropped away
BOARD_VIEW_MARGIN = 0.025


def board_keyframes(items: list[dict], board: tuple[int, int], seconds: float) -> list[tuple[float, float, float, float]]:
    """Camera keyframes (t, cx, cy, zoom): one slow push in, from the whole board towards the subject.

    It opens on the whole board and eases in towards the first pin's print, but only as far as
    keeps every print and label in frame (with a small margin), and never past BOARD_PUSH_MAX. So
    the move drifts one way and stops gently; nothing is ever cut off while its string draws.
    """
    bw, bh = board
    prints = [footprint(item) for item in items]
    x0 = min(p[0] for p in prints) - bw * BOARD_VIEW_MARGIN
    y0 = min(p[1] for p in prints) - bh * BOARD_VIEW_MARGIN
    x1 = max(p[2] for p in prints) + bw * BOARD_VIEW_MARGIN
    y1 = max(p[3] for p in prints) + bh * BOARD_VIEW_MARGIN
    zoom = clamp(min(bw / (x1 - x0), bh / (y1 - y0)), 1.0, BOARD_PUSH_MAX)
    vw, vh = bw / zoom, bh / zoom
    hub = items[0]
    cx = clamp(hub["cx"], x1 - vw / 2, x0 + vw / 2) if x1 - x0 <= vw else (x0 + x1) / 2
    cy = clamp(hub["cy"], y1 - vh / 2, y0 + vh / 2) if y1 - y0 <= vh else (y0 + y1) / 2
    v = view_box(cx, cy, zoom, board)
    return [(0.0, bw / 2, bh / 2, 1.0), (seconds, (v[0] + v[2]) / 2, (v[1] + v[3]) / 2, zoom)]


def camera_at(t: float, keys) -> tuple[float, float, float]:
    """The camera between keyframes: a Catmull-Rom curve whose end tangents are zero, so it eases
    out of its first position and into its last and never stops in between."""
    if t <= keys[0][0]:
        return keys[0][1:]
    if t >= keys[-1][0]:
        return keys[-1][1:]
    for i in range(len(keys) - 1):
        if keys[i][0] <= t <= keys[i + 1][0]:
            break
    t0, t1 = keys[i][0], keys[i + 1][0]
    u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
    result = []
    for axis in range(1, 4):
        p0 = keys[max(0, i - 1)][axis]
        p1, p2 = keys[i][axis], keys[i + 1][axis]
        p3 = keys[min(len(keys) - 1, i + 2)][axis]
        m1 = 0.0 if i == 0 else (p2 - p0) / 2
        m2 = 0.0 if i + 1 == len(keys) - 1 else (p3 - p1) / 2
        u2, u3 = u * u, u * u * u
        result.append((2 * u3 - 3 * u2 + 1) * p1 + (u3 - 2 * u2 + u) * m1
                      + (-2 * u3 + 3 * u2) * p2 + (u3 - u2) * m2)
    return tuple(result)


def _board_ground(size: tuple[int, int]) -> Image.Image:
    """A dark felt board: a soft pool of light from above, a fine fibre and a faint mottle."""
    w, h = size
    small = Image.new("L", (96, 54))
    for y in range(54):
        for x in range(96):
            d = math.hypot((x / 95 - 0.5) * 1.15, (y / 53 - 0.40) * 1.0)
            small.putpixel((x, y), round(25 - 16 * clamp(d / 0.75) ** 1.4))
    light = small.resize(size, Image.Resampling.BICUBIC)
    fibre = Image.effect_noise(size, 18)
    mottle = Image.effect_noise((max(1, w // 24), max(1, h // 24)), 40).resize(size, Image.Resampling.BICUBIC)
    fibre = fibre.point(lambda v: round(128 + (v - 128) * 0.045))
    mottle = mottle.point(lambda v: round(128 + (v - 128) * 0.035))
    value = ImageChops.add(ImageChops.add(light, fibre, 1.0, -128), mottle, 1.0, -128)
    return Image.merge("RGB", (value, value.point(lambda v: v * 0.99), value.point(lambda v: v * 0.97)))


def _print_image(path, w: int, h: int) -> Image.Image:
    """A library photograph as a print: lightly muted (colour still reads as colour), cool, bordered in off-white."""
    with Image.open(path) as source:
        if source.mode in ("RGBA", "LA", "P"):
            source = source.convert("RGBA")
            flat = Image.new("RGB", source.size, (235, 235, 235))
            flat.paste(source, mask=source.getchannel("A"))
            source = flat
        picture = source.convert("RGB")
    border = max(3, round(min(w, h) * 0.035))
    inner = ImageOps.fit(picture, (w - 2 * border, h - 2 * border), Image.Resampling.LANCZOS)
    inner = ImageEnhance.Color(inner).enhance(0.80)
    inner = ImageEnhance.Contrast(inner).enhance(1.05)
    r, g, b = inner.split()
    inner = Image.merge("RGB", (r.point(lambda v: v * 0.96), g.point(lambda v: v * 0.985), b))
    out = Image.new("RGB", (w, h), PRINT_BORDER)
    out.paste(inner, (border, border))
    return out


def _paste_with_shadow(board: Image.Image, image: Image.Image, centre, angle: float, *,
                       shadow_blur: float, offset: tuple[float, float], opacity: float) -> None:
    """Turn an object, then lay it on the board over a soft shadow falling down and right."""
    turned = image.convert("RGBA").rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    alpha = turned.getchannel("A")
    pad = round(shadow_blur * 3)
    shadow_mask = Image.new("L", (turned.width + 2 * pad, turned.height + 2 * pad), 0)
    shadow_mask.paste(alpha.point(lambda a: a * opacity), (pad, pad))
    shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(shadow_blur))
    x = round(centre[0] - turned.width / 2)
    y = round(centre[1] - turned.height / 2)
    black = Image.new("RGB", shadow_mask.size, (0, 0, 0))
    board.paste(black, (x - pad + round(offset[0]), y - pad + round(offset[1])), shadow_mask)
    board.paste(turned, (x, y), turned)


def _label_tag(label: str, sub: str | None, size: tuple[float, float], board_h: int) -> Image.Image:
    w, h = round(size[0]), round(size[1])
    tag = Image.new("RGB", (w, h), PAPER)
    grain = Image.effect_noise((w, h), 10).point(lambda v: 128 + (v - 128) * 0.35)
    tag = ImageChops.multiply(tag, Image.merge("RGB", [grain.point(lambda v: min(255, v + 127))] * 3))
    draw = ImageDraw.Draw(tag)
    fnt = graphics.font(graphics.TYPEWRITER, max(6, round(board_h * LABEL_SIZE)))
    sub_fnt = graphics.font(graphics.TYPEWRITER, max(6, round(board_h * SUB_SIZE)))
    pad_x, pad_y = board_h * LABEL_SIZE * 0.6, board_h * LABEL_SIZE * 0.42
    draw.text((pad_x, pad_y), label, font=fnt, fill=PAPER_INK)
    if sub:
        draw.text((pad_x, pad_y + graphics.line_height(fnt, 1.15)), sub, font=sub_fnt, fill=PAPER_GREY)
    return tag


def _pin_patch(radius: float) -> Image.Image:
    """A crimson push pin seen from the front: a domed head, a darker rim, one small highlight."""
    scale = 4
    side = math.ceil(radius * 2.6)
    patch = _aa_patch((side, side), scale)
    r, c = radius * scale, side * scale / 2
    shade = Image.new("L", patch.size, 0)
    ImageDraw.Draw(shade).ellipse((c - r * 0.95 + r * 0.30, c - r * 0.95 + r * 0.45,
                                   c + r * 0.95 + r * 0.30, c + r * 0.95 + r * 0.45), fill=150)
    shade = shade.filter(ImageFilter.GaussianBlur(r * 0.25))
    patch.paste((0, 0, 0, 255), (0, 0), shade)
    draw = ImageDraw.Draw(patch)
    rim = tuple(round(v * 0.55) for v in ACCENT)
    draw.ellipse((c - r, c - r, c + r, c + r), fill=rim + (255,))
    draw.ellipse((c - r * 0.86, c - r * 0.90, c + r * 0.80, c + r * 0.76), fill=ACCENT + (255,))
    glint = Image.new("L", patch.size, 0)
    ImageDraw.Draw(glint).ellipse((c - r * 0.55, c - r * 0.60, c - r * 0.10, c - r * 0.22), fill=120)
    glint = glint.filter(ImageFilter.GaussianBlur(r * 0.10))
    patch = Image.alpha_composite(patch, Image.merge("RGBA", [Image.new("L", patch.size, 255)] * 3 + [glint]))
    return patch.resize((side, side), Image.Resampling.LANCZOS)


def _string_patch(a, b, board_h: int):
    """A taut crimson string from a to b, with its shadow: (patch, origin), antialiased."""
    scale = 3
    width = board_h * 0.0029
    shadow_dx, shadow_dy, blur = board_h * 0.003, board_h * 0.006, board_h * 0.003
    pad = width * 2 + blur * 3 + max(shadow_dx, shadow_dy)
    x0, y0 = min(a[0], b[0]) - pad, min(a[1], b[1]) - pad
    x1, y1 = max(a[0], b[0]) + pad, max(a[1], b[1]) + pad
    size = (math.ceil(x1 - x0), math.ceil(y1 - y0))
    big = (size[0] * scale, size[1] * scale)

    def local(p, dx=0.0, dy=0.0):
        return ((p[0] - x0 + dx) * scale, (p[1] - y0 + dy) * scale)

    shadow = Image.new("L", big, 0)
    ImageDraw.Draw(shadow).line([local(a, shadow_dx, shadow_dy), local(b, shadow_dx, shadow_dy)],
                                fill=140, width=max(1, round(width * scale * 1.1)))
    shadow = shadow.resize(size, Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(blur))
    body = Image.new("L", big, 0)
    ImageDraw.Draw(body).line([local(a), local(b)], fill=255, width=max(1, round(width * scale)))
    body = body.resize(size, Image.Resampling.LANCZOS)
    # the lit side: a thinner core nudged up-left of the centre line
    span = math.hypot(b[0] - a[0], b[1] - a[1]) or 1
    nx, ny = -(b[1] - a[1]) / span, (b[0] - a[0]) / span
    if ny > 0:
        nx, ny = -nx, -ny
    lift = width * 0.18
    core = Image.new("L", big, 0)
    ImageDraw.Draw(core).line([local(a, nx * lift, ny * lift), local(b, nx * lift, ny * lift)],
                              fill=255, width=max(1, round(width * scale * 0.38)))
    core = core.resize(size, Image.Resampling.LANCZOS)

    patch = Image.new("RGBA", size, (0, 0, 0, 0))
    patch.paste((0, 0, 0, 255), (0, 0), shadow)
    patch.paste(ACCENT + (255,), (0, 0), body)
    patch.paste(STRING_CORE + (255,), (0, 0), core.point(lambda v: v * 0.8))
    return patch, (round(x0), round(y0)), (a[0] - round(x0), a[1] - round(y0)), (b[0] - round(x0), b[1] - round(y0))


def _partial(patch: Image.Image, a, b, fraction: float) -> Image.Image:
    """The string drawn from a only `fraction` of the way to b, with a clean cut end."""
    if fraction >= 1:
        return patch
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1
    ux, uy = dx / length, dy / length
    reach = fraction * length
    far = (patch.width + patch.height) * 2
    cut = (a[0] + ux * reach, a[1] + uy * reach)
    behind = (a[0] - ux * far, a[1] - uy * far)
    px, py = -uy * far, ux * far
    mask = Image.new("L", patch.size, 0)
    ImageDraw.Draw(mask).polygon([(behind[0] + px, behind[1] + py), (cut[0] + px, cut[1] + py),
                                  (cut[0] - px, cut[1] - py), (behind[0] - px, behind[1] - py)], fill=255)
    out = patch.copy()
    out.putalpha(ImageChops.multiply(patch.getchannel("A"), mask))
    return out


def compose_board(pins: list[dict], board: tuple[int, int], seed: int = 1978):
    """The board without its strings or pins, plus the layout: (image, items)."""
    aspects = []
    for pin in pins:
        with Image.open(pin["image"]) as image:
            aspects.append(image.width / image.height)
    items = board_layout(aspects, [(p.get("label", ""), p.get("sub")) for p in pins], board, seed)
    canvas = _board_ground(board)
    bh = board[1]
    for pin, item in zip(pins, items):
        print_image = _print_image(pin["image"], round(item["w"]), round(item["h"]))
        _paste_with_shadow(canvas, print_image, (item["cx"], item["cy"]), item["angle"],
                           shadow_blur=bh * 0.010, offset=(bh * 0.004, bh * 0.010), opacity=0.85)
        tag = _label_tag(pin.get("label", ""), pin.get("sub"), (item["label_w"], item["label_h"]), bh)
        _paste_with_shadow(canvas, tag, (item["label_cx"], item["label_cy"]), item["label_angle"],
                           shadow_blur=bh * 0.005, offset=(bh * 0.002, bh * 0.005), opacity=0.7)
    return canvas, items


def evidence_board(pins: list[dict], links, out_path, seconds: float = 9.0, *, width: int = 1920,
                   height: int = 1080, fps: int = FPS, seed: int = 1978, draw_strings: bool = True) -> Path:
    """Real photographs pinned to a dark board, joined by red string, filmed by a slow push in.

    `pins`: [{"image": path, "label": "Robert Piest", "sub": "15, missing Dec 11 1978"}], the
    subject first (it is placed largest, in the centre). `links`: pairs of pin indices or labels.
    The board is composed once at twice the frame size; the strings then draw on one at a time
    while the camera pushes slowly in towards the subject, and the result is encoded as an mp4 of `seconds`.
    """
    if not 3.0 <= seconds <= 30.0:
        raise ValueError(f"an evidence board runs 3-30 s (6-12 s reads best), not {seconds}")
    board = (width * SUPERSAMPLE, height * SUPERSAMPLE)
    links = resolve_links(links, pins)
    canvas, items = compose_board(pins, board, seed)
    bh = board[1]
    strings = [_string_patch(items[a]["pin"], items[b]["pin"], bh) for a, b in links]
    schedule = string_schedule(len(links), seconds) if draw_strings else [(0, 0)] * len(links)
    keys = board_keyframes(items, board, seconds)
    pin = _pin_patch(pin_radius(bh))
    total = frame_count(seconds, fps)

    baked, baked_count = canvas, 0
    with FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            t = n / fps
            while baked_count < len(strings) and t >= schedule[baked_count][1]:
                patch, origin, _, _ = strings[baked_count]
                baked = baked.copy() if baked is canvas else baked
                baked.paste(patch, origin, patch)
                baked_count += 1
            current = baked
            if baked_count < len(strings) and t > schedule[baked_count][0]:
                patch, origin, a, b = strings[baked_count]
                part = _partial(patch, a, b, ease_in_out_sine(progress(t, *schedule[baked_count])))
                current = baked.copy()
                current.paste(part, origin, part)
            else:
                current = baked.copy()
            for item in items:
                current.paste(pin, (round(item["pin"][0] - pin.width / 2), round(item["pin"][1] - pin.height / 2)), pin)
            cx, cy, zoom = camera_at(t, keys)
            out.write(shoot(current, view_box(cx, cy, zoom, board), (width, height)))
    return Path(out_path)


# --- animated map -------------------------------------------------------------

MAP_PANEL = (0.08, 0.17, 0.92, 0.88)  # left, top, right, bottom as frame fractions
PIN_DROP_S = 0.45
ROUTE_S = 0.65
LABEL_CHAR_S = 0.032
MAP_PUSH = 0.16  # the map pushes this much further in over the shot, towards the points


def map_schedule(count: int, seconds: float, lead: float = 0.7, hold: float = 1.6,
                 max_gap: float = 1.8, min_gap: float = 0.35) -> list[dict]:
    """When each pin drops, lands, and types its label, and when the route to it draws.

    Pins land evenly spaced, the first after `lead`, the last at least `hold` before the end so
    the finished map is held. The route into pin i draws over the ROUTE_S before it lands.
    """
    if count < 1:
        raise ValueError("a map needs at least one point")
    first = lead + PIN_DROP_S
    gap = 0.0
    if count > 1:
        gap = clamp((seconds - hold - first) / (count - 1), min_gap, max_gap)
    plan = []
    for i in range(count):
        land = first + i * gap
        plan.append({"drop_from": land - PIN_DROP_S, "land": land,
                     "route_from": land - min(ROUTE_S, max(gap, min_gap) * 0.9) if i else None,
                     "label_from": land + 0.05})
    return plan


def map_place(point: dict, size: tuple[int, int]) -> tuple[float, float]:
    left, top, right, bottom = (MAP_PANEL[0] * size[0], MAP_PANEL[1] * size[1],
                                MAP_PANEL[2] * size[0], MAP_PANEL[3] * size[1])
    x = clamp(float(point.get("x", 0.5)), 0.04, 0.96)
    y = clamp(float(point.get("y", 0.5)), 0.04, 0.96)
    return left + x * (right - left), top + y * (bottom - top)


def label_spots(spots, widths, height_px: float, radius: float, bounds) -> list[tuple[float, float]]:
    """Top-left corners for each label: right of its pin if that is clear, else above or below
    and still reading away from the pin, and only then to its left (a label typing towards its pin
    from the left reads as detached from it).

    A spot is clear when it overlaps no pin and no label already placed, and stays in `bounds`.
    """
    placed, out = [], []
    pins = [(x - radius * 1.6, y - radius * 1.6, x + radius * 1.6, y + radius * 1.6) for x, y in spots]
    gap = radius * 2.2
    for (x, y), w in zip(spots, widths):
        options = [(x + gap, y - height_px / 2), (x + gap * 0.5, y - gap - height_px),
                   (x + gap * 0.5, y + gap), (x - gap - w, y - height_px / 2),
                   (x - gap * 0.5 - w, y - gap - height_px), (x - gap * 0.5 - w, y + gap)]
        choice = None
        for lx, ly in options:
            rect = (lx, ly, lx + w, ly + height_px)
            if rect[0] < bounds[0] or rect[2] > bounds[2] or rect[1] < bounds[1] or rect[3] > bounds[3]:
                continue
            if all(overlap_fraction(rect, other) == 0 for other in placed + pins):
                choice = rect
                break
        if choice is None:
            lx, ly = options[0] if x + gap + w < bounds[2] else options[1]
            choice = (lx, ly, lx + w, ly + height_px)
        placed.append(choice)
        out.append(choice[:2])
    return out


def _map_base(size: tuple[int, int]) -> Image.Image:
    """The schematic map: a fine survey grid that fills the frame and fades into the ground at
    its edges, lit a little from the centre. No outline, so a push in never runs a border into
    anything; the title and north mark sit on the screen above it."""
    w, h = size
    small = Image.new("L", (64, 36))
    fall = Image.new("L", (64, 36))
    for y in range(36):
        for x in range(64):
            d = math.hypot((x / 63 - 0.5) * 1.0, (y / 35 - 0.5) * 0.9)
            small.putpixel((x, y), round(10 + 12 * (1 - smoothstep(d / 0.62))))
            edge = min(x / 63, 1 - x / 63, (y / 35) * 1.25, (1 - y / 35) * 1.25)
            fall.putpixel((x, y), round(255 * smoothstep(edge / 0.22)))
    light = small.resize(size, Image.Resampling.BICUBIC)
    image = Image.merge("RGB", (light, light, light.point(lambda v: v + 1)))
    falloff = fall.resize(size, Image.Resampling.BICUBIC)
    minor, major = Image.new("L", size, 0), Image.new("L", size, 0)
    minor_draw, major_draw = ImageDraw.Draw(minor), ImageDraw.Draw(major)
    left, right = MAP_PANEL[0] * w, MAP_PANEL[2] * w
    step = (right - left) / 24
    line_w = max(1, round(h * 0.0008))
    k0 = -math.ceil(left / step)
    for k in range(k0, k0 + math.ceil(w / step) + 2):
        x = left + k * step
        (major_draw if k % 4 == 0 else minor_draw).line((x, 0, x, h), fill=255, width=line_w)
    top = MAP_PANEL[1] * h
    k0 = -math.ceil(top / step)
    for k in range(k0, k0 + math.ceil(h / step) + 2):
        y = top + k * step
        (major_draw if k % 4 == 0 else minor_draw).line((0, y, w, y), fill=255, width=line_w)
    image.paste((0x26, 0x26, 0x28), (0, 0), ImageChops.multiply(minor, falloff.point(lambda v: v * 0.75)))
    image.paste((0x33, 0x33, 0x35), (0, 0), ImageChops.multiply(major, falloff.point(lambda v: v * 0.85)))
    return image


def _north(draw: ImageDraw.ImageDraw, width: int, height: int) -> None:
    """A small north mark, bottom right of the screen."""
    nx, ny = width * 0.925, height * 0.86
    arrow = height * 0.030
    line = max(1, round(height * 0.0016))
    draw.line((nx, ny + arrow / 2, nx, ny - arrow / 2), fill=MUTED, width=line)
    draw.polygon([(nx, ny - arrow / 2 - height * 0.008), (nx - height * 0.006, ny - arrow / 2 + height * 0.004),
                  (nx + height * 0.006, ny - arrow / 2 + height * 0.004)], fill=MUTED)
    n_font = graphics.font(graphics.LABEL, max(6, round(height * 0.018)))
    draw.text((nx - n_font.getlength("N") / 2, ny + arrow / 2 + height * 0.006), "N", font=n_font, fill=MUTED)


def _dot_patch(radius: float) -> Image.Image:
    """A map pin: a crimson dot ringed in off-white, on a soft shadow; antialiased."""
    scale = 4
    side = math.ceil(radius * 4)
    patch = _aa_patch((side, side), scale)
    r, c = radius * scale, side * scale / 2
    shade = Image.new("L", patch.size, 0)
    ImageDraw.Draw(shade).ellipse((c - r * 1.2, c - r * 1.0, c + r * 1.2, c + r * 1.4), fill=170)
    shade = shade.filter(ImageFilter.GaussianBlur(r * 0.4))
    patch.paste((0, 0, 0, 255), (0, 0), shade)
    draw = ImageDraw.Draw(patch)
    draw.ellipse((c - r, c - r, c + r, c + r), fill=FOREGROUND + (255,))
    inner = r * 0.74
    draw.ellipse((c - inner, c - inner, c + inner, c + inner), fill=ACCENT + (255,))
    return patch.resize((side, side), Image.Resampling.LANCZOS)


def _glow_patch(radius: float) -> Image.Image:
    side = math.ceil(radius * 9)
    mask = Image.new("L", (side, side), 0)
    c = side / 2
    ImageDraw.Draw(mask).ellipse((c - radius * 2.2, c - radius * 2.2, c + radius * 2.2, c + radius * 2.2), fill=80)
    mask = mask.filter(ImageFilter.GaussianBlur(radius * 1.1))
    patch = Image.new("RGBA", (side, side), ACCENT + (0,))
    patch.putalpha(mask)
    return patch


def _ring(canvas: Image.Image, centre, radius: float, alpha: float, width: float) -> None:
    scale = 3
    side = math.ceil(radius * 2 + width * 4)
    patch = _aa_patch((side, side), scale)
    c, r = side * scale / 2, radius * scale
    ImageDraw.Draw(patch).ellipse((c - r, c - r, c + r, c + r), outline=ACCENT + (round(255 * alpha),),
                                  width=max(1, round(width * scale)))
    patch = patch.resize((side, side), Image.Resampling.LANCZOS)
    canvas.paste(patch, (round(centre[0] - side / 2), round(centre[1] - side / 2)), patch)


def _route(canvas: Image.Image, a, b, fraction: float, width: float) -> None:
    if fraction <= 0:
        return
    end = (a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction)
    scale = 3
    x0, y0 = min(a[0], end[0]) - width * 3, min(a[1], end[1]) - width * 3
    x1, y1 = max(a[0], end[0]) + width * 3, max(a[1], end[1]) + width * 3
    size = (max(1, math.ceil(x1 - x0)), max(1, math.ceil(y1 - y0)))
    mask = Image.new("L", (size[0] * scale, size[1] * scale), 0)
    ImageDraw.Draw(mask).line([((a[0] - x0) * scale, (a[1] - y0) * scale),
                               ((end[0] - x0) * scale, (end[1] - y0) * scale)],
                              fill=255, width=max(1, round(width * scale)))
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    canvas.paste(ACCENT, (round(x0), round(y0)), mask)


def map_reveal(points: list[dict], out_path, seconds: float = 7.0, title: str | None = None, *,
               width: int = 1920, height: int = 1080, fps: int = FPS, route: bool = True) -> Path:
    """A schematic map where pins drop in order, each label types on, and a crimson route joins them.

    `points`: [{"name": "Des Plaines", "x": 0.30, "y": 0.38}], with x and y as fractions of the
    panel (as `graphics.map_card` takes them), in the order they are named. No tiles, no network.
    """
    if not points:
        raise ValueError("a map needs at least one point")
    if not 2.0 <= seconds <= 30.0:
        raise ValueError(f"a map runs 2-30 s, not {seconds}")
    size = (width * SUPERSAMPLE, height * SUPERSAMPLE)
    h = size[1]
    base = _map_base(size)
    spots = [map_place(p, size) for p in points]
    radius = h * 0.0095
    dot = _dot_patch(radius)
    glow = _glow_patch(radius)
    label_font = graphics.font(graphics.LABEL, max(6, round(h * 0.025)))
    names = [str(p.get("name", "")) for p in points]
    widths = [label_font.getlength(name) for name in names]
    text_h = graphics.line_height(label_font, 1.0)
    panel = (MAP_PANEL[0] * size[0] + h * 0.01, MAP_PANEL[1] * size[1] + h * 0.01,
             MAP_PANEL[2] * size[0] - h * 0.07, MAP_PANEL[3] * size[1] - h * 0.01)
    corners = label_spots(spots, widths, text_h, radius, panel)
    plan = map_schedule(len(points), seconds)
    # The title is fixed to the screen, not the map: the map moves under it.
    title_font = graphics.font(graphics.LABEL, max(6, round(height * 0.024)))
    tracking = height * 0.0055
    title_text = str(title).upper() if title else ""
    title_x, title_y = MAP_PANEL[0] * width, height * 0.085
    shade = _top_shade(width, height) if title_text else None
    route_w = max(1.0, h * 0.0026)

    # the frame is centred on the points and pushes in very slowly
    cx = sum(x for x, _ in spots) / len(spots) * 0.6 + size[0] / 2 * 0.4
    cy = sum(y for _, y in spots) / len(spots) * 0.6 + size[1] / 2 * 0.4
    total = frame_count(seconds, fps)
    with FrameWriter(out_path, width, height, fps, frames=total) as out:
        for n in range(total):
            t = n / fps
            canvas = base.copy()
            draw = ImageDraw.Draw(canvas)
            if route:
                for i in range(1, len(points)):
                    step = plan[i]
                    _route(canvas, spots[i - 1], spots[i],
                           ease_in_out_sine(progress(t, step["route_from"], step["land"])), route_w)
            for i, (step, (x, y)) in enumerate(zip(plan, spots)):
                if t < step["drop_from"]:
                    continue
                p = progress(t, step["drop_from"], step["land"])
                fall = (1 - ease_out_back(p, 1.6)) * h * 0.045
                alpha = clamp(p / 0.35)
                since = t - step["land"]
                if 0 <= since < 0.8:
                    q = since / 0.8
                    _ring(canvas, (x, y), radius * (1.2 + 3.2 * ease_out_cubic(q)), 0.55 * (1 - q), h * 0.0016)
                if since > 0:  # a landed pin sits in a faint crimson glow that comes up as it lands
                    halo = _fade(glow, clamp(since / 0.5))
                    canvas.paste(halo, (round(x - halo.width / 2), round(y - halo.height / 2)), halo)
                patch = dot if alpha >= 1 else _fade(dot, alpha)
                canvas.paste(patch, (round(x - patch.width / 2), round(y - fall - patch.height / 2)), patch)
                count = int(max(0.0, t - step["label_from"]) / LABEL_CHAR_S)
                if count > 0:
                    lx, ly = corners[i]
                    draw.text((lx, ly), names[i][:count], font=label_font, fill=FOREGROUND)
            zoom = 1.0 + MAP_PUSH * ease_in_out_sine(t / seconds)
            shot = shoot(canvas, view_box(cx, cy, zoom, size), (width, height))
            _north(ImageDraw.Draw(shot), width, height)
            if title_text:
                shot.paste((0, 0, 0), (0, 0), shade)
                screen = ImageDraw.Draw(shot)
                rule_p = ease_out_cubic(progress(t, 0.0, 0.5))
                screen.rectangle((title_x, title_y - height * 0.014, title_x + height * 0.06 * rule_p,
                                  title_y - height * 0.014 + max(1, round(height * 0.003)) - 1), fill=ACCENT)
                shown = title_text[:int(max(0.0, t - 0.15) / 0.028)]
                graphics.draw_tracked(screen, (title_x, title_y), shown, title_font, MUTED, tracking)
            out.write(shot)
    return Path(out_path)


def _top_shade(width: int, height: int) -> Image.Image:
    """A soft dark band at the top of the frame that keeps a fixed title clear of the map under it."""
    column = Image.new("L", (1, height), 0)
    for y in range(height):
        column.putpixel((0, y), round(215 * (1 - smoothstep(y / (height * 0.19)))))
    return column.resize((width, height))


def _fade(patch: Image.Image, alpha: float) -> Image.Image:
    out = patch.copy()
    out.putalpha(patch.getchannel("A").point(lambda a: round(a * alpha)))
    return out


# --- typed stamp ----------------------------------------------------------------

TYPE_FROM = 0.55        # the first key lands just after the overlay has faded in (0.3 + 0.4 s)
DATE_CPS = 18.0         # date characters per second
PLACE_CPS = 40.0        # the place line follows, quicker
TYPE_PAUSE_S = 0.14     # between the date and the place
CARET_PERIOD_S = 1.0    # an idle caret blinks once a second, half on and half off


def typing_plan(date_len: int, place_len: int, finish_by: float, type_from: float = TYPE_FROM) -> dict:
    """The typing speeds and times for a stamp that must be fully typed by `finish_by`."""
    date_cps, place_cps, pause = DATE_CPS, PLACE_CPS, TYPE_PAUSE_S
    natural = date_len / date_cps + (pause + place_len / place_cps if place_len else 0)
    room = max(0.2, finish_by - type_from)
    if natural > room:
        squeeze = natural / room
        date_cps, place_cps, pause = date_cps * squeeze, place_cps * squeeze, pause / squeeze
    date_done = type_from + date_len / date_cps
    place_from = date_done + (pause if place_len else 0)
    return {"type_from": type_from, "date_cps": date_cps, "date_done": date_done,
            "place_from": place_from, "place_cps": place_cps,
            "done": place_from + (place_len / place_cps if place_len else 0),
            "date_len": date_len, "place_len": place_len}


def typed_state(t: float, plan: dict) -> tuple[int, int, str | None]:
    """What the stamp shows at time t: (date characters, place characters, caret line or None).

    Before typing begins the caret blinks at the start of the date. While a line types, the
    caret is solid at its end. When everything is typed the caret blinks after the last character.
    """
    eps = 1e-9
    date = int(clamp((t - plan["type_from"]) * plan["date_cps"] + eps, 0, plan["date_len"]))
    place = int(clamp((t - plan["place_from"]) * plan["place_cps"] + eps, 0, plan["place_len"])) \
        if t >= plan["place_from"] else 0
    typing = plan["type_from"] <= t < plan["done"]
    line = "place" if plan["place_len"] and t >= plan["place_from"] else "date"
    if typing:
        return date, place, line
    phase = (t if t < plan["type_from"] else t - plan["done"]) % CARET_PERIOD_S
    return date, place, line if phase < CARET_PERIOD_S / 2 else None


def keystrokes(plan: dict) -> list[float]:
    """When each key lands, for typewriter sounds laid under the stamp."""
    out = [plan["type_from"] + (k + 1) / plan["date_cps"] for k in range(plan["date_len"])]
    out += [plan["place_from"] + (k + 1) / plan["place_cps"] for k in range(plan["place_len"])]
    return [round(t - 1e-6, 4) for t in out]


def typed_stamp(text: str, subtext: str | None, out_path, seconds: float = 4.6, *, width: int = 1920,
                height: int = 1080, fps: int = FPS, finish_by: float | None = None,
                type_from: float = TYPE_FROM) -> Path:
    """The lower-left date stamp, typing itself out with a caret, as a transparent qtrle .mov.

    Laid out exactly like `graphics.overlay_image("stamp", ...)`, so the finished frame matches
    the static stamp. Assembly applies its own alpha fades in and out over this. Typing is done
    by `finish_by` (default 55% of `seconds`), sped up if the text is long.
    """
    if seconds <= 0:
        raise ValueError("a typed stamp needs a positive length")
    date_len, place_len = graphics.stamp_text_length(text, subtext, width, height)
    plan = typing_plan(date_len, place_len, finish_by if finish_by is not None else seconds * 0.55, type_from)
    cloud = graphics.stamp_cloud(text, subtext, width, height)
    cache: dict = {}
    total = frame_count(seconds, fps)
    with FrameWriter(out_path, width, height, fps, alpha=True, frames=total) as out:
        for n in range(total):
            state = typed_state(n / fps, plan)
            if state not in cache:
                date, place, caret = state
                cache[state] = graphics.stamp_frame(text, subtext, width, height, date_chars=date,
                                                    place_chars=place, caret=caret, cloud=cloud).tobytes()
            out.write_bytes(cache[state])
    return Path(out_path)


# --- the owner's demo -----------------------------------------------------------

def demo(out_dir) -> None:
    root = Path(__file__).resolve().parents[2]
    library = root / "data" / "video" / "gacy" / "library"
    out_dir = Path(out_dir)
    began = time.monotonic()
    board = evidence_board(
        [{"image": library / "c0011.jpg", "label": "John Wayne Gacy", "sub": "Arrested Dec 21, 1978"},
         {"image": library / "c0035.png", "label": "Robert Piest", "sub": "15, missing Dec 11, 1978"},
         {"image": library / "c0007.jpg", "label": "8213 W. Summerdale", "sub": "Norwood Park Township"},
         {"image": library / "n0039.jpg", "label": "P.D.M. Contractors", "sub": "Gacy's company"},
         {"image": library / "n0037.png", "label": "Des Plaines Police", "sub": "The investigators"}],
        [("John Wayne Gacy", "Robert Piest"), ("John Wayne Gacy", "8213 W. Summerdale"),
         ("John Wayne Gacy", "P.D.M. Contractors"), ("Des Plaines Police", "John Wayne Gacy")],
        out_dir / "evidence_board.mp4", seconds=10.0)
    middle = time.monotonic()
    print(f"{board}  ({middle - began:.1f}s)")
    chart = map_reveal(
        [{"name": "Des Plaines", "x": 0.30, "y": 0.38},
         {"name": "Nisson Pharmacy", "x": 0.33, "y": 0.42},
         {"name": "8213 W. Summerdale", "x": 0.52, "y": 0.47},
         {"name": "Des Plaines River", "x": 0.40, "y": 0.70}],
        out_dir / "map_reveal.mp4", seconds=8.0, title="Cook County, Illinois")
    late = time.monotonic()
    print(f"{chart}  ({late - middle:.1f}s)")
    stamp = typed_stamp("December 11, 1978", "Des Plaines, Illinois", out_dir / "typed_stamp.mov")
    print(f"{stamp}  ({time.monotonic() - late:.1f}s)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        demo(sys.argv[2] if len(sys.argv) > 2 else
             Path(__file__).resolve().parents[2] / "data" / "video" / "samples" / "motion")
    else:
        sys.exit(__doc__)

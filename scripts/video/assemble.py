"""Cut the picture: treated stills, dissolves and dips locked to the narration, one graded look.

    .venv/bin/python scripts/video/assemble.py <shots.json> <out.mp4> [--subtitles burn|sidecar|none]
                                               [--limit N] [--start-index N] [--jobs N]
                                               [--keep-intermediates]

Rendered with the isolated ffmpeg 9 build (`/opt/miniconda3/envs/video/bin/`, overridable with
VIDEO_FFMPEG and VIDEO_FFPROBE), which has xfade, libass, gblur and noise. The voice pipeline's
ffmpeg 4.2.2 is never used here: its loudness behaviour is tuned and must stay untouched.

Three passes, each resumable:

1. Shots. Every shot renders to its own silent intermediate in parallel: the still is treated
   (`fill`, `print` or `pan`), graded, vignetted, given its overlay and any fade to or from black.
   A `media: "clip"` shot runs a stretch of real footage instead (trimmed from `clip_in`, cropped
   to fill the frame, at the film's frame rate, its own sound dropped) with the same grade,
   vignette, overlay and fades; a `media: "motion"` shot runs a rendered motion graphic.
   A shot followed by a dissolve carries that dissolve's length as an extra tail, so the incoming
   picture can start appearing exactly at its own planned start. Intermediates are named by a hash
   of everything that shapes them, so a changed shot re-renders and an unchanged one never does.
2. Groups. Shots are joined twenty at a time with `xfade` (dissolves) and `concat` (cuts and dips,
   whose fades are already inside the clips), at offsets equal to the planned cut times. Grain and
   any burned subtitles go on here, in the final-quality encode. Each group file covers exactly its
   own span of the timeline: a dissolve across a group boundary is rendered wholly inside the later
   group, from the earlier group's last intermediate trimmed to its tail, and the earlier group is
   trimmed to its last planned frame. So the group files abut frame-exactly.
3. Film. The group files are joined with the concat demuxer as a stream copy, and the narration is
   laid under. The length equals the narration's to within a frame.

The passes overlap: shots render in timeline order and each group joins as soon as its shots
exist. Scratch files are deleted as they are used up (see `render`'s `keep_intermediates`).

Footage that runs short. A clip shot needs its frames plus any dissolve tail from the source,
from `clip_in` to `clip_out` (or the end of the file). If the source is short, the clip is slowed
by up to 10%. If it is still short by no more than CLIP_REVERSE_MAX_S of screen time, the rest is
filled by running the last moments of the clip backwards from its end (a short tail only, so the
`reverse` filter never buffers more than a second of frames). Anything shorter than that is an
error the planner must avoid: it is reported for every offending shot before a single frame is
rendered. A clip never freezes. A motion graphic shorter than its shot holds its last frame,
which is the finished graphic.

Everything that builds an ffmpeg command or plans the cut is a pure function, tested without
rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import graphics  # noqa: E402
import motion  # noqa: E402

# --- tools ------------------------------------------------------------------

DEFAULT_FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
DEFAULT_FFPROBE = "/opt/miniconda3/envs/video/bin/ffprobe"


def resolve_tools(env=None) -> tuple[str, str]:
    """The ffmpeg and ffprobe this edit runs: the conda ffmpeg 9 unless the environment says so."""
    env = os.environ if env is None else env
    return env.get("VIDEO_FFMPEG") or DEFAULT_FFMPEG, env.get("VIDEO_FFPROBE") or DEFAULT_FFPROBE


FFMPEG, FFPROBE = resolve_tools()


def require_tools(ffmpeg: str | None = None, ffprobe: str | None = None) -> None:
    """Fail loudly when the ffmpeg 9 build is missing; never fall back to another ffmpeg."""
    for name, path in (("ffmpeg", ffmpeg or FFMPEG), ("ffprobe", ffprobe or FFPROBE)):
        if not (Path(path).is_file() and os.access(path, os.X_OK)):
            raise RuntimeError(
                f"{name} not found at {path}. The edit needs the isolated ffmpeg 9 build (xfade, libass, "
                f"gblur, noise) from the conda env at {Path(DEFAULT_FFMPEG).parent}; install it there or "
                f"point VIDEO_FFMPEG / VIDEO_FFPROBE at it. The voice pipeline's ffmpeg 4.2.2 is "
                f"deliberately not used.")


# --- constants: frame, motion, treatments -----------------------------------

BACKDROP = "0x0A0A0A"  # the film's ground (#0A0A0A): pan margins, missing pictures
DRAWN_KINDS = {"card", "chart", "map"}  # drawn in the film's palette already: never graded
SUPERSAMPLE = 2  # stills are composed at twice the frame size, so slow motion does not stair-step
ZOOM_TRAVEL = 0.10  # a Ken Burns move crosses no more than this much of the frame over a shot
FILL_TOLERANCE = 0.16  # aspect within this of 16:9 fills; 3:2 (35 mm, the commonest photo) sits inside
PRINT_FIT = 0.84  # a print fits within this much of the frame height (or width, if wider than 16:9)
PRINT_PUSH = 0.05  # the print's push-in over the whole shot: very slow
PRINT_BORDER = "0xE9E6DF"  # the print's thin off-white border
PRINT_DARKEN = 0.30  # the blurred ground behind a print, as a fraction of its own brightness
# A small archival scan is shown smaller, not blown up: a 181x255 newspaper portrait stretched to the
# full print size turns to mush. On screen a print is never more than this many times its own pixels.
PRINT_MAX_UPSCALE = 2.2
# With a name card or a date stamp on screen, a print moves into the frame's right-hand part, so the
# lower-left text never runs into its border: this fraction of the width is kept clear for the text.
PRINT_OVERLAY_CLEAR = 0.40
PRINT_RIGHT_MARGIN = 0.035
PAN_FIT = 0.90  # a document fills this much of the frame width (or height, for a wide one)
DOCUMENT_KINDS = {"document", "newspaper"}

# --- constants: moving pictures -----------------------------------------------

MEDIA = ("photo", "clip", "motion")
CLIP_MAX_STRETCH = 1.10  # footage may be slowed by at most 10% to cover a shot
CLIP_REVERSE_MAX_S = 1.0  # screen seconds a clip may run backwards from its end to cover a shot
CLIP_GUARD_FRAMES = 3  # frame-rate rounding can leave a clip a frame or two short; never more

# --- constants: the cut -----------------------------------------------------

DISSOLVE_S = 0.6
BLACK_S = 1.0  # a whole dip: half down to black before the cut, half up after it
FINAL_FADE_S = 1.0
GROUP_SIZE = 20  # shots per joining graph: enough to amortise, few enough to stay robust
JOBS = int(os.getenv("VIDEO_JOBS", "5"))

INTERMEDIATE_PRESET, INTERMEDIATE_CRF = "veryfast", 14  # fast and near-transparent
# Full quality is the owner's standing choice (2026-10-14): the final encode is not lowered to save disk.
FINAL_PRESET, FINAL_CRF, FINAL_TUNE = os.getenv("VIDEO_FINAL_PRESET", "medium"), 19, "film"
DURATION_TOLERANCE_S = 0.1
# Stills arrive full-range (JPEG) or RGB (PNG); everything leaves as tagged BT.709 limited range, so
# no two shots disagree about where black is and players read the film the same way.
TO_BT709 = "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p"
COLOUR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
               "-color_range", "tv"]

# --- constants: overlays and subtitles -----------------------------------------

OVERLAY_IN_AT, OVERLAY_IN_S = 0.3, 0.4
OVERLAY_OUT_BY, OVERLAY_OUT_S = 3.5, 0.3
TYPED_OUT_BY = 4.6  # a stamp that types itself stays up longer, so the finished date can be read
TYPED_HOLD_S = 1.6  # and is fully typed this long before it fades
OVERLAY_MIN_HOLD_S = 1.0  # a shot that cannot hold its overlay this long gets none
OVERLAY_END_MARGIN_S = 0.15

SUBTITLE_CHARS = 42  # characters per subtitle line, two lines at most: the broadcast norm
SUBTITLE_FONT = "Helvetica Neue"

# --- the look ---------------------------------------------------------------
#
# One look for the whole film: pictures come from a dozen places and a dozen decades, and without a
# common grade the cut reads as a slideshow. Cold, lightly desaturated (a colour photograph must
# still read as colour beside the black-and-white archive), with the blacks crushed: the toe of
# the curve sits under the diagonal so shadows sink into the ground. Black-and-white photographs are
# forced to true greyscale first, then given only a faint cool tone, so an old print never picks up
# a colour cast from the colour grade. Grain goes on last, over everything, in the final encode.

COLOUR_GRADE = (
    "eq=contrast=1.06:saturation=0.85,"
    "colorbalance=rs=-0.03:gs=-0.01:bs=0.045:rm=-0.02:bm=0.02:rh=-0.01:bh=0.005,"
    "curves=all='0/0 0.06/0.015 0.25/0.22 0.5/0.50 0.8/0.81 1/0.96'"
)
BW_GRADE = (
    "format=gray,format=rgb24,"
    "curves=all='0/0 0.06/0.015 0.25/0.22 0.5/0.50 1/0.97',"
    "colorbalance=rs=-0.015:bs=0.025:rm=-0.01:bm=0.018"
)
VIGNETTE = "vignette=angle=PI/7:eval=init"
GRAIN = "noise=c0s=4:c0f=t"  # luma only, renewed every frame: seen on a dark wall at 100%, not at a glance


def grade_filter(mode: str) -> str:
    """The colour treatment for a grade mode: 'colour', 'bw', or 'none' for drawn shots."""
    return {"colour": COLOUR_GRADE, "bw": BW_GRADE}.get(mode, "")


def is_monochrome(path) -> bool:
    """True when a picture carries no real colour: a scan of a black-and-white print."""
    with Image.open(path) as image:
        thumb = image.convert("RGB").resize((48, 48))
    return monochrome_pixels(thumb.tobytes())


def monochrome_pixels(rgb: bytes) -> bool:
    """True when packed RGB pixels carry no real colour (mean channel spread under 6 levels)."""
    pixels = [rgb[i:i + 3] for i in range(0, len(rgb) - 2, 3)]
    if not pixels:
        return False
    return sum(max(px) - min(px) for px in pixels) / len(pixels) < 6


def media_of(shot: dict) -> str:
    """What a shot shows: `photo` (the default), `clip` (footage) or `motion` (a motion graphic)."""
    media = (shot.get("media") or "photo").lower()
    return media if media in MEDIA else "photo"


def grade_mode(shot: dict, monochrome: bool = False) -> str:
    """'none' for drawn shots; otherwise the shot's own `colour`, else what the picture looks like."""
    if shot.get("kind") in DRAWN_KINDS or media_of(shot) == "motion":
        return "none"
    colour = shot.get("colour")
    if colour in ("bw", "colour"):
        return colour
    return "bw" if monochrome else "colour"


# --- treatments -------------------------------------------------------------

def _even(value: float) -> int:
    return 2 * round(value / 2)


def auto_treatment(width: int, height: int, kind_hint: str | None = None) -> str:
    """Documents pan; pictures near 16:9 fill; portrait, square, 4:3 and very wide ones are prints."""
    if kind_hint in DOCUMENT_KINDS:
        return "pan"
    ratio = (width / height) / (16 / 9)
    return "fill" if abs(ratio - 1) <= FILL_TOLERANCE else "print"


def choose_treatment(shot: dict, size: tuple[int, int]) -> str:
    """The shot's own treatment if it names one, else the rule. Drawn shots always fill."""
    if shot.get("kind") in DRAWN_KINDS:
        return "fill"
    if shot.get("treatment") in ("fill", "print", "pan"):
        return shot["treatment"]
    return auto_treatment(size[0], size[1], shot.get("kind_hint"))


def motion_expressions(motion: str, frames: int) -> tuple[str, str, str]:
    """The zoompan z, x and y expressions for a motion, as functions of the output frame number."""
    progress = f"on/{frames - 1}" if frames > 1 else "0"
    centre_x, centre_y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    held = f"{1.0 + ZOOM_TRAVEL:.4f}"
    if motion == "zoom_in":
        return f"{1.0:.4f}+{ZOOM_TRAVEL:.4f}*{progress}", centre_x, centre_y
    if motion == "zoom_out":
        return f"{1.0 + ZOOM_TRAVEL:.4f}-{ZOOM_TRAVEL:.4f}*{progress}", centre_x, centre_y
    if motion == "pan_left":  # the frame starts at the right edge and walks left
        return held, f"(iw-iw/zoom)*(1-{progress})", centre_y
    if motion == "pan_right":
        return held, f"(iw-iw/zoom)*({progress})", centre_y
    if motion == "pan_up":  # the frame starts at the bottom and rises
        return held, centre_x, f"(ih-ih/zoom)*(1-{progress})"
    if motion == "pan_down":
        return held, centre_x, f"(ih-ih/zoom)*({progress})"
    return f"{1.0:.4f}", "0", "0"  # still, and anything unrecognised


def fill_chain(motion: str, frames: int, width: int, height: int, fps: int) -> str:
    """Cover the frame (cropping the overhang) and move slowly; one still in, `frames` frames out."""
    box_w, box_h = _even(width * SUPERSAMPLE), _even(height * SUPERSAMPLE)
    zoom, pan_x, pan_y = motion_expressions(motion or "still", frames)
    return (f"scale={box_w}:{box_h}:force_original_aspect_ratio=increase,crop={box_w}:{box_h},"
            f"zoompan=z='{zoom}':x='{pan_x}':y='{pan_y}':d={frames}:s={width}x{height}:fps={fps},"
            f"setsar=1")


def print_chain(src_w: int, src_h: int, frames: int, width: int, height: int, fps: int,
                src: str = "[src]", out: str = "[pic]", clear_left: float = 0.0) -> str:
    """The whole photograph as an object: centred, bordered, shadowed, over a blurred copy of itself.

    The ground is the same picture scaled to cover the frame, heavily blurred and darkened. The
    print is fitted inside PRINT_FIT of the frame (the source shape decides whether height or width
    binds), never cropped, with a thin off-white border and a soft shadow falling down and right.
    Everything is composed once, at twice the frame size, then a very slow push-in moves the whole.
    A small source is held to PRINT_MAX_UPSCALE of its own size, so it sits smaller in the frame
    rather than being stretched soft.
    """
    box_w, box_h = _even(width * SUPERSAMPLE), _even(height * SUPERSAMPLE)
    fit_w, fit_h = _even(box_w * PRINT_FIT), _even(box_h * PRINT_FIT)
    if src_w > 0 and src_h > 0:
        fit_w = min(fit_w, _even(src_w * PRINT_MAX_UPSCALE * SUPERSAMPLE))
        fit_h = min(fit_h, _even(src_h * PRINT_MAX_UPSCALE * SUPERSAMPLE))
    # The print is centred in the part of the frame left free: all of it, or the right-hand part
    # when an overlay holds the lower left.
    region_x = _even(box_w * clear_left) if clear_left > 0 else 0
    region_w = box_w - region_x - (_even(box_w * PRINT_RIGHT_MARGIN) if clear_left > 0 else 0)
    fit_w = min(fit_w, _even(region_w * 0.96))
    place_x = f"{region_x}+({region_w}-w)/2"
    border = max(2, round(box_h * 0.0028))
    blur = round(box_h * 0.036)
    soft = max(2, round(box_h * 0.012))
    drop = round(box_h * 0.007)
    progress = f"on/{frames - 1}" if frames > 1 else "0"
    return ";".join([
        f"{src}split=2[bg0][fg0]",
        f"[fg0]scale={fit_w}:{fit_h}:force_original_aspect_ratio=decrease:force_divisible_by=2,"
        f"pad=iw+{2 * border}:ih+{2 * border}:{border}:{border}:color={PRINT_BORDER}[print]",
        "[print]split=2[p1][p2]",
        f"[p2]format=rgba,colorchannelmixer=rr=0:gg=0:bb=0:aa=0.78,"
        f"pad=iw+{4 * soft}:ih+{4 * soft}:{2 * soft}:{2 * soft}:color=black@0,gblur=sigma={soft}[shadow]",
        f"[bg0]scale={box_w}:{box_h}:force_original_aspect_ratio=increase,crop={box_w}:{box_h},"
        f"gblur=sigma={blur},"
        f"colorchannelmixer=rr={PRINT_DARKEN}:gg={PRINT_DARKEN}:bb={PRINT_DARKEN}[bg]",
        f"[bg][shadow]overlay={place_x}+{drop // 2}:(H-h)/2+{drop}[bgs]",
        f"[bgs][p1]overlay={place_x}:(H-h)/2,"
        f"zoompan=z='{1.0:.4f}+{PRINT_PUSH:.4f}*{progress}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={width}x{height}:fps={fps},setsar=1{out}",
    ])


def pan_chain(src_w: int, src_h: int, frames: int, width: int, height: int, fps: int) -> str:
    """A document read by the camera: fitted across the frame and walked from top to bottom.

    A page at or under 16:9 is fitted to PAN_FIT of the frame width over the dark ground and
    panned top to bottom; anything wider is fitted to the height and panned left to right. The
    walk covers the whole item over the shot, margins included, so the edges are seen too.
    """
    box_w, box_h = _even(width * SUPERSAMPLE), _even(height * SUPERSAMPLE)
    progress = f"n/{frames - 1}" if frames > 1 else "0"
    if src_w / src_h > width / height:
        margin = round(box_w * (1 - PAN_FIT) / 2)
        fit = f"scale=-2:{_even(box_h * PAN_FIT)}"
        canvas = (f"pad=w='max(iw+{2 * margin},{box_w})':h={box_h}:x='(ow-iw)/2':y='(oh-ih)/2'"
                  f":color={BACKDROP}")
        move = f"crop={box_w}:{box_h}:x='(iw-ow)*{progress}':y=0"
    else:
        margin = round(box_h * (1 - PAN_FIT) / 2)
        fit = f"scale={_even(box_w * PAN_FIT)}:-2"
        canvas = (f"pad=w={box_w}:h='max(ih+{2 * margin},{box_h})':x='(ow-iw)/2':y='(oh-ih)/2'"
                  f":color={BACKDROP}")
        move = f"crop={box_w}:{box_h}:x=0:y='(ih-oh)*{progress}'"
    return (f"{fit},{canvas},loop=loop={frames - 1}:size=1,settb=1/{fps},setpts=N,fps={fps},{move},"
            f"scale={width}:{height},setsar=1")


# --- overlays ---------------------------------------------------------------

def overlay_fades(duration: float, out_by: float = OVERLAY_OUT_BY) -> str | None:
    """Fade in over 0.4 s from 0.3 s, gone by 3.5 s; earlier if the shot ends sooner, none if too short."""
    gone = min(out_by, duration - OVERLAY_END_MARGIN_S)
    out_start = gone - OVERLAY_OUT_S
    if out_start - (OVERLAY_IN_AT + OVERLAY_IN_S) < OVERLAY_MIN_HOLD_S:
        return None
    return (f"fade=t=in:st={OVERLAY_IN_AT:.3f}:d={OVERLAY_IN_S:.3f}:alpha=1,"
            f"fade=t=out:st={out_start:.3f}:d={OVERLAY_OUT_S:.3f}:alpha=1")


def overlay_gone(duration: float, out_by: float = OVERLAY_OUT_BY) -> float:
    """When an overlay has fully faded out, in shot seconds."""
    return min(out_by, duration - OVERLAY_END_MARGIN_S)


def is_animated(card: dict | None) -> bool:
    """An overlay that types itself out: a stamp with `"animate": true`. Names never animate."""
    return bool(card) and bool(card.get("animate")) and (card.get("style") or "stamp") == "stamp"


# --- moving pictures: footage and motion graphics -------------------------------

class ClipTooShort(ValueError):
    """The footage cannot cover the shot without freezing: the planner must pick a longer segment."""


def clip_timing(needed_s: float, clip_in: float, clip_out: float | None, source_s: float | None) -> dict:
    """How a stretch of footage covers `needed_s` of screen time without ever freezing.

    The usable source runs from `clip_in` to `clip_out` (capped at the file's length). If it is long
    enough it plays at speed. If not, it is slowed (setpts) by at most CLIP_MAX_STRETCH. If it is
    still short by no more than CLIP_REVERSE_MAX_S of screen time, the remainder is the clip's own
    last moments run backwards from its end. Otherwise it raises ClipTooShort.

    Returns `start` (source seconds to seek to), `forward` (source seconds played forwards),
    `reverse` (source seconds played backwards from the end of the forward run) and `stretch`
    (the setpts factor, 1.0 to 1.1). Screen time is (forward + reverse) * stretch.
    """
    start = max(0.0, float(clip_in or 0.0))
    end = None
    for limit in (clip_out, source_s):
        if limit is not None:
            end = float(limit) if end is None else min(end, float(limit))
    if end is None:
        raise ValueError("a clip needs clip_out or a known source length")
    available = end - start
    if available <= 0:
        raise ClipTooShort(f"clip_in {start:.2f}s is at or past the end of the usable source ({end:.2f}s)")
    if available >= needed_s:
        return {"start": start, "forward": needed_s, "reverse": 0.0, "stretch": 1.0}
    if available * CLIP_MAX_STRETCH >= needed_s:
        return {"start": start, "forward": available, "reverse": 0.0, "stretch": needed_s / available}
    short = needed_s - available * CLIP_MAX_STRETCH
    reverse = short / CLIP_MAX_STRETCH
    if short > CLIP_REVERSE_MAX_S + 1e-9 or reverse > available:
        raise ClipTooShort(
            f"needs {needed_s:.2f}s but {start:.2f}-{end:.2f}s gives {available:.2f}s "
            f"({available * CLIP_MAX_STRETCH:.2f}s slowed by 10%); {short:.2f}s short, and at most "
            f"{CLIP_REVERSE_MAX_S:.1f}s may run backwards")
    return {"start": start, "forward": available, "reverse": reverse, "stretch": CLIP_MAX_STRETCH}


def motion_timing(needed_s: float, clip_in: float, source_s: float | None) -> dict:
    """A motion graphic plays at speed from `clip_in`; if it is shorter than the shot, its last
    (finished) frame holds."""
    start = max(0.0, float(clip_in or 0.0))
    available = needed_s if source_s is None else max(0.0, float(source_s) - start)
    return {"start": start, "forward": min(needed_s, available) if available > 0 else needed_s,
            "reverse": 0.0, "stretch": 1.0, "hold": True}


def moving_inputs(path, timing: dict) -> list[str]:
    """The ffmpeg inputs for a moving shot: the forward run, then the tail to reverse if any."""
    argv = ["-ss", f"{timing['start']:.3f}", "-t", f"{timing['forward']:.3f}", "-i", str(path)]
    if timing.get("reverse", 0) > 0:
        tail_from = timing["start"] + timing["forward"] - timing["reverse"]
        argv += ["-ss", f"{tail_from:.3f}", "-t", f"{timing['reverse']:.3f}", "-i", str(path)]
    return argv


def moving_chain(timing: dict, mode: str, width: int, height: int, fps: int) -> str:
    """Footage (or a motion graphic) into the frame: [0:v] (and [1:v], the reversed tail) to [pic].

    Deinterlaced where the source is interlaced; the reversed tail drops its first frame, which
    repeats the forward run's last; slowed by `stretch`; resampled to the film's frame rate;
    anamorphic pixels squared; graded at source size (cheaper); scaled to cover and centre-cropped
    to the frame, so 4:3 period footage fills 16:9 with no pillarbox. A few cloned frames guard
    against frame-rate rounding (the frame count is exact downstream); a motion graphic instead
    holds its last frame for as long as the shot needs.
    """
    deint = "bwdif=mode=send_frame:parity=auto:deint=interlaced"
    parts = [f"[0:v]{deint},setpts=PTS-STARTPTS[fwd]"]
    source = "[fwd]"
    if timing.get("reverse", 0) > 0:
        parts.append(f"[1:v]{deint},reverse,trim=start_frame=1,setpts=PTS-STARTPTS[bwd]")
        parts.append("[fwd][bwd]concat=n=2:v=1:a=0[run]")
        source = "[run]"
    grade = grade_filter(mode)
    hold = "stop=-1" if timing.get("hold") else f"stop={CLIP_GUARD_FRAMES}"
    parts.append(
        f"{source}setpts={timing['stretch']:.4f}*PTS,fps={fps},"
        f"scale=trunc(iw*sar/2)*2:ih,setsar=1,format=rgb24{',' + grade if grade else ''},"
        f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={width}:{height},setsar=1,tpad=stop_mode=clone:{hold}[pic]")
    return ";".join(parts)


# --- planning the cut ---------------------------------------------------------

def cut_plan(shots: list[dict], fps: int, *, opens_film: bool = True, closes_film: bool = True) -> list[dict]:
    """Frames, tails and fades for each shot, every cut on the frame nearest its planned time.

    Frames come from rounding the planned cut times, not the durations, so rounding never
    accumulates: a shot's first frame is always round(start * fps). A dissolve adds its length to
    the outgoing shot's tail and nothing else. A dip to black is centred on the cut: the outgoing
    shot fades down over its last half, the incoming one rises from black at its own first frame.
    Transitions shrink to fit a short shot, and one that cannot fit two frames becomes a cut.
    """
    starts = [round(float(s["start"]) * fps) for s in shots]
    ends = starts[1:] + [round(float(shots[-1]["end"]) * fps)]
    plan = [{"index": s.get("index", i + 1), "start_frame": a, "frames": b - a, "tail": 0,
             "fade_in": 0, "fade_out": 0, "transition": "cut", "duration": 0}
            for i, (s, a, b) in enumerate(zip(shots, starts, ends))]
    half_black = round(BLACK_S / 2 * fps)
    for i in range(1, len(plan)):
        before, clip = plan[i - 1], plan[i]
        wanted = (shots[i].get("transition_in") or "dissolve").lower()
        if wanted == "dissolve":
            length = min(round(DISSOLVE_S * fps), before["frames"] // 2, clip["frames"] // 2)
            if length >= 2:
                clip.update(transition="dissolve", duration=length)
                before["tail"] = length
        elif wanted == "black":
            length = min(half_black, before["frames"] // 3, clip["frames"] // 3)
            if length >= 2:
                clip.update(transition="black", duration=2 * length)
                before["fade_out"], clip["fade_in"] = length, length
    first, last = plan[0], plan[-1]
    if opens_film and (shots[0].get("transition_in") or "dissolve").lower() in ("dissolve", "black"):
        length = min(half_black, first["frames"] // 3)
        first["fade_in"] = length if length >= 2 else 0
    if closes_film:
        length = min(round(FINAL_FADE_S * fps), last["frames"] // 2)
        if length >= 2:
            last["fade_out"] = max(last["fade_out"], length)
    return plan


def span_frames(clips: list[dict]) -> int:
    """How many frames of the timeline a run of clips covers: overlaps never add length."""
    return sum(clip["frames"] for clip in clips)


def join_steps(clips: list[dict], fps: int) -> list[dict]:
    """How each clip after the first joins the one before, at its planned time from the run's start."""
    origin = clips[0]["start_frame"]
    steps = []
    for clip in clips[1:]:
        offset = round((clip["start_frame"] - origin) / fps, 3)
        if clip["transition"] == "dissolve":
            steps.append({"kind": "xfade", "offset": offset, "duration": clip["duration"] / fps})
        else:  # a cut, or a dip whose fades are already inside the two clips
            steps.append({"kind": "concat", "offset": offset})
    return steps


def join_filter(clips: list[dict], fps: int, lead: dict | None = None, finish: str = "") -> str:
    """The filtergraph that joins a group's intermediates into exactly its span of the timeline.

    Inputs are the intermediates in order, preceded by the lead-in when there is one: the previous
    group's last shot, trimmed to its tail, which dissolves into this group's first frame. Every
    stream is renumbered frame by frame first, so xfade offsets land on exact frames.
    """
    parts, shift = [], 0
    if lead is not None:
        parts.append(f"[0:v]trim=start_frame={lead['frames']},settb=1/{fps},setpts=N[lead]")
        shift = 1
    for i in range(len(clips)):
        parts.append(f"[{i + shift}:v]settb=1/{fps},setpts=N[v{i}]")
    acc = "v0"
    if lead is not None:
        parts.append(f"[lead][v0]xfade=transition=fade:duration={clips[0]['duration'] / fps:.3f}"
                     f":offset=0.000[j0]")
        acc = "j0"
    for i, step in enumerate(join_steps(clips, fps), start=1):
        if step["kind"] == "xfade":
            parts.append(f"[{acc}][v{i}]xfade=transition=fade:duration={step['duration']:.3f}"
                         f":offset={step['offset']:.3f}[j{i}]")
        else:
            parts.append(f"[{acc}][v{i}]concat=n=2:v=1:a=0,settb=1/{fps},setpts=N[j{i}]")
        acc = f"j{i}"
    tail = f",{finish}" if finish else ""
    parts.append(f"[{acc}]trim=end_frame={span_frames(clips)},setpts=PTS-STARTPTS{tail}[out]")
    return ";".join(parts)


def group_plan(plan: list[dict], size: int = GROUP_SIZE) -> list[dict]:
    """Consecutive groups of `size` clips; a group opening on a dissolve takes a lead-in clip."""
    groups = []
    for at in range(0, len(plan), size):
        clips = plan[at:at + size]
        lead = plan[at - 1] if at and clips[0]["transition"] == "dissolve" else None
        groups.append({"clips": clips, "lead": lead})
    return groups


# --- subtitles --------------------------------------------------------------

def wrap_subtitle(text: str, width: int = SUBTITLE_CHARS) -> str:
    """One line if it fits, else two balanced lines; never a dropped word.

    The break goes where the longer line is shortest, nudged towards punctuation, away from the
    middle of a name (two capitalised words), and towards a bottom-heavy shape. A cue too long for
    two lines of `width` still gets two lines: a wider subtitle beats losing narration.
    """
    words = text.split()
    if len(" ".join(words)) <= width:
        return " ".join(words)
    best, best_score = None, None
    for i in range(1, len(words)):
        top, bottom = " ".join(words[:i]), " ".join(words[i:])
        score = max(len(top), len(bottom)) + (100 if max(len(top), len(bottom)) > width else 0)
        if words[i - 1][-1] in ",.;:!?—":
            score -= 4
        if words[i - 1][:1].isupper() and words[i][:1].isupper():
            score += 8
        if len(top) > len(bottom):
            score += 1
        if best_score is None or score < best_score:
            best, best_score = f"{top}\n{bottom}", score
    return best


def _pack(pieces: list[str], limit: int) -> list[str]:
    """Greedily join pieces with spaces into chunks no longer than `limit` where possible."""
    chunks: list[str] = []
    for piece in pieces:
        if chunks and len(chunks[-1]) + 1 + len(piece) <= limit:
            chunks[-1] = f"{chunks[-1]} {piece}"
        else:
            chunks.append(piece)
    return chunks


def subtitle_cues(entries: list[dict]) -> list[dict]:
    """Aligned narration lines as subtitle cues of at most two short lines each.

    A line too long for two subtitle lines is split at sentences, then clauses, then words, and its
    time is shared out in proportion to the characters, so each cue stays on screen while it is
    being spoken. Two sentences share a cue only if together they fit on one line.
    """
    most = 2 * SUBTITLE_CHARS - 10  # leave the wrap some slack
    cues = []
    for entry in entries:
        text = " ".join(str(entry.get("text", "")).split())
        if not text:
            continue
        if len(text) <= most:
            chunks = [text]
        else:
            chunks = []
            for sentence in _pack(re.split(r"(?<=[.!?…])\s+", text), SUBTITLE_CHARS):
                if len(sentence) <= most:
                    chunks.append(sentence)
                    continue
                for clause in _pack(re.split(r"(?<=[,;:—])\s+", sentence), most):
                    chunks.extend(_pack(clause.split(), most) if len(clause) > most else [clause])
        start, end = float(entry["start"]), float(entry["end"])
        total = sum(len(chunk) for chunk in chunks)
        at = start
        for n, chunk in enumerate(chunks):
            until = end if n == len(chunks) - 1 else at + (end - start) * len(chunk) / total
            cues.append({"start": at, "end": until, "text": chunk})
            at = until
    return cues


def subtitle_font_size(height: int) -> int:
    return round(height * 0.041)


def subtitle_margin(height: int) -> int:
    return round(height * 0.046)


def subtitle_box_top(height: int) -> int:
    """The highest a two-line subtitle box reaches, box padding included."""
    size = subtitle_font_size(height)
    return height - subtitle_margin(height) - 2 * round(size * 1.25) - round(height * 0.011)


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    return f"{hours}:{minutes:02d}:{rest // 100:02d}.{rest % 100:02d}"


def _ass_text(text: str) -> str:
    body = wrap_subtitle(text).replace("\\", "⧵").replace("{", "(").replace("}", ")")
    return body.replace("\n", "\\N")


def subtitle_ass(entries: list[dict], start: float, end: float, width: int, height: int) -> str:
    """An ASS script of the cues touching [start, end), timed from the window's own zero.

    Off-white text on one soft translucent box per cue (libass BorderStyle 4), bottom centre, in
    the lower third and clear of the lower-left overlays.
    """
    size = subtitle_font_size(height)
    pad = round(height * 0.011)
    side = round(width * 0.12)
    header = "\n".join([
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{SUBTITLE_FONT},{size},&H00F2F2F2,&H00F2F2F2,&H70000000,&H70000000,"
        f"0,0,0,0,100,100,0,0,4,{pad},0,2,{side},{side},{subtitle_margin(height)},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ])
    events = []
    for cue in subtitle_cues(entries):
        if cue["end"] <= start or cue["start"] >= end:
            continue
        at, until = max(cue["start"], start) - start, min(cue["end"], end) - start
        events.append(f"Dialogue: 0,{_ass_time(at)},{_ass_time(until)},Default,,0,0,0,,{_ass_text(cue['text'])}")
    return header + "\n" + "\n".join(events) + "\n"


# --- escaping -----------------------------------------------------------------

def escape_filter_path(path) -> str:
    """A file path as a filter option value: protected from the option and the graph splitters."""
    value = str(path).replace("\\", "\\\\")
    for char in ("'", ":"):
        value = value.replace(char, "\\" + char)
    value = value.replace("\\", "\\\\")
    for char in ("'", ",", ";", "[", "]"):
        value = value.replace(char, "\\" + char)
    return value


# --- one shot ---------------------------------------------------------------

def image_size(path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def shot_clip(shot: dict, clip: dict, still, out_path, width: int, height: int, fps: int, *,
              overlay=None) -> list[str]:
    """The ffmpeg argv that renders one shot, tail included, to a silent intermediate.

    A photograph is read once; the treatment expands it to every frame. The grade is applied to the
    still itself (once, not per frame), the vignette to the moving frame, then the overlay, then
    any fade to or from black, so a dip takes the overlay down with the picture. A clip or motion
    shot (`still` is then the footage file) takes its timing from `clip["timing"]` and whether it
    is black-and-white from `clip["mono"]`, both worked out by `render`. An overlay is a still PNG,
    or a .mov with alpha for a stamp that types itself.
    """
    total = clip["frames"] + clip["tail"]
    parts = []
    media = clip.get("media") or media_of(shot)
    if still is not None and media in ("clip", "motion"):
        timing = clip["timing"]
        mode = grade_mode(shot, monochrome=bool(clip.get("mono")))
        source = moving_inputs(still, timing)
        parts.append(moving_chain(timing, mode, width, height, fps))
        parts.append(f"[pic]{VIGNETTE if mode != 'none' else 'null'}[base]")
    elif still is not None:
        still = Path(still)
        size = image_size(still)
        treatment = choose_treatment(shot, size)
        mono = shot.get("colour") is None and shot.get("kind") not in DRAWN_KINDS and is_monochrome(still)
        mode = grade_mode(shot, monochrome=mono)
        source = ["-i", str(still)]
        parts.append(f"[0:v]format=rgb24{',' + grade_filter(mode) if mode != 'none' else ''}[src]")
        if treatment == "print":
            parts.append(print_chain(size[0], size[1], total, width, height, fps,
                                     clear_left=PRINT_OVERLAY_CLEAR if overlay else 0.0))
        elif treatment == "pan":
            parts.append(f"[src]{pan_chain(size[0], size[1], total, width, height, fps)}[pic]")
        else:
            parts.append(f"[src]{fill_chain(shot.get('motion') or 'still', total, width, height, fps)}[pic]")
        parts.append(f"[pic]{VIGNETTE if mode != 'none' else 'null'}[base]")
    else:  # nothing was sourced and nothing to hold: the film's own dark ground
        source = ["-f", "lavfi", "-i", f"color=c={BACKDROP}:s={width}x{height}:r={fps}"]
        parts.append("[0:v]setsar=1[base]")

    animated = overlay is not None and Path(overlay).suffix.lower() == ".mov"
    out_by = TYPED_OUT_BY if animated else OVERLAY_OUT_BY
    fades = overlay_fades(clip["frames"] / fps, out_by) if overlay is not None else None
    current = "base"
    if fades:
        index = source.count("-i")
        if animated:  # plays once from the shot's first frame; gone (faded) before it ends
            source += ["-i", str(overlay)]
            parts.append(f"[{index}:v]format=rgba,setpts=PTS-STARTPTS,{fades}[ov]")
            parts.append("[base][ov]overlay=0:0:eof_action=pass:format=auto[comp]")
        else:
            source += ["-loop", "1", "-framerate", str(fps), "-i", str(overlay)]
            parts.append(f"[{index}:v]format=rgba,{fades}[ov]")
            parts.append("[base][ov]overlay=0:0:shortest=1:format=auto[comp]")
        current = "comp"
    finish = []
    if clip["fade_in"]:
        finish.append(f"fade=t=in:s=0:n={clip['fade_in']}")
    if clip["fade_out"]:  # down to black on the shot's last frame
        finish.append(f"fade=t=out:s={clip['frames'] - clip['fade_out']}:n={clip['fade_out'] - 1}")
    finish.append(TO_BT709)
    parts.append(f"[{current}]{','.join(finish)}[out]")
    return [
        FFMPEG, "-y", "-v", "error", *source,
        "-filter_complex", ";".join(parts), "-map", "[out]", "-frames:v", str(total), "-an",
        "-c:v", "libx264", "-preset", INTERMEDIATE_PRESET, "-crf", str(INTERMEDIATE_CRF),
        "-pix_fmt", "yuv420p", *COLOUR_TAGS, "-r", str(fps), str(out_path),
    ]


def group_command(clips: list[dict], lead: dict | None, out_path, fps: int, subtitles=None) -> list[str]:
    """The ffmpeg argv that joins one group, adds grain and any burned subtitles, at final quality."""
    inputs = []
    for clip in ([lead] if lead is not None else []) + clips:
        inputs += ["-i", str(clip["path"])]
    finish = GRAIN
    if subtitles is not None:
        finish += f",subtitles=filename={escape_filter_path(subtitles)}"
    finish += f",{TO_BT709}"
    return [
        FFMPEG, "-y", "-v", "error", *inputs,
        "-filter_complex", join_filter(clips, fps, lead=lead, finish=finish),
        "-map", "[out]", "-frames:v", str(span_frames(clips)), "-an",
        "-c:v", "libx264", "-preset", FINAL_PRESET, "-crf", str(FINAL_CRF), "-tune", FINAL_TUNE,
        "-pix_fmt", "yuv420p", *COLOUR_TAGS, "-r", str(fps), str(out_path),
    ]


# --- joining it all up ------------------------------------------------------

def concat_list(paths) -> str:
    """The concat demuxer's playlist; its quoting wants '\\'' for an apostrophe in a path."""
    return "".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in paths)


def probe_duration(path) -> float:
    """How long a rendered file really is, according to ffprobe."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def resolve_asset(base: Path, asset) -> Path | None:
    """Shot assets are written relative to shots.json, so that a project folder can be moved."""
    if not asset:
        return None
    path = Path(asset)
    return path if path.is_absolute() else base / path


LOOK = json.dumps([COLOUR_GRADE, BW_GRADE, VIGNETTE, SUPERSAMPLE, ZOOM_TRAVEL, PRINT_FIT, PRINT_PUSH,
                   PRINT_BORDER, PRINT_DARKEN, PRINT_MAX_UPSCALE, PRINT_OVERLAY_CLEAR, PRINT_RIGHT_MARGIN,
                   PAN_FIT, INTERMEDIATE_PRESET, INTERMEDIATE_CRF, TO_BT709, CLIP_GUARD_FRAMES,
                   TYPED_OUT_BY, 5])


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _file_key(path) -> list | None:
    if path is None:
        return None
    stat = Path(path).stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def shot_key(shot: dict, clip: dict, still, overlay, width: int, height: int, fps: int) -> list:
    """Everything that shapes a shot's intermediate: change any of it and the shot re-renders.

    The picture or footage file (path, size, modification time), the shot fields that steer its
    treatment, the footage range (`media`, `clip_in`, `clip_out`) and the timing worked out from
    it, its frames, tail and fades, the overlay file, the frame, and the look.
    """
    return [_file_key(still),
            {k: shot.get(k) for k in ("kind", "treatment", "kind_hint", "motion", "colour")},
            {"media": clip.get("media") or media_of(shot), "clip_in": shot.get("clip_in"),
             "clip_out": shot.get("clip_out"), "timing": clip.get("timing"), "mono": clip.get("mono")},
            {k: clip[k] for k in ("frames", "tail", "fade_in", "fade_out")}, _file_key(overlay),
            width, height, fps, LOOK]


def make_overlay(card: dict, duration: float, folder: Path, width: int, height: int, fps: int) -> Path:
    """The overlay file for a shot, drawn once and reused: a still PNG, or for a stamp with
    `"animate": true` a .mov in which the date types itself out before it is read."""
    graphics_key = _file_key(graphics.__file__)
    if is_animated(card):
        gone = overlay_gone(duration, TYPED_OUT_BY)
        finish_by = max(motion.TYPE_FROM + 0.5, gone - OVERLAY_OUT_S - TYPED_HOLD_S)
        path = folder / f"{_digest([card, width, height, fps, round(gone, 3), graphics_key, _file_key(motion.__file__)])}.mov"
        if not path.exists():
            graphics.overlay_image("stamp", card.get("text"), card.get("subtext"), width, height)  # validates
            motion.typed_stamp(card.get("text"), card.get("subtext"), path, seconds=gone, width=width,
                               height=height, fps=fps, finish_by=finish_by)
        return path
    path = folder / f"{_digest([card, width, height, graphics_key])}.png"
    if not path.exists():
        graphics.save_overlay(graphics.overlay_image(
            card.get("style"), card.get("text"), card.get("subtext"), width, height), path)
    return path


_DURATIONS: dict = {}


def probe_duration_cached(path) -> float | None:
    """A source file's length, probed once per file version; None if ffprobe cannot say."""
    key = tuple(_file_key(path))
    if key not in _DURATIONS:
        try:
            _DURATIONS[key] = probe_duration(path)
        except (subprocess.CalledProcessError, ValueError):
            _DURATIONS[key] = None
    return _DURATIONS[key]


def clip_is_monochrome(path, at: float) -> bool:
    """Whether footage is black and white, judged from one small frame at `at` seconds."""
    done = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", f"{max(0.0, at):.3f}", "-i", str(path), "-frames:v", "1",
         "-vf", "scale=48:48", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True)
    return done.returncode == 0 and monochrome_pixels(done.stdout)


def render(shots_path, out_path, *, subtitles=None, limit=None, start_index=None, progress=None,
           jobs=None, keep_intermediates: bool = False) -> Path:
    """Render shots.json to a finished film, resuming any intermediates already on disk.

    Unless `keep_intermediates`, each shot intermediate is deleted as soon as every group that
    uses it has been joined, and the group files (and typed-stamp .mov overlays) once the film is
    written and its length checked. A 98-minute film would otherwise hold about 15 GB of scratch.
    A render that stops part-way still resumes: groups already joined are kept until the end, and
    only the shots of groups not yet joined are drawn again. A finished render, re-run, starts over.
    """
    require_tools()
    began = time.monotonic()
    shots_path, out_path = Path(shots_path), Path(out_path)
    base = shots_path.parent
    spec = json.loads(shots_path.read_text(encoding="utf-8"))
    width, height = int(spec.get("width", 1920)), int(spec.get("height", 1080))
    fps = int(spec.get("fps", 30))
    mode = (subtitles or spec.get("subtitles") or "none").lower()
    jobs = max(1, int(jobs or JOBS))

    shots = spec["shots"]
    for position, shot in enumerate(shots, start=1):
        shot.setdefault("index", position)
    chosen = [s for s in shots if start_index is None or s["index"] >= start_index]
    chosen = chosen[:limit] if limit else chosen
    chosen = [s for s in chosen if round(float(s["end"]) * fps) > round(float(s["start"]) * fps)]
    if not chosen:
        raise ValueError(f"{shots_path}: no shots selected (start_index={start_index}, limit={limit})")

    # Every shot, so that a slice still knows the picture running into it: a shot whose sourcing
    # failed holds the last photograph that existed. Footage and motion graphics are never held
    # (a held clip would replay); a clip shot whose file is missing holds the last photograph too.
    pictures, held = {}, None
    for shot in shots:
        asset = resolve_asset(base, shot.get("asset"))
        found = asset is not None and asset.exists()
        if media_of(shot) == "photo":
            if found:
                held = asset
            pictures[shot["index"]] = (held, "photo")
        else:
            pictures[shot["index"]] = (asset, media_of(shot)) if found else (held, "photo")

    plan = cut_plan(chosen, fps, opens_film=chosen[0] is shots[0], closes_film=chosen[-1] is shots[-1])
    work = base / "work"
    (work / "overlays").mkdir(parents=True, exist_ok=True)

    # Footage timing is settled before anything renders, so a clip the planner made too long for
    # its segment is reported once, with every other offender, not an hour into the render.
    problems = []
    for shot, clip in zip(chosen, plan):
        asset, media = pictures[shot["index"]]
        clip["media"] = media
        if media == "photo":
            continue
        needed = (clip["frames"] + clip["tail"]) / fps
        length = probe_duration_cached(asset)
        try:
            if media == "clip":
                clip["timing"] = clip_timing(needed, shot.get("clip_in") or 0.0, shot.get("clip_out"), length)
                clip["mono"] = shot.get("colour") is None and clip_is_monochrome(
                    asset, clip["timing"]["start"] + clip["timing"]["forward"] / 2)
            else:
                clip["timing"] = motion_timing(needed, shot.get("clip_in") or 0.0, length)
        except ValueError as exc:
            problems.append(f"shot #{shot['index']} ({shot.get('clip_id') or asset.name}): {exc}")
    if problems:
        raise ClipTooShort("footage too short for its shot (slowed by 10% and a "
                           f"{CLIP_REVERSE_MAX_S:.0f} s reversed tail at most):\n  " + "\n  ".join(problems))

    for shot, clip in zip(chosen, plan):
        still = pictures[shot["index"]][0]
        overlay = None
        card = shot.get("overlay")
        animated = is_animated(card)
        duration = clip["frames"] / fps
        if card and overlay_fades(duration, TYPED_OUT_BY if animated else OVERLAY_OUT_BY):
            try:
                overlay = make_overlay(card, duration, work / "overlays", width, height, fps)
            except ValueError as exc:
                print(f"  WARNING: shot #{shot['index']} overlay skipped: {exc}", flush=True)
                overlay = None
        key = shot_key(shot, clip, still, overlay, width, height, fps)
        clip["path"] = (work / f"shot_{shot['index']:04d}_{_digest(key)}.mp4").resolve()
        clip["shot"] = shot
        clip["still"], clip["overlay"] = still, overlay

    lines = []
    if mode == "burn":
        lines_path = base / "lines.json"
        if lines_path.exists():
            lines = json.loads(lines_path.read_text(encoding="utf-8"))
        else:
            print(f"  WARNING: {lines_path} is missing; nothing to burn in", flush=True)

    # Groups are planned before any shot renders: a group already joined needs none of its shots,
    # so a resumed render never redraws shots whose intermediates were cleaned away.
    groups = group_plan(plan, GROUP_SIZE)
    for group in groups:
        clips, lead = group["clips"], group["lead"]
        first = clips[0]["start_frame"]
        group["ass"], ass_text = None, ""
        if lines:
            ass_text = subtitle_ass(lines, first / fps, (first + span_frames(clips)) / fps, width, height)
            group["ass"] = work / "subtitles" / f"{_digest(ass_text)}.ass"
            group["ass"].parent.mkdir(parents=True, exist_ok=True)
            group["ass"].write_text(ass_text, encoding="utf-8")
        key = [[c["path"].name for c in clips], lead["path"].name if lead else None,
               [(c["frames"], c["transition"], c["duration"]) for c in clips], ass_text, GRAIN,
               FINAL_PRESET, FINAL_CRF, FINAL_TUNE]
        group["path"] = (work / f"group_{clips[0]['index']:04d}-{clips[-1]['index']:04d}_{_digest(key)}.mp4").resolve()
        group["inputs"] = ([lead] if lead is not None else []) + clips
        group["done"] = exists(group["path"])
    waiting = [g for g in groups if not g["done"]]
    needed = {id(c): c for g in waiting for c in g["inputs"]}
    todo = [c for c in plan if id(c) in needed and not exists(c["path"])]
    pending = {id(clip) for clip in todo}

    def report(done: int, clip: dict, fresh: bool) -> None:
        shot = clip["shot"]
        if progress is not None:
            progress(done, len(plan), shot)
            return
        if fresh:
            still = clip["still"].name if clip["still"] is not None else f"colour {BACKDROP}"
            timing = clip.get("timing")
            moving = ""
            if timing:
                moving = f" {clip['media']} from {timing['start']:.1f}s"
                moving += f", slowed {timing['stretch']:.2f}x" if timing["stretch"] > 1.0001 else ""
                moving += f", {timing['reverse']:.2f}s reversed" if timing["reverse"] else ""
            print(f"  shot {done}/{len(plan)} (#{shot['index']}) {clip['frames'] / fps:.1f}s "
                  f"{clip['transition']} {still}{moving}", flush=True)

    done = len(plan) - len(todo)
    for n, clip in enumerate((c for c in plan if id(c) not in pending), start=1):
        report(n, clip, fresh=False)
    if progress is None and done:
        print(f"  {done}/{len(plan)} shots already rendered or joined", flush=True)

    def render_shot(clip):
        partial = clip["path"].with_suffix(".part.mp4")
        run(shot_clip(clip["shot"], clip, clip["still"], partial, width, height, fps, overlay=clip["overlay"]))
        partial.replace(clip["path"])
        return clip

    def render_group(group):
        partial = group["path"].with_suffix(".part.mp4")
        run(group_command(group["clips"], group["lead"], partial, fps, subtitles=group["ass"]))
        partial.replace(group["path"])
        return group

    def tidy(group) -> None:
        """A joined group's shots go, unless a group still to be joined needs one as its lead-in."""
        if keep_intermediates:
            return
        still_needed = {id(c) for g in groups if not g["done"] for c in g["inputs"]}
        for clip in group["inputs"]:
            if id(clip) not in still_needed:
                clip["path"].unlink(missing_ok=True)

    # Shots render in timeline order; each group joins as soon as its shots exist, so the joins
    # overlap the shot renders and, with intermediates cleaned as they are used, the scratch space
    # stays a few groups deep instead of the whole film.
    joined = len(groups) - len(waiting)
    shots_rendered = 0
    for group in groups:
        if group["done"]:
            tidy(group)
    with ThreadPoolExecutor(max_workers=jobs) as shot_pool, \
            ThreadPoolExecutor(max_workers=max(1, min(jobs, 4))) as group_pool:
        running = {shot_pool.submit(render_shot, clip): "shot" for clip in todo}

        def start_ready_groups() -> None:
            for group in waiting:
                if not group.get("started") and all(id(c) not in pending for c in group["inputs"]):
                    group["started"] = True
                    running[group_pool.submit(render_group, group)] = "group"

        start_ready_groups()
        while running:
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in finished:
                kind = running.pop(future)
                result = future.result()
                if kind == "shot":
                    pending.discard(id(result))
                    done += 1
                    shots_rendered += 1
                    report(done, result, fresh=True)
                else:
                    result["done"] = True
                    joined += 1
                    tidy(result)
                    if progress is None:
                        print(f"  joined group {joined}/{len(groups)} "
                              f"(#{result['clips'][0]['index']}-#{result['clips'][-1]['index']})", flush=True)
            start_ready_groups()
    joins_done = time.monotonic()

    listing = work / "concat.txt"
    listing.write_text(concat_list(group["path"] for group in groups), encoding="utf-8")
    span = span_frames(plan) / fps
    offset = plan[0]["start_frame"] / fps
    audio = resolve_asset(base, spec.get("audio"))
    argv = [FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio is not None and audio.exists():
        argv += ["-ss", f"{offset:.3f}", "-t", f"{span:.3f}", "-i", str(audio),
                 "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    else:
        print(f"  WARNING: no narration at {spec.get('audio')!r}; the cut will be silent", flush=True)
        argv += ["-map", "0:v:0", "-c:v", "copy", "-an"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(argv + ["-movflags", "+faststart", str(out_path)])

    if mode == "sidecar":
        srt = base / "script.srt"
        if srt.exists():
            shutil.copyfile(srt, out_path.with_suffix(".srt"))
            print(f"  subtitles alongside: {out_path.with_suffix('.srt')}", flush=True)
        else:
            print(f"  WARNING: {srt} is missing; no sidecar written", flush=True)

    check_duration(out_path, span)
    if not keep_intermediates:  # the film is written and checked: its scratch files can go
        for group in groups:
            group["path"].unlink(missing_ok=True)
        listing.unlink(missing_ok=True)
        for typed in (work / "overlays").glob("*.mov"):
            typed.unlink(missing_ok=True)
    wall = time.monotonic() - began
    print(f"  {span:.1f}s of film in {wall:.1f}s ({span / wall:.2f}x real time; "
          f"{shots_rendered} shots rendered and {len(waiting)} groups joined in {joins_done - began:.1f}s)",
          flush=True)
    return out_path


def exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def check_duration(out_path: Path, expected: float) -> None:
    """The cut must match the narration it was timed to; any drift means a lost or doubled frame."""
    actual = probe_duration(out_path)
    drift = actual - expected
    if abs(drift) > DURATION_TOLERANCE_S:
        print(f"  WARNING: {out_path.name} is {actual:.2f}s but the shot list wants {expected:.2f}s "
              f"({drift:+.2f}s); the picture and the narration have drifted apart", flush=True)
    else:
        print(f"  {out_path.name}: {actual:.2f}s ({drift:+.3f}s against the shot list)", flush=True)


def run(argv: list[str]) -> None:
    """ffmpeg, with its own complaint in the exception when it fails."""
    done = subprocess.run(argv, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.strip()[-2000:]}\n  {' '.join(argv)}")


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    args = [a for a in sys.argv[3:] if a != "--keep-intermediates"]
    options = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2)}
    render(
        Path(sys.argv[1]), Path(sys.argv[2]),
        subtitles=options.get("--subtitles"),
        limit=int(options["--limit"]) if "--limit" in options else None,
        start_index=int(options["--start-index"]) if "--start-index" in options else None,
        jobs=int(options["--jobs"]) if "--jobs" in options else None,
        keep_intermediates="--keep-intermediates" in sys.argv[3:],
    )


if __name__ == "__main__":
    main()

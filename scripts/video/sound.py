"""Sound for the documentary: a low music bed ducked under the narration, sparse effects tied to
picture events, and a master at YouTube loudness.

    .venv/bin/python scripts/video/sound.py [video_dir] [--preview N] [--probe]

      video_dir     default data/video/gacy; reads shots.json, writes sound/mix.wav
      --preview N   only the first N seconds, to sound/mix_preview.wav, with ducking numbers
      --probe       ask ElevenLabs which sound endpoints the key may use, then stop

The narration is always on top and keeps its delivery: it gets one fixed gain (the master's) and a
true-peak limiter that only touches its rare loudest peaks. The bed is keyed from the narration with
`sidechaincompress`, so it sits about 10 dB lower under speech and breathes back up in the pauses.
Effects are placed sample-exactly and sit 15 LU or more below the voice. The master is normalised
with two-pass `loudnorm` (linear) to -14 LUFS integrated, true peak at or below -1 dBTP, 48 kHz
stereo 16-bit WAV (triangular dither), about 1.1 GB for the full film.

Sources. The owner's ElevenLabs key lacks `sound_generation` and `music_generation` (checked
2026-10-14, see --probe), so the music is Kevin MacLeod's (incompetech.com, CC BY 4.0) and the two
recorded effects come from Wikimedia Commons (CC0 and CC BY 4.0). The whoosh and the drone are
synthesised here from sine waves and noise. Every file is listed in sound/credits.json.

Long ffmpeg graphs are kept cheap with lanes: a lane is one audio stream made by concatenating
silence and sounds that never overlap, so a 98-minute bed or a film's worth of effects costs a few
streams instead of one full-length stream per sound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"  # ffmpeg 9; never the voice pipeline's 4.2.2
FFPROBE = "/opt/miniconda3/envs/video/bin/ffprobe"
VIDEO_DIR = ROOT / "data" / "video" / "gacy"

RATE = 48000

# Master. The voice is lifted by one fixed gain to reach the target; a limiter run at 4x oversampling
# (so it catches true peaks) holds the rare peaks under LIMIT_DB, which leaves loudnorm's second pass
# room to stay linear.
TARGET_I, TARGET_TP, TARGET_LRA = -14.0, -1.0, 20.0
LIMIT_DB = -1.6
LIMITER = "attack=1:release=30:asc=0:level=0:latency=1"  # measured: touches ~2.5% of 10 ms windows, at most ~2.4 dB

# Bed. Each track is levelled to BED_LUFS (stereo, in the master's scale), with a gentle dip where
# speech is most intelligible so the music leaves a pocket for the voice.
BED_LUFS = -31.0
XFADE_S = 4.0
FADE_IN_S = 4.0
FADE_OUT_S = 10.0
LIFT_DB, LIFT_RAMP_S, LIFT_HOLD_S = 2.0, 1.0, 0.5  # the bed rises a little into each dip to black
TRACK_FILTER = ("highpass=f=32,equalizer=f=2600:t=o:w=1.6:g=-4,treble=g=-3:f=6000,"
                f"loudnorm=I={BED_LUFS}:LRA=7:TP=-9")

# Ducking. The key is the narration at its master gain. The threshold sits just above the pauses'
# floor and the ratio is high, so all speech, loud or soft, drives the compressor deep; `mix` then
# sets how deep the bed goes (1 - mix as a linear floor), which keeps the depth steady through a
# sentence instead of following each syllable. Attack and release are slow enough not to pump.
DUCK = {"threshold": 0.004, "ratio": 20, "attack": 120, "release": 450, "knee": 6,
        "detection": "rms", "link": "average", "mix": 0.70}

# Effects: placement rules, and each effect's loudest 400 ms (momentary LUFS) in the master's scale,
# against the voice's -14 integrated (about -10 at its loudest moments).
WHOOSH_S = 1.2
SHUTTER_LOOKBACK = 20
SHUTTER_GAP_S = 30.0
TYPE_DELAY_S = 0.3
MAX_DRONES = 5
DRONE_GAP_S = 120.0
KIT_REF_LUFS = -20.0  # every kit file is stored with its loudest moment here
EFFECT_LUFS = {"shutter": -31.0, "typewriter": -32.0, "whoosh": -29.0, "drone": -28.0}

# Big reveals, in story order. Each fires once, on the first shot whose narration says it; a reveal
# within DRONE_GAP_S of the previous drone is skipped, and no more than MAX_DRONES fire. For the Gacy
# script: the cold open's "remains of twenty-nine" (shot 8), the confession "That boy is dead" (376),
# the house search with the warrant (399), the verdict (573) and the execution (624). The arrest
# (395, "took him into custody") is left out: it falls 35 s before the search, which would then be
# skipped by the gap rule, and the search is the bigger picture.
REVEALS = [
    ("cold open", r"remains of twenty-nine"),
    ("confession", r"that boy is dead"),
    ("house search", r"cook county sheriff.{0,20}went to gacy"),
    ("verdict", r"jury found gacy guilty"),
    ("execution", r"put to death by lethal injection"),
]

INCOMPETECH = "https://incompetech.com/music/royalty-free/mp3-royaltyfree/"
CC_BY_4 = "https://creativecommons.org/licenses/by/4.0/"
CC0 = "https://creativecommons.org/publicdomain/zero/1.0/"

# The bed's rotation, in order, starting from the steadiest drone so the cold open sits still.
# Each is a slow pad or drone with no drums and no lead melody. ("Long Note Three" was tried and
# dropped: it sinks to near silence for half a minute and ends on a sudden swell.)
MUSIC = [
    {"file": "music/ossuary_6__air.mp3", "title": "Ossuary 6 - Air", "length": "4:10",
     "why": "slow-moving low drone"},
    {"file": "music/echoes_of_time_v2.mp3", "title": "Echoes of Time v2", "length": "4:45",
     "why": "deep soundscape, the version without the bell melody"},
    {"file": "music/ossuary_5__rest.mp3", "title": "Ossuary 5 - Rest", "length": "3:56",
     "why": "calm, dark synth pad"},
    {"file": "music/long_note_one.mp3", "title": "Long Note One", "length": "7:20",
     "why": "a long pad for use under dialogue (basses, violins)"},
    {"file": "music/dark_times.mp3", "title": "Dark Times", "length": "3:04",
     "why": "slow, somber low strings"},
    {"file": "music/thunder_dreams.mp3", "title": "Thunder Dreams", "length": "3:50",
     "why": "dark chordal soundscape, no rhythm"},
]
for _m in MUSIC:
    _m.update({"author": "Kevin MacLeod", "licence": "CC BY 4.0", "licence_url": CC_BY_4,
               "url": INCOMPETECH + _m["title"].replace(" ", "%20") + ".mp3",
               "page": "https://incompetech.com/music/royalty-free/music.html",
               "source": "incompetech.com", "used_for": "music bed",
               "attribution": f"\"{_m['title']}\" Kevin MacLeod (incompetech.com)\n"
                              "Licensed under Creative Commons: By Attribution 4.0 License\n"
                              "http://creativecommons.org/licenses/by/4.0/"})

COMMONS_UA = "Sinhala-English-Voice sound stage (https://github.com/ShalikaNuwan/sinhala-English-Voice-Generator)"
SFX = [
    {"file": "sources/shutter_5d.wav", "title": "261007 polecatz camera-canon-5d-iii.wav",
     "author": "Polecatz", "licence": "CC0", "licence_url": CC0, "source": "Wikimedia Commons",
     "url": "https://commons.wikimedia.org/wiki/File:261007_polecatz_camera-canon-5d-iii.wav",
     "download": "https://upload.wikimedia.org/wikipedia/commons/4/49/261007_polecatz_camera-canon-5d-iii.wav",
     "used_for": "camera shutter (one exposure, 1.12-1.52 s)"},
    {"file": "sources/typewriter_erika.ogg", "title": "WWS Typewriter.ogg",
     "author": "Work With Sounds / Konrad Gutkowski", "licence": "CC BY 4.0", "licence_url": CC_BY_4,
     "source": "Wikimedia Commons", "url": "https://commons.wikimedia.org/wiki/File:WWS_Typewriter.ogg",
     "download": "https://upload.wikimedia.org/wikipedia/commons/2/25/WWS_Typewriter.ogg",
     "used_for": "typewriter keys (1.55 s from 12.25 s; trimmed, filtered, faded)",
     "attribution": "\"WWS Typewriter\" by Work With Sounds / Konrad Gutkowski, CC BY 4.0, "
                    "via Wikimedia Commons (trimmed and filtered)"},
]
SYNTH_CREDIT = {"author": "synthesised by scripts/video/sound.py (ffmpeg aevalsrc)", "licence": "Original work",
                "licence_url": "", "source": "generated", "url": ""}

# The kit: how each effect file is made. Recorded effects are cut from a source; synthesised ones
# are expressions, deterministic because aevalsrc's random() is seeded from its variable.
KIT = {
    "shutter": {"source": "sources/shutter_5d.wav", "ss": 1.12, "t": 0.40,
                "af": "highpass=f=150,afade=t=in:d=0.004,afade=t=out:st=0.30:d=0.10"},
    "typewriter": {"source": "sources/typewriter_erika.ogg", "ss": 12.25, "t": 1.55,
                   "af": "highpass=f=120,lowpass=f=7000,afade=t=in:d=0.01,afade=t=out:st=1.30:d=0.25"},
    "whoosh": {"synth": (
        # a riser that ends on the cut: band-passed noise and a sub falling 95 -> 47 Hz, both swelling
        f"aevalsrc=exprs='(2*random(0)-1)*pow(t/{WHOOSH_S},3)|(2*random(1)-1)*pow(t/{WHOOSH_S},3)'"
        f":s={RATE}:d={WHOOSH_S},highpass=f=180,lowpass=f=1400,lowpass=f=1400[n];"
        f"aevalsrc=exprs='0.8*sin(2*PI*(95*t-20*t*t))*pow(t/{WHOOSH_S},2)"
        f"|0.8*sin(2*PI*(95*t-20*t*t))*pow(t/{WHOOSH_S},2)':s={RATE}:d={WHOOSH_S}[s];"
        f"[n][s]amix=inputs=2:normalize=0,afade=t=out:st={WHOOSH_S - 0.04:.2f}:d=0.04")},
    "drone": {"synth": (
        # a low hit: D1, A1, D2 and two quiet upper partials (audible on phone speakers), decaying
        # over a few seconds, under a short low-passed noise thump
        "aevalsrc=exprs='min(1,t/0.02)*exp(-t/2.2)*(0.5*sin(2*PI*36.71*t)+0.35*sin(2*PI*55*t+0.3)"
        "+0.25*sin(2*PI*73.42*t)+0.12*sin(2*PI*110*t)+0.06*sin(2*PI*146.83*t))"
        "|min(1,t/0.02)*exp(-t/2.2)*(0.5*sin(2*PI*36.71*t)+0.35*sin(2*PI*55.4*t+0.3)"
        f"+0.25*sin(2*PI*73.42*t)+0.12*sin(2*PI*110.3*t)+0.06*sin(2*PI*146.83*t))':s={RATE}:d=6[t];"
        f"aevalsrc=exprs='(2*random(0)-1)*exp(-t/0.12)*0.6|(2*random(1)-1)*exp(-t/0.12)*0.6':s={RATE}:d=6,"
        "lowpass=f=250,lowpass=f=250[k];"
        "[t][k]amix=inputs=2:normalize=0,afade=t=out:st=4.5:d=1.5")},
}


# ---------------------------------------------------------------- running ffmpeg

def run(cmd: list[str], quiet: bool = True) -> str:
    """Run a command; return stderr (where ffmpeg reports), raising with its tail on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed ({result.returncode}):\n{result.stderr[-3000:]}")
    return result.stderr


def duration_of(path: Path) -> float:
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def channels_of(path: Path) -> int:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return int(out.strip())


def loudnorm_json(stderr: str) -> dict:
    """loudnorm's print_format=json block, the last {...} in ffmpeg's log."""
    block = stderr[stderr.rindex("{"):stderr.rindex("}") + 1]
    return {k: (v if k == "normalization_type" else float(v)) for k, v in json.loads(block).items()}


def measure(path: Path, start: float = 0.0, length: float | None = None, af: str = "") -> dict:
    """Integrated loudness, true peak, LRA and threshold of a file (or a span of it), as loudnorm sees it."""
    span = ["-ss", f"{start}"] + (["-t", f"{length}"] if length else [])
    chain = (af + "," if af else "") + f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json"
    return loudnorm_json(run([FFMPEG, "-hide_banner", "-nostats", *span, "-i", str(path), "-af", chain,
                              "-f", "null", "-"]))


def max_momentary(path: Path) -> float:
    """The loudest 400 ms of a file, in LUFS: how loud a short effect sounds at its peak."""
    log = run([FFMPEG, "-hide_banner", "-nostats", "-i", str(path), "-af",
               "apad=pad_dur=0.4,ebur128=metadata=1,ametadata=mode=print:key=lavfi.r128.M", "-f", "null", "-"])
    values = [float(v) for v in re.findall(r"lavfi\.r128\.M=(-?[\d.]+|-inf)", log) if v != "-inf"]
    return max(values)


def window_levels(path: Path, window_s: float = 0.1, start: float = 0.0, length: float | None = None,
                  af: str = "") -> list[float]:
    """RMS level in dBFS of each consecutive window (both channels together)."""
    samples = int(round(window_s * RATE))
    span = ["-ss", f"{start}"] + (["-t", f"{length}"] if length else [])
    chain = (af + "," if af else "") + (f"aresample={RATE},pan=mono|c0=0.5*c0+0.5*c1," if channels_of(path) == 2
                                        else f"aresample={RATE},")
    chain += f"asetnsamples={samples},astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.RMS_level"
    log = run([FFMPEG, "-hide_banner", "-nostats", *span, "-i", str(path), "-af", chain, "-f", "null", "-"])
    return [(-150.0 if v in ("-inf", "inf", "nan") else float(v))
            for v in re.findall(r"RMS_level=(-?inf|nan|-?[\d.]+)", log)]


# ---------------------------------------------------------------- where the effects go

def _overlay_style(shot: dict) -> str | None:
    overlay = shot.get("overlay")
    if not isinstance(overlay, dict):
        return None
    return overlay.get("style") or overlay.get("type")


def _picture_key(shot: dict) -> str | None:
    return shot.get("library_id") or shot.get("asset") or None


def events_from_shots(shots: list[dict], whoosh_s: float = WHOOSH_S, reveals=REVEALS) -> list[dict]:
    """The film's sound effects, sparse and tied to what the picture does. Pure.

    - whoosh: ends exactly on each `black` transition's cut (the shot's start, the darkest frame of
      the dip); none for a dip too near the film's start to fit one.
    - shutter: the first frame of a case photograph not seen in the previous SHUTTER_LOOKBACK shots
      (`kind` archival, shown as a still), at most one per SHUTTER_GAP_S; never on a shot that rises
      out of black (its whoosh already marks it) or carries a date stamp (its keys do).
    - typewriter: TYPE_DELAY_S into a shot whose overlay is a `stamp` card, as the date types itself.
    - drone: the first shot of each big reveal in `reveals` (see REVEALS), at most MAX_DRONES, no two
      within DRONE_GAP_S.
    """
    shots = sorted(shots, key=lambda s: s["start"])
    events: list[dict] = []
    last_shutter = -math.inf
    last_drone = -math.inf
    drones = 0
    pending = list(reveals)
    for position, shot in enumerate(shots):
        start = float(shot["start"])
        transition = (shot.get("transition_in") or "dissolve").lower()
        style = _overlay_style(shot)
        whooshed = transition == "black" and start >= whoosh_s
        if whooshed:
            events.append({"kind": "whoosh", "t": round(start - whoosh_s, 3), "shot": shot.get("index"),
                           "reason": "into the dip to black"})
        if style == "stamp":
            events.append({"kind": "typewriter", "t": round(start + TYPE_DELAY_S, 3), "shot": shot.get("index"),
                           "reason": f"date stamp: {shot['overlay'].get('text', '')}"})
        key = _picture_key(shot)
        recent = {_picture_key(s) for s in shots[max(0, position - SHUTTER_LOOKBACK):position]}
        if (shot.get("kind") == "archival" and (shot.get("media") or "photo") == "photo" and key
                and key not in recent and start - last_shutter >= SHUTTER_GAP_S
                and not whooshed and style != "stamp"):
            events.append({"kind": "shutter", "t": round(start, 3), "shot": shot.get("index"),
                           "reason": f"new case photograph {key}"})
            last_shutter = start
        narration = (shot.get("narration") or "").lower().replace("’", "'")
        for name, pattern in list(pending):
            if re.search(pattern, narration):
                pending.remove((name, pattern))
                if drones < MAX_DRONES and start - last_drone >= DRONE_GAP_S:
                    events.append({"kind": "drone", "t": round(start, 3), "shot": shot.get("index"),
                                   "reason": f"reveal: {name}"})
                    last_drone, drones = start, drones + 1
    return sorted(events, key=lambda e: (e["t"], e["kind"]))


# ---------------------------------------------------------------- lanes

def lanes(items: list[dict]) -> list[list[dict]]:
    """Share items ({"start", "length"} in samples) among as few streams as possible, none overlapping
    within a stream. Greedy in start order, which is optimal for intervals."""
    out: list[list[dict]] = []
    ends: list[int] = []
    for item in sorted(items, key=lambda i: i["start"]):
        for k, end in enumerate(ends):
            if end <= item["start"]:
                out[k].append(item)
                ends[k] = item["start"] + item["length"]
                break
        else:
            out.append([item])
            ends.append(item["start"] + item["length"])
    return out


def lane_filter(lane: list[dict], out_label: str, prefix: str) -> str:
    """One lane as a filtergraph: silence up to each item, the item, and so on, joined by concat.

    Each item names the already-built label of its sound ("label") and its own exact length in
    samples; gaps are cut from anullsrc to the sample, so every sound starts where it was placed.
    """
    parts, labels, cursor = [], [], 0
    for n, item in enumerate(lane):
        gap = item["start"] - cursor
        if gap < 0:
            raise ValueError("overlapping items in one lane")
        if gap:
            parts.append(f"anullsrc=r={RATE}:cl=stereo,atrim=end_sample={gap}[{prefix}g{n}]")
            labels.append(f"[{prefix}g{n}]")
        labels.append(f"[{item['label']}]")
        cursor = item["start"] + item["length"]
    if len(labels) == 1:
        parts.append(f"{labels[0]}anull[{out_label}]")
    else:
        parts.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[{out_label}]")
    return ";".join(parts)


def sum_lanes(labels: list[str], out_label: str, duration_samples: int) -> str:
    """Add lanes together (unscaled) and pad or cut the result to exactly the film's length."""
    tail = f"apad=whole_len={duration_samples},atrim=end_sample={duration_samples}[{out_label}]"
    if len(labels) == 1:
        return f"[{labels[0]}]{tail}"
    return "".join(f"[{x}]" for x in labels) + f"amix=inputs={len(labels)}:normalize=0:duration=longest," + tail


# ---------------------------------------------------------------- the music bed

def bed_plan(duration_s: float, lengths: list[float], xfade_s: float = XFADE_S) -> list[dict]:
    """Tracks in rotation, each crossfading into the next over xfade_s, until the film is covered.

    Returns segments {"track", "at", "length", "fade_in"} in seconds: segment k+1 starts xfade_s
    before segment k ends. The first starts at 0 with no crossfade in (the bed's own fade-in covers
    it); the last is cut to end at the film's end (the bed's fade-out covers that).
    """
    if not lengths or min(lengths) <= 2 * xfade_s:
        raise ValueError("every track must be longer than two crossfades")
    segments, at, k = [], 0.0, 0
    while True:
        track = k % len(lengths)
        length = lengths[track]
        last = at + length >= duration_s
        segments.append({"track": track, "at": round(at, 6), "length": round(duration_s - at if last else length, 6),
                         "fade_in": 0.0 if k == 0 else xfade_s})
        if last:
            return segments
        at += length - xfade_s
        k += 1


def lift_expr(cuts: list[float], db: float = LIFT_DB, ramp: float = LIFT_RAMP_S, hold: float = LIFT_HOLD_S) -> str:
    """A volume expression that raises the bed by `db` around each cut: a ramp up, a short hold either
    side of the cut, a ramp down. Bumps that meet never stack above `db`."""
    if not cuts:
        return "1"
    bumps = []
    for c in cuts:
        a, b = c - hold - ramp, c + hold + ramp
        bumps.append(f"clip(min((t-{a:.3f})/{ramp},({b:.3f}-t)/{ramp}),0,1)")
    peak = 10 ** (db / 20) - 1
    return f"1+{peak:.4f}*min(1,{'+'.join(bumps)})"


def bed_filter(plan: list[dict], duration_s: float, lifts: list[float] = ()) -> str:
    """The filtergraph that turns the plan into the bed. Input i is segment i's track, already levelled.

    Each segment is cut from its track's start to the sample, rises over its crossfade with a quarter
    sine (and the segment it overlaps falls with the same curve, so the crossfade is equal-power),
    then sits in a lane at its start. The lanes are summed, lifted at the dips, faded in and out.
    """
    total = int(round(duration_s * RATE))
    parts, items = [], []
    starts = [int(round(seg["at"] * RATE)) for seg in plan]
    for i, seg in enumerate(plan):
        end = starts[i + 1] + int(round(plan[i + 1]["fade_in"] * RATE)) if i + 1 < len(plan) else total
        length = end - starts[i]
        chain = f"[{i}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,atrim=end_sample={length}"
        if seg["fade_in"]:
            chain += f",afade=t=in:d={seg['fade_in']}:curve=qsin"
        if i + 1 < len(plan) and plan[i + 1]["fade_in"]:
            fade = plan[i + 1]["fade_in"]
            chain += f",afade=t=out:st={(length / RATE) - fade:.6f}:d={fade}:curve=qsin"
        parts.append(chain + f"[b{i}]")
        items.append({"label": f"b{i}", "start": starts[i], "length": length})
    lane_labels = []
    for n, lane in enumerate(lanes(items)):
        parts.append(lane_filter(lane, f"bl{n}", f"bl{n}_"))
        lane_labels.append(f"bl{n}")
    parts.append(sum_lanes(lane_labels, "bsum", total))
    fade_out_at = max(0.0, duration_s - FADE_OUT_S)
    parts.append(f"[bsum]volume='{lift_expr(list(lifts))}':eval=frame,"
                 f"afade=t=in:d={min(FADE_IN_S, duration_s / 2)},"
                 f"afade=t=out:st={fade_out_at:.6f}:d={min(FADE_OUT_S, duration_s / 2)}[bed]")
    return ";".join(parts)


def prepare_tracks(sound_dir: Path, music=MUSIC) -> list[Path]:
    """Each track trimmed of its silent head and tail, given a pocket for the voice and levelled to
    BED_LUFS (dynamic loudnorm to even out swells, then one linear trim onto the mark), as 48 kHz
    stereo FLAC. Cached under sound/work/."""
    work = sound_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    out = []
    for track in music:
        src = sound_dir / track["file"]
        tag = hashlib.sha1(TRACK_FILTER.encode()).hexdigest()[:8]
        dst = work / f"bed_{src.stem}_{tag}.flac"
        if not dst.exists():
            trim = "silenceremove=start_periods=1:start_threshold=-60dB"
            af = f"{trim},areverse,{trim},areverse,{TRACK_FILTER},aresample={RATE}"
            levelled = work / f"{dst.stem}.part.flac"
            run([FFMPEG, "-y", "-hide_banner", "-i", str(src), "-vn", "-af", af, "-ac", "2", "-ar", str(RATE),
                 "-sample_fmt", "s32", "-c:a", "flac", str(levelled)])
            # loudnorm's dynamic mode lands within a couple of LU; one linear gain puts it on the mark
            correction = BED_LUFS - measure(levelled)["input_i"]
            run([FFMPEG, "-y", "-hide_banner", "-i", str(levelled), "-af", f"volume={correction:.2f}dB",
                 "-sample_fmt", "s32", "-c:a", "flac", str(dst)])
            levelled.unlink()
        out.append(dst)
    return out


def music_bed(duration_s: float, tracks: list[Path], out: Path, lifts: list[float] = ()) -> Path:
    """A continuous bed for the whole film, from levelled tracks in rotation (see bed_plan)."""
    lengths = [duration_of(t) for t in tracks]
    plan = bed_plan(duration_s, lengths)
    inputs = []
    for seg in plan:
        inputs += ["-i", str(tracks[seg["track"]])]
    graph = bed_filter(plan, duration_s, lifts)
    out.parent.mkdir(parents=True, exist_ok=True)
    script = out.with_suffix(".filter.txt")
    script.write_text(graph, encoding="utf-8")
    run([FFMPEG, "-y", "-hide_banner", *inputs, "-/filter_complex", str(script), "-map", "[bed]",
         "-ar", str(RATE), "-sample_fmt", "s16", "-c:a", "flac", str(out)])  # 16-bit: ample for a -31 LUFS bed
    script.unlink()
    return out


# ---------------------------------------------------------------- the kit

def build_kit(sound_dir: Path, kit=KIT) -> dict[str, Path]:
    """Each effect as 48 kHz stereo WAV under sound/kit/, its loudest moment set to KIT_REF_LUFS."""
    kit_dir = sound_dir / "kit"
    kit_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, recipe in kit.items():
        dst = kit_dir / f"{kind}.wav"
        if not dst.exists():
            raw = kit_dir / f"{kind}.raw.wav"
            if "synth" in recipe:
                cmd = [FFMPEG, "-y", "-hide_banner", "-filter_complex", recipe["synth"] + "[o]", "-map", "[o]"]
            else:
                cmd = [FFMPEG, "-y", "-hide_banner", "-ss", str(recipe["ss"]), "-t", str(recipe["t"]),
                       "-i", str(sound_dir / recipe["source"]), "-af", recipe["af"]]
            run(cmd + ["-ac", "2", "-ar", str(RATE), "-c:a", "pcm_f32le", str(raw)])
            gain = KIT_REF_LUFS - max_momentary(raw)
            run([FFMPEG, "-y", "-hide_banner", "-i", str(raw), "-af", f"volume={gain:.2f}dB",
                 "-c:a", "pcm_s24le", str(dst)])
            raw.unlink()
        out[kind] = dst
    return out


# ---------------------------------------------------------------- credits

def credits_entries(sound_dir: Path, music=MUSIC, sfx=SFX, kit=KIT) -> list[dict]:
    """One entry per file the sound is made from, and per kit file with what it was made from.

    Raises if a file under music/ or sources/ has no entry, so nothing reaches the film uncredited.
    """
    known = {m["file"]: m for m in [*music, *sfx]}
    for folder in ("music", "sources"):
        for path in sorted((sound_dir / folder).glob("*")) if (sound_dir / folder).exists() else []:
            rel = f"{folder}/{path.name}"
            if path.is_file() and not path.name.startswith(".") and rel not in known:
                raise ValueError(f"{rel} has no credit entry")
    entries = []
    for meta in [*music, *sfx]:
        if (sound_dir / meta["file"]).exists():
            entries.append({k: meta[k] for k in ("file", "title", "author", "licence", "licence_url", "url",
                                                  "source", "used_for") if k in meta}
                           | {"attribution": meta.get("attribution") or f"\"{meta['title']}\" by {meta['author']} "
                              f"({meta['licence']}, {meta['source']})"})
    for kind, recipe in kit.items():
        entry = {"file": f"kit/{kind}.wav", "title": f"{kind} (kit)", "used_for": f"{kind} effect"}
        if "synth" in recipe:
            entry.update(SYNTH_CREDIT, attribution="")
        else:
            origin = known[recipe["source"]]
            entry.update({"derived_from": recipe["source"], "author": origin["author"], "licence": origin["licence"],
                          "licence_url": origin["licence_url"], "url": origin["url"], "source": origin["source"],
                          "attribution": origin.get("attribution", "")})
        entries.append(entry)
    return entries


def write_credits(sound_dir: Path) -> Path:
    path = sound_dir / "credits.json"
    path.write_text(json.dumps(credits_entries(sound_dir), ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def credits_markdown(entries: list[dict]) -> str:
    """A "Music and sound" section for the film's credits.md. Attribution is only owed for CC BY."""
    lines = ["## Music and sound", ""]
    for e in entries:
        if e.get("used_for", "").startswith("alternative") or e.get("derived_from"):
            continue
        line = f"- {e['title']} — {e['author']} — {e['licence']}"
        if e.get("url"):
            line += f" — {e['url']}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def fetch_sources(sound_dir: Path) -> None:
    """Download any music or effect source not already on disk (resume-safe)."""
    import httpx

    for meta in [*MUSIC, *SFX]:
        dst = sound_dir / meta["file"]
        url = meta.get("download") or (meta["url"] if meta["url"].endswith(".mp3") else "")
        if dst.exists() or not url:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with httpx.stream("GET", url, headers={"User-Agent": COMMONS_UA}, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(dst.with_suffix(".part"), "wb") as fh:
                for chunk in r.iter_bytes():
                    fh.write(chunk)
        dst.with_suffix(".part").rename(dst)


# ---------------------------------------------------------------- the mix

def duck_filter(bed_in: str, key_in: str, out_label: str, duck=DUCK) -> str:
    opts = ":".join(f"{k}={v}" for k, v in duck.items())
    return f"[{bed_in}][{key_in}]sidechaincompress={opts}[{out_label}]"


def voice_chain(voice_gain_db: float, mono: bool, duration_s: float) -> str:
    """The narration: cut to length, at 48 kHz, centred in stereo, at its one fixed gain. Nothing else."""
    pan = ",pan=stereo|c0=c0|c1=c0" if mono else ""
    return (f"[0:a]atrim=end={duration_s:.6f},asetpts=PTS-STARTPTS,aresample={RATE}"
            f"{pan},aformat=sample_fmts=fltp:channel_layouts=stereo,volume={voice_gain_db:.3f}dB")


def mix_filter(events: list[dict], kinds: list[str], lengths: dict[str, int], voice_gain_db: float,
               duration_s: float, mono: bool = True) -> str:
    """The whole pre-master graph. Inputs: 0 narration, 1 bed, then one per kind in `kinds` (kit files).

    Voice (unprocessed but for its gain) + ducked bed + effect lanes, summed unscaled, then a true-peak
    limiter at 4x oversampling with its delay compensated. Output label [bus].
    """
    total = int(round(duration_s * RATE))
    parts = [voice_chain(voice_gain_db, mono, duration_s) + ",asplit=2[voice][key]",
             f"[1:a]atrim=end={duration_s:.6f},asetpts=PTS-STARTPTS,"
             f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo[bedraw]",
             duck_filter("bedraw", "key", "bed")]
    placed = [e for e in events if 0 <= e["t"] and int(round(e["t"] * RATE)) < total]
    items = []
    for kind_index, kind in enumerate(kinds):
        mine = [e for e in placed if e["kind"] == kind]
        if not mine:
            continue
        copies = "".join(f"[k{kind_index}_{n}]" for n in range(len(mine)))
        parts.append(f"[{kind_index + 2}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo,"
                     f"asplit={len(mine)}{copies}" if len(mine) > 1 else
                     f"[{kind_index + 2}:a]aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo{copies}")
        gain = EFFECT_LUFS[kind] - KIT_REF_LUFS
        for n, e in enumerate(mine):
            start = int(round(e["t"] * RATE))
            length = min(lengths[kind], total - start)
            parts.append(f"[k{kind_index}_{n}]atrim=end_sample={length},volume={gain:.2f}dB[e{kind_index}_{n}]")
            items.append({"label": f"e{kind_index}_{n}", "start": start, "length": length})
    sources = ["[voice]", "[bed]"]
    for n, lane in enumerate(lanes(items)):
        parts.append(lane_filter(lane, f"fx{n}", f"fx{n}_"))
        sources.append(f"[fx{n}]")
    limit = 10 ** (LIMIT_DB / 20)
    parts.append("".join(sources) + f"amix=inputs={len(sources)}:normalize=0:duration=first:dropout_transition=0,"
                 f"apad=whole_len={total},atrim=end_sample={total},"
                 f"aresample={RATE * 4},alimiter=limit={limit:.4f}:{LIMITER},aresample={RATE}[bus]")
    return ";".join(parts)


def second_pass(measured: dict) -> str:
    """loudnorm's second pass: a single linear gain onto TARGET_I, given the first pass's numbers."""
    return (f"loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}:measured_I={measured['input_i']}"
            f":measured_TP={measured['input_tp']}:measured_LRA={measured['input_lra']}"
            f":measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}"
            f":linear=true:print_format=json,aresample={RATE}:osf=s16:dither_method=triangular")


def voice_gain(narration: Path, duration_s: float) -> tuple[float, dict]:
    """The one gain that brings the narration (centred in stereo) to the master's loudness."""
    mono = channels_of(narration) == 1
    measured = measure(narration, 0, duration_s, af="pan=stereo|c0=c0|c1=c0" if mono else "")
    return TARGET_I - measured["input_i"], measured


def mix(narration: Path, bed: Path, events: list[dict], kit: dict[str, Path], out: Path,
        duration_s: float | None = None) -> dict:
    """Narration on top, the bed ducked under it, effects at their times; mastered to -14 LUFS, -1 dBTP.

    Two-pass loudnorm on the same deterministic graph: pass 1 runs it to measure the pre-master bus,
    pass 2 runs it again and applies the one linear gain. Nothing but the master touches the disk,
    so the 98-minute mix needs only its own 1.1 GB. Returns the numbers of each stage.
    """
    duration_s = duration_s or duration_of(narration)
    gain_db, voice_measured = voice_gain(narration, duration_s)
    kinds = [k for k in kit if any(e["kind"] == k for e in events)]
    lengths = {k: int(round(duration_of(kit[k]) * RATE)) for k in kinds}
    graph = mix_filter(events, kinds, lengths, gain_db, duration_s, mono=channels_of(narration) == 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["-i", str(narration), "-i", str(bed)]
    for k in kinds:
        inputs += ["-i", str(kit[k])]
    t0 = time.time()
    script = out.parent / f".{out.stem}.measure.txt"
    script.write_text(graph.replace("[bus]", f",loudnorm=I={TARGET_I}:TP={TARGET_TP}:LRA={TARGET_LRA}"
                                              ":print_format=json[bus]"), encoding="utf-8")
    first = loudnorm_json(run([FFMPEG, "-hide_banner", "-nostats", *inputs, "-/filter_complex", str(script),
                               "-map", "[bus]", "-f", "null", "-"]))
    t_pass1 = time.time() - t0
    script.write_text(graph.replace("[bus]", "," + second_pass(first) + "[bus]"), encoding="utf-8")
    applied = loudnorm_json(run([FFMPEG, "-y", "-hide_banner", "-nostats", *inputs, "-/filter_complex", str(script),
                                 "-map", "[bus]", "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_s16le", str(out)]))
    script.unlink()
    if applied.get("normalization_type") != "linear":
        raise RuntimeError(f"loudnorm fell back to {applied.get('normalization_type')}; lower LIMIT_DB")
    final = measure(out)
    return {"voice": voice_measured, "voice_gain_db": round(gain_db, 2), "bus": first, "final": final,
            "events": len([e for e in events if e["t"] < duration_s]),
            "pass_seconds": [round(t_pass1, 1), round(time.time() - t0 - t_pass1, 1)]}


# ---------------------------------------------------------------- proving the ducking

def ducked_bed(narration: Path, bed: Path, out: Path, duration_s: float, voice_gain_db: float,
               ducked: bool = True) -> Path:
    """The bed alone as it sits in the mix (ducked by the narration, or not, for comparison)."""
    mono = channels_of(narration) == 1
    graph = [voice_chain(voice_gain_db, mono, duration_s) + "[key]",
             f"[1:a]atrim=end={duration_s:.6f},asetpts=PTS-STARTPTS,"
             f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts=stereo[bedraw]"]
    graph.append(duck_filter("bedraw", "key", "o") if ducked else "[bedraw]anull[o];[key]anullsink")
    run([FFMPEG, "-y", "-hide_banner", "-i", str(narration), "-i", str(bed), "-filter_complex", ";".join(graph),
         "-map", "[o]", "-c:a", "pcm_f32le", str(out)])
    return out


def power_mean_db(levels: list[float]) -> float:
    if not levels:
        return float("nan")
    return 10 * math.log10(sum(10 ** (v / 10) for v in levels) / len(levels))


def ducking_numbers(narration: Path, bed: Path, duration_s: float, voice_gain_db: float, work: Path,
                    window_s: float = 0.1, speech_db: float = -45.0, settle_s: float = 0.6) -> dict:
    """How far the bed drops under speech, measured from the gain the ducker applied per window.

    Speech windows: narration RMS above speech_db. Pause windows: at least settle_s into a pause of
    the narration (so the release has had time), which is when a listener hears the bed come up.
    """
    wet = ducked_bed(narration, bed, work / "duck_wet.wav", duration_s, voice_gain_db, True)
    dry = ducked_bed(narration, bed, work / "duck_dry.wav", duration_s, voice_gain_db, False)
    voice = window_levels(narration, window_s, 0, duration_s)
    w, d = window_levels(wet, window_s), window_levels(dry, window_s)
    n = min(len(voice), len(w), len(d))
    settle = int(round(settle_s / window_s))
    speech, pause, gr_speech, gr_pause = [], [], [], []
    quiet_run = 0
    for i in range(n):
        if voice[i] > speech_db:
            quiet_run = 0
            speech.append(w[i]); gr_speech.append(w[i] - d[i])
        else:
            quiet_run += 1
            if quiet_run > settle:
                pause.append(w[i]); gr_pause.append(w[i] - d[i])
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    sd = lambda xs: math.sqrt(sum((x - mean(xs)) ** 2 for x in xs) / len(xs)) if xs else float("nan")
    wet.unlink(); dry.unlink()
    return {"speech_windows": len(speech), "pause_windows": len(pause),
            "bed_db_speech": round(power_mean_db(speech), 2), "bed_db_pause": round(power_mean_db(pause), 2),
            "gain_reduction_speech_db": round(mean(gr_speech), 2), "gain_reduction_speech_sd": round(sd(gr_speech), 2),
            "gain_reduction_pause_db": round(mean(gr_pause), 2)}


def limiter_share(narration: Path, duration_s: float, voice_gain_db: float, window_s: float = 0.01) -> dict:
    """How much of the narration the limiter can touch: 10 ms windows whose peak, at the voice's
    gain, would pass LIMIT_DB."""
    samples = int(round(window_s * RATE))
    mono = channels_of(narration) == 1
    af = (f"aresample={RATE},volume={voice_gain_db:.3f}dB,asetnsamples={samples},astats=metadata=1:reset=1,"
          "ametadata=mode=print:key=lavfi.astats.Overall.Peak_level")
    log = run([FFMPEG, "-hide_banner", "-nostats", "-t", f"{duration_s}", "-i", str(narration), "-af", af,
               "-f", "null", "-"])
    peaks = [float(v) for v in re.findall(r"Peak_level=(-?[\d.]+)", log)]
    over = [p for p in peaks if p > LIMIT_DB]
    return {"windows": len(peaks), "over_limit": len(over), "share_pct": round(100 * len(over) / max(1, len(peaks)), 3),
            "worst_db_over": round(max(over) - LIMIT_DB, 2) if over else 0.0, "mono": mono}


# ---------------------------------------------------------------- ElevenLabs

def probe_elevenlabs() -> dict[str, str]:
    """Ask ElevenLabs, with the smallest possible requests, which sound endpoints this key may use."""
    import httpx

    sys.path.insert(0, str(ROOT))
    from app.config import settings

    headers = {"xi-api-key": settings.elevenlabs_api_key or ""}
    probes = {
        "sound_generation": ("https://api.elevenlabs.io/v1/sound-generation",
                             {"text": "a single soft camera shutter click", "duration_seconds": 0.5}),
        "music_generation": ("https://api.elevenlabs.io/v1/music",
                             {"prompt": "dark low ambient pad", "music_length_ms": 3000, "force_instrumental": True}),
    }
    out = {}
    for name, (url, body) in probes.items():
        r = httpx.post(url, headers=headers, json=body, params={"output_format": "mp3_44100_128"}, timeout=180)
        if r.status_code == 200:
            out[name] = "allowed"
        else:
            try:
                out[name] = f"{r.status_code}: {r.json()['detail']['message']}"
            except Exception:
                out[name] = f"{r.status_code}: {r.text[:200]}"
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("video_dir", nargs="?", default=str(VIDEO_DIR))
    parser.add_argument("--preview", type=float, default=None, help="mix only the first N seconds")
    parser.add_argument("--probe", action="store_true", help="check the ElevenLabs key's sound permissions")
    args = parser.parse_args()
    if args.probe:
        for name, status in probe_elevenlabs().items():
            print(f"{name}: {status}")
        return

    video_dir = Path(args.video_dir)
    sound_dir = video_dir / "sound"
    plan = json.loads((video_dir / "shots.json").read_text(encoding="utf-8"))
    narration = Path(plan["audio"])
    full = duration_of(narration)
    duration = min(args.preview, full) if args.preview else full
    started = time.time()

    fetch_sources(sound_dir)
    credits = write_credits(sound_dir)
    kit = build_kit(sound_dir)
    tracks = prepare_tracks(sound_dir)
    events = [e for e in events_from_shots(plan["shots"]) if e["t"] < duration]
    counts = {k: sum(1 for e in events if e["kind"] == k) for k in KIT}
    print(f"{duration / 60:.1f} min; effects: {counts}; credits: {credits}")
    stem = "mix_preview" if args.preview else "mix"
    (sound_dir / f"{stem}.events.json").write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")

    lifts = [float(s["start"]) for s in plan["shots"]
             if (s.get("transition_in") or "") == "black" and 0 < float(s["start"]) < duration]
    bed = music_bed(duration, tracks, sound_dir / "work" / f"bed_{int(duration)}.flac", lifts)
    print(f"  bed: {measure(bed)['input_i']:.1f} LUFS ({time.time() - started:.0f} s)")
    out = sound_dir / f"{stem}.wav"
    need = duration * RATE * 2 * 2 * 1.05
    free = shutil.disk_usage(sound_dir).free
    if free < need + 2e9:
        sys.exit(f"only {free / 1e9:.1f} GB free; the master needs {need / 1e9:.1f} GB and 2 GB is kept spare")
    report = mix(narration, bed, events, kit, out, duration)
    final = report["final"]
    print(f"  mix: {out}\n  master {final['input_i']:.2f} LUFS, true peak {final['input_tp']:.2f} dBTP, "
          f"LRA {final['input_lra']:.1f} LU; voice gain {report['voice_gain_db']:+.2f} dB "
          f"({time.time() - started:.0f} s)")
    if args.preview:
        work = sound_dir / "work"
        report["ducking"] = ducking_numbers(narration, bed, duration, report["voice_gain_db"], work)
        report["limiter"] = limiter_share(narration, duration, report["voice_gain_db"])
        report["effects_max_momentary"] = {k: EFFECT_LUFS[k] for k in counts if counts[k]}
        print(f"  ducking: {report['ducking']}\n  limiter: {report['limiter']}")
    else:
        bed.unlink()  # 13 s to rebuild; not worth its disk
    report["seconds"] = round(time.time() - started, 1)
    (sound_dir / f"{stem}.report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

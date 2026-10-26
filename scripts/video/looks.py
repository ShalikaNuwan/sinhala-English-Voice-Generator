"""The film's looks: one grade per kind of scene, chosen by the storyboard, all within one family.

A storyboard scene names its look (story_bible.LOOKS). Each look has a grade for colour pictures and one
for black-and-white ones, a vignette, and the effects it allows. The looks are deliberately subtle: they
shift temperature, saturation and contrast around the film's base grade (`investigation`, the cold,
lightly desaturated grade the first films used), so a cut between scenes reads as a change of mood, not a
different film. Grades are ffmpeg filter chains applied to the still (or to each frame of footage).

Effects are short, at the start of a shot, and only where the storyboard placed them:
- flash: the shot opens from a brief white frame, as on a hit.
- shake: a small camera jolt that settles within half a second.
- flicker: the light stutters for the first fraction of a second, like a tube catching.
"""

from __future__ import annotations

# The base grade, unchanged from the earlier films: investigation is the film's home look.
INVESTIGATION_COLOUR = (
    "eq=contrast=1.06:saturation=0.85,"
    "colorbalance=rs=-0.03:gs=-0.01:bs=0.045:rm=-0.02:bm=0.02:rh=-0.01:bh=0.005,"
    "curves=all='0/0 0.06/0.015 0.25/0.22 0.5/0.50 0.8/0.81 1/0.96'"
)
INVESTIGATION_BW = (
    "format=gray,format=rgb24,"
    "curves=all='0/0 0.06/0.015 0.25/0.22 0.5/0.50 1/0.97',"
    "colorbalance=rs=-0.015:bs=0.025:rm=-0.01:bm=0.018"
)

GRADES: dict[str, dict[str, str]] = {
    "investigation": {"colour": INVESTIGATION_COLOUR, "bw": INVESTIGATION_BW},
    # Childhood, ordinary life: warmer and a little faded (blacks lifted, highlights softened), like old prints.
    "warm_memory": {
        "colour": ("eq=contrast=1.02:saturation=0.9,"
                   "colorbalance=rs=0.035:gs=0.012:bs=-0.03:rm=0.03:gm=0.008:bm=-0.025:rh=0.02:bh=-0.02,"
                   "curves=all='0/0.035 0.25/0.26 0.5/0.52 0.8/0.80 1/0.94'"),
        "bw": ("format=gray,format=rgb24,curves=all='0/0.035 0.25/0.26 0.5/0.51 1/0.95',"
               "colorbalance=rs=0.03:gs=0.012:bs=-0.025:rm=0.03:bm=-0.02"),
    },
    # Suspicion building: colder, less colour, deeper shadows.
    "cold_dread": {
        # Shadows are deepened, not crushed: a night scene must still show what is in it.
        "colour": ("eq=contrast=1.08:saturation=0.70,"
                   "colorbalance=rs=-0.04:gs=-0.01:bs=0.05:rm=-0.025:bm=0.028:rh=-0.012:bh=0.012,"
                   "curves=all='0/0 0.08/0.035 0.25/0.215 0.5/0.485 0.8/0.79 1/0.95'"),
        "bw": ("format=gray,format=rgb24,curves=all='0/0 0.08/0.035 0.25/0.215 0.5/0.485 1/0.96',"
               "colorbalance=rs=-0.03:bs=0.045:rm=-0.02:bm=0.03"),
    },
    # The turning point: hard contrast, colour drained except the reds.
    "revelation": {
        "colour": ("colorhold=color=0xb0141e:similarity=0.2:blend=0.04,"   # only crimson keeps its colour
                   "eq=contrast=1.16:saturation=0.9,"
                   "curves=all='0/0 0.1/0.01 0.3/0.22 0.5/0.48 0.75/0.80 1/0.97'"),
        "bw": "format=gray,format=rgb24,curves=all='0/0 0.1/0.01 0.3/0.22 0.5/0.48 0.75/0.80 1/0.97'",
    },
    # Trial and aftermath: muted and steady, a little warmer than investigation, softer contrast.
    "courtroom": {
        "colour": ("eq=contrast=1.0:saturation=0.62,"
                   "colorbalance=rs=0.01:bs=-0.005:rm=0.012:bm=-0.008,"
                   "curves=all='0/0.02 0.25/0.24 0.5/0.50 0.8/0.79 1/0.95'"),
        "bw": "format=gray,format=rgb24,curves=all='0/0.02 0.25/0.24 0.5/0.50 1/0.95'",
    },
    # Remembering the victims: soft, neutral black and white for everything, no tint, lifted blacks.
    "memorial": {
        "colour": "format=gray,format=rgb24,curves=all='0/0.04 0.25/0.27 0.5/0.52 0.8/0.80 1/0.93'",
        "bw": "format=gray,format=rgb24,curves=all='0/0.04 0.25/0.27 0.5/0.52 0.8/0.80 1/0.93'",
    },
}

VIGNETTES = {
    "investigation": "vignette=angle=PI/7:eval=init",
    "warm_memory": "vignette=angle=PI/8:eval=init",
    "cold_dread": "vignette=angle=PI/5.5:eval=init",
    "revelation": "vignette=angle=PI/5:eval=init",
    "courtroom": "vignette=angle=PI/7:eval=init",
    "memorial": "vignette=angle=PI/9:eval=init",
}

EFFECTS = ("flash", "shake", "flicker")
EFFECTS_ORDER = ("shake", "flicker", "flash")


def look_of(shot: dict) -> str:
    look = shot.get("look") or "investigation"
    return look if look in GRADES else "investigation"


def grade(mode: str, look: str = "investigation") -> str:
    """The grade chain for a picture of `mode` ('colour' or 'bw') in `look`; '' for drawn shots ('none')."""
    if mode not in ("colour", "bw"):
        return ""
    return GRADES.get(look, GRADES["investigation"])[mode]


def vignette(look: str = "investigation") -> str:
    return VIGNETTES.get(look, VIGNETTES["investigation"])


def effect_chain(effects: list[str] | None, width: int, height: int) -> str:
    """Filters for the shot's effects, applied to the moving frame at the shot's start ('' for none)."""
    chain = []
    wanted = set(effects or [])
    for fx in (e for e in EFFECTS_ORDER if e in wanted):   # the picture moves, then the light: a fixed order
        if fx == "shake":
            m = max(8, round(width * 0.008))           # the jolt, in pixels, settling within ~0.5 s
            chain.append(f"crop=w=iw-{2 * m}:h=ih-{2 * m}:x='{m}+{m}*sin(n*2.3)*exp(-n/6)'"
                         f":y='{m}+{m}*cos(n*1.9)*exp(-n/6)',scale={width}:{height},setsar=1")
        elif fx == "flicker":
            chain.append("eq=brightness='if(lt(t,0.6),0.09*sin(t*52)*(1-t/0.6),0)':eval=frame")
        elif fx == "flash":
            chain.append("fade=t=in:st=0:d=0.18:color=white")
    return ",".join(chain)

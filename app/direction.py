"""Directs the voice: who the narrator is, how they speak, what this passage needs, and how much
silence separates passages.

Everything here is a pure function of dictionaries so it can be unit-tested without touching the
network. The pipeline sends the brief to the TTS model as `instructions` and the gaps to the
assembler. Inputs are trusted: style dictionaries come from the validated adaptation schema and
profiles are machine-written by `speaking_profile`, so malformed values are allowed to raise.
"""

from __future__ import annotations


DEFAULT_PERSONA = (
    "You are a real person telling one listener, late at night, about a murder that actually "
    "happened. You are not a presenter and nothing is performed: you are remembering the case "
    "and telling it carefully because the people in it were real. You care about the victims. "
    "Your voice is quiet, close to the microphone, and unhurried."
)

DELIVERY_RULES = (
    "Talk, do not read. Let the rhythm be slightly uneven, the way real speech is.",
    "End statements low and settled. Never lift the pitch at the end of a statement and never sing-song.",
    "Breathe. Take a short breath before a long sentence. An ellipsis is a beat you hold. A dash is a change of thought.",
    "Stay quiet and close. Intensity comes from getting quieter and slower, not louder.",
    "Stay fully intelligible and present. Quiet means close and controlled, never breathy, mumbled, or trailing off.",
    "Say names, dates, times, and numbers carefully, as if you want the listener to remember them.",
    "Deliver quoted speech from 911 calls and witnesses as restrained reportage. Do not act it out.",
    "Sound like someone talking, not presenting: plain and matte, with no announcer lift, no advert brightness, and no smile in the voice.",
)

DEFAULT_BEAT = "build"
DEFAULT_EMOTION = "neutral"

# Silence between assembled segments.
MIN_GAP_MS = 300
DEFAULT_GAP_MS = 600
DEFAULT_MAX_GAP_MS = 2500
# The TTS endpoint accepts speed 0.25-4.0.
SPEED_RANGE = (0.25, 4.0)
DEFAULT_SPEED = 1.0


def _character(profile: dict) -> str | None:
    """Describe the original speaker so the English narrator can match them."""
    derived = profile.get("derived") or {}
    parts: list[str] = []
    if profile.get("described"):
        parts.append(profile["described"][:600])
    if derived:
        parts.append(
            f"Baseline pace is {derived.get('pace') or 'moderate'}, "
            f"with {derived.get('pause_style') or 'deliberate'} pauses and "
            f"{derived.get('dynamics') or 'controlled'} delivery."
        )
    return " ".join(parts) if parts else None


def _moment(style: dict | None) -> str:
    """What this passage needs, from the adaptation model."""
    style = style or {}
    text = (
        f"This passage is a {style.get('beat') or DEFAULT_BEAT} beat: "
        f"{style.get('emotion') or DEFAULT_EMOTION} in tone, pace {style.get('pace') or 'moderate'}."
    )
    note = str(style.get("delivery") or "")[:300].strip()
    if note:
        terminal = "" if note.endswith((".", "!", "?", "…")) else "."
        text += f" Direction for this passage, within the rules above: {note}{terminal}"
    raw = style.get("emphasis") or []
    if isinstance(raw, str):
        raw = [raw]
    emphasis = ", ".join(str(x) for x in raw if x)
    if emphasis:
        text += f" Emphasise: {emphasis}."
    return text


def _continuity(previous_style: dict) -> str:
    return (
        f"Continuity: the previous passage was a {previous_style.get('beat') or DEFAULT_BEAT} beat and "
        f"ended {previous_style.get('emotion') or DEFAULT_EMOTION} in tone. Carry that mood into your "
        "first line; do not reset to neutral."
    )


def build_instructions(
    style: dict | None,
    profile: dict | None,
    previous_style: dict | None = None,
    persona: str | None = None,
) -> str:
    """Build the TTS direction: persona, the original speaker's character, how to speak, this moment."""
    sections = [persona or DEFAULT_PERSONA]
    character = _character(profile) if profile else None
    if character:
        sections.append("Voice character, matched to the original narrator: " + character)
    sections.append("How to speak:\n- " + "\n- ".join(DELIVERY_RULES))
    sections.append(_moment(style))
    if previous_style:
        sections.append(_continuity(previous_style))
    return "\n\n".join(sections)


def segment_gap_ms(previous_style: dict | None, next_style: dict | None, profile: dict | None) -> int:
    """How much silence to leave between two assembled segments.

    The adaptation model's pause fields win. When they say nothing, fall back to the source
    narrator's measured mean pause, and never leave a gap longer than their longest pause
    (unless that is below the MIN_GAP_MS floor).
    """
    previous_style = previous_style or {}
    next_style = next_style or {}
    measured = (profile or {}).get("measured") or {}

    gap = int(previous_style.get("pause_after_ms") or 0) + int(next_style.get("pause_before_ms") or 0)
    if gap == 0:
        gap = int(measured.get("mean_pause_ms") or 0) or DEFAULT_GAP_MS

    longest = int(measured.get("longest_pause_ms") or 0)
    ceiling = max(longest, MIN_GAP_MS) if longest else DEFAULT_MAX_GAP_MS
    return max(MIN_GAP_MS, min(gap, ceiling))


def speaking_speed(requested: float | None) -> float:
    """Clamp a configured speed to what the API accepts."""
    if requested is None:
        requested = DEFAULT_SPEED
    low, high = SPEED_RANGE
    return min(high, max(low, float(requested)))

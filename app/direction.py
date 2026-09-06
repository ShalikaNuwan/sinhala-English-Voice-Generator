"""Directs the voice: who the narrator is, how they speak, and what this passage needs.

Everything here is a pure function of dictionaries so it can be unit-tested without
touching the network. The pipeline passes the result to the TTS model as `instructions`.
"""

from __future__ import annotations


DEFAULT_PERSONA = (
    "You are a real person telling one listener, late at night, about a murder that actually "
    "happened. You are not a presenter and nothing is performed: you are remembering the case "
    "and telling it carefully because the people in it were real. You care about the victims. "
    "Your voice is quiet, close to the microphone, and unhurried."
)

DELIVERY_RULES = [
    "Talk, do not read. Let the rhythm be slightly uneven, the way real speech is.",
    "End statements low and settled. Never lift the pitch at the end of a sentence and never sing-song.",
    "Breathe. Take a short breath before a long sentence. An ellipsis is a beat you hold. A dash is a change of thought.",
    "Stay quiet and close. Intensity comes from getting quieter and slower, not louder.",
    "Say names, dates, times, and numbers carefully, as if you want the listener to remember them.",
    "Deliver quoted speech from 911 calls and witnesses as restrained reportage. Do not act it out.",
    "Never sound like an announcer, a voice assistant, or an advert. No brightness, no polish, no smile in the voice.",
]


def _character(profile: dict) -> str | None:
    """Describe the original speaker so the English narrator can match them."""
    derived = profile.get("derived") or {}
    measured = profile.get("measured") or {}
    parts: list[str] = []
    if profile.get("described"):
        parts.append(profile["described"])
    if derived:
        parts.append(
            f"Baseline pace is {derived.get('pace') or 'moderate'}, "
            f"with {derived.get('pause_style') or 'deliberate'} pauses and "
            f"{derived.get('dynamics') or 'controlled'} delivery."
        )
    if measured.get("mean_pause_ms"):
        parts.append(
            f"Leave roughly {measured['mean_pause_ms']}ms between sentences, "
            f"stretching to about {measured.get('longest_pause_ms') or measured['mean_pause_ms']}ms "
            "at the heaviest beats."
        )
    return " ".join(parts) if parts else None


def _moment(style: dict) -> str:
    """What this passage needs, from the adaptation model."""
    text = (
        f"This passage is a {style.get('beat') or 'build'} beat: "
        f"{style.get('emotion') or 'neutral'} in tone, pace {style.get('pace') or 'moderate'}."
    )
    if style.get("delivery"):
        text += f" Direction: {style['delivery']}"
    emphasis = ", ".join(style.get("emphasis") or [])
    if emphasis:
        text += f" Emphasise: {emphasis}."
    return text


def _continuity(previous_style: dict) -> str:
    return (
        f"Continuity: the previous passage was a {previous_style.get('beat') or 'build'} beat and "
        f"ended {previous_style.get('emotion') or 'neutral'} in tone. Carry that mood into your "
        "first line; do not reset to neutral."
    )


def build_instructions(
    style: dict,
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

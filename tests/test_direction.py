from __future__ import annotations

from app import direction


PROFILE = {
    "measured": {"mean_pause_ms": 1321, "longest_pause_ms": 3547},
    "derived": {"pace": "moderate", "pause_style": "deliberate", "dynamics": "moderately dynamic"},
    "described": "Steady, moderate energy. Neutral and factual tone with little fluctuation.",
}
STYLE = {
    "pace": "moderate",
    "emotion": "calm, suspenseful",
    "emphasis": ["dead inside her house"],
    "beat": "reveal",
    "delivery": "Lower your voice as the call cuts out and let the last line hang.",
}
PREVIOUS = {"beat": "build", "emotion": "tense"}


def test_brief_opens_with_the_storyteller_persona():
    text = direction.build_instructions(STYLE, PROFILE)

    assert text.startswith(direction.DEFAULT_PERSONA)


def test_brief_carries_the_narrator_profile_and_the_moment():
    text = direction.build_instructions(STYLE, PROFILE)

    # The original speaker's character.
    assert "Steady, moderate energy" in text
    assert "deliberate" in text
    assert "moderately dynamic" in text
    assert "1321" in text and "3547" in text
    # What this passage needs.
    assert "reveal beat" in text
    assert "calm, suspenseful" in text
    assert "Lower your voice as the call cuts out" in text
    assert "dead inside her house" in text


def test_brief_includes_every_delivery_rule():
    text = direction.build_instructions(STYLE, PROFILE)

    for rule in direction.DELIVERY_RULES:
        assert rule in text


def test_brief_adds_continuity_only_when_a_previous_passage_exists():
    without = direction.build_instructions(STYLE, PROFILE)
    with_previous = direction.build_instructions(STYLE, PROFILE, previous_style=PREVIOUS)

    assert "Continuity" not in without
    assert "Continuity" in with_previous
    assert "build beat" in with_previous
    assert "tense" in with_previous
    assert "do not reset to neutral" in with_previous


def test_brief_falls_back_to_persona_rules_and_moment_without_a_profile():
    text = direction.build_instructions(STYLE, None)

    assert direction.DEFAULT_PERSONA in text
    assert "calm, suspenseful" in text
    assert "Voice character" not in text


def test_brief_uses_measurements_when_the_tone_analysis_failed():
    profile = {**PROFILE, "described": None}

    text = direction.build_instructions(STYLE, profile)

    assert "deliberate" in text
    assert "1321" in text
    assert "None" not in text


def test_brief_never_prints_none_for_missing_style_fields():
    text = direction.build_instructions({"narration_text": "x"}, None, previous_style={})

    assert "None" not in text
    assert "build beat" in text  # default beat


def test_brief_uses_a_project_persona_when_one_is_given():
    text = direction.build_instructions(STYLE, PROFILE, persona="You are a calm history lecturer.")

    assert text.startswith("You are a calm history lecturer.")
    assert direction.DEFAULT_PERSONA not in text

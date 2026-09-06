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
    assert "Baseline pace is moderate, with deliberate pauses and moderately dynamic delivery." in text
    assert "Voice character" in text


def test_brief_uses_documented_defaults_for_missing_style_fields():
    text = direction.build_instructions({}, None)

    assert "This passage is a build beat: neutral in tone, pace moderate." in text


def test_brief_tolerates_a_missing_style_entirely():
    text = direction.build_instructions(None, None)

    assert "This passage is a build beat: neutral in tone, pace moderate." in text


def test_continuity_defaults_when_the_previous_style_is_sparse():
    text = direction.build_instructions(STYLE, None, previous_style={"emotion": "tense"})

    assert "previous passage was a build beat and ended tense in tone" in text


def test_direction_note_is_hedged_and_terminated():
    text = direction.build_instructions({"delivery": "Slow down here", "emphasis": "x"}, None)

    assert "Direction for this passage, within the rules above: Slow down here." in text
    assert "Emphasise: x." in text


def test_direction_note_keeps_its_own_terminal_punctuation():
    hangs = direction.build_instructions({"delivery": "Let the last line hang..."}, None)
    asks = direction.build_instructions({"delivery": "Who would do that?"}, None)

    assert "within the rules above: Let the last line hang..." in hangs
    assert "hang...." not in hangs
    assert "within the rules above: Who would do that?" in asks
    assert "that?." not in asks


def test_direction_note_accepts_a_single_character_ellipsis_and_an_exclamation():
    ellipsis = direction.build_instructions({"delivery": "Let it hang…"}, None)
    shout = direction.build_instructions({"delivery": "Do not rush this!"}, None)

    assert "within the rules above: Let it hang…" in ellipsis
    assert "hang…." not in ellipsis
    assert "within the rules above: Do not rush this!" in shout
    assert "this!." not in shout


def test_blank_direction_note_is_omitted():
    text = direction.build_instructions({"delivery": "   "}, None)

    assert "Direction for this passage" not in text


def test_brief_fills_gaps_in_a_partial_profile():
    text = direction.build_instructions({}, {"derived": {"pace": None}, "measured": {"mean_pause_ms": 900}})

    assert "Baseline pace is moderate, with deliberate pauses and controlled delivery." in text
    assert "the longest pause is about 900ms" in text


def test_brief_uses_a_project_persona_when_one_is_given():
    text = direction.build_instructions(STYLE, PROFILE, persona="You are a calm history lecturer.")

    assert text.startswith("You are a calm history lecturer.")
    assert direction.DEFAULT_PERSONA not in text


MEASURED_PROFILE = {"measured": {"mean_pause_ms": 773, "longest_pause_ms": 2732}}


def test_gap_is_the_pause_after_plus_the_pause_before():
    gap = direction.segment_gap_ms({"pause_after_ms": 800}, {"pause_before_ms": 400}, MEASURED_PROFILE)

    assert gap == 1200


def test_gap_falls_back_to_the_source_narrators_mean_pause_when_styles_say_nothing():
    gap = direction.segment_gap_ms({"pause_after_ms": 0}, {"pause_before_ms": 0}, MEASURED_PROFILE)

    assert gap == 773


def test_gap_falls_back_to_a_default_without_a_profile():
    assert direction.segment_gap_ms({}, {}, None) == direction.DEFAULT_GAP_MS
    assert direction.segment_gap_ms(None, None, {"measured": {"mean_pause_ms": 0}}) == direction.DEFAULT_GAP_MS


def test_gap_never_drops_below_the_minimum():
    gap = direction.segment_gap_ms({"pause_after_ms": 100}, {"pause_before_ms": 50}, MEASURED_PROFILE)

    assert gap == direction.MIN_GAP_MS


def test_gap_never_exceeds_the_source_narrators_longest_pause():
    gap = direction.segment_gap_ms({"pause_after_ms": 3000}, {"pause_before_ms": 3000}, MEASURED_PROFILE)

    assert gap == 2732


def test_gap_uses_the_default_ceiling_without_a_measured_longest_pause():
    gap = direction.segment_gap_ms({"pause_after_ms": 3000}, {"pause_before_ms": 3000}, None)

    assert gap == direction.DEFAULT_MAX_GAP_MS


def test_speed_is_clamped_to_the_api_range():
    assert direction.speaking_speed(1.0) == 1.0
    assert direction.speaking_speed(0.92) == 0.92
    assert direction.speaking_speed(0.1) == 0.25
    assert direction.speaking_speed(9.0) == 4.0


def test_gap_floor_wins_over_an_implausibly_short_longest_pause():
    gap = direction.segment_gap_ms({"pause_after_ms": 0}, {"pause_before_ms": 0}, {"measured": {"mean_pause_ms": 100, "longest_pause_ms": 100}})

    assert gap == direction.MIN_GAP_MS


def test_speed_boundaries_are_inclusive():
    assert direction.speaking_speed(0.25) == 0.25
    assert direction.speaking_speed(4.0) == 4.0

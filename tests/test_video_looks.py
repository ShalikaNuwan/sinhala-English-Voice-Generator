from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import looks  # noqa: E402


def test_every_look_has_a_colour_and_a_black_and_white_grade_and_a_vignette():
    for look in ("investigation", "warm_memory", "cold_dread", "revelation", "courtroom", "memorial"):
        assert looks.grade("colour", look) and looks.grade("bw", look)
        assert looks.vignette(look).startswith("vignette=")


def test_drawn_shots_get_no_grade_and_an_unknown_look_falls_back_to_the_base_grade():
    assert looks.grade("none", "revelation") == ""
    assert looks.grade("colour", "neon") == looks.INVESTIGATION_COLOUR
    assert looks.look_of({"look": "neon"}) == "investigation" and looks.look_of({}) == "investigation"


def test_a_memorial_turns_even_colour_pictures_to_soft_black_and_white():
    assert looks.grade("colour", "memorial").startswith("format=gray")


def test_effects_become_filters_in_the_order_given_and_nothing_for_none():
    chain = looks.effect_chain(["shake", "flash"], 1920, 1080)
    assert chain.index("crop=") < chain.index("fade=t=in") and "color=white" in chain
    assert "scale=1920:1080" in chain
    assert looks.effect_chain([], 1920, 1080) == "" and looks.effect_chain(None, 1920, 1080) == ""

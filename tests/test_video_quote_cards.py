from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import graphics  # noqa: E402
import quote_cards  # noqa: E402


def quote(**kw):
    q = {"id": "q1", "text": "I didn't do any of this, and I don't know anything about it.", "start": 12.0, "end": 15.0,
         "speaker": "John Wayne Gacy", "setting": "to detectives", "date": "", "style": "card",
         "lines": [{"text": "I didn't do any of this,", "at": 12.0}, {"text": "and I don't know", "at": 13.2},
                   {"text": "anything about it.", "at": 14.1}], "pages": [[0, 1], [2]]}
    q.update(kw)
    return q


def test_the_card_comes_up_just_before_the_first_word_and_leaves_after_the_last_or_before_the_shot_ends():
    assert quote_cards.window(quote(), shot_start=10.0, shot_s=20.0) == (1.88, 6.5)
    begin, end = quote_cards.window(quote(), shot_start=10.0, shot_s=5.5)
    assert begin == 1.88 and end == 5.4


def test_lines_appear_as_they_are_said_and_only_the_current_page_shows():
    q = quote()
    assert quote_cards.line_alphas(q, 12.0) == [1.0, 0.0, 0.0]
    assert quote_cards.line_alphas(q, 13.6) == [1.0, 1.0, 0.0]
    assert quote_cards.line_alphas(q, 14.6) == [0.0, 0.0, 1.0]     # the second page replaces the first


def test_the_attribution_uses_only_what_the_narration_states():
    assert quote_cards.attribution(quote()) == "— John Wayne Gacy, to detectives"
    assert quote_cards.attribution(quote(speaker="", setting="")) == ""


def test_a_long_attribution_wraps_to_two_lines_and_ends_in_an_ellipsis():
    fnt = graphics.font(graphics.LABEL, 28)
    lines = quote_cards.attribution_lines("— " + "word " * 60, fnt, 400)
    assert len(lines) == 2 and lines[-1].endswith("…") and all(fnt.getlength(l) <= 400 for l in lines)


def test_nothing_is_drawn_outside_the_cards_window_and_the_card_is_drawn_inside_it():
    state, draw = quote_cards.frame_at([quote()], 0.5, 10.0, 20.0, 640, 360)
    assert state == ("empty",) and draw().getbbox() is None
    state, draw = quote_cards.frame_at([quote()], 3.0, 10.0, 20.0, 640, 360)
    image = draw()
    assert state[0] == "c" and image.getbbox() is not None
    assert image.getpixel((20, 180))[3] == 0              # the speaker's side of the frame stays clear
    assert image.getpixel((600, 180))[3] > 150            # the panel is dark behind the words


def test_written_words_are_drawn_as_an_exhibit_slip():
    state, draw = quote_cards.frame_at([quote(style="exhibit", text="JAS")], 3.0, 10.0, 20.0, 640, 360)
    assert state[0] == "x" and draw().getbbox() is not None


def test_an_attribution_names_without_narrating():
    q = {"speaker": "Robert Jerome Piest", "setting": "to his mother, who was waiting in her car", "date": ""}
    assert quote_cards.attribution(q) == "— Robert Jerome Piest, to his mother"

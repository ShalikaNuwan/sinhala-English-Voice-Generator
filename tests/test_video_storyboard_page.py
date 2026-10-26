from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import storyboard_page  # noqa: E402


def board():
    chapter = {"id": 1, "title": "Rob's Last Shift", "first": 0, "last": 3, "start": 0.0, "end": 20.0, "era": "1978",
               "look": "cold_dread", "intensity": 5, "purpose": "Rob goes missing.", "tone": "uneasy"}
    scene = {"id": 1, "chapter": 1, "title": "The pharmacy", "first": 0, "last": 3, "start": 0.0, "end": 20.0,
             "setting": "Nisson Pharmacy", "story_time": "December 11, 1978", "purpose": "Meet Rob.", "mood": "ordinary",
             "visual_idea": "Snowy street, then the shop.", "look": "cold_dread", "intensity": 5, "beats": [1]}
    beat = {"index": 1, "start": 0.0, "end": 20.0, "narration": "Rob told her, “Wait here.” <b>", "kind": "archival",
            "intent": "A quiet evening turns.", "shot_type": "establishing", "motion": "zoom_in", "transition_in": "black",
            "effects": ["flicker"], "quote_ids": ["q001"], "media": "photo", "asset": None}
    quotes = [{"id": "q001", "text": "Wait here.", "start": 2.0, "kind": "spoken", "speaker": "Robert Piest",
               "show": "card", "style": "card", "sensitive": False},
              {"id": "q002", "text": "A threat.", "start": 9.0, "kind": "spoken", "speaker": "John Wayne Gacy",
               "show": "narration", "style": "card", "sensitive": True}]
    bible = {"logline": "A story.", "chapters": [chapter], "review": {"narration_notes": ["Check the date."]}}
    return {"duration_s": 20.0, "bible": bible, "scenes": [scene], "beats": [beat], "quotes": quotes}


def test_the_page_lays_out_chapters_scenes_beats_and_quotes_and_escapes_the_text():
    page = storyboard_page.render(board(), {1: None})
    assert "1. Rob&#x27;s Last Shift" in page and "The pharmacy" in page and "Nisson Pharmacy" in page
    assert "A quiet evening turns." in page and "Wait here.” &lt;b&gt;" in page      # narration text is escaped
    assert "flicker" in page and "in: black" in page
    assert "shown as a card" in page and "narration only" in page and 'class="flag">sensitive' in page
    assert "Check the date." in page and "1 quote cards" in page


def test_the_tension_curve_has_a_step_per_scene_in_its_looks_colour():
    svg = storyboard_page.tension_svg(board())
    assert svg.count("<rect") == 1 and "#5b7a99" in svg


def test_a_device_beat_takes_its_thumbnail_from_its_subject_or_first_pin():
    library = {"c0011": {"file": "library/c0011.jpg"}, "c0037": {"file": "library/c0037.jpg"}}
    assert storyboard_page.thumb_source({"kind": "profile", "subject_id": "c0011", "media": "motion"}, library) == ("library/c0011.jpg", None)
    assert storyboard_page.thumb_source({"kind": "board", "media": "motion", "pins": [{"library_id": "c0037"}]}, library) == ("library/c0037.jpg", None)
    assert storyboard_page.thumb_source({"media": "clip", "asset": "clips/v1.mp4", "clip_in": 3.0}, library) == ("clips/v1.mp4", 3.5)


def test_a_pictures_check_shows_on_its_beat():
    sb = board()
    sb["beats"][0]["check"] = {"fits": "partly", "era_ok": False, "care_ok": True, "problem": "Modern cars."}
    page = storyboard_page.render(sb, {1: None})
    assert "check: partly, wrong era" in page and "Modern cars." in page
    sb["beats"][0]["check"] = {"fits": "no", "era_ok": True, "care_ok": True, "replaced": "c0155"}
    assert "picture swapped after check (was c0155)" in storyboard_page.render(sb, {1: None})

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import plan_edit  # noqa: E402


def item(i, **extra):
    return {"id": i, "file": f"library/{i}.jpg", "caption": f"picture {i}", "graphic": False,
            "kind": "photo", "colour": "bw", "era": "1978", "quality": 4, "subjects": [], "places": [], **extra}


def shot(index, start=0.0, end=8.0, narration="", **extra):
    return {"index": index, "start": start, "end": end, "narration": narration, "kind": "archival",
            "choices": [], "overlay": None, "transition_in": "dissolve", **extra}


def test_graphic_and_uncaptioned_pictures_never_reach_the_catalogue():
    items = [item("c1"), item("c2", graphic=True), item("c3", caption=""), item("c4"),
             item("c5", excluded=True, excluded_reason="shows Waterloo, Belgium")]
    lines = plan_edit.catalogue(items).splitlines()
    assert [line.split(" |")[0] for line in lines] == ["c1", "c4"]


def test_a_picture_is_not_repeated_inside_the_window_and_falls_back_to_the_next_choice():
    library = {i: item(i) for i in ("a", "b", "c")}
    shots = [shot(n, choices=["a", "b"]) for n in range(1, 4)]
    plan_edit.pick_assets(shots, library)
    # The third shot's choices are both inside the window, so the free picture "c" runs instead.
    assert [s["library_id"] for s in shots] == ["a", "b", "c"]


def test_unknown_or_graphic_choices_are_ignored():
    library = {"a": item("a", graphic=True), "b": item("b")}
    shots = [shot(1, choices=["zzz", "a", "b"])]
    plan_edit.pick_assets(shots, library)
    assert shots[0]["library_id"] == "b"


def test_a_shot_with_no_valid_choice_keeps_no_picture():
    shots = [shot(1, choices=["nope"])]
    plan_edit.pick_assets(shots, {})
    assert shots[0]["library_id"] is None


def test_a_stamp_must_be_spoken_in_its_shot():
    narration = "On December 11th, 1978, a fifteen-year-old boy disappeared."
    assert plan_edit.overlay_is_stated({"style": "stamp", "text": "December 11th, 1978"}, narration)
    assert plan_edit.overlay_is_stated({"style": "stamp", "text": "December 11, 1978"}, narration)
    assert not plan_edit.overlay_is_stated({"style": "stamp", "text": "December 12, 1978"}, narration)
    assert not plan_edit.overlay_is_stated({"style": "stamp", "text": "January 1979"}, narration)
    assert not plan_edit.overlay_is_stated({"style": "stamp", "text": "THE MAN NEXT DOOR"}, narration)


def test_a_day_said_as_a_word_still_matches_a_numbered_stamp():
    assert plan_edit.overlay_is_stated({"style": "stamp", "text": "March 11, 1980"},
                                       "It was the eleventh of March, 1980.")


def test_a_name_card_needs_the_name_in_the_narration():
    narration = "Lieutenant Kozenczak took the case."
    assert plan_edit.overlay_is_stated({"style": "name", "text": "Joseph Kozenczak"}, narration)
    assert not plan_edit.overlay_is_stated({"style": "name", "text": "John Wayne Gacy"}, narration)


def test_invented_text_is_dropped_and_names_appear_once():
    shots = [
        shot(1, narration="Think about it.", overlay={"style": "stamp", "text": "THE MAN NEXT DOOR", "subtext": ""}),
        shot(2, narration="John Wayne Gacy owned a business.", overlay={"style": "name", "text": "John Wayne Gacy", "subtext": "Contractor"}),
        shot(3, narration="Nothing new here."),
        shot(4, narration="Gacy again.", overlay={"style": "name", "text": "John Wayne Gacy", "subtext": "Contractor"}),
    ]
    plan_edit.thin_overlays(shots)
    assert [bool(s["overlay"]) for s in shots] == [False, True, False, False]


def test_overlays_never_land_on_two_shots_in_a_row():
    shots = [
        shot(1, narration="Robert Piest disappeared.", overlay={"style": "name", "text": "Robert Piest", "subtext": "Pharmacy worker, 15"}),
        shot(2, narration="Kim Byers was at the register.", overlay={"style": "name", "text": "Kim Byers", "subtext": "Cashier"}),
    ]
    plan_edit.thin_overlays(shots)
    assert [bool(s["overlay"]) for s in shots] == [True, False]


def test_a_long_or_empty_chart_becomes_a_picture():
    shots = [
        shot(1, start=0, end=5, kind="map", points=[{"name": "A", "x": 0.1, "y": 0.1}, {"name": "B", "x": 0.5, "y": 0.5}]),
        shot(2, start=5, end=17, kind="chart", events=[{"label": "1978", "text": "x"}]),
        shot(3, start=17, end=21, kind="map", points=[]),
    ]
    plan_edit.limit_drawn(shots, duration_s=600)
    assert [s["kind"] for s in shots] == ["map", "period", "period"]
    assert all(s.get("query") for s in shots[1:])


def test_the_film_opens_from_black_and_never_cuts_twice_in_a_row():
    shots = [shot(1, transition_in="cut"), shot(2, transition_in="cut"), shot(3, transition_in="cut")]
    plan_edit.finish_transitions(shots)
    assert [s["transition_in"] for s in shots] == ["black", "cut", "dissolve"]


def test_a_name_card_only_sits_on_a_picture_of_that_person():
    library = {"gacy": item("gacy", subjects=["John Wayne Gacy"]), "shop": item("shop", subjects=[])}
    shots = [
        shot(1, library_id="gacy", overlay={"style": "name", "text": "John Wayne Gacy", "subtext": "Contractor"}),
        shot(2, library_id="shop", overlay={"style": "name", "text": "Kim Byers", "subtext": "Cashier"}),
        shot(3, library_id="shop", overlay={"style": "stamp", "text": "December 11th, 1978", "subtext": ""}),
    ]
    plan_edit.names_match_pictures(shots, library)
    assert [bool(s["overlay"]) for s in shots] == [True, False, True]


def test_verifier_reviews_exclude_and_correct_without_touching_the_library(tmp_path):
    import json
    (tmp_path / "exclusions.json").write_text(json.dumps({"a": "logo"}))
    (tmp_path / "verify_a.json").write_text(json.dumps({
        "exclude": {"b": "shows Waterloo, Belgium"},
        "subjects": {"c": ["Robert Piest"]},
        "caption": {"c": "Robert Piest in a school photograph."},
    }))
    items = [item("a"), item("b"), item("c", subjects=["unknown boy"])]
    plan_edit.apply_reviews(items, tmp_path)
    assert [bool(i.get("excluded")) for i in items] == [True, True, False]
    assert items[2]["subjects"] == ["Robert Piest"] and items[2]["caption"].startswith("Robert Piest")


def test_no_picture_runs_more_than_its_share_of_the_film():
    library = {f"p{n}": item(f"p{n}") for n in range(30)}
    shots = [shot(n, start=n * 9.0, end=n * 9.0 + 9, choices=["p0"]) for n in range(80)]
    plan_edit.pick_assets(shots, library, duration_s=12 * 60)
    assert sum(1 for s in shots if s["library_id"] == "p0") <= plan_edit.max_uses(12 * 60) == 4


def test_when_every_picture_is_spent_the_overflow_is_shared_evenly():
    library = {i: item(i) for i in ("a", "b", "c", "d", "e")}
    shots = [shot(n, start=n * 9.0, end=n * 9.0 + 9, choices=["a"]) for n in range(80)]
    plan_edit.pick_assets(shots, library, duration_s=12 * 60)
    counts = [sum(1 for s in shots if s["library_id"] == pid) for pid in library]
    assert max(counts) - min(counts) <= 1


def test_a_spent_shot_falls_back_to_the_most_relevant_free_picture():
    library = {
        "house": item("house", caption="The brick house on Summerdale Avenue in snow", places=["Norwood Park"]),
        "court": item("court", caption="A courtroom in Chicago"),
        "used": item("used"),
    }
    shots = [shot(1, choices=["used"]), shot(2, narration="Snow fell on the house in Norwood Park.", choices=["used"])]
    plan_edit.pick_assets(shots, library)
    assert [s["library_id"] for s in shots] == ["used", "house"]


def test_clip_segments_become_pickable_entries_and_unusable_ones_are_left_out():
    clips = [{"id": "v0001", "file": "clips/v0001.mp4", "source": "archive.org", "rights": "free", "segments": [
        {"in": 10.0, "out": 19.0, "caption": "Snow on a street at night.", "era": "1970s", "quality": 3, "graphic": False},
        {"in": 30.0, "out": 34.0, "caption": "A crash.", "era": "1970s", "quality": 3, "graphic": True},
        {"in": 50.0, "out": 58.0, "caption": "Blurred.", "era": "1970s", "quality": 1, "graphic": False},
    ]}]
    entries = plan_edit.clip_entries(clips)
    assert list(entries) == ["v0001#0"]
    assert entries["v0001#0"]["length"] == 9.0 and entries["v0001#0"]["media"] == "clip"


def test_a_clip_runs_once_only_where_it_is_long_enough_and_never_as_a_fallback():
    library = {"photo": item("photo"), "other": item("other")}
    library.update(plan_edit.clip_entries([{"id": "v0001", "file": "clips/v0001.mp4", "source": "archive.org", "rights": "free",
        "segments": [{"in": 0.0, "out": 8.0, "caption": "Snow falling.", "era": "1970s", "quality": 3, "graphic": False}]}]))
    shots = [
        shot(1, start=0, end=12, choices=["v0001#0", "photo"]),   # too long for an 8 s clip
        shot(2, start=12, end=19, choices=["v0001#0", "photo"]),  # fits
        shot(3, start=19, end=26, choices=["v0001#0"]),           # already used: a photo fallback, never the clip
    ]
    plan_edit.pick_assets(shots, library)
    assert [s["library_id"] for s in shots] == ["photo", "v0001#0", "other"]


def test_finalise_marks_media_and_carries_clip_ranges():
    library = {"photo": item("photo")}
    library.update(plan_edit.clip_entries([{"id": "v0002", "file": "clips/v0002.mp4", "source": "archive.org", "rights": "free",
        "segments": [{"in": 40.0, "out": 49.0, "caption": "Police car at night.", "era": "1970s", "quality": 4, "graphic": False}]}]))
    shots = [shot(1, start=0, end=8, choices=["v0002#0"]), shot(2, start=8, end=16, choices=["photo"])]
    plan_edit.finalise(shots, library, 16)
    assert shots[0]["media"] == "clip" and shots[0]["asset"] == "clips/v0002.mp4"
    assert (shots[0]["clip_in"], shots[0]["clip_out"]) == (40.0, 49.0)
    assert shots[1]["media"] == "photo" and shots[1]["asset"] == "library/photo.jpg"


def test_a_board_needs_three_real_photographs_and_six_seconds():
    library = {i: item(i) for i in ("a", "b", "c")}
    pins = [{"library_id": i, "label": i, "sub": ""} for i in ("a", "b", "c")]
    shots = [
        shot(1, start=0, end=8, kind="board", pins=pins, links=[{"a": 0, "b": 1}, {"a": 0, "b": 9}]),
        shot(2, start=8, end=12, kind="board", pins=pins, links=[]),          # too short
        shot(3, start=12, end=20, kind="board", pins=pins[:2], links=[]),     # too few pins
    ]
    plan_edit.limit_boards(shots, library, 60 * 30)
    assert [s["kind"] for s in shots] == ["board", "archival", "archival"]
    assert shots[0]["links"] == [{"a": 0, "b": 1}]  # a link to a missing pin is dropped


def test_below_the_colour_target_a_colour_choice_moves_ahead_of_a_grey_one():
    library = {"g1": item("g1", colour="bw"), "g2": item("g2", colour="bw"), "c1": item("c1", colour="colour"),
               "c2": item("c2", colour="colour")}
    shots = [shot(1, choices=["g1", "c1"]), shot(2, choices=["g2", "c2"])]
    plan_edit.pick_assets(shots, library)
    # Nothing shown yet counts as below target, so colour leads; once at 100% the model's order holds.
    assert [s["library_id"] for s in shots] == ["c1", "g2"]


def test_a_profile_needs_stated_fields_the_right_photo_and_happens_once_per_person():
    library = {"piest": item("piest", subjects=["Robert Piest"]), "shop": item("shop", subjects=[])}
    narration = "On December 11th, 1978, a fifteen-year-old boy named Robert Piest disappeared. He was 15."
    fields = [{"label": "NAME", "value": "Robert Piest"}, {"label": "AGE", "value": "15"},
              {"label": "HOBBY", "value": "Chess club captain"}]
    shots = [
        shot(1, start=0, end=8, narration=narration, kind="profile", subject_id="piest", fields=fields),
        shot(2, start=8, end=16, narration=narration, kind="profile", subject_id="piest", fields=fields),
        shot(3, start=16, end=24, narration="Kim Byers was seventeen.", kind="profile", subject_id="shop",
             fields=[{"label": "NAME", "value": "Kim Byers"}, {"label": "AGE", "value": "17"}]),
    ]
    plan_edit.limit_devices(shots, library)
    assert [s["kind"] for s in shots] == ["profile", "archival", "archival"]
    # The invented field is dropped; the stated ones stay.
    assert [f["label"] for f in shots[0]["fields"]] == ["NAME", "AGE"]


def test_exhibits_and_highlights_are_spaced_and_need_their_content():
    library = {"diagram": item("diagram", kind="diagram")}
    shots = [
        shot(1, start=0, end=6, kind="exhibit", subject_id="diagram", tag="Diagram of the house, Dec 1978"),
        shot(2, start=30, end=36, kind="exhibit", subject_id="diagram", tag="Diagram again"),   # too soon
        shot(3, start=40, end=46, kind="highlight", subject_id="diagram", detail="the crawl space"),
        shot(4, start=200, end=206, kind="highlight", subject_id="diagram", detail=""),        # nothing to circle
    ]
    plan_edit.limit_devices(shots, library)
    assert [s["kind"] for s in shots] == ["exhibit", "archival", "highlight", "archival"]
    assert shots[1]["choices"][0] == "diagram"  # a refused device still shows its subject


def test_the_board_grows_from_one_return_to_the_next():
    first = shot(1, kind="board", pins=[{"library_id": "gacy", "label": "Gacy"}, {"library_id": "piest", "label": "Piest"}],
                 links=[{"a": 0, "b": 1}])
    second = shot(2, kind="board", pins=[{"library_id": "house", "label": "House"}, {"library_id": "gacy", "label": "Gacy"}],
                  links=[{"a": 1, "b": 0}])
    plan_edit.grow_boards([first, second])
    assert [p["library_id"] for p in second["pins"]] == ["piest", "house", "gacy"]
    pairs = {(second["pins"][l["a"]]["library_id"], second["pins"][l["b"]]["library_id"]) for l in second["links"]}
    assert pairs == {("gacy", "piest"), ("gacy", "house")}



def _clip_library(n, length=12.0):
    return plan_edit.clip_entries([{"id": f"v{i:04d}", "file": f"clips/v{i:04d}.mp4", "source": "archive.org", "rights": "free",
        "segments": [{"in": 0.0, "out": length, "caption": f"A 1970s city street at night, take {i}.", "era": "1970s",
                      "quality": 3, "graphic": False}]} for i in range(n)])


def test_a_long_stretch_without_footage_gets_a_clip_on_an_atmosphere_shot_only():
    library = {"street": item("street", subjects=[]), "gacy": item("gacy", subjects=["John Wayne Gacy"])}
    library.update(_clip_library(3))
    shots = [shot(n, start=n * 10.0, end=n * 10.0 + 9, kind="period", choices=["gacy" if n % 2 else "street"])
             for n in range(30)]
    plan_edit.pick_assets(shots, library)
    plan_edit.insert_clips(shots, library)
    with_clips = [s for s in shots if s["kind"] == "clip"]
    assert with_clips and all(library[s["library_id"]].get("media") == "clip" for s in with_clips)
    # Never on a shot showing a named person.
    assert all(s["choices"] == ["street"] for s in with_clips)
    starts = [s["start"] for s in with_clips]
    assert all(b - a >= plan_edit.CLIP_MAX_GAP_S for a, b in zip(starts, starts[1:]))


def test_a_fitting_clip_in_the_choices_beats_the_colour_lean_once_clips_are_due():
    library = {"colour": item("colour", colour="colour")}
    library.update(_clip_library(1))
    shots = [shot(1, start=100, end=109, choices=["colour", "v0000#0"])]
    plan_edit.pick_assets(shots, library)
    assert shots[0]["library_id"] == "v0000#0"


def test_a_map_demoted_by_an_old_rule_is_restored_and_maps_are_spaced():
    pts = [{"name": "Des Plaines", "x": 0.3, "y": 0.4}, {"name": "Chicago", "x": 0.7, "y": 0.6}]
    shots = [shot(1, start=0, end=9, kind="period", points=pts), shot(2, start=60, end=69, kind="map", points=pts),
             shot(3, start=400, end=420, kind="map", points=pts)]
    plan_edit.limit_drawn(shots, 600)
    assert [s["kind"] for s in shots] == ["map", "period", "period"]  # too soon; then too long


def test_the_board_returns_at_a_scene_change_with_what_the_film_has_established():
    library = {i: item(i) for i in ("gacy", "piest", "diagram", "card", "street")}
    shots = [
        shot(1, start=0, end=8, kind="profile", subject_id="gacy",
             fields=[{"label": "NAME", "value": "John Wayne Gacy"}, {"label": "ROLE", "value": "Contractor"}]),
        shot(2, start=8, end=16, kind="profile", subject_id="piest",
             fields=[{"label": "NAME", "value": "Robert Piest"}, {"label": "AGE", "value": "15"}]),
        shot(3, start=16, end=24, kind="exhibit", subject_id="diagram", tag="Diagram of the house, Dec 1978"),
        shot(4, start=400, end=409, kind="archival", library_id="street", transition_in="black"),
        shot(5, start=1000, end=1009, kind="archival", library_id="street"),  # the film runs on past the board
    ]
    plan_edit.insert_boards(shots, library)
    board = shots[3]
    assert board["kind"] == "board"
    assert [p["label"] for p in board["pins"]] == ["John Wayne Gacy", "Robert Piest", "Diagram of the house"]
    assert board["links"] == [{"a": 0, "b": 1}, {"a": 0, "b": 2}]



def test_the_board_leaves_out_other_killers_scenery_repeats_and_the_outro():
    library = {i: item(i) for i in ("gacy", "piest", "corll", "card", "postcard", "a", "b", "c")}
    profile = lambda n, sid, name, role: shot(n, start=n * 10.0, end=n * 10.0 + 8, kind="profile", subject_id=sid,
                                              fields=[{"label": "NAME", "value": name}, {"label": "ROLE", "value": role}])
    shots = [
        profile(1, "gacy", "John Wayne Gacy", "Contractor"),
        profile(2, "piest", "Robert Piest", "Pharmacy worker"),
        profile(3, "corll", "Dean Corll", "Serial killer, Houston"),
        shot(4, start=40, end=48, kind="exhibit", subject_id="card", tag="Business card, P.D.M. Contractors"),
        shot(5, start=50, end=58, kind="exhibit", subject_id="postcard", tag="Postcard of a business college"),
        shot(6, start=400, end=409, kind="archival", library_id="a", transition_in="black"),
        shot(7, start=1200, end=1209, kind="archival", library_id="b", transition_in="black"),   # nothing new
        shot(8, start=2990, end=2999, kind="archival", library_id="c", transition_in="black"),   # the outro
    ]
    plan_edit.insert_boards(shots, library)
    assert [p["label"] for p in shots[5]["pins"]] == ["John Wayne Gacy", "Robert Piest", "Business card"]
    assert shots[6]["kind"] == "archival" and shots[7]["kind"] == "archival"

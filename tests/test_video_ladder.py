from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import ladder  # noqa: E402

BIBLE = {"people": [{"name": "Robert Piest"}], "places": [{"name": "Nisson Pharmacy"}, {"name": "Des Plaines"}]}


def beat(index, start, end, link, scene=1, **kw):
    b = {"index": index, "start": start, "end": end, "link": link, "scene": scene, "media": "photo",
         "narration": f"N{index}.", "sentences": [index, index], "choices": [], "library_id": f"p{index}", "quote_ids": []}
    b.update(kw)
    return b


def library(tmp_path):
    lib = {}
    for i in range(1, 6):
        Image.new("RGB", (1600, 1200), (90, 90, 90)).save(tmp_path / f"p{i}.jpg")
        lib[f"p{i}"] = {"id": f"p{i}", "file": f"p{i}.jpg", "caption": "x"}
    return lib


def test_a_hold_continues_the_picture_before_it_and_the_beats_are_renumbered(tmp_path):
    entries = [beat(1, 0, 8, "subject"), beat(2, 8, 15, "hold"), beat(3, 15, 22, "subject")]
    done = ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: None)
    assert done["hold"] == 1
    assert [(e["index"], e["start"], e["end"], e["library_id"]) for e in entries] == [(1, 0, 15, "p1"), (2, 15, 22, "p3")]
    assert entries[0]["narration"] == "N1. N2."


def test_a_hold_too_long_to_merge_reframes_the_picture_before_or_repeats_it_never_its_stale_pick(tmp_path):
    entries = [beat(1, 0, 12, "subject"), beat(2, 12, 30, "hold")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: (0.3, 0.2, 0.6, 0.6))
    # The hold reframes the picture before it; the 12 s portrait itself also cuts to a closer framing (over 10 s).
    assert {e["library_id"] for e in entries} == {"p1"} and entries[-1]["link"] == "reframe" and entries[-1]["crop"]
    entries = [beat(1, 0, 12, "subject"), beat(2, 12, 30, "hold")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: None)
    assert [e["library_id"] for e in entries] == ["p1", "p1"] and not entries[1].get("crop")


def test_a_detail_that_is_the_whole_picture_is_not_a_reframe(tmp_path):
    entries = [beat(1, 0, 8, "subject"), beat(2, 8, 15, "reframe", choices=["p1"])]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: (0.0, 0.0, 1.0, 1.0))
    assert len(entries) == 1 and entries[0]["end"] == 15             # it holds instead


def test_a_long_hold_on_one_photograph_cuts_halfway_to_a_closer_framing(tmp_path):
    entries = [beat(1, 0, 22, "subject")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: (0.4, 0.1, 0.6, 0.5))
    assert [(e["start"], e["end"], bool(e.get("crop"))) for e in entries] == [(0, 11.0, False), (11.0, 22, True)]


def test_a_reframe_runs_the_located_detail_of_a_relevant_photograph(tmp_path):
    entries = [beat(1, 0, 8, "subject"), beat(2, 8, 15, "reframe", choices=["p1"], reframe_on="his face")]
    seen = []
    done = ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}},
                               locate=lambda path, what: seen.append(what) or (0.4, 0.1, 0.6, 0.5))
    assert done["reframe"] == 1 and seen == ["his face"]
    assert entries[1]["library_id"] == "p1" and entries[1]["crop"] == [0.36, 0.052, 0.64, 0.548]


def test_a_reframe_that_cannot_be_located_holds_instead(tmp_path):
    entries = [beat(1, 0, 8, "subject"), beat(2, 8, 15, "reframe", choices=["p1"])]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: None)
    assert len(entries) == 1 and entries[0]["end"] == 15


def test_generated_details_show_objects_and_settings_only_and_are_rationed(tmp_path):
    made = []

    def fake(detail, era, video_dir):
        made.append(detail)
        out = video_dir / "assets" / "generated" / f"d{len(made)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 40)).save(out)
        return out

    entries = [beat(1, 0, 30, "generated", detail="a 1970s cash register on a counter at night"),
               beat(2, 40, 50, "generated", detail="a winter coat on a hook", scene=2),
               beat(3, 300, 310, "generated", detail="a man waiting in a car", scene=3),
               beat(4, 400, 410, "generated", detail="the front of Nisson Pharmacy", scene=4)]
    done = ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {i: {"id": i} for i in range(1, 5)},
                               locate=lambda *a: None, generate=fake)
    assert done["generated"] == 1 and made == ["a 1970s cash register on a counter at night"]
    assert entries[0]["kind"] == "generated" and entries[0]["rights"] == "generated"
    assert [e.get("kind") for e in entries[1:]] == [None, None, None]   # rationed, a person, a named place


def test_a_located_detail_is_framed_with_room_and_never_too_small():
    assert ladder.frame_box((0.4, 0.4, 0.45, 0.45)) == [0.285, 0.285, 0.565, 0.565]


def test_the_illustration_tag_is_drawn_inside_the_frame(tmp_path):
    path = tmp_path / "g.png"
    Image.new("RGB", (1536, 1024), (40, 40, 40)).save(path)
    ladder.tag_illustration(path)
    img = Image.open(path)
    assert img.getpixel((int(1536 * 0.95), int(1024 * 0.95))) == (40, 40, 40)
    assert img.crop((int(1536 * 0.75), int(1024 * 0.88), int(1536 * 0.93), int(1024 * 0.95))).getextrema()[0][1] > 100


def test_a_detail_at_the_edge_is_framed_inside_the_picture():
    assert ladder.frame_box((0.95, 0.0, 1.0, 0.1)) == [0.72, 0.0, 1.0, 0.28]


def test_back_to_back_beats_on_one_persons_photograph_rotate_through_their_others(tmp_path):
    bible = {"people": [{"name": "Robert Piest", "photo_ids": ["p1", "p2", "p3"]}], "places": []}
    entries = [beat(1, 0, 8, "subject", library_id="p1"), beat(2, 8, 16, "subject", library_id="p1"),
               beat(3, 16, 24, "subject", library_id="p1", scene=2)]
    done = ladder.apply_ladder(entries, library(tmp_path), tmp_path, bible, {1: {"id": 1}, 2: {"id": 2}}, locate=lambda *a: None)
    assert done["varied"] == 1 and [e["library_id"] for e in entries] == ["p1", "p2", "p1"]


def test_a_framing_of_a_small_scan_is_refused_as_too_small_to_show(tmp_path):
    Image.new("RGB", (180, 250), (90, 90, 90)).save(tmp_path / "small.jpg")
    assert ladder.located(lambda *a: (0.3, 0.2, 0.7, 0.6), tmp_path / "small.jpg", "face") is None
    Image.new("RGB", (1600, 2000), (90, 90, 90)).save(tmp_path / "big.jpg")
    assert ladder.located(lambda *a: (0.3, 0.2, 0.7, 0.6), tmp_path / "big.jpg", "face") == (0.3, 0.2, 0.7, 0.6)


def test_a_hold_never_stretches_a_clip_past_its_footage(tmp_path):
    clip = dict(media="clip", clip_in=1.0, clip_out=6.5, library_id="v1")
    entries = [beat(1, 0, 5, "action", **clip), beat(2, 5, 12, "hold")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: None)
    assert len(entries) == 2 and entries[0]["end"] == 5


def test_a_hold_after_short_footage_repeats_the_last_relevant_still_not_the_clip(tmp_path):
    entries = [beat(1, 0, 8, "subject", library_id="p1"),
               beat(2, 8, 12, "action", media="clip", library_id="v1", clip_in=1.0, clip_out=5.5),
               beat(3, 12, 30, "hold")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, BIBLE, {1: {"id": 1}}, locate=lambda *a: None)
    assert entries[2]["library_id"] == "p1" and entries[2]["media"] == "photo"


def test_a_long_hold_on_a_photograph_too_small_to_frame_cuts_to_another_of_the_same_person(tmp_path):
    bible = {"people": [{"name": "Robert Piest", "photo_ids": ["p1", "p2"]}], "places": []}
    entries = [beat(1, 0, 24, "subject", library_id="p1")]
    ladder.apply_ladder(entries, library(tmp_path), tmp_path, bible, {1: {"id": 1}}, locate=lambda *a: None)
    assert [(e["library_id"], e["start"], e["end"]) for e in entries] == [("p1", 0, 12.0), ("p2", 12.0, 24)]


def fake_generate(made):
    def gen(detail, era, video_dir):
        made.append(detail)
        out = video_dir / "assets" / "generated" / f"g{len(made)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 40)).save(out)
        return out
    return gen


def test_a_long_hold_on_a_place_gives_its_second_half_to_an_illustration(tmp_path):
    made = []
    entries = [beat(1, 0, 30, "place_era", library_id="aerial"), beat(2, 30, 40, "subject", library_id="p1")]
    details = iter(["a 1950s kitchen table with a radio", "a 1950s hallway with coats on hooks"])
    n = ladder.illustrate_holds(entries, tmp_path, {"people": [{"name": "Robert Piest", "photo_ids": ["p1"]}], "places": []},
                                {1: {"id": 1, "story_time": "1950s"}}, describe=lambda w, s: next(details),
                                generate=fake_generate(made))
    # The aerial keeps 8 s; the 22 s after it is more than one illustration should hold, so it is two.
    assert n == 2 and made == ["a 1950s kitchen table with a radio", "a 1950s hallway with coats on hooks"]
    assert [(e["start"], e["end"], e.get("kind")) for e in entries][:3] == [
        (0, 8.0, None), (8.0, 19.0, "generated"), (19.0, 30, "generated")]
    assert entries[2]["transition_in"] == "cut"


def test_a_long_illustration_whose_second_half_would_be_the_same_picture_runs_on(tmp_path):
    made = []
    entries = [beat(1, 0, 30, "place_era", library_id="aerial")]
    n = ladder.illustrate_holds(entries, tmp_path, {"people": [], "places": []}, {1: {"id": 1}},
                                describe=lambda w, s: "a coat on a hook", generate=fake_generate(made))
    assert n == 1 and [(e["start"], e["end"]) for e in entries] == [(0, 8.0), (8.0, 30)]
    assert entries[1]["rights"] == "generated" and entries[1]["link"] == "generated"


def test_a_persons_photograph_is_never_replaced_and_unsafe_details_are_refused(tmp_path):
    made = []
    bible = {"people": [{"name": "Robert Piest", "photo_ids": ["p1"]}], "places": [{"name": "Des Plaines"}]}
    entries = [beat(1, 0, 30, "subject", library_id="p1"), beat(2, 200, 230, "place_era", library_id="aerial")]
    n = ladder.illustrate_holds(entries, tmp_path, bible, {1: {"id": 1}}, describe=lambda w, s: "a Des Plaines street at night",
                                generate=fake_generate(made))
    assert n == 0 and made == [] and "refused" in entries[1]["ladder_note"]


def test_a_pictures_third_showing_within_two_minutes_becomes_an_illustration_and_they_are_rationed(tmp_path):
    made = []
    entries = [beat(1, 0, 8, "action", library_id="x"), beat(2, 30, 38, "action", library_id="x"),
               beat(3, 60, 68, "action", library_id="x"), beat(4, 72, 80, "action", library_id="x")]
    n = ladder.illustrate_holds(entries, tmp_path, {"people": [], "places": []}, {1: {"id": 1}},
                                describe=lambda w, s: "a coat on a hook", generate=fake_generate(made))
    assert n == 1 and [e.get("kind") for e in entries] == [None, None, "generated", None]   # the 4th is rationed


def test_an_illustration_may_use_ordinary_words_from_a_place_name_but_not_its_proper_name():
    bible = {"people": [{"name": "John Wayne Gacy"}], "places": [{"name": "Gacy's house on Summerdale Avenue"},
                                                                 {"name": "Nisson Pharmacy"}]}
    assert ladder.safe_detail("a 1970s suburban house in winter, porch light on", bible)
    assert ladder.safe_detail("a 1970s pharmacy counter with a cash register", bible)
    assert not ladder.safe_detail("the front of Nisson Pharmacy at night", bible)
    assert not ladder.safe_detail("a snowy Summerdale lawn", bible)


def test_a_misleading_picture_is_replaced_on_every_beat_ahead_of_the_ration(tmp_path):
    library = {"c1": {"id": "c1", "file": "c1.jpg", "caption": "A CVS storefront."},
               "r1": {"id": "r1", "file": "r1.jpg", "caption": "Rob."}}
    bible = {"people": [{"name": "Rob Piest", "photo_ids": ["r1"]}], "places": []}
    beats = [{"index": 1, "scene": 1, "start": 0, "end": 5, "media": "photo", "library_id": "c1", "narration": "a",
              "check": {"misleads": True}},
             {"index": 2, "scene": 1, "start": 5, "end": 10, "media": "photo", "library_id": "r1", "narration": "b"},
             {"index": 3, "scene": 1, "start": 10, "end": 15, "media": "photo", "library_id": "c1", "narration": "c"},
             {"index": 4, "scene": 2, "start": 15, "end": 20, "media": "photo", "library_id": "c1", "narration": "d"}]
    made = []

    def generate(detail, era, video_dir):
        out = video_dir / "assets" / "generated" / f"d{len(made)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        made.append(detail)
        return out

    n = ladder.illustrate_holds(beats, tmp_path, bible, {1: {"id": 1}, 2: {"id": 2}},
                                describe=lambda words, scene: "a pharmacy counter at night", generate=generate,
                                library=library)
    assert n == 3                                   # the CVS sign is wrong in the next scene too
    assert [b.get("kind") for b in beats] == ["generated", None, "generated", "generated"]
    assert beats[1]["library_id"] == "r1"


def test_an_a_b_a_b_loop_takes_another_choice_or_a_new_framing(tmp_path):
    from PIL import Image
    for name in ("a", "b"):
        Image.new("RGB", (2000, 1500)).save(tmp_path / f"{name}.jpg")
    library = {"a": {"id": "a", "file": "a.jpg", "caption": ""}, "b": {"id": "b", "file": "b.jpg", "caption": ""},
               "c": {"id": "c", "file": "c.jpg", "caption": "", "era": "1978"}}
    scenes = {1: {"id": 1, "story_time": "1964"}}

    def beat(i, pid, choices=()):
        return {"index": i, "scene": 1, "start": i * 5, "end": i * 5 + 5, "media": "photo", "library_id": pid,
                "choices": list(choices)}
    beats = [beat(1, "a"), beat(2, "b"), beat(3, "a", ["a", "c"]), beat(4, "b")]
    n = ladder.break_loops(beats, library, tmp_path, lambda photo, what: (0.1, 0.1, 0.5, 0.6), scenes)
    assert n == 1 and beats[3]["library_id"] == "b" and beats[3]["crop"] and beats[3]["link"] == "reframe"
    assert beats[2]["library_id"] == "a"          # c is dated after the scene, so it is not the way out


def test_a_possessive_is_an_object_not_a_person():
    assert ladder.safe_detail("a boy’s blue jacket on a chair", {"people": [], "places": []})
    assert not ladder.safe_detail("a boy waiting by a car", {"people": [], "places": []})


def test_a_persons_third_showing_in_two_minutes_gives_way_and_a_photo_dated_after_its_scene_is_replaced(tmp_path):
    made = []
    library = {"p1": {"id": "p1", "file": "p1.jpg", "caption": "Rob."},
               "late": {"id": "late", "file": "l.jpg", "caption": "A street in 1990."}}
    bible = {"people": [{"name": "Rob", "photo_ids": ["p1"]}], "places": []}
    entries = [beat(1, 0, 6, "subject", library_id="p1"), beat(2, 20, 26, "subject", library_id="p1"),
               beat(3, 40, 46, "subject", library_id="p1"), beat(4, 100, 106, "place_era", library_id="late")]
    n = ladder.illustrate_holds(entries, tmp_path, bible, {1: {"id": 1, "story_time": "1978"}},
                                describe=lambda w, s: f"a coat on a hook {len(made)}", generate=fake_generate(made),
                                library=library)
    assert n == 2 and [e.get("kind") for e in entries] == [None, None, "generated", "generated"]


def test_illustrations_run_on_rather_than_pile_up_and_a_crowded_scene_falls_back_to_its_subject(tmp_path):
    made = []
    library = {"bad": {"id": "bad", "file": "b.jpg", "caption": "A CVS sign."},
               "p1": {"id": "p1", "file": "p.jpg", "caption": "Rob.", "subjects": ["Rob"]}}
    bible = {"people": [{"name": "Rob", "photo_ids": ["p1"]}], "places": []}
    entries = [beat(k + 1, k * 6, k * 6 + 6, "hold", library_id="bad", check={"misleads": True}) for k in range(5)]
    entries[4]["choices"] = ["p1"]
    n = ladder.illustrate_holds(entries, tmp_path, bible, {1: {"id": 1, "start": 0, "end": 30}},
                                describe=lambda w, s: f"a pharmacy counter {len(made)}", generate=fake_generate(made),
                                library=library)
    # 6 s + 6 s run on as one illustration; past 40% of the scene it may run on to 18 s; the 4th beat has no
    # other picture, so one more is drawn; the 5th has Rob's photograph, which a crowded scene prefers.
    assert [(e["start"], e["end"], e.get("kind") or e.get("library_id")) for e in entries] == [
        (0, 18, "generated"), (18, 24, "generated"), (24, 30, "p1")]
    assert n == 2


def test_after_24_seconds_of_one_persons_photographs_the_next_beat_cuts_away(tmp_path):
    made = []
    library = {f"p{k}": {"id": f"p{k}", "file": "p.jpg", "caption": "Rob.", "subjects": ["Rob"]} for k in range(3)}
    bible = {"people": [{"name": "Rob", "photo_ids": ["p0", "p1", "p2"]}], "places": []}
    entries = [beat(k + 1, k * 9, k * 9 + 9, "subject", library_id=f"p{k % 3}") for k in range(3)]
    n = ladder.illustrate_holds(entries, tmp_path, bible, {1: {"id": 1}}, describe=lambda w, s: "a pharmacy counter",
                                generate=fake_generate(made), library=library)
    assert n == 1 and [e.get("kind") for e in entries] == [None, None, "generated"]



def test_a_near_duplicate_of_the_picture_before_runs_on_as_one_shot(tmp_path):
    from PIL import Image, ImageDraw
    for name, shift in (("a", 0), ("b", 2), ("c", 300)):
        img = Image.new("RGB", (800, 600), "white")
        ImageDraw.Draw(img).rectangle((100 + shift, 100, 400 + shift, 500), fill="black")
        img.save(tmp_path / f"{name}.jpg")
    library = {k: {"id": k, "file": f"{k}.jpg", "caption": ""} for k in "abc"}
    entries = [beat(1, 0, 5, "place_era", library_id="a"), beat(2, 5, 9, "place_era", library_id="b"),
               beat(3, 9, 14, "place_era", library_id="c")]
    ladder.illustrate_holds(entries, tmp_path, {"people": [], "places": []}, {1: {"id": 1}},
                            describe=lambda w, s: "x", generate=fake_generate([]), library=library)
    assert [(e["start"], e["end"], e["library_id"]) for e in entries] == [(0, 9, "a"), (9, 14, "c")]

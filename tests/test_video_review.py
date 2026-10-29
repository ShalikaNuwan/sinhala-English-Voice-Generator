import json
import subprocess
import time
from pathlib import Path

import pytest

from app import video_review as review


def make_film(root: Path, name="demo") -> Path:
    d = root / name
    (d / "library").mkdir(parents=True)
    (d / "assets" / "generated").mkdir(parents=True)
    items = [
        {"id": "c1", "file": "library/c1.jpg", "caption": "Rex Heuermann in 2023.", "found_by": "case:Category:Rex",
         "rights": "free", "licence": "Public domain", "source": "commons"},
        {"id": "c2", "file": "library/c2.jpg", "caption": "A Long Island street, 1970s.", "found_by": "period",
         "rights": "free", "licence": "Public domain", "source": "commons"},
        {"id": "c3", "file": "library/c3.jpg", "caption": "Oak Beach, 2010.", "found_by": "need:oak beach",
         "rights": "free", "licence": "CC0", "source": "openverse"},
        {"id": "bad", "file": "library/bad.jpg", "caption": "Remains.", "found_by": "case", "rights": "free",
         "licence": "Public domain", "graphic": True},
    ]
    for i in items:
        (d / i["file"]).write_bytes(b"jpg")
    (d / "library.json").write_text(json.dumps({"items": items}))
    (d / "clips.json").write_text(json.dumps({"items": [{"id": "v1", "file": "clips/v1.mp4", "source": "archive",
        "licence": "Public domain", "rights": "free", "segments": [
            {"in": 0.0, "out": 3.0, "caption": "Police car, 1970s.", "quality": 4},
            {"in": 0.0, "out": 20.0, "caption": "Traffic, 1970s.", "quality": 4}]}]}))
    (d / "assets" / "generated" / "detail_a.png").write_bytes(b"png")
    beats = [
        {"index": 1, "scene": 1, "start": 0.0, "end": 5.0, "kind": "archival", "media": "photo", "link": "subject",
         "library_id": "c1", "asset": "library/c1.jpg", "narration": "Rex Heuermann was arrested.",
         "choices": ["c1", "c2", "c3", "bad", "v1#0", "v1#1"], "link_reason": "Shows Rex.",
         "check": {"fits": "yes", "misleads": False, "problem": ""}},
        {"index": 2, "scene": 1, "start": 5.0, "end": 10.0, "kind": "period", "media": "photo", "link": "place_era",
         "library_id": "c2", "asset": "library/c2.jpg", "narration": "Long Island at night.", "choices": ["c2"]},
        {"index": 3, "scene": 1, "start": 10.0, "end": 14.0, "kind": "generated", "media": "photo", "link": "generated",
         "library_id": None, "asset": "assets/generated/detail_a.png", "detail": "a cash register at dusk",
         "narration": "She worked the register.", "choices": []},
        {"index": 4, "scene": 1, "start": 14.0, "end": 17.0, "kind": "keyfact", "media": "photo", "link": "reframe",
         "library_id": None, "asset": "assets/cards/keyfact_a.png", "key_fact": "blue jacket",
         "narration": "He wore a blue jacket.", "choices": []},
        {"index": 5, "scene": 1, "start": 17.0, "end": 20.0, "kind": "archival", "media": "photo", "link": "reframe",
         "library_id": "c3", "asset": "library/c3.jpg", "crop": [0.1, 0.1, 0.6, 0.6], "narration": "Oak Beach.",
         "choices": ["c3"]},
        {"index": 6, "scene": 1, "start": 20.0, "end": 28.0, "kind": "clip", "media": "clip", "link": "action",
         "library_id": "v1#1", "asset": "clips/v1.mp4", "clip_id": "v1", "clip_in": 0.0, "clip_out": 20.0,
         "narration": "Cars passed.", "choices": ["v1#1"]},
    ]
    sb = {"duration_s": 28.0, "bible": {"people": [], "places": []},
          "scenes": [{"id": 1, "title": "The arrest", "story_time": "2023", "look": "investigation", "start": 0.0,
                      "end": 28.0, "beats": [1, 2, 3, 4, 5, 6]}],
          "quotes": [], "beats": beats}
    (d / "storyboard.json").write_text(json.dumps(sb))
    (d / "shots.json").write_text(json.dumps({"audio": "", "width": 1920, "height": 1080, "fps": 30, "shots": beats}))
    return d


def test_categories_follow_what_each_shot_shows(tmp_path):
    d = make_film(tmp_path)
    view = review.film_view(d)
    cats = {s["n"]: s["category"] for sc in view["scenes"] for s in sc["shots"]}
    assert cats == {1: "actual", 2: "stand-in", 3: "illustration", 4: "graphic", 5: "stand-in", 6: "stand-in"}
    assert view["summary"]["actual"] == 1 and view["summary"]["stand-in"] == 3
    shot3 = view["scenes"][0]["shots"][2]
    assert shot3["illustration"] == "a cash register at dusk" and shot3["kind"] == "illustration"


CASE = {"id": "k", "file": "library/k.jpg", "caption": "Him.", "found_by": "case:Category:Rex"}
PERIOD = {"id": "p", "file": "library/p.jpg", "caption": "A street.", "found_by": "period"}
CASE_CLIP = {"id": "v#0", "media": "clip", "file": "clips/v.mp4", "caption": "Him, filmed.", "found_by": "case"}
LIBRARY = {"k": CASE, "p": PERIOD, "v#0": CASE_CLIP}


@pytest.mark.parametrize("beat, expected", [
    ({"kind": "profile", "media": "motion", "link": "subject", "library_id": None, "subject_id": "k"}, "actual"),
    ({"kind": "archival", "media": "photo", "link": "reframe", "library_id": "p", "asset": "library/p.jpg"},
     "stand-in"),
    ({"kind": "archival", "media": "photo", "link": "reframe", "library_id": "k", "asset": "library/k.jpg"},
     "actual"),
    ({"kind": "archival", "media": "photo", "link": "place_era", "library_id": "k", "asset": "library/k.jpg",
      "owner": {"choice": "k", "at": 1.0}}, "actual"),
    ({"kind": "clip", "media": "clip", "link": "action", "library_id": "v#0", "asset": "clips/v.mp4"}, "actual"),
    ({"kind": "archival", "media": "photo", "link": "subject", "library_id": None}, "missing"),
])
def test_category_rules(beat, expected):
    assert review.category(beat, LIBRARY) == expected


def test_list_films_finds_folders_with_a_storyboard(tmp_path):
    make_film(tmp_path, "one")
    (tmp_path / "not_a_film").mkdir()
    films = review.list_films(tmp_path)
    assert [f["film"] for f in films] == ["one"] and films[0]["shots"] == 6


def test_film_dir_refuses_unknown_and_escaping_names(tmp_path):
    make_film(tmp_path)
    assert review.film_dir(tmp_path, "demo") == tmp_path / "demo"
    for bad in ("nope", "../demo", "", ".hidden"):
        with pytest.raises(review.FilmError):
            review.film_dir(tmp_path, bad)


def test_marks_are_kept_and_dropped_when_the_picture_changes(tmp_path):
    d = make_film(tmp_path)
    review.set_mark(d, 2, "not_ok", "wrong town")
    shot = review.film_view(d)["scenes"][0]["shots"][1]
    assert shot["mark"] == {"status": "not_ok", "note": "wrong town"}
    sb = json.loads((d / "storyboard.json").read_text())
    sb["beats"][1]["asset"] = "library/c3.jpg"
    (d / "storyboard.json").write_text(json.dumps(sb))
    assert review.film_view(d)["scenes"][0]["shots"][1]["mark"] is None
    review.set_mark(d, 2, None, "")
    assert "2" not in json.loads((d / "review.json").read_text())


def test_alternatives_are_the_shots_usable_candidates_and_long_enough_clips(tmp_path):
    d = make_film(tmp_path)
    alt = review.alternatives(d, 1)
    ids = [o["id"] for o in alt["options"]]
    assert ids == ["c2", "c3", "v1#1"]          # not itself, not the graphic one, not the 3 s clip for a 5 s shot
    assert alt["options"][0]["category"] == "stand-in" and alt["illustration"]["allowed"] is True
    graphic = review.alternatives(d, 4)
    assert graphic["options"] == [] and graphic["illustration"]["allowed"] is False


def test_choosing_another_picture_updates_both_files_and_marks_it_the_owners(tmp_path):
    d = make_film(tmp_path)
    shot = review.choose(d, 1, "c2")
    assert shot["picture"]["id"] == "c2" and shot["owner"]["choice"] == "c2"
    sb = json.loads((d / "storyboard.json").read_text())
    shots = json.loads((d / "shots.json").read_text())["shots"]
    assert sb["beats"][0]["library_id"] == "c2" == shots[0]["library_id"]
    assert sb["beats"][0]["asset"] == "library/c2.jpg" and sb["beats"][0]["owner"]["choice"] == "c2"
    clip = review.choose(d, 2, "v1#1")
    assert clip["picture"]["media"] == "clip"
    beat = json.loads((d / "storyboard.json").read_text())["beats"][1]
    assert beat["media"] == "clip" and beat["clip_in"] == 0.0 and beat["clip_out"] == 20.0
    with pytest.raises(review.FilmError):
        review.choose(d, 1, "bad")


def test_a_new_illustration_is_drawn_and_an_unsafe_one_refused(tmp_path):
    d = make_film(tmp_path)

    def generate(detail, era, video_dir):
        out = video_dir / "assets" / "generated" / "detail_new.png"
        out.write_bytes(b"png")
        return out

    shot = review.choose(d, 2, "illustration", describe=lambda words, scene: "an empty road at night",
                         generate=generate)
    assert shot["category"] == "illustration" and shot["illustration"] == "an empty road at night"
    with pytest.raises(review.FilmError):
        review.choose(d, 1, "illustration", describe=lambda words, scene: "a man waiting by a car",
                      generate=generate)


def test_frames_are_current_only_after_a_rebuild(tmp_path):
    d = make_film(tmp_path)
    (d / "animatic.mp4").write_bytes(b"mp4")
    view = review.film_view(d)
    assert view["animatic"] and all(s["frame_current"] for s in view["scenes"][0]["shots"])
    review.choose(d, 1, "c2")
    time.sleep(0.01)
    shots = review.film_view(d)["scenes"][0]["shots"]
    assert shots[0]["frame_current"] is False and shots[1]["frame_current"] is True


def test_no_frame_without_a_review_cut(tmp_path):
    d = make_film(tmp_path)
    assert review.frame(d, 1) is None


@pytest.mark.skipif(not Path(review.VIDEO_FFMPEG).exists(), reason="video ffmpeg not installed")
def test_a_frame_is_grabbed_from_the_review_cut_and_cached(tmp_path):
    d = make_film(tmp_path)
    subprocess.run([review.VIDEO_FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x180:d=30",
                    "-pix_fmt", "yuv420p", str(d / "animatic.mp4")], check=True)
    first = review.frame(d, 2)
    assert first is not None and first.exists() and first.parent == d / "review" / "frames"
    assert review.frame(d, 2) == first
    review.choose(d, 2, "c3")
    assert review.frame(d, 2) is None                 # the cut is older than the owner's choice


def test_thumb_is_the_library_file_for_a_photo(tmp_path):
    d = make_film(tmp_path)
    assert review.thumb(d, "c1") == d / "library" / "c1.jpg"
    assert review.thumb(d, "nope") is None



def test_a_picture_chosen_while_a_rebuild_runs_is_not_in_that_cut(tmp_path):
    d = make_film(tmp_path)
    (d / "review").mkdir()
    started = time.time()
    (d / "review" / "job.json").write_text(json.dumps({"kind": "rebuild", "started_at": started}))
    review.choose(d, 1, "c2")                         # chosen after the rebuild read the storyboard
    time.sleep(0.01)
    (d / "animatic.mp4").write_bytes(b"mp4")          # the cut is written afterwards
    shots = review.film_view(d)["scenes"][0]["shots"]
    assert shots[0]["frame_current"] is False and shots[1]["frame_current"] is True
    assert review.frame(d, 1) is None


def test_the_final_film_is_found_under_its_own_name_but_not_a_sample(tmp_path):
    d = make_film(tmp_path)
    (d / "sample2_1080p.mp4").write_bytes(b"mp4")
    assert review.film_view(d)["final"] is None
    (d / "demo_documentary_1080p.mp4").write_bytes(b"mp4")
    assert review.film_view(d)["final"] == "demo_documentary_1080p.mp4"


def test_a_picture_on_its_own_is_actual_only_when_the_case_searches_found_it():
    assert review.picture_category({"id": "c", "found_by": "case:Category:Rex"}) == "actual"
    assert review.picture_category({"id": "n", "found_by": "news:Arrest, 2023"}) == "actual"
    assert review.picture_category({"id": "w", "found_by": "wikipedia:Volkswagen Beetle"}) == "stand-in"
    assert review.picture_category({"id": "e", "found_by": "entity:Oak Beach"}) == "stand-in"
    assert review.picture_category({"id": "v#0", "media": "clip"}) == "stand-in"
    assert review.found_words({"found_by": "wikipedia:Volkswagen Beetle"}) == "wikipedia: Volkswagen Beetle"


def test_summary_counts_approved_shots(tmp_path):
    d = make_film(tmp_path)
    review.set_mark(d, 1, "ok")
    review.set_mark(d, 2, "ok")
    review.set_mark(d, 3, "not_ok")
    summary = review.film_view(d)["summary"]
    assert summary["approved"] == 2 and summary["regenerating"] == 0
    assert all(s["regen"] is None for s in review.film_view(d)["scenes"][0]["shots"])


def test_approve_unreviewed_skips_marked_shots_and_shots_with_a_review(tmp_path):
    d = make_film(tmp_path)
    review.set_mark(d, 1, "not_ok")
    (d / "review").mkdir()
    (d / "review" / "regen.json").write_text(json.dumps({
        "2": {"review": "show the house", "state": "failed", "error": "x", "at": 1.0},
        "3": {"review": "  ", "state": "done", "at": 1.0}}))
    assert review.approve_unreviewed(d) == 4
    marks = review.load_marks(d)
    assert {k: v["status"] for k, v in marks.items()} == {"1": "not_ok", "3": "ok", "4": "ok", "5": "ok", "6": "ok"}
    assert review.approve_unreviewed(d) == 0

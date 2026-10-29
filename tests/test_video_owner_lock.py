import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import ladder  # noqa: E402
import storyboard  # noqa: E402

OWNER = {"choice": "x", "at": 1.0}


def item(i, caption="A street.", **kw):
    return {"id": i, "file": f"library/{i}.jpg", "caption": caption, **kw}


def beat(i, start, end, pid, **kw):
    return {"index": i, "scene": 1, "start": start, "end": end, "media": "photo", "library_id": pid,
            "asset": f"library/{pid}.jpg", "narration": f"words {i}", "choices": [], **kw}


def test_respect_links_puts_a_beat_back_on_its_own_choice_but_not_the_owners():
    library = {"a": item("a"), "x": item("x")}
    planned = beat(1, 0, 5, "x", link="subject", choices=["a"])
    chosen = beat(2, 5, 10, "x", link="subject", choices=["a"], owner=OWNER)
    storyboard.respect_links([planned, chosen], library)
    assert planned["library_id"] == "a"                  # the rule works
    assert chosen["library_id"] == "x"                   # and leaves the owner's pick


def test_guard_dates_leaves_the_owners_pick():
    library = {"late": item("late", "A street in 1990.", era="1990"), "ok": item("ok", "A street in 1975.")}
    scenes = {1: {"id": 1, "story_time": "1975"}}
    planned = beat(1, 0, 5, "late", choices=["ok"])
    chosen = beat(2, 5, 10, "late", choices=["ok"], owner=OWNER)
    storyboard.guard_dates([planned, chosen], library, scenes)
    assert planned["library_id"] == "ok" and chosen["library_id"] == "late"
    assert "guarded" not in chosen


def test_guard_victims_leaves_the_owners_pick():
    library = {"v": item("v", "Portrait of John Doe."), "ok": item("ok")}
    bible = {"people": [{"name": "John Doe", "victim": True, "photo_ids": ["v"]}]}
    planned = beat(1, 0, 5, "v", choices=["ok"])
    chosen = beat(2, 5, 10, "v", choices=["ok"], owner=OWNER)
    storyboard.guard_victims([planned, chosen], library, bible)
    assert planned["library_id"] == "ok" and chosen["library_id"] == "v"


def test_guard_mourning_leaves_the_owners_pick():
    library = {"f": item("f", "A funeral in the rain."), "ok": item("ok")}
    planned = beat(1, 0, 5, "f", choices=["ok"])
    chosen = beat(2, 5, 10, "f", choices=["ok"], owner=OWNER)
    storyboard.guard_mourning([planned, chosen], library, {"people": []})
    assert planned["library_id"] == "ok" and chosen["library_id"] == "f"


def test_speaker_pictures_leaves_the_owners_pick():
    library = {"s": item("s", "Portrait of Jane Roe."), "x": item("x")}
    bible = {"people": [{"name": "Jane Roe", "photo_ids": ["s"]}]}
    card = [{"speaker": "Jane Roe", "style": "card"}]
    planned = beat(1, 0, 5, "x", quotes=card)
    chosen = beat(2, 100, 105, "x", quotes=card, owner=OWNER)
    storyboard.speaker_pictures([planned, chosen], bible, library)
    assert planned["library_id"] == "s" and chosen["library_id"] == "x"


def test_merge_repeats_never_joins_an_owners_pick():
    scenes = [{"id": 1, "beats": []}]
    plain = [beat(1, 0, 5, "x"), beat(2, 5, 10, "x")]
    storyboard.merge_repeats(plain, scenes)
    assert len(plain) == 1                               # the rule works
    into = [beat(1, 0, 5, "x"), beat(2, 5, 10, "x", owner=OWNER)]
    storyboard.merge_repeats(into, scenes)
    out_of = [beat(1, 0, 5, "x", owner=OWNER), beat(2, 5, 10, "x")]
    storyboard.merge_repeats(out_of, scenes)
    assert len(into) == 2 and len(out_of) == 2


def test_excluded_pictures_are_replaced_even_when_the_owner_chose_them():
    library = {"bad": item("bad", excluded=True), "ok": item("ok")}
    chosen = beat(1, 0, 5, "bad", choices=["ok"], owner=OWNER)
    storyboard.replace_excluded([chosen], library)
    assert chosen["library_id"] == "ok"


def test_vary_subjects_leaves_the_owners_pick():
    library = {"r1": item("r1"), "r2": item("r2"), "r3": item("r3")}
    bible = {"people": [{"name": "Rob", "photo_ids": ["r1", "r2", "r3"]}]}
    plain = [beat(1, 0, 5, "r1"), beat(2, 5, 10, "r1")]
    ladder.vary_subjects(plain, bible, library)
    chosen = [beat(1, 0, 5, "r1"), beat(2, 5, 10, "r1", owner=OWNER)]
    ladder.vary_subjects(chosen, bible, library)
    assert plain[1]["library_id"] != "r1" and chosen[1]["library_id"] == "r1"


def test_split_long_holds_leaves_the_owners_pick(tmp_path):
    library = {"r1": item("r1"), "r2": item("r2")}
    bible = {"people": [{"name": "Rob", "photo_ids": ["r1", "r2"]}]}
    plain = ladder.split_long_holds([beat(1, 0, 20, "r1")], library, tmp_path, lambda *a: None, bible)
    chosen = ladder.split_long_holds([beat(1, 0, 20, "r1", owner=OWNER)], library, tmp_path, lambda *a: None, bible)
    assert len(plain) == 2 and len(chosen) == 1 and chosen[0]["library_id"] == "r1"


def test_break_loops_leaves_the_owners_pick(tmp_path):
    library = {"a": item("a"), "b": item("b"), "c": item("c")}

    def loop(**kw):
        return [beat(1, 0, 5, "a"), beat(2, 5, 10, "b"), beat(3, 10, 15, "a"), beat(4, 15, 20, "b", choices=["c"], **kw)]
    plain, chosen = loop(), loop(owner=OWNER)
    ladder.break_loops(plain, library, tmp_path, lambda *a: None)
    ladder.break_loops(chosen, library, tmp_path, lambda *a: None)
    assert plain[3]["library_id"] == "c" and chosen[3]["library_id"] == "b"


def test_illustrate_holds_leaves_an_owners_long_stand_in(tmp_path):
    made = []

    def generate(detail, era, video_dir):
        out = video_dir / "assets" / "generated" / f"d{len(made)}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x")
        made.append(detail)
        return out

    entries = [beat(1, 0, 30, "aerial", link="place_era", owner=OWNER),
               beat(2, 30, 60, "street", link="place_era")]
    ladder.illustrate_holds(entries, tmp_path, {"people": [], "places": []}, {1: {"id": 1}},
                            describe=lambda w, s: "a kitchen table", generate=generate, library={})
    assert entries[0]["library_id"] == "aerial" and (entries[0]["start"], entries[0]["end"]) == (0, 30)
    assert made == ["a kitchen table"]                    # the plain long hold beside it was illustrated
    assert [(e["library_id"], e.get("kind")) for e in entries[1:]] == [("street", None), (None, "generated")]


def test_apply_ladder_leaves_the_owners_pick(tmp_path):
    Image.new("RGB", (1600, 1200), (90, 90, 90)).save(tmp_path / "a.jpg")
    library = {"a": {"id": "a", "file": "a.jpg", "caption": "x"}, "x": item("x")}
    plain = [beat(1, 0, 8, "a", link="subject"), beat(2, 8, 15, "x", link="hold")]
    ladder.apply_ladder(plain, library, tmp_path, {"people": []}, {1: {"id": 1}}, locate=lambda *a: None)
    chosen = [beat(1, 0, 8, "a", link="subject"), beat(2, 8, 15, "x", link="hold", owner=OWNER)]
    ladder.apply_ladder(chosen, library, tmp_path, {"people": []}, {1: {"id": 1}}, locate=lambda *a: None)
    assert len(plain) == 1                               # a hold runs the picture before on
    assert len(chosen) == 2 and chosen[1]["library_id"] == "x"


def test_apply_ladder_does_not_run_the_next_shot_on_over_the_owners_pick(tmp_path):
    Image.new("RGB", (1600, 1200), (90, 90, 90)).save(tmp_path / "a.jpg")
    library = {"a": {"id": "a", "file": "a.jpg", "caption": "x"}, "x": item("x")}
    chosen = [beat(1, 0, 8, "a", link="subject", owner=OWNER), beat(2, 8, 15, "x", link="hold")]
    ladder.apply_ladder(chosen, library, tmp_path, {"people": []}, {1: {"id": 1}}, locate=lambda *a: None)
    assert chosen[0]["end"] == 8                         # the owner's shot keeps its length


def test_a_replaced_excluded_picture_is_no_longer_the_owners_choice():
    library = {"bad": item("bad", excluded=True), "ok": item("ok")}
    chosen = beat(1, 0, 5, "bad", choices=["ok"], owner=OWNER)
    storyboard.replace_excluded([chosen], library)
    assert "owner" not in chosen

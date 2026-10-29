from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import storyboard  # noqa: E402


def sentences(n: int, length: float = 4.0):
    return [{"index": i, "text": f"Sentence {i}.", "start": i * length, "end": (i + 1) * length} for i in range(n)]


def beat(first, last, **kw):
    base = dict(first=first, last=last, kind="archival", choices=[], intent="See it.", subjects=[], shot_type="period")
    base.update(kw)
    return storyboard.StoryBeat(**base)


def scene(first, last, beats, look="investigation", **kw):
    base = dict(title="S", first=first, last=last, setting="Des Plaines", story_time="1978", purpose="p", mood="m",
                visual_idea="v", look=look, intensity=5, beats=beats)
    base.update(kw)
    return storyboard.Scene(**base)


def bible(chapters, people=()):
    return {"chapters": chapters, "people": list(people)}


def chapter(id, first, last, look="investigation"):
    return {"id": id, "first": first, "last": last, "title": f"C{id}", "purpose": "p", "tone": "t",
            "intensity": 5, "era": "1978", "look": look}


def test_a_long_chapter_is_planned_in_even_parts():
    old = storyboard.PART_SENTENCES
    storyboard.PART_SENTENCES = 10
    try:
        assert storyboard.chapter_parts({"first": 0, "last": 24}) == [(0, 8), (9, 17), (18, 24)]
        assert storyboard.chapter_parts({"first": 5, "last": 9}) == [(5, 9)]
    finally:
        storyboard.PART_SENTENCES = old


def test_every_sentence_lands_in_one_scene_and_one_beat_even_when_the_model_leaves_gaps():
    board = storyboard.ChapterBoard(scenes=[
        scene(0, 4, [beat(0, 2), beat(3, 4)]),
        scene(7, 9, [beat(7, 9)], look="revelation"),            # sentences 5-6 left out by the model
    ])
    entries, scenes = storyboard.build_entries({1: [((0, 9), board)]}, bible([chapter(1, 0, 9)]), sentences(10))
    assert [(s["first"], s["last"], s["look"]) for s in scenes] == [(0, 6, "investigation"), (7, 9, "revelation")]
    covered = [i for e in entries for i in range(e["sentences"][0], e["sentences"][1] + 1)]
    assert covered == list(range(10))
    assert {e["scene"] for e in entries} == {1, 2} and all(e["chapter"] == 1 for e in entries)


def test_a_chapter_the_model_failed_on_still_gets_scenes_and_beats_in_its_own_look():
    entries, scenes = storyboard.build_entries({1: [((0, 3), None)]}, bible([chapter(1, 0, 3, look="memorial")]), sentences(4))
    assert len(scenes) == 1 and scenes[0]["look"] == "memorial"
    assert entries and entries[0]["look"] == "memorial"


def test_a_beat_showing_a_victim_is_marked_so_it_gets_no_effects():
    board = storyboard.ChapterBoard(scenes=[scene(0, 3, [beat(0, 3, subjects=["Robert Piest"], effects=["flash"])],
                                                   look="revelation")])
    people = [{"name": "Robert Piest", "victim": True}]
    entries, _ = storyboard.build_entries({1: [((0, 3), board)]}, bible([chapter(1, 0, 3)], people), sentences(4))
    storyboard.keep_budget(entries)
    assert entries[0]["victim_on_screen"] is True and entries[0]["effects"] == []


def test_effects_keep_to_each_looks_budget_per_scene():
    entries = [{"chapter": 1, "scene": 1, "look": "revelation", "effects": ["flash", "shake"]},
               {"chapter": 1, "scene": 1, "look": "revelation", "effects": ["flash", "flicker"]},
               {"chapter": 1, "scene": 2, "look": "courtroom", "effects": ["shake"]},
               {"chapter": 1, "scene": 3, "look": "revelation", "effects": ["flash"]}]
    storyboard.keep_budget(entries)
    assert [e["effects"] for e in entries] == [["flash", "shake"], ["flicker"], [], ["flash"]]


def test_a_quote_card_belongs_to_the_beat_its_first_word_is_spoken_in():
    entries = [{"start": 0.0, "end": 10.0}, {"start": 10.0, "end": 20.0}]
    quotes = [{"id": "q1", "start": 12.5, "show": "card"}, {"id": "q2", "start": 3.0, "show": "narration"},
              {"id": "q3", "start": 9.9, "show": "card"}]
    storyboard.attach_quotes(entries, quotes)
    assert [e["quote_ids"] for e in entries] == [["q3"], ["q1"]]


def test_the_bible_brief_carries_people_with_their_photographs_and_the_care_rules():
    text = storyboard.bible_brief({
        "logline": "L.", "people": [{"name": "Robert Piest", "role": "a victim", "victim": True, "photo_ids": ["c0037"]}],
        "places": [{"name": "Nisson Pharmacy", "era": "1978", "description": "Where Rob worked.", "photo_ids": []}],
        "motifs": [], "chapters": [chapter(1, 0, 9)], "sensitivities": ["Never show remains."]})
    assert "Robert Piest - a victim (victim) | photos: c0037" in text
    assert "Never show remains." in text and "1. C1 (1978; investigation, intensity 5)" in text


def test_a_quote_outside_the_planned_range_is_left_out_of_a_preview():
    entries = [{"start": 0.0, "end": 10.0}]
    storyboard.attach_quotes(entries, [{"id": "q9", "start": 500.0, "show": "card"}])
    assert entries[0]["quote_ids"] == []


def test_the_colourists_looks_reach_the_beats_but_a_memorial_scene_stays_a_memorial():
    scenes = [{"id": 1, "look": "investigation"}, {"id": 2, "look": "memorial"}, {"id": 3, "look": "investigation"}]
    entries = [{"scene": 1, "look": "investigation"}, {"scene": 2, "look": "memorial"}, {"scene": 3, "look": "investigation"}]
    storyboard.apply_looks(scenes, entries, {1: "warm_memory", 2: "revelation"})
    assert [s["look"] for s in scenes] == ["warm_memory", "memorial", "investigation"]
    assert [e["look"] for e in entries] == ["warm_memory", "memorial", "investigation"]


def test_a_victims_photograph_never_sits_under_a_beat_that_does_not_name_them():
    bible_ = {"people": [
        {"name": "Michael Bonnin", "victim": True, "photo_ids": ["c0033"], "aliases": []},
        {"name": "Michael Albrecht", "victim": False, "photo_ids": [], "aliases": ["Mike"]},
        {"name": "John Wayne Gacy", "victim": False, "importance": 3, "photo_ids": ["c0011"], "aliases": ["Gacy"]},
        {"name": "Robert Piest", "victim": True, "photo_ids": ["c0037"], "aliases": ["Rob"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0033", "c0011", "c0037", "p1")}
    entries = [
        {"library_id": "c0033", "narration": "Gacy turned to him: Mike, why is this? We are friends.", "choices": ["c0033"]},
        {"library_id": "c0033", "narration": "Michael Bonnin was seventeen.", "choices": []},
        {"library_id": "c0037", "narration": "Rob told his mother to wait.", "choices": []},
        {"library_id": "c0037", "narration": "Michael was the first to arrive.", "choices": ["p1"]},
    ]
    assert storyboard.guard_victims(entries, library, bible_) == 2
    assert entries[0]["library_id"] == "c0011" and "Bonnin" in entries[0]["guarded"]   # Gacy is named: his photo
    assert entries[1]["library_id"] == "c0033" and entries[2]["library_id"] == "c0037"  # named, so kept
    assert entries[3]["library_id"] == "p1"                                             # a shared first name is not enough


def test_a_family_member_in_the_bible_does_not_stop_a_victims_own_name_counting():
    bible_ = {"people": [
        {"name": "Robert Jerome Piest", "victim": True, "photo_ids": ["c0037"], "aliases": ["Rob", "Robert Piest"]},
        {"name": "Rob Piest’s mother", "victim": False, "photo_ids": [], "aliases": []},
        {"name": "Robert Schultz", "victim": False, "photo_ids": [], "aliases": ["Schultz"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0037", "p1")}
    entries = [{"library_id": "c0037", "narration": "Rob told her to wait here.", "choices": []},
               {"library_id": "c0037", "narration": "Gacy began explaining how he had killed Rob Piest.", "choices": []},
               {"library_id": "c0037", "narration": "Robert Schultz went into the bathroom.", "choices": ["p1"]}]
    assert storyboard.guard_victims(entries, library, bible_) == 1
    assert [e["library_id"] for e in entries] == ["c0037", "c0037", "p1"]


def test_a_beat_that_names_a_different_victim_shows_that_victims_photograph():
    bible_ = {"people": [
        {"name": "John Butkovich", "victim": True, "photo_ids": ["c0031"], "aliases": ["Butkovich"]},
        {"name": "John Anthony Mowery", "victim": True, "photo_ids": ["c0032"], "aliases": ["John Mowery", "Mowery"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0031", "c0032", "p1")}
    entries = [{"library_id": "c0031", "narration": "John Mowery, one of Gacy's victims, and Mike Rossi.", "choices": ["p1"]}]
    storyboard.guard_victims(entries, library, bible_)
    assert entries[0]["library_id"] == "c0032"


def test_a_beat_quoting_someone_shows_them_when_the_bible_has_their_photograph():
    bible_ = {"people": [{"name": "John Wayne Gacy", "aliases": ["Gacy"], "photo_ids": ["c0011", "c0018"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0011", "c0018", "p1")}
    entries = [{"start": 0, "library_id": "c0011", "quotes": []},
               {"start": 20, "library_id": "p1", "quotes": [{"speaker": "John Wayne Gacy", "style": "card"}]},
               {"start": 200, "library_id": "p1", "quotes": [{"speaker": "Nobody Known", "style": "card"}]},
               {"start": 300, "library_id": "p1", "quotes": [{"speaker": "John Wayne Gacy", "style": "exhibit"}]}]
    assert storyboard.speaker_pictures(entries, bible_, library) == 1
    assert entries[1]["library_id"] == "c0018"      # c0011 already runs within the minute
    assert entries[2]["library_id"] == "p1" and entries[3]["library_id"] == "p1"


def test_a_quote_never_sits_on_a_photograph_its_speaker_shares_with_someone_else():
    bible_ = {"people": [{"name": "John Wayne Gacy", "aliases": [], "photo_ids": ["c0001", "c0011"]},
                         {"name": "Rosalynn Carter", "aliases": [], "photo_ids": ["c0001"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0001", "c0011", "p1")}
    entries = [{"start": 0, "library_id": "c0001", "quotes": [{"speaker": "John Wayne Gacy", "style": "card"}]}]
    assert storyboard.speaker_pictures(entries, bible_, library) == 1 and entries[0]["library_id"] == "c0011"


def test_a_minor_persons_photograph_needs_their_name_but_a_central_persons_does_not():
    bible_ = {"people": [
        {"name": "John Wayne Gacy", "importance": 3, "photo_ids": ["c0011", "c0001"], "aliases": ["Gacy"]},
        {"name": "Rosalynn Carter", "importance": 1, "photo_ids": ["c0001", "w0046"], "aliases": []},
        {"name": "John Butkovich", "importance": 2, "victim": True, "photo_ids": ["c0031"], "aliases": ["Butkovich"]}]}
    library = {i: {"id": i, "file": f"library/{i}.jpg", "caption": "x"} for i in ("c0011", "c0001", "w0046", "c0031", "p1")}
    entries = [{"library_id": "w0046", "narration": "John Butkovich was eighteen.", "choices": []},
               {"library_id": "c0001", "narration": "He shook hands with the First Lady.", "choices": []},
               {"library_id": "w0046", "narration": "Rosalynn Carter signed the photograph.", "choices": []}]
    storyboard.guard_victims(entries, library, bible_)
    assert entries[0]["library_id"] == "c0031"                       # the victim it names, not Carter
    assert entries[1]["library_id"] == "c0001" and entries[2]["library_id"] == "w0046"


def test_funeral_and_grave_pictures_stay_in_memorial_passages_or_where_the_person_is_named():
    bible_ = {"people": [{"name": "Timothy McCoy", "victim": True, "photo_ids": [], "aliases": ["McCoy"]}]}
    library = {"g1": {"id": "g1", "file": "g1.jpg", "caption": "A grave marker for Timothy Jack McCoy."},
               "f1": {"id": "f1", "file": "f1.jpg", "caption": "Mourners beside coffins at a funeral."},
               "p1": {"id": "p1", "file": "p1.jpg", "caption": "A 1978 street at night."}}
    entries = [{"library_id": "f1", "narration": "The defence prepared its case.", "choices": [], "look": "investigation"},
               {"library_id": "g1", "narration": "Timothy McCoy was the first.", "choices": [], "look": "investigation"},
               {"library_id": "f1", "narration": "Thirty-three young men.", "choices": [], "look": "memorial"}]
    assert storyboard.guard_mourning(entries, library, bible_) == 1
    assert [e["library_id"] for e in entries] == ["p1", "g1", "f1"]


def test_a_picture_excluded_after_planning_leaves_every_beat_it_is_on():
    library = {"c0003": {"id": "c0003", "file": "x.jpg", "caption": "a stretcher", "excluded": True},
               "p1": {"id": "p1", "file": "p1.jpg", "caption": "a street"}}
    entries = [{"library_id": "c0003", "narration": "n", "choices": ["c0003", "p1"]},
               {"library_id": "p1", "narration": "n", "choices": []}]
    assert storyboard.replace_excluded(entries, library) == 1
    assert [e["library_id"] for e in entries] == ["p1", "p1"]


def test_back_to_back_beats_on_one_photograph_become_one_hold_and_the_scenes_are_renumbered():
    entries = [{"index": 1, "start": 0, "end": 8, "library_id": "c0031", "media": "photo", "scene": 1, "narration": "A.", "sentences": [0, 1]},
               {"index": 2, "start": 8, "end": 15, "library_id": "c0031", "media": "photo", "scene": 1, "narration": "B.", "sentences": [2, 3]},
               {"index": 3, "start": 15, "end": 24, "library_id": "c0031", "media": "photo", "scene": 2, "narration": "C.", "sentences": [4, 4]},
               {"index": 4, "start": 24, "end": 30, "library_id": "p1", "media": "photo", "scene": 2, "narration": "D.", "sentences": [5, 5]}]
    scenes = [{"id": 1, "beats": [1, 2]}, {"id": 2, "beats": [3, 4]}]
    assert storyboard.merge_repeats(entries, scenes) == 1
    assert [(e["index"], e["start"], e["end"], e["sentences"]) for e in entries][:2] == [(1, 0, 15, [0, 3]), (2, 15, 24, [4, 4])]
    assert entries[0]["narration"] == "A. B." and scenes == [{"id": 1, "beats": [1]}, {"id": 2, "beats": [2, 3]}]


def test_a_clip_marked_as_a_still_is_replaced_and_guards_never_swap_in_a_clip():
    library = {"v1#0": {"id": "v1#0", "file": "clips/v1.mp4", "caption": "a clip", "media": "clip", "length": 9.0},
               "p1": {"id": "p1", "file": "p1.jpg", "caption": "a street"}}
    entries = [{"library_id": "v1#0", "media": "photo", "narration": "n", "choices": ["v1#0", "p1"]}]
    assert storyboard.replace_excluded(entries, library) == 1 and entries[0]["library_id"] == "p1"
    bible_ = {"people": [{"name": "Robert Piest", "victim": True, "photo_ids": ["c0037"], "aliases": ["Rob"]}]}
    library["c0037"] = {"id": "c0037", "file": "c0037.jpg", "caption": "Rob"}
    entries = [{"library_id": "c0037", "narration": "Someone else.", "choices": ["v1#0", "p1"]}]
    storyboard.guard_victims(entries, library, bible_)
    assert entries[0]["library_id"] == "p1"


def test_a_linked_beat_runs_its_own_choice_or_holds_never_a_stand_in():
    library = {i: {"id": i, "file": f"{i}.jpg", "caption": "x"} for i in ("c0035", "c0140", "p1")}
    entries = [{"link": "subject", "library_id": "c0140", "choices": ["c0035"], "start": 0, "end": 8},
               {"link": "action", "library_id": "p1", "choices": ["gone"], "start": 8, "end": 16},
               {"link": "place_era", "library_id": "p1", "choices": ["p1"], "start": 16, "end": 20},
               {"link": "hold", "library_id": "c0140", "choices": [], "start": 20, "end": 24}]
    assert storyboard.respect_links(entries, library) == 2
    assert entries[0]["library_id"] == "c0035"                    # its own choice, even as a repeat
    assert entries[1]["link"] == "hold"                           # nothing of its own: holds
    assert entries[2]["library_id"] == "p1" and entries[3]["link"] == "hold"


def test_a_picture_the_ladder_or_a_guard_chose_on_purpose_is_left_alone():
    library = {i: {"id": i, "file": f"{i}.jpg", "caption": "x"} for i in ("a", "b")}
    entries = [{"link": "subject", "library_id": "b", "choices": ["a"], "varied": "a"},
               {"link": "reframe", "library_id": "b", "choices": ["a"], "crop": [0, 0, 0.5, 0.5]},
               {"link": "subject", "library_id": "b", "choices": ["a"], "guarded": "..."}]
    assert storyboard.respect_links(entries, library) == 0 and {e["library_id"] for e in entries} == {"b"}


def test_a_photograph_dated_after_its_scene_is_replaced_or_the_beat_holds():
    library = {"c0001": {"id": "c0001", "file": "a.jpg", "caption": "Rosalynn Carter and Gacy pose together in 1978.", "era": "1970s"},
               "c0010": {"id": "c0010", "file": "b.jpg", "caption": "Gacy and Marlynn serve food in Waterloo.", "era": "1960s"},
               "p1": {"id": "p1", "file": "c.jpg", "caption": "A street.", "era": ""}}
    scenes = {1: {"id": 1, "story_time": "1964-1967"}, 2: {"id": 2, "story_time": "December 1978"}}
    entries = [{"scene": 1, "library_id": "c0001", "choices": ["c0001", "c0010"]},
               {"scene": 1, "library_id": "c0001", "choices": []},
               {"scene": 2, "library_id": "c0001", "choices": []},
               {"scene": 1, "library_id": "p1", "choices": []}]
    assert storyboard.guard_dates(entries, library, scenes) == 2
    assert entries[0]["library_id"] == "c0010" and entries[1]["link"] == "hold"
    assert entries[2]["library_id"] == "c0001" and entries[3]["library_id"] == "p1"
    assert storyboard.years_of("the 1960s, then 1978") == [1960, 1969, 1978]


def test_a_later_portrait_may_identify_a_person_but_not_a_record_or_a_group_photograph():
    scene = {"story_time": "Childhood–1963"}
    portrait = {"caption": "John Wayne Gacy in a head-and-shoulders portrait.", "era": "1970s", "subjects": ["Gacy"]}
    record = {"caption": "A full-length photograph of Gacy beside his parole face sheet, 1968 to 1970.", "era": "1970",
              "subjects": ["Gacy"]}
    group = {"caption": "Rosalynn Carter and Gacy pose together.", "era": "1978", "subjects": ["Carter", "Gacy"]}
    assert not storyboard.from_the_future(portrait, scene)
    assert storyboard.from_the_future(record, scene) and storyboard.from_the_future(group, scene)

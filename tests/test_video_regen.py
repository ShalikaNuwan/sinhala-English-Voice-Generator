import json
import threading
from pathlib import Path

import pytest
from PIL import Image

from app import video_regen as regen
from app import video_review as review
from app.video_regen import Decision, Verdict
from tests.test_video_review import make_film

BOX = (0.3, 0.2, 0.6, 0.7)


def film(tmp_path: Path) -> Path:
    """The review fixture with real photographs (the crop and the checker's picture need them)."""
    d = make_film(tmp_path)
    for name in ("c1", "c2", "c3"):
        Image.new("RGB", (1600, 1200), (90, 90, 90)).save(d / "library" / f"{name}.jpg", "JPEG")
    return d


def files(d: Path) -> tuple:
    return json.loads((d / "storyboard.json").read_text()), json.loads((d / "shots.json").read_text())


def entry(d: Path, n: int) -> dict:
    return review.load_regen(d)[str(n)]


class Fakes:
    """The models and the engine's steps, answering in turn and remembering the shot's step at each call."""

    def __init__(self, d, n, *answers, verdicts=None, box=BOX, new_ids=()):
        self.d, self.n = d, n
        self.answers = list(answers)
        self.verdicts = list(verdicts or [Verdict(satisfied=True, problem="")])
        self.box, self.new_ids = box, list(new_ids)
        self.asked, self.checked, self.searched, self.rendered, self.located = [], [], [], [], []
        self.steps = []

    def step(self, name):
        e = entry(self.d, self.n)
        self.steps.append((name, e["state"], e["step"]))

    def ask(self, system, user):
        self.step("ask")
        self.asked.append((system, user))
        return self.answers.pop(0)

    def check(self, image, review_text, beat, scene, look):
        self.step("check")
        self.checked.append((image, review_text, beat["library_id"], look))
        return self.verdicts.pop(0) if len(self.verdicts) > 1 else self.verdicts[0]

    def search(self, d, queries):
        self.step("search")
        self.searched.append(list(queries))
        lib = json.loads((d / "library.json").read_text())
        for i in self.new_ids:
            Image.new("RGB", (1600, 1200)).save(d / "library" / f"{i}.jpg", "JPEG")
            lib["items"].append({"id": i, "file": f"library/{i}.jpg", "caption": "His house in 1978.",
                                 "found_by": "need:1970s house", "rights": "free", "licence": "Public domain",
                                 "source": "commons", "era": "1978"})
        (d / "library.json").write_text(json.dumps(lib))
        return list(self.new_ids)

    def render(self, d, n):
        self.step("render")
        self.rendered.append(n)
        return f"review/clips/{n}_1.mp4"

    def locate(self, photo, what):
        self.located.append((Path(photo).name, what))
        return self.box

    def generate(self, detail, era, video_dir):
        out = Path(video_dir) / "assets" / "generated" / "detail_new.png"
        out.write_bytes(b"png")
        return out

    def run(self, review_text):
        regen.regenerate(self.d, self.n, review_text, ask=self.ask, check=self.check, search=self.search,
                         render=self.render, locate=self.locate, generate=self.generate)
        return entry(self.d, self.n)


def test_options_are_the_shots_own_pictures_first_then_the_scenes_then_the_rest(tmp_path):
    d = film(tmp_path)
    lib = json.loads((d / "library.json").read_text())
    lib["items"] += [
        {"id": "c4", "file": "library/c4.jpg", "caption": "A courthouse.", "found_by": "entity:Court",
         "rights": "free", "licence": "CC0", "era": "1970s"},
        {"id": "c5", "file": "library/c5.jpg", "caption": "Not free.", "found_by": "case", "rights": "free",
         "licence": "CC BY-NC 2.0"},
        {"id": "c6", "file": "library/c6.jpg", "caption": "Excluded.", "found_by": "case", "rights": "free",
         "licence": "Public domain"}]
    (d / "library.json").write_text(json.dumps(lib))
    (d / "library" / "exclusions.json").write_text(json.dumps({"c6": "wrong man"}))
    assert [o["id"] for o in regen.options(d, 1)] == ["c1", "c2", "c3", "v1#1", "c4"]
    assert [o["id"] for o in regen.options(d, 2)] == ["c2", "c1", "c3", "v1#1", "c4"]
    first = regen.options(d, 1)[0]
    assert first["category"] == "actual" and first["found"] == "case: Category:Rex"


def test_options_are_capped(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(regen, "MAX_OPTIONS", 2)
    assert [o["id"] for o in regen.options(d, 1)] == ["c1", "c2"]


def test_the_choose_prompt_states_the_rules_and_passes_the_review_verbatim(tmp_path):
    d = film(tmp_path)
    sb = review.load_storyboard(d)
    beat, scene = review.find_beat(sb, 2), sb["scenes"][0]
    seen = []
    long_caption = {"id": "c9", "category": "stand-in", "found": "period", "era": "", "media": "photo",
                    "caption": "x" * 300}
    opts = regen.options(d, 2) + [long_caption]

    def ask(system, user):
        seen.append((system, user))
        return Decision(action="library", library_id="c3", explanation="Used c3")

    review_text = 'Too modern; needs the 1970s. "Show the house," not a street!'
    decision = regen.decide(review_text, beat, scene, review.load_library(d)["c2"], opts, True, "", ask)
    assert decision.library_id == "c3"
    system, user = seen[0]
    for rule in ("public domain", "remains", "dignity", "never a person", "named", "stand-in", "pretend",
                 "actual", "era"):
        assert rule in system.lower(), rule
    assert review_text in user
    assert "Long Island at night." in user and "The arrest" in user and "2023" in user
    assert "c2 | stand-in | period | - | A Long Island street, 1970s." in user
    assert "c9 | stand-in | period | - | " + "x" * 160 + "\n" in user + "\n"
    assert "x" * 161 not in user


def test_the_choice_is_told_what_the_shots_around_it_show_so_it_does_not_repeat_them(tmp_path):
    d = film(tmp_path)
    sb = review.load_storyboard(d)
    for b in sb["beats"]:
        b["detail"] = {1: "a late-1960s living room with one armchair", 3: "two armchairs turned apart"}.get(b["index"], "")
    library = review.load_library(d)
    nearby = regen.nearby_pictures(sb, 2, library)
    assert nearby[0] == "1 shot before: a late-1960s living room with one armchair"
    assert "1 shot after: two armchairs turned apart" in nearby
    seen = []

    def ask(system, user):
        seen.append(user)
        return Decision(action="keep", explanation=".")

    beat = review.find_beat(sb, 2)
    regen.decide("change it", beat, sb["scenes"][0], None, [], True, "", ask, nearby=nearby)
    assert "Shots around this one show" in seen[0] and "two armchairs turned apart" in seen[0]


def test_decide_refuses_an_id_not_offered_and_a_search_when_none_is_allowed(tmp_path):
    d = film(tmp_path)
    sb = review.load_storyboard(d)
    beat, scene = review.find_beat(sb, 2), sb["scenes"][0]
    opts = regen.options(d, 2)
    with pytest.raises(regen.RegenError):
        regen.decide("x", beat, scene, None, opts, True, "",
                     lambda s, u: Decision(action="library", library_id="bad", explanation=""))
    seen = []

    def search(system, user):
        seen.append(user)
        return Decision(action="search", queries=["1970s house"], explanation="")

    with pytest.raises(regen.RegenError):
        regen.decide("x", beat, scene, None, opts, False, "", search)
    assert "search is not available" in seen[0].lower()


def test_a_library_pick_is_applied_as_the_owners_and_its_approval_cleared(tmp_path):
    d = film(tmp_path)
    review.set_mark(d, 2, "ok")
    f = Fakes(d, 2, Decision(action="library", library_id="c3", explanation="Used c3, Oak Beach (stand-in)"))
    done = f.run("show the beach")
    beat = review.find_beat(review.load_storyboard(d), 2)
    shots = json.loads((d / "shots.json").read_text())["shots"]
    assert beat["library_id"] == "c3" == shots[1]["library_id"] and beat["owner"]["choice"] == "c3"
    assert "2" not in review.load_marks(d)
    assert done["state"] == "done" and done["clip"] == "review/clips/2_1.mp4" and f.rendered == [2]
    assert done["explanation"] == "Used c3, Oak Beach (stand-in)" and done["review"] == "show the beach"
    assert done["check"] == {"satisfied": True, "problem": ""} and done["error"] == ""
    assert f.checked[0][2] == "c3" and f.checked[0][0][:2] == b"\xff\xd8"     # the checker saw the new JPEG


def test_regen_json_walks_through_its_steps(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 1, Decision(action="search", queries=["1970s house", "", "a", "b", "c"], explanation=""),
              Decision(action="library", library_id="c9", explanation="Used c9"), new_ids=["c9"])
    f.run("show his house")
    assert f.steps == [("ask", "running", "Choosing a picture…"), ("search", "running", "Searching the archives and the web…"),
                       ("ask", "running", "Choosing a picture…"), ("check", "running", "Checking…"),
                       ("render", "running", "Rendering the shot…")]
    assert f.searched == [["1970s house", "a", "b"]]            # empty phrases dropped, three at most


def test_a_focus_frames_a_photo_closer_and_the_same_picture_loses_its_approval(tmp_path):
    d = film(tmp_path)
    review.set_mark(d, 1, "ok")
    f = Fakes(d, 1, Decision(action="library", library_id="c1", focus="his face", explanation="Closer on his face"))
    assert f.run("zoom in on his face")["state"] == "done"
    review.engine()
    import ladder
    beat = review.find_beat(review.load_storyboard(d), 1)
    assert f.located == [("c1.jpg", "his face")] and beat["crop"] == ladder.frame_box(BOX)
    assert "1" not in review.load_marks(d)
    assert review.film_view(d)["scenes"][0]["shots"][0]["kind"] == "closer framing of a photo"


def test_a_focus_that_cannot_be_found_keeps_the_whole_picture_and_says_so(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="library", library_id="c3", focus="the pier", explanation="Used c3"), box=None)
    done = f.run("show the pier")
    assert review.find_beat(review.load_storyboard(d), 2)["crop"] is None
    assert done["state"] == "done" and "the pier" in done["explanation"]


def test_footage_is_never_cropped(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="library", library_id="v1#1", focus="the car", explanation="Used footage"))
    assert f.run("show traffic")["state"] == "done"
    beat = review.find_beat(review.load_storyboard(d), 2)
    assert f.located == [] and beat["crop"] is None and beat["media"] == "clip"


def test_an_illustration_is_drawn_from_the_decisions_detail(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="search", queries=["road at night"], explanation=""),
              Decision(action="illustration", detail="an empty 1970s road at night", explanation="Drew it"))
    assert f.run("night, not day")["state"] == "done"
    beat = review.find_beat(review.load_storyboard(d), 2)
    assert beat["kind"] == "generated" and beat["detail"] == "an empty 1970s road at night"
    assert beat["asset"] == "assets/generated/detail_new.png"


def test_a_refused_drawing_gets_one_more_choice_told_why(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="search", queries=["man by a car"], explanation=""),
              Decision(action="illustration", detail="a man waiting by a car", explanation="Drew him"),
              Decision(action="illustration", detail="an empty parked car at night", explanation="Drew the car"))
    done = f.run("show him waiting")
    assert done["state"] == "done" and done["explanation"] == "Drew the car"
    assert "refused by the illustration rules: 'a man waiting by a car'" in f.asked[2][1]
    assert "search is not available" in f.asked[2][1].lower()


def test_two_refused_drawings_fail_and_leave_the_shot_as_it_was(tmp_path):
    d = film(tmp_path)
    before = files(d)
    f = Fakes(d, 2, Decision(action="search", queries=["man by a car"], explanation=""),
              Decision(action="illustration", detail="a man waiting by a car", explanation="Drew him"),
              Decision(action="illustration", detail="a woman by a car", explanation="Drew her"))
    failed = f.run("show him waiting")
    assert failed["state"] == "failed" and "refused" in failed["error"] and failed["clip"] is None
    assert files(d) == before and f.rendered == []


def test_a_search_offers_only_the_new_pictures_and_no_second_search(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 1, Decision(action="search", queries=["Rex Heuermann house 1978"], explanation=""),
              Decision(action="library", library_id="c9", explanation="Used c9, his house"), new_ids=["c9"])
    assert f.run("show his actual house")["state"] == "done"
    first, second = f.asked[0][1], f.asked[1][1]
    assert "search is not available" not in first.lower() and "c2 |" in first
    assert "search is not available" in second.lower()
    offered = second.split("Options (id | category | where found | era | caption):\n")[1]
    assert offered == "c9 | stand-in | need: 1970s house | 1978 | His house in 1978."
    assert review.find_beat(review.load_storyboard(d), 1)["library_id"] == "c9"


def test_a_search_that_finds_nothing_fails_cleanly(tmp_path):
    d = film(tmp_path)
    before = files(d)
    f = Fakes(d, 1, Decision(action="search", queries=["his house"], explanation=""),
              Decision(action="library", library_id="c2", explanation="Used c2"))
    failed = f.run("show his house")
    assert failed["state"] == "failed" and failed["error"] and files(d) == before


def test_a_rejected_shot_is_chosen_again_once_with_the_checkers_problem(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="library", library_id="c1", explanation="Used c1"),
              Decision(action="library", library_id="c3", explanation="Used c3"),
              verdicts=[Verdict(satisfied=False, problem="this is a man, the review asked for the beach"),
                        Verdict(satisfied=False, problem="still not quite the beach")])
    done = f.run("show the beach")
    assert len(f.asked) == 2 and len(f.checked) == 2 and f.rendered == [2]
    assert "this is a man, the review asked for the beach" in f.asked[1][1]
    assert "this is a man" not in f.asked[0][1]
    assert done["state"] == "done" and done["explanation"] == "Used c3"
    assert done["check"] == {"satisfied": False, "problem": "still not quite the beach"}
    assert review.find_beat(review.load_storyboard(d), 2)["library_id"] == "c3"


def test_a_failure_after_the_picture_changed_restores_the_shot_exactly(tmp_path):
    d = film(tmp_path)
    review.set_mark(d, 1, "ok")
    before = files(d)
    f = Fakes(d, 1, Decision(action="library", library_id="c1", focus="his face", explanation="Closer"))

    def broken(d, n):
        raise RuntimeError("assembly failed: ffmpeg died")

    f.render = broken
    failed = f.run("closer")
    assert failed["state"] == "failed" and "ffmpeg died" in failed["error"]
    assert files(d) == before
    assert review.load_marks(d)["1"]["status"] == "ok"           # still the picture it approved


def test_a_shot_the_engine_makes_cannot_be_regenerated(tmp_path):
    d = film(tmp_path)
    with pytest.raises(review.FilmError):
        regen.Regenerator(run=lambda *a: None).submit(d, 4, "make it red")
    with pytest.raises(review.FilmError):
        regen.Regenerator(run=lambda *a: None).submit(d, 99, "anything")
    with pytest.raises(review.FilmError):
        regen.Regenerator(run=lambda *a: None).submit(d, 1, "   ")


def test_the_worker_runs_one_regeneration_at_a_time_in_order(tmp_path):
    d = film(tmp_path)
    gate, order = threading.Event(), []

    def run(d, n, text):
        order.append((n, text))
        if n == 1:
            assert gate.wait(10)

    worker = regen.Regenerator(run=run)
    queued = worker.submit(d, 1, "first")
    assert queued["state"] == "queued" and queued["step"] == "Queued" and queued["review"] == "first"
    worker.submit(d, 2, "second")
    assert worker.busy(d)
    with pytest.raises(regen.RegenBusy):
        worker.submit(d, 2, "again")
    view = review.film_view(d)
    assert view["summary"]["regenerating"] == 2
    gate.set()
    assert worker.join(10)
    assert order == [(1, "first"), (2, "second")] and not worker.busy(d)


def test_a_crash_in_the_worker_marks_the_shot_failed(tmp_path):
    d = film(tmp_path)

    def run(d, n, text):
        raise ValueError("boom")

    worker = regen.Regenerator(run=run)
    worker.submit(d, 1, "x")
    assert worker.join(10)
    e = entry(d, 1)
    assert e["state"] == "failed" and "boom" in e["error"] and not worker.busy(d)


def test_a_regeneration_left_running_by_another_server_reads_as_interrupted(tmp_path):
    d = film(tmp_path)
    (d / "review").mkdir()
    (d / "review" / "regen.json").write_text(json.dumps({
        "1": {"review": "closer", "state": "running", "step": "Checking…", "explanation": "", "check": None,
              "clip": None, "error": "", "at": 1.0, "worker": "an-older-server"},
        "2": {"review": "beach", "state": "queued", "step": "Queued", "explanation": "", "check": None,
              "clip": None, "error": "", "at": 1.0, "worker": review.WORKER}}))
    view = review.film_view(d)
    shots = view["scenes"][0]["shots"]
    assert shots[0]["regen"] == {"review": "closer", "state": "failed", "step": "", "explanation": "", "check": None,
                                 "clip": None, "error": "interrupted: regenerate again", "at": 1.0}
    assert shots[1]["regen"]["state"] == "queued" and "worker" not in shots[1]["regen"]
    assert shots[2]["regen"] is None
    assert view["summary"]["regenerating"] == 1


def test_render_preview_renders_exactly_that_shot(tmp_path, monkeypatch):
    d = film(tmp_path)
    seen = []

    def run_step(cmd, d, timeout):
        seen.append(cmd)
        Path(cmd[cmd.index("--out") + 1]).write_bytes(b"mp4")

    monkeypatch.setattr(regen, "run_step", run_step)
    (d / "review" / "clips").mkdir(parents=True)
    (d / "review" / "clips" / "5_100.mp4").write_bytes(b"old")
    clip = regen.render_preview(d, 5)
    cmd = seen[0]
    assert cmd[cmd.index("--from") + 1] == "5" and cmd[cmd.index("--shots") + 1] == "1" and "--animatic" in cmd
    assert cmd[2].endswith("scripts/video/build.py") and cmd[3] == str(d)
    assert clip.startswith("review/clips/5_") and clip.endswith(".mp4") and (d / clip).exists()
    assert not (d / "review" / "clips" / "5_100.mp4").exists()          # the older version of this shot goes


def test_search_online_adds_the_needs_runs_the_harvest_and_captions_only_the_new(tmp_path, monkeypatch):
    d = film(tmp_path)
    (d / "library" / "needs.json").write_text(json.dumps([{"query": "1970s pharmacy counter", "beats": [5]}]))
    seen = []

    def run_step(cmd, d, timeout):
        seen.append(cmd)
        if cmd[2].endswith("harvest.py"):
            # Only this shot's searches are in the file while the harvest runs.
            assert [x["query"] for x in json.loads((d / "library" / "needs.json").read_text())] == [
                "1970s house", "1970s pharmacy counter"]
            lib = json.loads((d / "library.json").read_text())
            lib["items"].insert(0, {"id": "c7", "file": "library/c7.jpg", "found_by": "need:1970s house"})
            (d / "library.json").write_text(json.dumps(lib))

    monkeypatch.setattr(regen, "run_step", run_step)
    new = regen.search_online(d, ["1970s house", "1970S PHARMACY COUNTER"], web=lambda q, about: [])
    assert new == ["c7"]
    needs = json.loads((d / "library" / "needs.json").read_text())
    assert needs == [{"query": "1970s house", "beats": []}, {"query": "1970s pharmacy counter", "beats": [5]}]
    harvest, caption = seen
    assert harvest[3:] == [str(d), "--stages", "needs"]
    assert caption[2].endswith("caption.py") and caption[3:] == [str(d), "--only", "c7"]


def test_search_online_runs_only_its_own_searches_and_keeps_the_rest_of_the_list(tmp_path, monkeypatch):
    d = film(tmp_path)
    (d / "library" / "needs.json").write_text(json.dumps([{"query": "film-wide search", "beats": [9]}]))
    during = []
    monkeypatch.setattr(regen, "run_step", lambda cmd, d, timeout: during.append(
        json.loads((d / "library" / "needs.json").read_text())))
    regen.search_online(d, ["oak beach"], web=lambda q, about: [])
    assert during == [[{"query": "oak beach", "beats": []}]]
    assert json.loads((d / "library" / "needs.json").read_text()) == [
        {"query": "oak beach", "beats": []}, {"query": "film-wide search", "beats": [9]}]


def test_search_online_with_nothing_new_does_not_caption(tmp_path, monkeypatch):
    d = film(tmp_path)
    seen = []
    monkeypatch.setattr(regen, "run_step", lambda cmd, d, timeout: seen.append(cmd))
    assert regen.search_online(d, ["anything"], web=lambda q, about: []) == []
    assert len(seen) == 1 and seen[0][2].endswith("harvest.py")


def test_a_name_card_is_drawn_by_the_film_over_the_kept_picture_never_inside_an_illustration(tmp_path):
    d = film(tmp_path)
    decision = regen.Decision(action="keep", name_card="Samuel Cowell", name_role="Ted's grandfather",
                              explanation="Kept the picture and named him.")
    regen.apply(d, 3, decision, locate=None)
    sb, shots = files(d)
    beat = next(b for b in sb["beats"] if b["index"] == 3)
    assert beat["asset"] == "assets/generated/detail_a.png"                       # the picture is kept
    assert beat["overlay"] == {"style": "name", "text": "Samuel Cowell", "subtext": "Ted's grandfather"}
    assert beat["owner"] and next(s for s in shots["shots"] if s["index"] == 3)["overlay"] == beat["overlay"]
    assert "name card drawn over it as clean text: Samuel Cowell (Ted's grandfather)" in regen.look_words(
        beat, decision, {})
    assert "Never writing of any kind" in " ".join(regen.CHOOSE.split())


def test_a_name_card_asked_for_the_whole_shot_holds(tmp_path):
    d = film(tmp_path)
    decision = regen.Decision(action="keep", name_card="Ann Rule", name_role="crime writer", name_hold=True,
                              explanation="Kept her portrait, named for the whole shot.")
    regen.apply(d, 3, decision, locate=None)
    beat = next(b for b in files(d)[0]["beats"] if b["index"] == 3)
    assert beat["overlay"] == {"style": "name", "text": "Ann Rule", "subtext": "crime writer", "hold": True}


def test_a_card_held_for_the_whole_shot_stays_held_when_the_same_name_is_asked_for_again(tmp_path):
    d = film(tmp_path)
    regen.apply(d, 3, regen.Decision(action="keep", name_card="Edmund Kemper", name_hold=True, explanation="."),
                locate=None)
    regen.apply(d, 3, regen.Decision(action="keep", name_card="Edmund Kemper", name_role="named in IQ lists",
                                     explanation="."), locate=None)
    beat = next(b for b in files(d)[0]["beats"] if b["index"] == 3)
    assert beat["overlay"]["hold"] is True and beat["overlay"]["subtext"] == "named in IQ lists"


def test_a_review_asking_to_remove_the_card_takes_it_off_and_keeps_the_picture(tmp_path):
    d = film(tmp_path)
    regen.apply(d, 3, regen.Decision(action="keep", name_card="Arthur Fletcher", explanation="."), locate=None)
    regen.apply(d, 3, regen.Decision(action="keep", remove_card=True, explanation="Took the name off."), locate=None)
    sb, shots = files(d)
    beat = next(b for b in sb["beats"] if b["index"] == 3)
    assert beat["overlay"] is None and beat["asset"] == "assets/generated/detail_a.png"
    assert next(s for s in shots["shots"] if s["index"] == 3)["overlay"] is None


def test_keep_without_a_name_card_changes_nothing(tmp_path):
    d = film(tmp_path)
    before = files(d)
    regen.apply(d, 2, regen.Decision(action="keep", explanation="Nothing better."), locate=None)
    assert files(d) == before


def test_a_drawing_is_never_the_first_answer_a_real_picture_is_searched_for_first(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="illustration", detail="a 1970s tavern doorway at night", explanation="Drew it"),
              Decision(action="illustration", detail="a 1970s tavern doorway at night", explanation="Drew it"),
              Decision(action="library", library_id="c9", explanation="Used c9, the tavern"), new_ids=["c9"])
    done = f.run("show the Sandpiper Tavern")
    assert "an illustration is not available yet" in f.asked[0][1].lower()
    assert "a real picture must be searched for first" in f.asked[1][1]
    assert f.searched == [["a 1970s tavern doorway at night"]]
    assert done["state"] == "done" and review.find_beat(review.load_storyboard(d), 2)["library_id"] == "c9"


def test_a_web_find_is_offered_and_chosen_and_credited_as_fair_use(tmp_path):
    d = film(tmp_path)

    def search(d, queries):
        lib = json.loads((d / "library.json").read_text())
        Image.new("RGB", (1600, 1200)).save(d / "library" / "n1.jpg", "JPEG")
        lib["items"].append({"id": "n1", "file": "library/n1.jpg", "caption": "Liz Kloepfer in 1975.",
                             "found_by": "news:Liz Kloepfer", "rights": "fair_use", "licence": "Copyrighted",
                             "attribution": "example.com", "source": "web", "era": "1975"})
        (d / "library.json").write_text(json.dumps(lib))
        return ["n1"]

    f = Fakes(d, 1, Decision(action="search", queries=["Elizabeth Kloepfer"], explanation=""),
              Decision(action="library", library_id="n1", explanation="Used n1, her photo"))
    f.search = search
    assert f.run("find an actual image of Liz")["state"] == "done"
    assert "n1 | actual | news: Liz Kloepfer | 1975 | Liz Kloepfer in 1975." in f.asked[1][1]
    beat = review.find_beat(review.load_storyboard(d), 1)
    assert beat["library_id"] == "n1" and beat["rights"] == "fair_use"
    assert "n1" not in {o["id"] for o in regen.options(d, 2)}       # not a free picture,
    assert "n1" in {o["id"] for o in regen.options(d, 2, regen.web_finds(d))}  # but offered to later regenerations
    with pytest.raises(review.FilmError):
        review.choose(d, 2, "n1")                                     # and never chosen by hand as a free one


def test_search_online_gives_the_web_search_pages_to_the_harvest(tmp_path, monkeypatch):
    d = film(tmp_path)
    seen = []

    def run_step(cmd, d, timeout):
        if cmd[2].endswith("harvest.py"):
            seen.append((cmd[3:], json.loads((d / "library" / "pages.json").read_text())))

    monkeypatch.setattr(regen, "run_step", run_step)
    pages = [{"page_url": "https://news.example/liz", "subject": "Liz Kloepfer"}]
    asked = []
    regen.search_online(d, ["Elizabeth Kloepfer"], about="Liz",
                        web=lambda q, about: asked.append((q, about)) or pages)
    assert asked == [(["Elizabeth Kloepfer"], "Liz")]
    assert seen == [([str(d), "--stages", "needs,pages"], pages)]
    assert not (d / "library" / "pages.json").exists()


def test_a_failed_web_search_still_searches_the_archives(tmp_path, monkeypatch):
    d = film(tmp_path)
    seen = []
    monkeypatch.setattr(regen, "run_step", lambda cmd, d, timeout: seen.append(cmd[3:]))

    def broken(q, about):
        raise RuntimeError("no network")

    regen.search_online(d, ["oak beach"], web=broken)
    assert seen == [[str(d), "--stages", "needs"]]


def test_web_pages_from_stock_and_social_sites_are_dropped(monkeypatch):
    class R:
        output_parsed = regen.WebFinds(pages=[
            regen.WebPage(page_url="https://www.gettyimages.com/photo/1", subject="x"),
            regen.WebPage(page_url="https://news.example/a", image_url="https://news.example/a.jpg",
                          subject="Liz ([news.example](https://news.example/a))"),
            regen.WebPage(page_url="https://www.pinterest.com/pin/2", subject="y"),
            regen.WebPage(page_url="https://wiki.example/b", image_url="not a url", subject="Ted")])

    class C:
        class responses:
            @staticmethod
            def parse(**kw):
                assert kw["tools"] == [{"type": "web_search"}]
                return R

    monkeypatch.setattr(regen, "client", lambda: C)
    assert regen.find_web_pages(["Liz Kloepfer"]) == [
        {"page_url": "https://news.example/a", "subject": "Liz", "image_url": "https://news.example/a.jpg"},
        {"page_url": "https://wiki.example/b", "subject": "Ted"}]


def test_a_picture_the_review_names_by_id_is_offered_first_even_past_the_cap(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(regen, "MAX_OPTIONS", 1)
    assert [o["id"] for o in regen.options(d, 1)] != ["c3"]
    assert [o["id"] for o in regen.options(d, 1, named=["c3"])] == ["c3"]
    f = Fakes(d, 1, Decision(action="library", library_id="c3", explanation="Used c3"))
    assert f.run("use c3, the beach")["state"] == "done"
    assert f.asked[0][1].split("Options (id | category | where found | era | caption):\n")[1].startswith("c3 |")


def test_two_workers_run_two_shots_side_by_side_and_render_in_turn(tmp_path):
    d = film(tmp_path)
    both_running = threading.Barrier(2, timeout=10)
    rendering, overlap = [], []

    def run(d, n, text):
        both_running.wait()                 # both shots are in flight at once, or this times out
        with regen.RENDER_LOCK:
            rendering.append(n)
            overlap.append(len(rendering))
            rendering.remove(n)
        regen.note(d, n, state="done", step="Done")

    worker = regen.Regenerator(run=run, workers=2)
    worker.submit(d, 1, "a")
    worker.submit(d, 2, "b")
    assert worker.join(10)
    assert max(overlap) == 1
    assert {entry(d, 1)["state"], entry(d, 2)["state"]} == {"done"}


def test_with_drawings_off_a_shot_with_no_real_picture_keeps_its_picture_and_gets_its_card(tmp_path):
    d = film(tmp_path)
    before = review.find_beat(review.load_storyboard(d), 2)["library_id"]
    f = Fakes(d, 2, Decision(action="search", queries=["Sandpiper Tavern"], explanation=""),
              Decision(action="illustration", detail="a 1960s tavern doorway at night", name_card="Seattle",
                       name_role="1969", explanation="Drew it"))
    regen.regenerate(d, 2, "show the tavern", ask=f.ask, check=f.check, search=f.search, render=f.render,
                     locate=f.locate, generate=f.generate, drawings=False)
    done = entry(d, 2)
    beat = review.find_beat(review.load_storyboard(d), 2)
    assert done["state"] == "done" and "drawings are off" in done["explanation"]
    assert beat["library_id"] == before and beat["kind"] != "generated"
    assert beat["overlay"]["text"] == "Seattle"
    assert "Drawings are switched off" in f.asked[0][1] and "keep the current picture" in f.asked[1][1]

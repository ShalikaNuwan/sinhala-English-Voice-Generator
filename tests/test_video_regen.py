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
         "licence": "CC BY-SA 4.0"},
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
    assert f.steps == [("ask", "running", "Choosing a picture…"), ("search", "running", "Searching the archives…"),
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
    f = Fakes(d, 2, Decision(action="illustration", detail="an empty 1970s road at night", explanation="Drew it"))
    assert f.run("night, not day")["state"] == "done"
    beat = review.find_beat(review.load_storyboard(d), 2)
    assert beat["kind"] == "generated" and beat["detail"] == "an empty 1970s road at night"
    assert beat["asset"] == "assets/generated/detail_new.png"


def test_a_refused_drawing_gets_one_more_choice_told_why(tmp_path):
    d = film(tmp_path)
    f = Fakes(d, 2, Decision(action="illustration", detail="a man waiting by a car", explanation="Drew him"),
              Decision(action="illustration", detail="an empty parked car at night", explanation="Drew the car"))
    done = f.run("show him waiting")
    assert done["state"] == "done" and done["explanation"] == "Drew the car"
    assert "refused by the illustration rules: 'a man waiting by a car'" in f.asked[1][1]
    assert "search is not available" in f.asked[1][1].lower()


def test_two_refused_drawings_fail_and_leave_the_shot_as_it_was(tmp_path):
    d = film(tmp_path)
    before = files(d)
    f = Fakes(d, 2, Decision(action="illustration", detail="a man waiting by a car", explanation="Drew him"),
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
            lib = json.loads((d / "library.json").read_text())
            lib["items"].insert(0, {"id": "c7", "file": "library/c7.jpg", "found_by": "need:1970s house"})
            (d / "library.json").write_text(json.dumps(lib))

    monkeypatch.setattr(regen, "run_step", run_step)
    new = regen.search_online(d, ["1970s house", "1970S PHARMACY COUNTER"])
    assert new == ["c7"]
    needs = json.loads((d / "library" / "needs.json").read_text())
    assert needs == [{"query": "1970s house", "beats": []}, {"query": "1970s pharmacy counter", "beats": [5]}]
    harvest, caption = seen
    assert harvest[3:] == [str(d), "--stages", "needs"]
    assert caption[2].endswith("caption.py") and caption[3:] == [str(d), "--only", "c7"]


def test_search_online_with_nothing_new_does_not_caption(tmp_path, monkeypatch):
    d = film(tmp_path)
    seen = []
    monkeypatch.setattr(regen, "run_step", lambda cmd, d, timeout: seen.append(cmd))
    assert regen.search_online(d, ["anything"]) == []
    assert len(seen) == 1 and seen[0][2].endswith("harvest.py")

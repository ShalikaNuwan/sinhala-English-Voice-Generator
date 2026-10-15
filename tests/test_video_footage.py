from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "video" / "footage.py"


def _load():
    """scripts/ is not a package, so the module is loaded from its path."""
    spec = importlib.util.spec_from_file_location("video_footage", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


footage = _load()


# --- licences ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://creativecommons.org/licenses/publicdomain/", "Public domain"),
        ("https://creativecommons.org/licenses/publicdomain/", "Public domain"),
        ("http://creativecommons.org/publicdomain/mark/1.0/", "Public Domain Mark 1.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0 1.0"),
        ("http://creativecommons.org/publicdomain/zero/1.0/legalcode", "CC0 1.0"),
        ("https://creativecommons.org/licenses/by/4.0/", "CC BY 4.0"),
        ("http://creativecommons.org/licenses/by/3.0/us/", "CC BY 3.0"),
        ("http://creativecommons.org/licenses/by/2.5", "CC BY 2.5"),
    ],
)
def test_archive_licences_that_allow_commercial_reuse(url, expected):
    assert footage.archive_licence(url, ["ephemera"]) == expected


@pytest.mark.parametrize(
    "url",
    [
        "http://creativecommons.org/licenses/by-sa/3.0/",
        "http://creativecommons.org/licenses/by-nc/3.0/",
        "https://creativecommons.org/licenses/by-nd/4.0/",
        "http://creativecommons.org/licenses/by-nc-sa/2.0/",
        "http://creativecommons.org/licenses/by-nc-nd/3.0/",
        "http://rightsstatements.org/vocab/InC/1.0/",
        "https://example.com/my-licence",
    ],
)
def test_archive_licences_that_are_refused(url):
    assert footage.archive_licence(url, ["prelinger"]) is None


def test_prelinger_without_a_licence_url_is_public_domain():
    assert footage.archive_licence(None, ["prelingerhomemovies", "prelinger"]) == "Public domain (Prelinger Archives)"
    assert footage.archive_licence("", "prelinger") == "Public domain (Prelinger Archives)"


def test_no_licence_url_outside_prelinger_is_refused():
    assert footage.archive_licence(None, ["ephemera"]) is None
    assert footage.archive_licence("", ["opensource_movies"]) is None


def test_untrusted_collections_are_refused_even_when_marked_public_domain():
    # Anyone can upload to opensource_movies and tick "public domain"; studio films turn up there.
    assert footage.archive_licence("http://creativecommons.org/publicdomain/mark/1.0/", ["opensource_movies"]) is None
    assert footage.trusted_collection(["FedFlix", "usgovfilms"])
    assert footage.trusted_collection("avgeeks")
    assert not footage.trusted_collection(["opensource_movies"])


@pytest.mark.parametrize(
    "name, ok",
    [
        ("Public domain", True),
        ("CC0", True),
        ("CC BY 4.0", True),
        ("CC BY 3.0", True),
        ("Public Domain Mark 1.0", True),
        ("CC BY-SA 4.0", False),
        ("CC BY-SA 3.0", False),
        ("CC BY-NC 2.0", False),
        ("GFDL", False),
        ("", False),
    ],
)
def test_commons_licence_names(name, ok):
    if ok:
        assert footage.commons_licence(name) == name
    else:
        assert footage.commons_licence(name) is None


# --- choosing what to download ----------------------------------------------------------------

def test_archive_file_choice_is_the_smallest_at_least_480p():
    files = [
        {"name": "a.mpeg", "format": "MPEG2", "size": "569055236", "width": "640", "height": "480"},
        {"name": "a.mp4", "format": "h.264", "size": "63509231", "width": "640", "height": "480"},
        {"name": "a.ogv", "format": "Ogg Video", "size": "45837236", "width": "400", "height": "300"},
        {"name": "a_512kb.mp4", "format": "512Kb MPEG4", "size": "20000000", "width": "320", "height": "240"},
        {"name": "a.gif", "format": "Animated GIF", "size": "302205"},
    ]
    assert footage.pick_archive_file(files)["name"] == "a.mp4"


def test_archive_file_choice_respects_the_size_cap_and_falls_back_to_sd():
    files = [
        {"name": "b.mov", "format": "QuickTime", "size": "838985127", "width": "1920", "height": "1080"},
        {"name": "b.mp4", "format": "h.264", "size": "97395031", "width": "640", "height": "360"},
    ]
    choice = footage.pick_archive_file(files, max_bytes=300_000_000)
    assert choice["name"] == "b.mp4"
    assert footage.pick_archive_file(files[:1], max_bytes=300_000_000) is None


def test_commons_derivative_choice_prefers_480p():
    derivatives = [
        {"src": "orig.webm", "type": 'video/webm; codecs="vp9"', "width": 1920, "height": 1080},
        {"src": "x.240p.vp9.webm", "transcodekey": "240p.vp9.webm", "width": 426, "height": 240},
        {"src": "x.480p.vp9.webm", "transcodekey": "480p.vp9.webm", "width": 854, "height": 480},
        {"src": "x.720p.vp9.webm", "transcodekey": "720p.vp9.webm", "width": 1280, "height": 720},
    ]
    assert footage.pick_commons_derivative(derivatives)["src"] == "x.480p.vp9.webm"


def test_year_of_reads_the_first_plausible_year():
    assert footage.year_of('circa 1974<div style="display: none;">date QS:P,+1974</div>') == 1974
    assert footage.year_of("2016-12-09") == 2016
    assert footage.year_of("") is None


# --- scene detection --------------------------------------------------------------------------

SHOWINFO_LOG = """
[Parsed_showinfo_1 @ 0x600] config in time_base: 1/30000, frame_rate: 30000/1001
[Parsed_showinfo_1 @ 0x600] n:   0 pts:  123123 pts_time:4.1041  duration:   1001 fmt:yuv420p
[Parsed_showinfo_1 @ 0x600] n:   1 pts:  360360 pts_time:12.012  duration:   1001 fmt:yuv420p
[Parsed_showinfo_1 @ 0x600] n:   2 pts:  360360 pts_time:12.012  duration:   1001 fmt:yuv420p
[Parsed_showinfo_1 @ 0x600] n:   3 pts:  900900 pts_time:30.03   duration:   1001 fmt:yuv420p
"""

SCDET_LOG = """
[scdet @ 0x7f] lavfi.scd.score: 41.200, lavfi.scd.time: 6.5
[scdet @ 0x7f] lavfi.scd.score: 55.010, lavfi.scd.time: 20.25
"""


def test_parse_scene_cuts_from_showinfo():
    assert footage.parse_scene_cuts(SHOWINFO_LOG) == [4.1041, 12.012, 30.03]


def test_parse_scene_cuts_from_scdet():
    assert footage.parse_scene_cuts(SCDET_LOG) == [6.5, 20.25]


def test_scenes_from_cuts_covers_the_film_and_drops_cuts_too_close_together():
    scenes = footage.scenes_from_cuts([4.0, 4.1, 12.0, 0.05, 40.0, 39.9], duration=35.0)
    assert scenes == [(0.0, 4.0), (4.0, 12.0), (12.0, 35.0)]


def test_split_long_cuts_scenes_over_fifteen_seconds_into_equal_pieces():
    pieces = footage.split_long([(0.0, 4.0), (4.0, 40.0)])
    assert pieces[0] == (0.0, 4.0)
    long = pieces[1:]
    assert len(long) == 3
    assert long[0][0] == 4.0 and long[-1][1] == 40.0
    assert all(abs((b - a) - 12.0) < 1e-6 for a, b in long)
    assert all(long[i][1] == long[i + 1][0] for i in range(len(long) - 1))


def piece(a, b, caption="A street.", quality=3, era="1970s"):
    return {"in": a, "out": b, "caption": caption, "era": era, "quality": quality, "graphic": False}


def test_merge_short_joins_adjacent_short_pieces():
    merged = footage.merge_short([piece(0, 2, "Cars."), piece(2, 3.5, "Snow."), piece(3.5, 6, "Cars.")])
    assert len(merged) == 1
    seg = merged[0]
    assert (seg["in"], seg["out"]) == (0, 6)
    assert seg["caption"] == "Cars. Snow."  # distinct captions, in order
    assert seg["graphic"] is False


def test_merge_short_drops_short_pieces_with_no_contiguous_neighbour():
    merged = footage.merge_short([piece(0, 2), piece(10, 18), piece(30, 32)])
    assert [(s["in"], s["out"]) for s in merged] == [(10, 18)]


def test_merge_short_never_exceeds_fifteen_seconds():
    merged = footage.merge_short([piece(0, 9), piece(9, 12), piece(12, 21), piece(21, 23)])
    assert all(4 <= s["out"] - s["in"] <= 15 for s in merged)
    assert [(s["in"], s["out"]) for s in merged] == [(0, 12), (12, 23)]


def test_merge_short_takes_the_worst_quality_and_the_commonest_era():
    merged = footage.merge_short([piece(0, 2, quality=4, era="1970s"), piece(2, 3, quality=2, era="1960s"),
                                  piece(3, 5, quality=5, era="1970s")])
    assert merged[0]["quality"] == 2
    assert merged[0]["era"] == "1970s"


def judgement(**over):
    base = {"usable": True, "caption": "A police car on a wet street.", "era": "1970s", "quality": 3,
            "graphic": False, "violence": False, "on_screen_text": False, "modern_elements": False,
            "close_up_face": False}
    base.update(over)
    return base


@pytest.mark.parametrize(
    "over, ok",
    [
        ({}, True),
        ({"usable": False}, False),
        ({"graphic": True}, False),
        ({"violence": True}, False),
        ({"on_screen_text": True}, False),
        ({"modern_elements": True}, False),
        ({"close_up_face": True}, False),
        ({"quality": 1}, False),
        ({"caption": ""}, False),
    ],
)
def test_acceptable(over, ok):
    assert footage.acceptable(judgement(**over)) is ok


# --- the catalogue ----------------------------------------------------------------------------

def test_write_atomic_leaves_the_old_file_when_the_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "clips.json"
    footage.write_atomic(path, {"items": [{"id": "v0001"}]})
    assert json.loads(path.read_text())["items"][0]["id"] == "v0001"
    assert not list(tmp_path.glob("*.tmp"))

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(footage.os, "replace", boom)
    with pytest.raises(OSError):
        footage.write_atomic(path, {"items": []})
    assert json.loads(path.read_text())["items"][0]["id"] == "v0001"


def test_library_ids_are_v_and_four_digits_and_never_reused(tmp_path):
    lib = footage.Library(tmp_path)
    first = lib.reserve_id("archive.org:a")
    second = lib.reserve_id("commons:File:B.webm")
    assert (first, second) == ("v0001", "v0002")
    assert lib.reserve_id("archive.org:a") == "v0001"  # the same source keeps its id
    lib.add_item({"id": "v0001", "file": "clips/v0001.mp4", "segments": []})
    again = footage.Library(tmp_path)
    assert again.reserve_id("archive.org:c") == "v0003"
    assert [i["id"] for i in again.items()] == ["v0001"]
    data = json.loads((tmp_path / "clips.json").read_text())
    assert list(data) == ["items"]


def test_library_add_item_replaces_an_entry_with_the_same_id(tmp_path):
    lib = footage.Library(tmp_path)
    lib.add_item({"id": "v0001", "title": "old"})
    lib.add_item({"id": "v0001", "title": "new"})
    assert [i["title"] for i in lib.items()] == ["new"]


def test_downloads_are_counted_against_the_budget(tmp_path):
    lib = footage.Library(tmp_path, budget_bytes=1000)
    assert lib.can_download(600)
    lib.count_download(600)
    assert not lib.can_download(600)
    assert footage.Library(tmp_path, budget_bytes=1000).downloaded_bytes() == 600


# --- one source end to end, with fake steps ---------------------------------------------------

class FakeSteps:
    def __init__(self, fail_at=None, fail_on=1):
        self.calls = {"download": 0, "probe": 0, "detect": 0, "frames": 0, "judge": 0, "verify": 0, "cut": 0}
        self.fail_at = fail_at
        self.fail_on = fail_on  # which call of fail_at raises
        self.cuts = []

    def _tick(self, name):
        self.calls[name] += 1
        if self.fail_at == name and self.calls[name] == self.fail_on:
            raise RuntimeError(f"{name} failed")

    def download(self, candidate, destination):
        self._tick("download")
        destination.write_bytes(b"x" * 100)
        return 100

    def probe(self, path):
        self._tick("probe")
        return {"duration_s": 40.0, "width": 640, "height": 480, "fps": 29.97}

    def detect(self, path, duration):
        self._tick("detect")
        return [10.0, 20.0, 30.0]

    def frames(self, path, spans, folder):
        self._tick("frames")
        out = []
        for n, _ in enumerate(spans):
            frame = folder / f"f{n:03d}.jpg"
            frame.write_bytes(b"jpg")
            out.append(frame)
        return out

    def judge(self, candidate, frames, spans):
        self._tick("judge")
        # The third scene shows a title card and is refused.
        return [judgement(caption=f"Scene {n}.", on_screen_text=(n == 2)) for n in range(len(spans))]

    def verify(self, candidate, source, segments, folder):
        self._tick("verify")
        return [judgement(caption=s["caption"]) for s in segments]

    def cut(self, source, start, end, destination):
        self._tick("cut")
        self.cuts.append((start, end))
        destination.write_bytes(b"mp4")
        return {"duration_s": round(end - start, 3), "width": 640, "height": 480, "fps": 29.97}


CANDIDATE = {
    "key": "archive.org:winter_streets", "source": "archive.org", "title": "Winter Streets",
    "source_url": "https://archive.org/details/winter_streets", "licence": "Public domain",
    "attribution": "Prelinger Archives", "download_url": "https://archive.org/download/winter_streets/w.mp4",
    "size": 100, "suffix": ".mp4",
}


def test_process_source_cuts_each_segment_to_its_own_small_file_shaped_like_the_spec(tmp_path):
    lib = footage.Library(tmp_path)
    steps = FakeSteps()
    items = footage.process_source(CANDIDATE, lib, steps)
    assert [i["id"] for i in items] == ["v0001", "v0002", "v0003"]
    for item in items:
        assert set(item) >= {"id", "file", "source", "title", "source_url", "licence", "attribution", "rights",
                             "duration_s", "width", "height", "fps", "segments"}
        assert item["file"] == f"clips/{item['id']}.mp4" and item["rights"] == "free"
        assert item["source"] == "archive.org" and item["title"] == "Winter Streets"
        assert len(item["segments"]) == 1
        seg = item["segments"][0]
        assert set(seg) >= {"in", "out", "caption", "era", "quality", "graphic"}
        assert 4 <= seg["out"] - seg["in"] <= 15
        assert 0 <= seg["in"] < seg["out"] <= item["duration_s"]
        assert (tmp_path / item["file"]).exists()
    assert [i["segments"][0]["caption"] for i in items] == ["Scene 0.", "Scene 1.", "Scene 3."]
    # A second of margin either side, clamped to the film; in/out re-timed to the small file.
    assert steps.cuts == [(0.0, 10.8), (9.2, 20.8), (29.2, 40.0)]
    assert [(i["segments"][0]["in"], i["segments"][0]["out"]) for i in items] == [(0.2, 9.8), (1.0, 10.6), (1.0, 10.6)]
    assert items[1]["segments"][0]["source_in"] == 10.2 and items[1]["segments"][0]["source_out"] == 19.8
    assert not (lib.work_dir(CANDIDATE["key"]) / "source.mp4").exists()  # the full film is deleted
    assert lib.items() == items


def test_rerun_never_downloads_or_captions_twice(tmp_path):
    lib = footage.Library(tmp_path)
    footage.process_source(CANDIDATE, lib, FakeSteps())
    again = FakeSteps()
    assert footage.process_source(CANDIDATE, footage.Library(tmp_path), again) is None
    assert all(n == 0 for n in again.calls.values())


def test_resume_after_a_failed_caption_does_not_download_again(tmp_path):
    lib = footage.Library(tmp_path)
    with pytest.raises(RuntimeError):
        footage.process_source(CANDIDATE, lib, FakeSteps(fail_at="judge"))
    steps = FakeSteps()
    items = footage.process_source(CANDIDATE, footage.Library(tmp_path), steps)
    assert steps.calls["download"] == 0 and steps.calls["judge"] == 1
    assert items[0]["id"] == "v0001"
    assert lib.downloaded_bytes() == 100


def test_resume_after_a_failed_cut_redoes_only_the_missing_files_and_keeps_ids(tmp_path):
    lib = footage.Library(tmp_path)
    with pytest.raises(RuntimeError):
        footage.process_source(CANDIDATE, lib, FakeSteps(fail_at="cut", fail_on=2))
    assert [i["id"] for i in lib.items()] == ["v0001"]  # the first segment was stored before the failure
    steps = FakeSteps()
    items = footage.process_source(CANDIDATE, footage.Library(tmp_path), steps)
    assert steps.calls == {"download": 0, "probe": 0, "detect": 0, "frames": 0, "judge": 0, "verify": 0, "cut": 2}
    assert [i["id"] for i in items] == ["v0002", "v0003"]
    assert [i["id"] for i in lib.items()] == ["v0001", "v0002", "v0003"]


def test_a_film_with_nothing_usable_is_remembered_and_not_fetched_again(tmp_path):
    class Useless(FakeSteps):
        def judge(self, candidate, frames, spans):
            self._tick("judge")
            return [judgement(usable=False) for _ in spans]

    lib = footage.Library(tmp_path)
    assert footage.process_source(CANDIDATE, lib, Useless()) is None
    assert lib.source_state(CANDIDATE["key"])["status"] == "rejected"
    assert lib.items() == []
    assert not lib.work_dir(CANDIDATE["key"]).exists()
    again = FakeSteps()
    assert footage.process_source(CANDIDATE, footage.Library(tmp_path), again) is None
    assert again.calls["download"] == 0


def test_over_budget_sources_are_skipped_without_downloading(tmp_path):
    lib = footage.Library(tmp_path, budget_bytes=50)
    steps = FakeSteps()
    assert footage.process_source(CANDIDATE, lib, steps) is None
    assert steps.calls["download"] == 0
    assert lib.source_state(CANDIDATE["key"]) is None  # not remembered: a bigger budget may take it later


def test_a_partial_download_is_never_mistaken_for_a_whole_one(tmp_path):
    lib = footage.Library(tmp_path)
    work = lib.work_dir(CANDIDATE["key"])
    work.mkdir(parents=True)
    (work / "source.mp4.part").write_bytes(b"half")  # a download cut short by a full disk
    steps = FakeSteps()
    footage.process_source(CANDIDATE, lib, steps)
    assert steps.calls["download"] == 1


# --- generated clips --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "A photo of John Wayne Gacy standing in the snow",
        "Close-up of a man's face lit by police lights",
        "A portrait of a teenage boy outside a pharmacy",
        "Robert Piest's face in a crowd",
        "Headshot of a detective at his desk",
    ],
)
def test_generated_clip_prompts_that_are_refused(prompt):
    assert footage.refusal(prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "Snow falling past a streetlight on an empty 1970s Chicago street at night",
        "Police lights reflected on a wet road at night, no people",
        "A suburban ranch house in Norwood Park under winter snow, 1970s",
    ],
)
def test_generated_clip_prompts_that_are_allowed(prompt):
    assert footage.refusal(prompt) is None


class FakeVideos:
    def __init__(self, statuses, error=None):
        self.statuses = list(statuses)
        self.error = error
        self.created = []
        self.retrieved = 0
        self.downloaded = []

    def _video(self, status, progress=0):
        return SimpleNamespace(id="video_123", status=status, progress=progress, seconds="4", size="1280x720",
                               model="sora-2", created_at=1000, completed_at=1060 if status == "completed" else None,
                               error=self.error if status == "failed" else None, expires_at=None,
                               model_dump=lambda: {"id": "video_123", "status": status})

    def create(self, **kwargs):
        self.created.append(kwargs)
        return self._video("queued")

    def retrieve(self, video_id):
        self.retrieved += 1
        return self._video(self.statuses.pop(0), progress=50)

    def download_content(self, video_id, variant="video"):
        self.downloaded.append((video_id, variant))
        return SimpleNamespace(content=b"\x00\x00\x00\x18ftypmp42")


def test_generate_clip_polls_until_complete_and_downloads(tmp_path):
    videos = FakeVideos(["in_progress", "in_progress", "completed"])
    client = SimpleNamespace(videos=videos)
    sleeps = []
    target = tmp_path / "clips" / "v0099.mp4"
    result = footage.generate_clip("Snow falling past a streetlight at night", 4, target,
                                   client=client, sleep=sleeps.append, poll_s=2.0)
    assert videos.created == [{"model": "sora-2", "prompt": "Snow falling past a streetlight at night",
                               "seconds": "4", "size": "1280x720"}]
    assert videos.retrieved == 3 and len(sleeps) == 3
    assert videos.downloaded == [("video_123", "video")]
    assert target.read_bytes().startswith(b"\x00\x00\x00\x18ftyp")
    assert result["id"] == "video_123" and result["status"] == "completed" and result["seconds"] == "4"
    assert "elapsed_s" in result


def test_generate_clip_raises_when_the_job_fails(tmp_path):
    videos = FakeVideos(["in_progress", "failed"], error=SimpleNamespace(code="moderation", message="blocked"))
    with pytest.raises(RuntimeError, match="blocked"):
        footage.generate_clip("Rain on a dark road", 4, tmp_path / "x.mp4",
                              client=SimpleNamespace(videos=videos), sleep=lambda s: None)
    assert not (tmp_path / "x.mp4").exists()


def test_generate_clip_gives_up_after_the_timeout(tmp_path):
    videos = FakeVideos(["in_progress"] * 50)
    clock = iter(range(0, 10_000, 100))
    with pytest.raises(TimeoutError):
        footage.generate_clip("Rain on a dark road", 4, tmp_path / "x.mp4", client=SimpleNamespace(videos=videos),
                              sleep=lambda s: None, timeout_s=500, clock=lambda: next(clock))


def test_generate_clip_refuses_before_calling_the_api(tmp_path):
    videos = FakeVideos(["completed"])
    with pytest.raises(ValueError):
        footage.generate_clip("A close-up of a man's face", 4, tmp_path / "x.mp4",
                              client=SimpleNamespace(videos=videos), sleep=lambda s: None)
    assert videos.created == []


def test_generate_clip_only_accepts_the_lengths_the_api_offers(tmp_path):
    with pytest.raises(ValueError):
        footage.generate_clip("Rain on a dark road", 5, tmp_path / "x.mp4",
                              client=SimpleNamespace(videos=FakeVideos([])), sleep=lambda s: None)


def test_a_failed_download_is_not_counted_against_the_budget_and_leaves_no_part_file(tmp_path):
    lib = footage.Library(tmp_path)
    with pytest.raises(RuntimeError):
        footage.process_source(CANDIDATE, lib, FakeSteps(fail_at="download"))
    assert lib.downloaded_bytes() == 0
    assert not list(lib.work_dir(CANDIDATE["key"]).glob("*.part"))
    assert lib.source_state(CANDIDATE["key"])["status"] == "downloading"  # retried on the next run


# --- keeping a few of the best seconds per film -----------------------------------------------

def test_best_segments_prefers_quality_spreads_across_the_film_and_keeps_film_order():
    segs = [piece(t, t + 10, caption=f"s{t}", quality=q) for t, q in
            [(0, 3), (10, 4), (20, 4), (30, 2), (100, 3), (200, 4), (300, 3), (310, 5)]]
    best = footage.best_segments(segs, 4)
    assert [s["caption"] for s in best] == ["s10", "s100", "s200", "s310"]
    assert [s["in"] for s in best] == sorted(s["in"] for s in best)


def test_best_segments_fills_up_with_neighbours_when_the_film_is_short():
    segs = [piece(t, t + 10, caption=f"s{t}", quality=3) for t in (0, 10, 20)]
    assert len(footage.best_segments(segs, 5)) == 3
    assert len(footage.best_segments(segs, 2)) == 2


def test_process_source_keeps_at_most_the_per_film_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(footage, "MAX_SEGMENTS_PER_FILM", 2)
    items = footage.process_source(CANDIDATE, footage.Library(tmp_path), FakeSteps())
    assert len(items) == 2


def test_prune_keeps_the_best_per_film_deletes_the_rest_and_renumbers(tmp_path):
    lib = footage.Library(tmp_path)
    footage.process_source(CANDIDATE, lib, FakeSteps())
    other = {**CANDIDATE, "key": "archive.org:other", "title": "Other"}
    footage.process_source(other, lib, FakeSteps())
    assert [i["id"] for i in lib.items()] == [f"v000{n}" for n in range(1, 7)]
    footage.cmd_prune(SimpleNamespace(video_dir=str(tmp_path), per_film=1))
    items = lib.items()
    assert [i["id"] for i in items] == ["v0001", "v0002"]
    assert [i["title"] for i in items] == ["Winter Streets", "Other"]
    assert sorted(p.name for p in (tmp_path / "clips").glob("*.mp4")) == ["v0001.mp4", "v0002.mp4"]
    assert lib.source_state("archive.org:other")["clip_ids"] == ["v0002"]
    assert lib.reserve_id("archive.org:new#0") == "v0003"


def test_the_three_still_check_drops_segments_and_rewrites_captions(tmp_path):
    class Strict(FakeSteps):
        def verify(self, candidate, source, segments, folder):
            self._tick("verify")
            # The middle segment hides a title card the single scene frame missed.
            return [judgement(caption=f"Checked {n}.", quality=2, on_screen_text=(n == 1))
                    for n, _ in enumerate(segments)]

    items = footage.process_source(CANDIDATE, footage.Library(tmp_path), Strict())
    assert [i["segments"][0]["caption"] for i in items] == ["Checked 0.", "Checked 2."]
    assert all(i["segments"][0]["quality"] == 2 for i in items)  # the lower of the two looks


def test_a_failed_check_is_resumed_without_judging_again(tmp_path):
    lib = footage.Library(tmp_path)
    with pytest.raises(RuntimeError):
        footage.process_source(CANDIDATE, lib, FakeSteps(fail_at="verify"))
    steps = FakeSteps()
    footage.process_source(CANDIDATE, footage.Library(tmp_path), steps)
    assert steps.calls["judge"] == 0 and steps.calls["verify"] == 1 and steps.calls["download"] == 0


# --- black borders ----------------------------------------------------------------------------

CROPDETECT_LOG = """
[Parsed_cropdetect_0 @ 0x6] x1:93 x2:758 y1:0 y2:478 w:664 h:478 x:94 y:0 pts:180224 t:14.66 limit:40.0 crop=664:478:94:0
[Parsed_cropdetect_0 @ 0x6] x1:93 x2:758 y1:0 y2:479 w:664 h:480 x:94 y:0 pts:180736 t:14.70 limit:40.0 crop=664:480:94:0
"""


def test_parse_crop_reads_the_last_estimate():
    assert footage.parse_crop(CROPDETECT_LOG) == (664, 480, 94, 0)
    assert footage.parse_crop("no estimate here") is None


def test_border_crop_only_when_the_borders_are_real():
    assert footage.border_crop(852, 480, (664, 480, 94, 0)) == (664, 480, 94, 0)  # pillarboxed 4:3 in 16:9
    assert footage.border_crop(640, 480, (630, 480, 6, 0)) == (630, 480, 6, 0)  # thin film-gate edges
    assert footage.border_crop(640, 480, (638, 480, 2, 0)) is None  # noise, not a border
    assert footage.border_crop(640, 480, (200, 160, 220, 160)) is None  # a dark scene, not a border
    assert footage.border_crop(640, 480, None) is None


def test_border_crop_keeps_symmetric_pillarbox_bars_when_the_picture_is_dark_at_the_bottom():
    # A dark rail yard fills the lower half, so the full estimate looks like a dark scene; the
    # matching 82 px bars either side are still bars.
    assert footage.border_crop(640, 360, (476, 242, 82, 0)) == (476, 360, 82, 0)
    assert footage.border_crop(640, 480, (480, 200, 80, 0)) == (480, 480, 80, 0)

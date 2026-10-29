import importlib
import os
from pathlib import Path

from tests.test_video_review import make_film


def client_for(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    import app.config
    import app.main
    import app.video_api
    importlib.reload(app.config)
    importlib.reload(app.video_api)
    module = importlib.reload(app.main)
    from fastapi.testclient import TestClient
    return TestClient(module.app), app.video_api


class FakeJobs:
    def __init__(self):
        self.started = []
        self.state = None

    def status(self, d):
        return self.state

    def start(self, d, kind, sound=False):
        from app.video_jobs import JobBusy
        if self.state and self.state["state"] == "running":
            raise JobBusy("a rebuild is already running for this film")
        self.started.append((d.name, kind, sound))
        self.state = {"kind": kind, "state": "running", "progress": 0.0, "tail": []}
        return self.state


def test_films_and_film_view(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    films = client.get("/api/video/films").json()["films"]
    assert [f["film"] for f in films] == ["demo"]
    view = client.get("/api/video/films/demo").json()
    assert view["summary"]["shots"] == 6 and view["job"] is None
    assert client.get("/api/video/films/nope").status_code == 404
    assert client.get("/api/video/films/nope/job").status_code == 404
    assert client.get("/api/video/films/demo/job").json() == {"state": "none"}


def test_no_video_folder_lists_no_films(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    assert client.get("/api/video/films").json() == {"films": []}


def test_mark_alternatives_and_choose(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    r = client.put("/api/video/films/demo/shots/2/mark", json={"status": "not_ok", "note": "wrong town"})
    assert r.status_code == 200 and r.json()["mark"]["status"] == "not_ok"
    cleared = client.put("/api/video/films/demo/shots/2/mark", json={"status": None})
    assert cleared.status_code == 200 and cleared.json()["mark"] is None
    assert client.put("/api/video/films/demo/shots/2/mark", json={"status": "maybe"}).status_code == 422
    assert client.put("/api/video/films/demo/shots/99/mark", json={"status": "ok"}).status_code == 400
    assert client.put("/api/video/films/nope/shots/2/mark", json={"status": "ok"}).status_code == 404
    alt = client.get("/api/video/films/demo/shots/1/alternatives").json()
    assert [o["id"] for o in alt["options"]] == ["c2", "c3", "v1#1"]
    assert client.get("/api/video/films/demo/shots/99/alternatives").status_code == 404
    assert client.get("/api/video/films/nope/shots/1/alternatives").status_code == 404
    chosen = client.post("/api/video/films/demo/shots/1/choose", json={"choice": "c3"})
    assert chosen.status_code == 200 and chosen.json()["picture"]["id"] == "c3"
    refused = client.post("/api/video/films/demo/shots/1/choose", json={"choice": "bad"})
    assert refused.status_code == 400
    assert client.post("/api/video/films/nope/shots/1/choose", json={"choice": "c3"}).status_code == 404


def test_illustration_refused_for_a_graphic_and_drawn_otherwise(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    d = make_film(tmp_path / "data" / "video")
    refused = client.post("/api/video/films/demo/shots/4/choose", json={"choice": "illustration"})
    assert refused.status_code == 400 and "cards" in refused.json()["detail"]

    from app import video_review
    video_review.engine()
    import ladder

    def generate(detail, era, video_dir):
        path = Path(video_dir) / "assets" / "generated" / "detail_new.png"
        path.write_bytes(b"png")
        return path

    monkeypatch.setattr(ladder, "describe_detail", lambda words, scene: "a parked car at dusk")
    monkeypatch.setattr(ladder, "generate_detail", generate)
    drawn = client.post("/api/video/films/demo/shots/2/choose", json={"choice": "illustration"})
    assert drawn.status_code == 200
    assert drawn.json()["category"] == "illustration" and drawn.json()["asset"] == "assets/generated/detail_new.png"
    assert (d / "assets" / "generated" / "detail_new.png").exists()


def test_rebuild_render_and_busy(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    fake = FakeJobs()
    monkeypatch.setattr(api, "jobs", fake)
    assert client.post("/api/video/films/demo/rebuild").status_code == 202
    assert client.post("/api/video/films/demo/render", json={"sound": True}).status_code == 409
    assert fake.started == [("demo", "rebuild", False)]
    assert client.get("/api/video/films/demo/job").json()["state"] == "running"
    assert client.get("/api/video/films/demo").json()["job"]["state"] == "running"
    assert client.post("/api/video/films/nope/rebuild").status_code == 404


def test_render_body_is_optional(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    fake = FakeJobs()
    monkeypatch.setattr(api, "jobs", fake)
    assert client.post("/api/video/films/demo/render").status_code == 202
    fake.state = None
    assert client.post("/api/video/films/demo/render", json={"sound": True}).status_code == 202
    assert fake.started == [("demo", "render", False), ("demo", "render", True)]


def test_render_refused_without_disk_space(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")

    class Full(FakeJobs):
        def start(self, d, kind, sound=False):
            from app.video_jobs import NoSpace
            raise NoSpace("the final render needs about 5 GB free; 1 GB is free")

    monkeypatch.setattr(api, "jobs", Full())
    r = client.post("/api/video/films/demo/render")
    assert r.status_code == 409 and "GB" in r.json()["detail"]


def test_files_are_served_inside_the_film_only(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    assert client.get("/api/video/films/demo/file/library/c1.jpg").status_code == 200
    assert client.get("/api/video/films/demo/file/library/none.jpg").status_code == 404
    assert client.get("/api/video/films/demo/file/../storyboard.json").status_code in (403, 404)
    assert client.get("/api/video/films/demo/file/%2E%2E/%2E%2E/app.db").status_code == 403
    assert client.get("/api/video/films/demo/file/library/%2E%2E/%2E%2E/%2E%2E/app.db").status_code == 403
    assert client.get("/api/video/films/demo/file//etc/hosts").status_code == 403
    assert client.get("/api/video/films/demo/file/%2Fetc%2Fhosts").status_code == 403
    assert client.get("/api/video/films/nope/file/library/c1.jpg").status_code == 404


def test_files_reached_through_a_symlink_inside_the_film_are_served(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    d = make_film(tmp_path / "data" / "video")
    elsewhere = tmp_path / "shared_library"
    elsewhere.mkdir()
    (elsewhere / "far.jpg").write_bytes(b"far away")
    os.symlink(elsewhere, d / "shared")
    r = client.get("/api/video/films/demo/file/shared/far.jpg")
    assert r.status_code == 200 and r.content == b"far away"


def test_a_range_request_gets_part_of_the_file(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    d = make_film(tmp_path / "data" / "video")
    (d / "animatic.mp4").write_bytes(bytes(range(256)) * 4)
    r = client.get("/api/video/films/demo/file/animatic.mp4", headers={"Range": "bytes=10-19"})
    assert r.status_code == 206
    assert r.content == bytes(range(10, 20))
    assert r.headers["content-range"] == "bytes 10-19/1024"
    assert r.headers["content-type"] == "video/mp4"


def test_thumbs_and_frames(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    r = client.get("/api/video/films/demo/thumb/c1.jpg")
    assert r.status_code == 200 and r.content == b"jpg"
    assert client.get("/api/video/films/demo/thumb/zzz.jpg").json()["detail"] == "No picture"
    # A clip id carries '#', which the browser sends as %23; with no clip file there is no still.
    seen = []
    original = api.review.thumb
    monkeypatch.setattr(api.review, "thumb", lambda d, item_id: seen.append(item_id) or original(d, item_id))
    clip = client.get("/api/video/films/demo/thumb/v1%231.jpg")
    assert clip.status_code == 404 and clip.json()["detail"] == "No picture"
    assert seen == ["v1#1"]
    assert client.get("/api/video/films/nope/thumb/c1.jpg").status_code == 404
    none_yet = client.get("/api/video/films/demo/frame/1.jpg")
    assert none_yet.status_code == 404 and "rebuild" in none_yet.json()["detail"]
    assert client.get("/api/video/films/demo/frame/99.jpg").status_code == 404
    assert client.get("/api/video/films/demo/frame/x.jpg").status_code == 422


def blocking_regen(api, monkeypatch):
    """A real worker whose regeneration waits until the test lets it go."""
    import threading
    from app import video_regen
    gate = threading.Event()
    worker = video_regen.Regenerator(run=lambda d, n, text: gate.wait(10))
    monkeypatch.setattr(api, "regen", worker)
    return worker, gate


def test_regenerate_queues_a_shot_and_refuses_what_it_cannot_do(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    worker, gate = blocking_regen(api, monkeypatch)
    try:
        r = client.post("/api/video/films/demo/shots/2/regenerate", json={"review": "show the house"})
        assert r.status_code == 202
        assert r.json()["state"] == "queued" and r.json()["review"] == "show the house"
        assert client.post("/api/video/films/demo/shots/2/regenerate", json={"review": "again"}).status_code == 409
        assert client.post("/api/video/films/demo/shots/1/regenerate", json={"review": "  "}).status_code == 400
        assert client.post("/api/video/films/demo/shots/99/regenerate", json={"review": "x"}).status_code == 400
        assert client.post("/api/video/films/demo/shots/4/regenerate", json={"review": "x"}).status_code == 400
        assert client.post("/api/video/films/nope/shots/1/regenerate", json={"review": "x"}).status_code == 404
        view = client.get("/api/video/films/demo").json()
        assert view["summary"]["regenerating"] == 1
        assert view["scenes"][0]["shots"][1]["regen"]["state"] in ("queued", "running")
        assert view["scenes"][0]["shots"][0]["regen"] is None
        # A full render or rebuild waits for the regeneration.
        fake = FakeJobs()
        monkeypatch.setattr(api, "jobs", fake)
        assert client.post("/api/video/films/demo/rebuild").status_code == 409
        assert client.post("/api/video/films/demo/render").status_code == 409
        assert fake.started == []
    finally:
        gate.set()
        assert worker.join(10)
    assert client.post("/api/video/films/demo/rebuild").status_code == 202


def test_regenerate_is_refused_while_a_rebuild_runs(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    fake = FakeJobs()
    fake.state = {"kind": "rebuild", "state": "running", "progress": 0.1, "tail": []}
    monkeypatch.setattr(api, "jobs", fake)
    worker, gate = blocking_regen(api, monkeypatch)
    r = client.post("/api/video/films/demo/shots/2/regenerate", json={"review": "show the house"})
    assert r.status_code == 409 and not worker.busy(tmp_path / "data" / "video" / "demo")


def test_approve_unreviewed_and_the_approved_count(tmp_path, monkeypatch):
    client, api = client_for(tmp_path, monkeypatch)
    make_film(tmp_path / "data" / "video")
    client.put("/api/video/films/demo/shots/1/mark", json={"status": "not_ok"})
    r = client.post("/api/video/films/demo/approve-unreviewed")
    assert r.status_code == 200 and r.json() == {"approved": 5}
    view = client.get("/api/video/films/demo").json()
    assert view["summary"]["approved"] == 5 and view["summary"]["regenerating"] == 0
    assert client.post("/api/video/films/nope/approve-unreviewed").status_code == 404

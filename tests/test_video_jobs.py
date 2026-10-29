import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from app import video_jobs


def film(tmp_path: Path) -> Path:
    d = tmp_path / "demo"
    d.mkdir()
    (d / "storyboard.json").write_text(json.dumps({"duration_s": 60.0, "beats": []}))
    return d


def wait_for(jobs, d, state, seconds=30):
    end = time.time() + seconds
    while time.time() < end:
        s = jobs.status(d)
        if s and s["state"] == state:
            return s
        time.sleep(0.1)
    raise AssertionError(f"job never reached {state}: {jobs.status(d)}")


def test_a_job_runs_reports_progress_and_finishes(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs, "commands", lambda d, kind, sound=False: [
        sys.executable, "-c", "for k in range(1, 4):\n print(f'  shot {k}/3 (#{k}) 1.0s cut', flush=True)"])
    jobs = video_jobs.Jobs()
    started = jobs.start(d, "rebuild")
    assert started["kind"] == "rebuild"
    done = wait_for(jobs, d, "done")
    assert done["progress"] == 1.0


def test_a_failed_job_reports_its_last_lines_and_a_second_start_is_refused(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs, "commands", lambda d, kind, sound=False: [
        sys.executable, "-c", "import sys, time\nprint('  shot 1/4 (#1) 1.0s cut', flush=True)\ntime.sleep(2)\n"
                              "print('boom', flush=True)\nsys.exit(3)"])
    jobs = video_jobs.Jobs()
    jobs.start(d, "rebuild")
    with pytest.raises(video_jobs.JobBusy):
        jobs.start(d, "render")
    failed = wait_for(jobs, d, "failed")
    assert failed["progress"] == 0.25 and "boom" in failed["tail"][-1]


def test_a_job_killed_without_its_exit_line_reads_as_stopped(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs, "commands", lambda d, kind, sound=False: [
        sys.executable, "-c", "import time\nprint('  shot 2/5 (#2) 1.0s cut', flush=True)\ntime.sleep(60)"])
    jobs = video_jobs.Jobs()
    pid = jobs.start(d, "rebuild")["pid"]
    end = time.time() + 30
    while jobs.status(d)["progress"] != 0.4 and time.time() < end:
        time.sleep(0.1)
    assert jobs.status(d)["state"] == "running"
    os.killpg(pid, signal.SIGKILL)          # the shell and build.py together: no exit line is written
    stopped = wait_for(jobs, d, "stopped")  # the dead shell is reaped, not left a zombie that looks alive
    assert stopped["progress"] == 0.4


def test_status_is_empty_before_any_job(tmp_path):
    assert video_jobs.Jobs().status(film(tmp_path)) is None


def test_a_render_is_refused_without_enough_disk(tmp_path, monkeypatch):
    d = film(tmp_path)
    monkeypatch.setattr(video_jobs.shutil, "disk_usage", lambda p: type("U", (), {"free": 10 ** 9})())
    with pytest.raises(video_jobs.NoSpace):
        video_jobs.Jobs().start(d, "render")


def test_progress_reads_the_assemblers_lines_only():
    log = ("527 shots, 89.9 min\n  527/527 shots have a picture\n  joined group 3/27 (#41-#60)\n"
           "  2/9 kind: ok\n")
    assert video_jobs.progress(log) == 0.0
    assert video_jobs.progress(log + "  169/585 shots already rendered or joined\n") == pytest.approx(169 / 585)
    assert video_jobs.progress(log + "  169/585 shots already rendered or joined\n"
                               "  shot 170/585 (#2) 6.5s dissolve c0144.jpg\n  joined group 1/30 (#1-#20)\n"
                               ) == pytest.approx(170 / 585)


def test_commands_are_the_engines_own():
    d = Path("/data/video/demo")
    rebuild = video_jobs.commands(d, "rebuild")
    render = video_jobs.commands(d, "render", sound=True)
    assert rebuild[-4:] == [str(d), "--animatic", "--out", str(d / "animatic.mp4")]
    assert render[-4:] == [str(d), "--out", str(d / "demo_1080p.mp4"), "--sound"]
    assert video_jobs.commands(d, "render")[-1] == str(d / "demo_1080p.mp4")
    assert rebuild[2].endswith("scripts/video/build.py")

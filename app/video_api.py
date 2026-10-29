"""HTTP endpoints for the storyboard review (Video tab), under /api/video.

The film work is app/video_review.py; background renders are app/video_jobs.py; regenerating a shot from the
owner's review is app/video_regen.py (one at a time, never while a rebuild or render runs, and the other way
round). Unknown films are 404
everywhere; a request about a film that cannot be met (unknown shot, unusable picture, refused drawing)
is 400, except asking for an unknown shot's alternatives, which is 404.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import config
from . import video_review as review
from .video_jobs import JobBusy, Jobs, NoSpace
from .video_regen import RegenBusy, Regenerator

router = APIRouter(prefix="/api/video")
jobs = Jobs()
regen = Regenerator()


class MarkIn(BaseModel):
    status: Literal["ok", "not_ok"] | None = None
    note: str = ""


class ChoiceIn(BaseModel):
    choice: str


class RenderIn(BaseModel):
    sound: bool = False


class ReviewIn(BaseModel):
    review: str


def video_root() -> Path:
    # Read through the module so a reloaded app.config (tests, DATA_DIR) is always the one used.
    return config.settings.data_dir / "video"


def film_or_404(film: str) -> Path:
    try:
        return review.film_dir(video_root(), film)
    except review.FilmError:
        raise HTTPException(404, "Film not found")


def still(make) -> Path | None:
    """A frame or thumbnail; None when ffmpeg is missing or stalls rather than a server error."""
    try:
        return make()
    except (OSError, subprocess.SubprocessError):
        return None


@router.get("/films")
def films() -> dict:
    return {"films": review.list_films(video_root())}


@router.get("/films/{film}")
def film_view(film: str) -> dict:
    d = film_or_404(film)
    return review.film_view(d) | {"job": jobs.status(d)}


@router.get("/films/{film}/shots/{n}/alternatives")
def alternatives(film: str, n: int) -> dict:
    d = film_or_404(film)
    try:
        return review.alternatives(d, n)
    except review.FilmError as exc:
        raise HTTPException(404, str(exc))


@router.put("/films/{film}/shots/{n}/mark")
def mark(film: str, n: int, payload: MarkIn) -> dict:
    d = film_or_404(film)
    try:
        return {"shot": n, "mark": review.set_mark(d, n, payload.status, payload.note)}
    except review.FilmError as exc:
        raise HTTPException(400, str(exc))


@router.post("/films/{film}/shots/{n}/choose")
def choose(film: str, n: int, payload: ChoiceIn) -> dict:
    d = film_or_404(film)
    try:
        return review.choose(d, n, payload.choice)
    except review.FilmError as exc:
        raise HTTPException(400, str(exc))


def job_running(d: Path) -> dict | None:
    job = jobs.status(d)
    return job if job and job["state"] == "running" else None


def not_regenerating(d: Path) -> None:
    if regen.busy(d):
        raise HTTPException(409, "a shot is being regenerated: wait for it to finish")


@router.post("/films/{film}/shots/{n}/regenerate", status_code=202)
def regenerate(film: str, n: int, payload: ReviewIn) -> dict:
    d = film_or_404(film)
    if not payload.review.strip():
        raise HTTPException(400, "write what is wrong with the shot first")
    running = job_running(d)
    if running:
        raise HTTPException(409, f"a {running['kind']} is running for this film: regenerate when it is done")
    try:
        return regen.submit(d, n, payload.review)
    except review.FilmError as exc:
        raise HTTPException(400, str(exc))
    except RegenBusy as exc:
        raise HTTPException(409, str(exc))


@router.post("/films/{film}/approve-unreviewed")
def approve_unreviewed(film: str) -> dict:
    return {"approved": review.approve_unreviewed(film_or_404(film))}


@router.post("/films/{film}/rebuild", status_code=202)
def rebuild(film: str) -> dict:
    d = film_or_404(film)
    not_regenerating(d)
    try:
        return jobs.start(d, "rebuild")
    except JobBusy as exc:
        raise HTTPException(409, str(exc))


@router.post("/films/{film}/render", status_code=202)
def render(film: str, payload: RenderIn | None = None) -> dict:
    d = film_or_404(film)
    not_regenerating(d)
    try:
        return jobs.start(d, "render", sound=bool(payload and payload.sound))
    except (JobBusy, NoSpace) as exc:
        raise HTTPException(409, str(exc))


@router.get("/films/{film}/job")
def job(film: str) -> dict:
    return jobs.status(film_or_404(film)) or {"state": "none"}


@router.get("/films/{film}/frame/{n}.jpg")
def frame(film: str, n: int) -> FileResponse:
    d = film_or_404(film)
    try:
        path = still(lambda: review.frame(d, n))
    except review.FilmError as exc:
        raise HTTPException(404, str(exc))
    if not path:
        raise HTTPException(404, "No frame yet: rebuild the review cut")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/films/{film}/thumb/{item_id}.jpg")
def thumb(film: str, item_id: str) -> FileResponse:
    d = film_or_404(film)
    path = still(lambda: review.thumb(d, item_id))
    if not path:
        raise HTTPException(404, "No picture")
    return FileResponse(path)


@router.get("/films/{film}/file/{path:path}")
def film_file(film: str, path: str) -> FileResponse:
    """A file inside the film folder. `..` and absolute paths are refused; symlinks the film itself holds
    (a library folder kept elsewhere) are followed."""
    d = film_or_404(film)
    relative = Path(path)
    parts = relative.parts
    if not parts or relative.is_absolute() or ".." in parts or "\\" in path:
        raise HTTPException(403, "Outside the film")
    target = d.joinpath(*parts)
    if not target.is_file():
        raise HTTPException(404, "No such file")
    return FileResponse(target)

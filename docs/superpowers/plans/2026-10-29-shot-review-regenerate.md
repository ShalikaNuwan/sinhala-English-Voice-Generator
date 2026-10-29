# Shot review → regenerate → approve: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD for every step.

**Goal:** per-shot free-text review in the Video tab; an AI model makes a new version of only that shot, a
second model checks it, the owner plays the regenerated shot and approves it; approve-all-unreviewed and
finalize.

**Spec:** `docs/superpowers/specs/2026-10-29-shot-review-regenerate-design.md`.
**Builds on:** `app/video_review.py`, `app/video_jobs.py`, `app/video_api.py`, `app/static/video.js`.
**Owner rule:** never commit or push; the owner approves the diff.
**Tests:** `.venv/bin/python -m pytest -p no:warnings -o addopts="" -q <files>`.

## Shared contract (both tasks code against this)

Film view (`GET /api/video/films/{film}`), additions:
- `summary.approved`: shots whose current mark is `ok`.
- `summary.regenerating`: shots whose regen state is `queued` or `running`.
- each shot gains `regen`: `null` or
  `{"review": str, "state": "queued"|"running"|"done"|"failed", "step": str, "explanation": str,
    "check": {"satisfied": bool, "problem": str} | null, "clip": str | null, "error": str, "at": float}`
  - `clip` is a path relative to the film folder (`review/clips/12_1793253364.mp4`), served by the
    existing `/file/{path}` endpoint;
  - `step` is plain words for the UI ("Choosing a picture…", "Searching the archives…", "Checking…",
    "Rendering the shot…");
  - a `queued`/`running` entry with no live worker (server restarted) reads as `failed` with error
    "interrupted: regenerate again".
- Approval is the existing mark: `PUT /shots/{n}/mark {"status": "ok"}` approves, `{"status": null}` removes.

New endpoints:
- `POST /api/video/films/{film}/shots/{n}/regenerate` body `{"review": str}` → 202 with the shot's `regen`;
  400 empty review or unknown shot; 409 while a rebuild/render job runs.
- `POST /api/video/films/{film}/approve-unreviewed` → 200 `{"approved": k}`: every shot with no current mark
  and no regen review text (or an empty one) gets mark `ok`.
- `POST /rebuild` and `POST /render` → 409 while any regeneration for that film is queued or running.

## Task A (back end): `app/video_regen.py`, small changes to `app/video_review.py` and `app/video_api.py`

- `video_regen.py`:
  - pydantic `Decision{action: "library"|"illustration"|"search", library_id: str="", focus: str="",
    detail: str="", queries: list[str]=[], explanation: str}` and `Verdict{satisfied: bool, problem: str}`.
  - `options(d, n)`: usable, free-only library pictures and long-enough clips (as `video_review.alternatives`
    but the whole library, the shot's choices first then the scene's then the rest; excluded/graphic never).
  - `decide(review, beat, scene, current, options, allow_search, feedback, ask)` → Decision; `ask` is the
    text-model call (default: OpenAI responses.parse with `settings.text_model`), injectable.
  - `verify(image_bytes, review, beat, scene, look)` → Verdict with `settings.qa_model` vision; injectable.
  - `search_online(d, queries)` → new library ids: append `{"query": q, ...}` entries to
    `library/needs.json` (match its real format), run `harvest.py <d> --stages needs` then `caption.py <d>`
    (same python, project env), diff library ids before/after; injectable.
  - `apply(d, n, decision, locate)`: library → `video_review.choose(d, n, id)`, then focus → crop with
    `ladder.located`/`frame_box` on photos (skip for clips); illustration → `choose(d, n, "illustration",
    describe=lambda *_: decision.detail)` (so `safe_detail` still decides).
  - `render_preview(d, n)` → `build.py <d> --animatic --from <x> --shots 1 --out review/clips/<n>_<t>.mp4`
    (check what `--from` means in `build.chosen_shots`: index or position).
  - `regenerate(d, n, review, *, ask, check, search, render, locate)`: decide → (search → decide on the new
    pictures only, allow_search False) → apply → verify → if not satisfied, decide once more with the
    checker's problem as feedback → apply → verify → render → state done. Every step updates
    `review/regen.json` (state, step). Errors → state failed with the reason; the shot keeps whatever
    `choose` already applied only if it succeeded fully (otherwise restore the beat as it was).
  - `Regenerator`: one worker thread, FIFO queue; `submit(d, n, review)`, `busy(d)`; restart → interrupted.
- `video_review.py`: shot view `regen` field and summary `approved`/`regenerating` (read `review/regen.json`).
- `video_api.py`: the endpoints above.
- Tests: `tests/test_video_regen.py` (fake ask/check/search/render; never the network), additions to
  `tests/test_video_api.py`.

## Task B (front end): `app/static/video.js`, `index.html`, `style.css`

- Per shot: a Review textarea (prefilled from `regen.review`), **Regenerate** (disabled if empty or the
  shot is queued/running), status line from `regen.step`/`state`/`error`, when `done`: a `<video controls>`
  of `regen.clip` shown in place of the stale frame, the explanation, the checker verdict (warn style if
  not satisfied). **Approve** toggles mark `ok`; the Not OK button and the not-OK note go (the review text
  replaces them). Choose another… stays, always available.
- Top: "Approved X of N", **Approve all unreviewed**, and the render button renamed **Finalize video**
  with a confirm that names the number of unapproved shots.
- Filters: All · Not actual only · Illustration · Stand-in · Graphic · Not approved · Has a review.
- Poll the film view every 3 s while `summary.regenerating > 0` (and as before while a job runs); keep
  the typed review text and playing videos intact across refreshes (only re-render rows that changed).

## Task C: real check on the Bundy test film, full suite, show the diff.

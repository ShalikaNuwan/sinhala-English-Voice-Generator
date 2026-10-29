# Storyboard review in the app: see every shot, judge the ones that are not real, then render

Date: 2026-10-29. Status: design, approved in conversation; awaiting review of this write-up.

## Goal

Before the long final render, the owner wants to see how a film made by the storyboard engine will look,
shot by shot: which picture, footage or drawing each shot uses, how the frame will actually appear, and
above all which shots are **not real images or real footage**. The owner marks each shot OK or not OK,
picks another picture for a shot that is not OK (or asks for a new illustration), rebuilds the review
cut, and then starts the final render from the app or from the terminal.

Out of scope: starting a new film from the browser (still `documentary.py prepare` in the terminal),
editing the narration, the voice pipeline, running on the AWS server. The page runs in the local app
(`uvicorn app.main:app`, http://localhost:8000) on this Mac, where the video tools and disk are.

## What the owner sees

A **Video** tab beside the existing voice studio (same look, `app/static/`).

1. **Films**: every folder in `data/video/` that has a `storyboard.json`, with its review progress
   ("12 of 18 shots reviewed, 2 not OK") and whether a final film exists.
2. **Review page for one film**:
   - **Player** at the top: the review cut (`animatic.mp4`, 854x480), or the final film once rendered.
   - **Summary**: "18 shots: 11 actual · 3 stand-in · 3 illustration · 1 graphic", and how many are
     reviewed / not OK.
   - **Filters**: All · Not actual only · Illustration · Stand-in · Graphic · Marked not OK · Not reviewed.
   - **Shots**, grouped by scene (scene title, time and look as a heading). Each shot shows:
     - its time span; ▶ plays the player from the shot's start;
     - **the frame as it will look**: a still from the review cut at the middle of the shot (crop,
       colour grade, labels and cards included);
     - the **source picture** (library image or a frame of the clip) when there is one;
     - a **category badge** (below) and the shot kind in plain words ("illustration", "archival photo",
       "period footage", "name card", "key-fact label", "map", "profile card");
     - the narration words spoken over it;
     - for a library picture: its caption, where it came from (source and link), licence; the reason
       the storyboard gave for choosing it; the automatic check's verdict and problem note, if any;
     - for an illustration: the drawn subject ("A 1970s pharmacy counter with a cash register…");
     - **✓ OK** and **✗ Not OK** buttons, and a note field for not OK.
   - For a shot marked not OK, **Choose another** opens the shot's alternatives:
     - the shot's own candidate pictures (`choices`: photos and clips the storyboard considered), each
       with thumbnail, caption, category and licence; clips only when long enough for the shot;
     - **New illustration**, when the rules allow it here (objects and settings only; never a person
       or a named place; refused otherwise with the reason shown).
     Picking one replaces the shot's picture and marks the shot as the owner's choice.
   - **Rebuild review cut**: re-renders `animatic.mp4` with the changes, in the background, with a
     progress bar; frames refresh when it finishes.
   - **Final render**: a **Render final film** button (option: "mix music and effects under the
     narration") with progress and, when done, the film's link; next to it the equivalent terminal
     command (`documentary.py render data/video/<film> [--sound]`) for rendering outside the app.

## Categories

Every shot is sorted into one of four categories from what the storyboard already records (beat
`kind`, `media`, `link`, `library_id`, and the library item's `found_by`):

| Category | Meaning | Rule |
|---|---|---|
| Actual | a real photo or document of the person, place or thing named | real library picture (or a profile/exhibit/highlight built on one) whose `link` is `subject` |
| Stand-in | real archival photo or footage, but not of this exact moment | real library picture or clip whose `link` is `place_era` or `action`; any archival footage clip; a `hold` that kept a stand-in |
| Illustration | drawn by the image model | `kind == "generated"` |
| Graphic | made for the film | `kind` in card, keyfact, map, board, chart (roll-call cards, quote cards, labels, maps) |

A `reframe` (a closer framing of a picture already in the scene) takes the category of the picture it
frames. A shot with no picture at all is listed as **Missing** (it should not happen; shown so it is
noticed). The rule lives in one function with its own tests.

## How the owner's decisions are kept

- **Marks** (OK / not OK, note, time) are saved in `<film>/review.json`, keyed by shot. They never
  change the film; they are the owner's checklist. If the storyboard is rebuilt and a shot's picture
  changes, its mark is cleared (the mark was about the old picture).
- **Choices** (another picture, a new illustration) change `storyboard.json` and `shots.json` the same
  way the engine does (`storyboard.set_picture`, clip fields from the clip catalogue; an illustration
  through `ladder.describe_detail`, `ladder.safe_detail` and `ladder.generate_detail`), and record
  `owner: {"choice": "<library id>" | "illustration", "at": "<time>"}` on the shot.
- **Engine change**: the automatic rules that can move a picture (`respect_links`, `guard_dates`,
  `guard_victims`, `merge_repeats`, the ladder's swaps and loop breaking, `illustrate_holds`) skip a
  shot that carries `owner`. The safety rules still apply: an excluded picture (remains, bodies) is
  never offered as an alternative and is still replaced by `replace_excluded`.

## Server

A new module `app/video_review.py`, mounted in `app/main.py` under `/api/video`:

| Endpoint | Does |
|---|---|
| `GET /api/video/films` | films with a storyboard, review progress, which outputs exist |
| `GET /api/video/films/{film}` | scenes and shots with category, kind, times, narration, picture details, check verdict, owner choice, mark |
| `GET /api/video/films/{film}/shots/{n}/alternatives` | the shot's candidate pictures with category and thumbnail URL, and whether a new illustration is allowed |
| `PUT /api/video/films/{film}/shots/{n}/mark` | save OK / not OK and note |
| `POST /api/video/films/{film}/shots/{n}/choose` | replace the picture (a library id, or `illustration`) |
| `POST /api/video/films/{film}/rebuild` | start rebuilding the review cut (`build.py <film> --animatic`) |
| `POST /api/video/films/{film}/render` | start the final render (`documentary.py render <film> [--sound]`) |
| `GET /api/video/films/{film}/job` | the running or last job: kind, progress (from its log: shots rendered / total), state, last lines on failure |
| `GET /api/video/films/{film}/frame/{n}.jpg` | the shot's frame from the review cut (cached) |
| `GET /api/video/films/{film}/file/{path}` | a film file: the review cut, the final film, library pictures, illustrations |

- **Jobs**: one background process per film at a time (a second request is refused while one runs),
  started with the same Python and environment as the terminal commands, output to
  `<film>/logs/review_<kind>.log`. State survives a server restart because it is read from the log and
  the process id file; a job found dead without its finish line is reported as stopped.
- **Frames**: `ffmpeg` (the `video` conda env's) grabs one 480p frame at each shot's middle from
  `animatic.mp4`; cached as `<film>/review/frames/<n>_<key>.jpg`, where the key changes when the review
  cut or the shot's picture changes.
- **Files**: film names must be existing folders in `data/video/`; file paths are resolved inside the
  film folder and refused outside it. Video files are served with HTTP range requests so the player can
  seek.

## Front end

- `index.html` gets two tabs, **Voice** (the current studio, unchanged) and **Video**.
- A new `app/static/video.js` (the voice studio's `app.js` stays as it is) and a few styles in
  `style.css`: badge colours (actual green, stand-in amber, illustration violet, graphic grey, missing
  red), the shot row layout, the alternatives panel.
- The page polls `/job` every few seconds only while a job runs.

## Errors

- No review cut yet: the page says so and offers **Rebuild review cut**; frames show the source picture.
- A job fails: the job panel shows "failed" with the last log lines; marks and choices are kept.
- A new illustration is refused by the rules or the image call fails: the reason is shown; the shot is
  unchanged.
- Disk space: before a final render the server checks free space (about 20 GB for a 90-minute film)
  and refuses with the number if there is not enough.

## Testing

- Categories: a table of sample shots (each kind/link/media/found_by combination, reframes, missing).
- Owner choices survive the rules: a shot with `owner` is left alone by `respect_links`, `guard_dates`,
  `guard_victims`, `merge_repeats`, `illustrate_holds`; an excluded picture is still replaced.
- Endpoints with FastAPI's test client on a small sample film folder and a fake job runner: listing,
  shot details, marks, choose (library id and refused illustration), rebuild/render refused while a
  job runs, file path escape refused.
- One real run on the Bundy test film: open the page, mark shots, swap one picture, rebuild the review
  cut, confirm the frame changed.

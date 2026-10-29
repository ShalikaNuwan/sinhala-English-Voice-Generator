# Shot review: write what is wrong, an AI makes a new version, approve it

Date: 2026-10-29. Builds on `2026-10-29-storyboard-review-ui-design.md` (the Video tab).

## Goal

Under every shot in the Video tab the owner can write a free-text review ("show the house, not a street",
"zoom in on his face", "too modern, needs the 1970s"). An AI model reads that review and makes a new version
of **only that shot**; a second AI model checks the new version against the review; the owner watches the
regenerated shot and approves it. An approved shot is final. When the shots are approved, the owner
finalizes the film (the 1080p render).

Out of scope: version history and undo (Choose another… still lets the owner pick any picture), changing
narration or timing, editing several shots with one review.

## What the owner sees (per shot, in the existing shot list)

- **Review** text box (kept between visits) and a **Regenerate** button.
- While it works: "Queued", then "Choosing a picture…", "Searching the archives…", "Checking…",
  "Rendering the shot…". Other shots can be reviewed meanwhile; regenerations run one at a time.
- When done: the **regenerated shot plays in place** (480p, with its own narration), the AI's one-line
  explanation ("Used c0022, Bundy's former house (case photo): the review asked for the house"), the
  checker's verdict, and the new category badge.
- **Approve** (replaces the earlier OK / Not OK buttons). Approved shots show a green bar. Changing an
  approved shot (regenerate, choose another) removes the approval, because the picture changed.
- If it fails (nothing fitting, a refused drawing, an error): the reason is shown and the shot is unchanged.
- **Choose another…** stays, for picking a picture by hand.

At the top: "Approved 120 of 527", **Approve all unreviewed** (every shot with no review text and no
approval), and **Finalize video** (the existing final render, with the music/effects option). Finalize
warns how many shots are not approved and lets the owner go on anyway.

## How a review becomes a new shot

1. **Choose** (text model, `settings.text_model`): given the review, the narration over the shot, the
   scene (title, time, place, mood), the current picture's caption, and the film's usable free pictures and
   footage (id, caption, where found, actual / stand-in, era; the shot's own candidates first, clips only if
   long enough), it answers with one of:
   - `library`: a picture or clip id, optionally `focus` ("his face") for a closer framing of a photo;
   - `illustration`: what to draw (an object or setting only; never a person or a named place);
   - `search`: 1-3 search phrases, when nothing in the film fits;
   plus a one-line `explanation`. It is told the film's rules: free pictures only, no remains or bodies,
   victims shown with dignity, a stand-in must not pretend to be the real thing.
2. **Search** (only for `search`): the phrases are added to the film's `library/needs.json` and the
   engine's own steps run: `harvest.py --stages needs` (Commons, Openverse, Internet Archive; public domain
   and CC0 only), then `caption.py` (captions and flags graphic pictures). Then step 1 again with only the new
   pictures offered and no further search. Nothing new found: an illustration may be chosen instead.
3. **Apply** through the review module's `choose()` (owner lock, mark cleared), plus the crop when a focus
   was given (the engine's `ladder.located` and `frame_box`). Illustrations pass `ladder.safe_detail`.
4. **Check** (a second model, `settings.qa_model`, with vision): it sees the new picture (a clip's frame,
   or the drawing) with the review and the narration, and says whether the review is satisfied and why not.
   If not, step 1 runs once more with the checker's reason; the second result is kept whatever the verdict,
   and the verdict is shown to the owner.
5. **Render** just that shot: `build.py <film> --animatic --from <shot> --shots 1 --out
   review/clips/<n>_<time>.mp4` (854x480, its own narration).

## Saved where

- `<film>/review/regen.json`: per shot `{review, state (queued|running|done|failed), step, explanation,
  check {satisfied, problem}, clip, error, at}`. The review text stays here even after the picture changes.
- `<film>/review.json` (existing marks): an approval is the mark `ok`, tied to the shot's picture.
- `storyboard.json` / `shots.json`: the new picture, as with Choose another.

## Server

- `app/video_regen.py`: the steps above (each model call and each external step injectable for tests)
  and a one-at-a-time worker thread per server. A shot left `queued`/`running` by a server restart is
  reported as failed ("interrupted: regenerate again").
- Endpoints: `POST /api/video/films/{film}/shots/{n}/regenerate` `{review}` (202; 409 while a rebuild or
  final render runs), `POST /api/video/films/{film}/approve-unreviewed`. The film view's shots gain a
  `regen` field and the summary an `approved` count. Rebuild / render are refused (409) while a
  regeneration is queued or running.

## Testing

- Choose/check/apply with fake models: library pick, focus crop, illustration, refused drawing, search
  then pick among the new pictures, checker rejection leading to one retry.
- Worker: queue order, state transitions, interrupted state after a restart, refusal while a full job runs.
- Endpoints with the fake worker; approve-unreviewed.
- One real run on the Bundy test film: a review on one shot, regenerate with the real models, play the clip,
  approve.

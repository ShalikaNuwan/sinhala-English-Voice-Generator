# Sentence fix and fact check: two agents after the English is written

Date: 2026-10-29. Approved by the owner in chat ("yes go ahead").

## Why

Segments are pieces of the Sinhala audio cut at silences about every 45 s. A cut sometimes falls inside a
sentence, so the start of a sentence ends one segment and its end begins the next. The owner wants every
segment's English to hold whole sentences only. The owner also wants every fact in the English checked on
the web, with wrong facts shown (never corrected) together with the evidence links.

## 1. Sentence fixer (`app/sentences.py`)

- Text only. The audio pieces, their times and the Sinhala transcript never change.
- Looks at each pair of neighbouring narration segments (indices n and n+1, both `kind == "narration"`,
  both with English). A pair with a recording (`kind == "original"`) between or on either side is skipped.
- A model (`settings.text_model`, today's model, never the older one recorded on the job) reads the end of segment n and the start of n+1 (the last two narration
  lines of n, the first two of n+1, and the tail and head of the faithful English) and decides **each text
  on its own**: the narration and the faithful English can be cut differently. Real case: the faithful
  English of segment 15 ends "...did not belong to Gacy, and" while the adaptation had already carried
  "along with another identification card." into segment 15's narration. For each text the answer is no
  split, or a direction and the fragment to move:
  - `back`: the fragment is the **start** of n+1 and joins the end of n;
  - `forward`: the fragment is the **end** of n and joins the start of n+1.
  - `merge`: the cut left n ending in a broken sentence (often marked with … or ...) that n+1 finishes
    or repeats in other words (real case, Bundy 63/64: "…at that time, Ted had never…" | "He'd never
    been caught by police for anything."). The answer names the broken sentence at the end of n and the
    words at the start of n+1 that finish it, both exactly, plus one joined sentence and where it goes
    (`a` or `b`). The joined sentence replaces the broken half; the echo is removed from the other side.
  It moves the smaller half of the split sentence. The model is told that the narration may already be
  whole when the faithful English is split, that a fragment whose words already appear on the other side
  must never move (that would duplicate what the adaptation carried over), and that a text reading as
  whole sentences on both sides is "no split".
- Each fragment must be an exact whole-word prefix (back) or suffix (forward) of the text it comes from,
  compared with whitespace collapsed, and must not be that segment's whole text. Otherwise that text's
  move is rejected, that text stays unchanged and a job warning says why; a valid move of the other text
  still applies. A move cannot add, drop or reword words. A move is also refused when n already ends a
  sentence, unless the text marks the break itself (a trailing "..."/"…" or "[incomplete]", or "…"
  opening n+1).
- A merge must name exact edges of both texts and leave both segments some text, and its joined sentence
  may only use the words of the two halves: every word of four letters or more, and every number, must
  appear there (another ending of the same word, or a possessive, counts as the same word). So no new
  name, date, number or fact can enter. A segment that loses its first words starts with a capital.
- Lines (`lines_json`, the break-marked spoken runs), only when the narration moved: the moved words join
  the edge line (the last line of n, or the first line of n+1) with a space; a line left empty is removed
  together with its break. Other lines and their breaks are untouched. An edited edge line's break is
  recomputed from its measured silence as a script edit does. `narration_en` is rebuilt as the lines
  joined by blank lines (or is the moved plain text when the segment has no lines).
- Each changed segment records a note in `sentence_fix_json` naming the text that moved: `{"notes": ["End
  of sentence moved in from segment 16 (narration and faithful)"], "at": iso}` (or "(narration)",
  "(faithful)"). Only when the narration changed, a segment that already has voice audio is marked
  `qa_status='needs_review'` with the issue "Text changed by the sentence fix; regenerate the voice".
- Model calls for the pairs run in parallel (decisions are made on the text before any move; two moves on
  the two ends of one segment do not clash, and a move that would empty a segment is rejected).
- Up to three passes: the first asks about every cut; later passes ask again only about cuts that still
  look broken (n not ending a sentence or ending on a break mark, n+1 starting with a small letter or a
  break mark), until a pass changes nothing. The model is told when a break mark shows at the cut.
  Real result on Bundy: 57 broken cuts after the first version, 0 after this one.

## 2. Fact checker (`app/fact_check.py`)

- Runs on the final English narration of each narration segment. A model (`settings.text_model`) with
  OpenAI's `web_search` tool lists the segment's checkable facts (names, dates, numbers, places, events;
  not opinions or style), searches each, and gives a verdict `correct`, `wrong` or `unverified`, a short
  `evidence` sentence (what the sources say) and source links.
- Links are only those the search actually returned (`include=["web_search_call.action.sources"]` and the
  answer's URL citations); tracking parameters such as `utm_source` are removed. A `wrong` verdict with no
  surviving link becomes `unverified` ("no source link found").
- The previous segment's English is passed as context so pronouns resolve; the project topic too.
- It never edits any text.
- Stored per segment in `facts_json`: `{"state": "checking"|"done"|"failed", "checked_text": str,
  "at": iso, "error": str, "items": [{"claim", "verdict", "evidence", "sources": [{"url", "title"}]}]}`.
  The API adds `facts_stale: true` when `checked_text` differs from the current `narration_en`.
- Segments are checked four at a time.

## Pipeline

`process_job` becomes: per segment, transcribe → translate → adapt (text pass); then **fixing sentences**;
then per segment voice (when the voice stage is on) → QA; then **checking facts**; then the review gate as
before. Job config flags `fix_sentences` and `check_facts` (default true) switch the two stages off.
Regenerating one segment does not rerun either agent; the owner uses the buttons.

## API

- `POST /api/jobs/{job_id}/fix-sentences` → 202; 409 while it is already running for that job.
- `POST /api/jobs/{job_id}/check-facts` → 202 (all narration segments); 409 while running.
- `POST /api/segments/{segment_id}/check-facts` → 202 (that segment).
- `GET /api/jobs/{job_id}/tasks` → `{"sentences": null | {"state": "running"|"done"|"failed", "done": k,
  "total": n, "error": str, "changed": [segment_index]}, "facts": null | {same, without "changed"}}`.
  Held in memory; a restart forgets it (per-segment fact state stays in the database).
- Segments gain `sentence_fix`, `facts` and `facts_stale`.

## Front end (`app/static/app.js`, `style.css`)

- Above the segments: **Fix split sentences** and **Check all facts** buttons with progress
  ("Checking facts 12/40…"), and a fact summary: "Fact check: all 84 facts correct" or
  "3 facts look wrong · 2 couldn't verify" (plus "5 segments out of date" when stale).
- On each narration segment: the sentence-fix note; a facts block with wrong facts in red (claim, what
  the sources say, links opening in a new tab), unverified in amber, and the correct ones folded as
  "12 facts correct" (expandable, with sources); "Checking facts…" while running; "Out of date: the text
  changed since the check" when stale; a **Check facts again** button.

## Testing

Fake models only, no network: no split, back and forward moves, multi-line fragments, emptied line,
rejected (not a prefix, whole segment, faithful mismatch), recordings skipped, notes and needs-review;
fact parsing, link filtering, `wrong` without a link downgraded, state transitions, stale flag; pipeline
order (fix runs before voicing and QA, facts after); endpoints and 409s. One real run on an owner project
copy with the real models.

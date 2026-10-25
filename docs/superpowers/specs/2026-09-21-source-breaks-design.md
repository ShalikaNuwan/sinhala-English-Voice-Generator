# Source-matched breaks and the ElevenLabs hand-off — design

Date: 2026-09-21
Status: approved by the project owner in conversation; implementation plan to follow.
Builds on: `main` (storyteller voice, kept recordings, pause shaping, ElevenLabs provider).

## Goal

The owner voices the English narration in the ElevenLabs web app (Multilingual v2) rather than in
the pipeline. The pipeline's job becomes: produce a reviewed English script whose breaks are the
owner's own breaks, measured from the Sinhala recording at the places they happened, and hand it
over as ordered parts around the kept recordings, with the breaks written into the text as
`<break time="x.xs" />` marks that ElevenLabs v2 honours.

Three things change:

1. **Text-only processing.** A job can stop after transcription, detection, translation,
   adaptation and QA, voicing nothing. The existing voice path stays as the fallback.
2. **Phrase runs.** Every narration segment keeps the narrator's phrases as the diarizer heard them,
   grouped into *runs* by the owner's rule: phrases that follow one another with almost no gap are
   one flowing run; a real pause ends a run. The adaptation writes the English as one line per run,
   in order, so every line has a measured silence after it.
3. **Parts export.** The reviewed script is written as numbered narration text files, one per stretch
   between kept recordings (split further under ElevenLabs' limits), each line followed by its break
   mark, plus the recording clips and an order sheet with source times and a target length per part.

## Decisions taken with the owner

| Question | Decision |
|---|---|
| Where the voice is made | ElevenLabs web app, Multilingual v2 (break tags supported; v3 is not). |
| Stitching | By the owner, in the editor, from the numbered parts and clips. No return step in this version. |
| Phrase merging | By flow: phrases with almost no gap between them are one paragraph; a break is written only where the narrator paused. |
| Timing | The order sheet prints the source time span and target length per part; no enforcement. |

## Where the phrases come from

`_create_segments` already diarizes every chunk with the narrator reference and has each span's
start, end and Sinhala text. Spans the narrator spoke (labelled `narrator` or non-Latin text) that
fall inside a narration piece become that segment's phrases, stored in a new `phrases_json` column
as `[{"start_ms", "end_ms", "text"}]` in absolute source time. When detection is off or the chunk's
diarization failed, the segment has no phrases; the adaptation then runs as today (one passage) and
the export writes default breaks from a small table, and the job carries a warning naming the
segment.

## Runs

`app/breaks.py` (new, pure):

- `runs(phrases, segment_end_ms, next_phrase_start_ms=None) -> list[Run]` where a run is
  `{"start_ms", "end_ms", "text", "gap_after_ms"}`. Consecutive phrases with a gap under
  `FLOW_GAP_MS = 150` are merged into one run (text joined with a space). `gap_after_ms` is the
  silence to the next run; for the last run it is the silence to the next segment's first phrase
  when known (the segments are contiguous in source time), else to the segment end.
- `break_seconds(gap_ms, line_text) -> float | None`: the table below, returning `None` when no
  mark should be written and otherwise a value rounded to 0.1 s.

| Measured gap | Line ends a sentence (`. ! ? …`) | Line does not |
|---|---|---|
| under 300 ms | 0.4 s (a sentence end is lifted first) | none; the comma carries it |
| 300 to 3000 ms | max(gap, 400) | gap |
| over 3000 ms | 3.0 s | 3.0 s |

  Constants: `FLOW_GAP_MS = 150` (run merging), `MIN_MARK_MS = 300` (ElevenLabs v2 already pauses
  about that long at a comma, so a smaller mark only spends the tag budget), `SENTENCE_MIN_MS = 400`,
  `BREAK_MAX_MS = 3000`, all in one place. Values are rounded to 0.1 s.
- `mark(seconds) -> str` renders `<break time="0.8s" />`.

## Adaptation with runs

`AIClient.adapt` gains an optional `runs: list[str]` (the Sinhala text of each run, numbered). When
given, the prompt adds: "The Sinhala was spoken as these numbered runs. Write the script as one line
per run, in the same order, each line rendering exactly that run's content; merge two runs into one
line only when English grammar leaves no other choice, and say which runs the line covers." The
structured output gains `lines: list[Line]` with `Line = {"text": str, "runs": list[int]}`. The
existing fields stay; `narration_text` becomes the lines joined by blank lines so every existing
consumer (QA, the fallback voice path, the review page) keeps working.

Validation, in the pipeline after adaptation: the union of `runs` over the lines must be exactly
`1..n` in order, each run used once. A merged line (two or more runs) is allowed; its break is the
last covered run's gap. A violation is a QA issue ("Script lines do not follow the spoken runs")
and the segment goes to review with the lines kept as returned.

The segment stores `lines_json`: `[{"text", "runs", "gap_after_ms", "break_s"}]`.

## Text-only processing

`voice_stage` (job config; `ProcessRequest.voice_stage`; `VOICE_STAGE` env, default on). When off,
`_narrate_segment` skips synthesis, shaping and mastering; QA runs on the text; the segment has no
audio and no duration ratio; the job stops at review. The review page gets a "Text only (voice
elsewhere)" checkbox.

## Parts export

`Pipeline.export_script_parts(project_id)` writes `exports/script_parts/`:

- Narration parts: every stretch of narration between two kept recordings, in order. Within a
  stretch, lines follow one another as `text <break time="x.xs" />` with a newline after each line;
  the break between two segments comes from the last line of the earlier segment. A stretch is split
  into further parts at a line boundary when it would exceed `PART_MAX_CHARS = 9000` or
  `PART_MAX_BREAKS = 40` (ElevenLabs destabilises with too many marks in one generation). Segments
  without `lines_json` contribute their `narration_en` paragraphs with default breaks
  (sentence 500 ms, paragraph 900 ms).
- Recording clips: copied as numbered WAVs.
- `index.md`: playing order, one line per file: source time span, and for narration parts the
  character count, the number of break marks, and the target length (the source span's duration).

Endpoint `POST /api/projects/{id}/export/parts` returns the files as media URLs; a script
`scripts/export_script_parts.py` does the same locally. The review page gets an "Export script
parts" button.

## Error handling

| Case | Behaviour |
|---|---|
| Detection off or diarization failed on a chunk | No phrases; passage-mode adaptation; default breaks in the export; job warning. |
| Adaptation lines do not cover the runs | Lines kept; QA issue; needs review. |
| A run's gap is unknown (last run of the last segment) | Break from the segment's trailing silence, capped. |
| Export with a recording that has no audio | 409 with "regenerate it first". |
| Old rows (no `phrases_json`/`lines_json`) | Export uses `narration_en` with default breaks. |

## Testing

- `tests/test_breaks.py`: runs from phrases (merge under 150 ms, gap after each run, last gap from
  the next segment or the segment end), the break table (none, clause, sentence lift, cap, rounding),
  mark rendering, part splitting by characters and by marks.
- `tests/test_ai.py`: `adapt` with runs sends the numbered runs and asks for lines; structured output
  parses `lines`.
- `tests/test_pipeline.py`: phrases stored per narration segment from the fake diarizer; runs and
  lines stored; a text-only job voices nothing and stops at review; validation flags a bad line
  set; export writes parts with marks in the right order around a recording; old rows export with
  default breaks.
- `tests/test_api.py`: `voice_stage` in the job config; the export endpoint.
- Paid check: one real run in text-only mode on the 130-second slice, then the export folder for the
  owner to paste into ElevenLabs v2.

## Out of scope

- A return step that assembles the ElevenLabs files (may follow if hand stitching proves tedious).
- Enforcing English length against the Sinhala.
- v3 audio-tag output.

## Addendum, 2026-09-21: diarizer coverage

The first real-model run (project `5f327b64-…`, 130 s slice) showed that one diarization call can
drop narrator spans another call returns (5 s and 3 s stretches of chunk 1 in that run). A dropped
span becomes a false long break, and the adaptation is pushed to squeeze the missing Sinhala into
the neighbouring lines. FFmpeg silence detection cannot substitute: the narration sits on a music
bed and nothing drops below the noise floor. Three changes, all approved by the owner, who asked
for the best output regardless of model cost:

1. **Second diarizer pass, gap-filling.** Every chunk is diarized twice. The first pass stays the
   base for recording detection and clip transcripts. Narrator spans from the second pass are added
   only where the first pass has no span at all, trimmed to the uncovered stretch, and only when the
   uncovered stretch is at least 300 ms (`recordings.fill_gaps`). A third pass runs when the
   narrator's coverage of the chunk's narration pieces is still under `COVERAGE_MIN` (0.85). A
   failed extra pass is a warning, never a chunk failure.
2. **Coverage guard.** For each narration segment, coverage is the phrases' total duration over
   the segment's duration. Under `COVERAGE_MIN`, every break over `GUARDED_MAX_MS` (1000 ms) is
   capped at 1.0 s and the job records "Phrases cover N% of segment K; breaks over 1.0 s capped",
   so the reviewer knows which part to check by ear. Lines store `coverage` so the export can show it.
3. **Prompt.** The runs-mode adaptation is told that run texts are rough machine transcripts,
   possibly garbled or loosely translated, that show only where each run starts and ends; every fact
   and sentence comes from the faithful translation, and nothing that appears only in a run text is
   added.

## Addendum 2, 2026-09-21: the pause map comes from Realtime voice detection

The second real-model run showed the diarizer's other instability: the same chunk came back as 5,
8 or 12 narrator spans on repeat calls, and a 17 s stretch as one span, so those stretches got no
breaks. Its server-VAD chunking setting does not change the returned boundaries. Whisper cannot time
Sinhala (20 words in 45 s), the GPT transcription models return no word times, and FFmpeg silence
detection finds nothing under the music bed.

OpenAI's Realtime API, in a transcription session with `server_vad`, reports every moment speech
starts and stops in milliseconds, and transcribes each turn with `gpt-4o-transcribe`. Probed on the
same chunks (threshold 0.8, silence 200 ms, no prefix padding): identical boundaries on repeat calls,
two different settings within 60 ms of each other, the 17 s blob split into three turns, and the
turn transcripts in Sinhala script. `audio_end_ms` includes the configured silence, so a turn ends
at `audio_end_ms - silence_duration_ms`; a turn that is still open when the audio ends closes at the
audio's end. Turns and transcripts pair by `item_id` (transcripts arrive out of order).

Design:
- `AIClient.speech_turns(pcm_path, threshold, silence_ms, model)` streams 24 kHz mono 16-bit PCM
  into one transcription session and returns `[{start_ms, end_ms, text}]`, file-relative, sorted.
  The pure pairing lives in `app/turns.py` (`pair_events(events, silence_ms, duration_ms)`), tested
  without a network.
- `_create_segments` builds every narration piece's phrases from its own audio's turns (the piece
  file, or the chunk file shifted by the chunk lead), not from the diarizer's narrator spans. The
  diarizer still finds the narrator and the recordings; its narrator spans remain the fallback,
  with a warning "Pause map unavailable for segment N: …", when the Realtime call fails.
- Settings: `PAUSE_SOURCE` (`realtime`, default; `diarizer` restores the old behaviour),
  `REALTIME_VAD_THRESHOLD=0.8`, `REALTIME_VAD_SILENCE_MS=200`, `TURN_TRANSCRIBE_MODEL=gpt-4o-transcribe`.
- Runs, lines, the coverage guard and the export are unchanged; they simply see better phrases.
  A 45 s chunk takes about 23 s of wall-clock in one session; the owner accepted the time.

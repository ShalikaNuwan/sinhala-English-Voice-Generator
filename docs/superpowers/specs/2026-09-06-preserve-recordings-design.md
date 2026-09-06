# Preserve embedded recordings — design

Date: 2026-09-06
Status: approved by the project owner in conversation; implementation plan to follow.
Builds on: `natural-narration` branch (segment gaps, continuity, WAV segments).

## Goal

Creator recordings sometimes contain evidence audio played inside the video: 911 calls, interrogation
tape, news clips. That audio is real and must reach the English export untouched. Today the pipeline
transcribes it, translates it, writes it into the script, and re-voices it with the English narrator.
The Gilgo Beach project shows the failure: the transcript of segment 2 contains the 911 dispatcher's
English lines inside the Sinhala narration, and the generated audio speaks them in the narrator's voice.

After this change, embedded recordings are detected, cut out as their own segments, passed through
from the source with only loudness alignment, confirmed by the reviewer, and assembled in place. Only
the narrator's own speech is translated and voiced.

## Decisions taken with the owner

| Question | Decision |
|---|---|
| Overlap | The narrator is silent while a clip plays. Overlapping speech is out of scope. |
| Clip language | Always English. Language is used as a second signal alongside speaker identity. |
| Confirmation | Detect, then the reviewer confirms on the review page before assembly. |
| Loudness | Clips are level-aligned to the narration target; no other processing. |
| Approach | Diarize every segment with the narrator as a known speaker (approach 1 of 3). |

## How detection works

OpenAI's `gpt-4o-transcribe-diarize` returns every span of speech with `speaker`, `start`, `end`, and
`text`, and accepts up to four reference clips (2–10 s, base64 data URLs) under `known_speaker_names`
and `known_speaker_references`; spans matching a reference carry that name, other speakers get letters.
Files are limited to 25 MB; the pipeline's 20–60 s chunks at 24 kHz mono are under 3 MB.

1. **Narrator reference, once per project.** The 45 s delivery sample already extracted for the
   speaking profile is diarized with no references. The speaker with the most total speech is the
   narrator; their longest continuous span, trimmed to 10 s, is cut to
   `original/narrator_reference.wav` and recorded in the project's speaking profile JSON as
   `narrator_reference`. If no span reaches 2 s, detection is disabled for the job with a warning.
2. **Per chunk.** After the existing silence-based split, each chunk file is diarized with the
   narrator reference. Spans whose text contains letters from a non-Latin script are the narrator
   speaking (the diarizer renders Sinhala in Sinhala or, occasionally, Devanagari script). A
   recording opens at a span that is Latin-only or empty **and** not labelled `narrator`, and stays
   open across every following span, including silences of several seconds and short Latin-only
   spans mislabelled as narrator, until a non-Latin span closes it or a gap exceeds 10 s. Requiring
   the script test protects against a failed reference match. Recordings under 1.5 s are ignored;
   each remaining one is padded 150 ms into the surrounding silence, clamped to the chunk. (A paid
   probe on the Gilgo Beach 911-call chunk shaped this rule: the call has 3–4.5 s silences between
   dispatcher lines and one "Okay." mislabelled as narrator.)
3. **Re-cut.** The chunk becomes an ordered list of pieces, each `narration` or `original`. A
   narration piece is kept only if it overlaps a span of narrator speech; a piece with no narrator
   speech (leading or trailing silence, a call's own pause) is absorbed into the neighbouring
   recording. A chunk with no recordings stays as it is, one narration segment, using the existing
   chunk file. A recording that runs across a chunk boundary becomes two adjacent original
   segments; that is accepted for this version.

## Data model

- `segments.kind TEXT NOT NULL DEFAULT 'narration'` (values `narration`, `original`), added through
  the existing `ADDED_COLUMNS` migration so old rows read as narration.
- `jobs.warnings_json TEXT NOT NULL DEFAULT '[]'`: non-fatal problems such as "recording detection
  unavailable", shown in the progress line.
- `projects.speaking_profile_json` gains `narrator_reference: {path, start_s, end_s}`.
- Original segments store the clip's diarized text in `transcript_si` (it is English; the column is
  the segment's source text), leave `faithful_en`/`narration_en` empty, and store the pass-through
  clip in `tts_audio_path`/`tts_duration_ms` so preview and assembly work unchanged.

## Units

### `app/recordings.py` (new, pure)

- `has_foreign_script(text) -> bool`: any alphabetic character outside the Latin range (code point
  0x0250 and above). `is_english(text)`: has a Latin letter and no foreign script.
- `choose_reference(spans) -> tuple[float, float] | None`: dominant speaker's longest span, trimmed
  to `REFERENCE_MAX_S = 10.0`, `None` if under `REFERENCE_MIN_S = 2.0`.
- `original_spans(spans, chunk_ms) -> list[tuple[int, int]]`: the rule in step 2 above, in ms
  relative to the chunk. Constants `MAX_INTERNAL_GAP_MS = 10000`, `MIN_SPAN_MS = 1500`,
  `PAD_MS = 150`.
- `narration_spans(spans) -> list[tuple[int, int]]`: spans with foreign script, in ms.
- `cut_plan(chunk_start_ms, chunk_end_ms, originals, narration) -> list[Piece]` where
  `Piece = (start_ms, end_ms, kind)` in absolute source time; the rule in step 3 above.
- `english_run(text, minimum_words=8) -> bool`: true when the text contains a run of at least
  `minimum_words` consecutive Latin-script words. Used as the safety net.
- `clip_text(spans, start_ms, end_ms) -> str`: the diarized text inside a span, for `transcript_si`.

### `app/ai.py`

- `AIClient.diarize(audio_path, model, reference_path=None) -> list[dict]`: calls
  `audio.transcriptions.create(model=model, file=..., response_format="diarized_json",
  chunking_strategy="auto", known_speaker_names=["narrator"],
  known_speaker_references=["data:audio/wav;base64,…"])` when a reference is given, and returns
  `[{"speaker", "start", "end", "text"}]`. No prompt: the model does not accept one.

### `app/audio.py`

- `AudioService.extract(source, start_ms, end_ms, destination, pad_ms=200) -> AudioInfo`: cut a
  piece from the normalised source, mono 24 kHz, used for narration pieces (replaces the inline
  extraction in `split`, which is refactored to call it).
- `AudioService.extract_levelled(source, start_ms, end_ms, destination) -> AudioInfo`: the same cut
  followed by `loudnorm=I=-16:TP=-1.5:LRA=11`, used for original clips.

### `app/config.py`, `app/schemas.py`, `app/main.py`

- `DIARIZE_MODEL` setting, default `gpt-4o-transcribe-diarize`; `ProcessRequest.diarize_model`
  override; `ProcessRequest.detect_recordings: bool = True` and the matching job-config key, as a
  kill switch.
- `PATCH /api/segments/{id}/kind` with `{"kind": "narration" | "original"}`.
- `POST /api/segments/{id}/confirm`: marks an original segment confirmed (`qa_status = passed`).
- Job responses include `warnings`; segment responses include `kind`.

### `app/pipeline.py`

- `_narrator_reference(job_id, project, normalized, profile, config) -> Path | None`: reuse the
  stored reference if its file exists; otherwise diarize the delivery sample, choose, cut, store.
  Any failure returns `None` and appends a job warning. Tracked as stage `diarization`.
- After `split`, for each chunk with detection enabled: diarize (tracked, stage `diarization`),
  compute originals and the cut plan, and insert one segment row per piece with `kind`. Original
  pieces are cut with `extract_levelled` to `generated/{job}/NNNN_original.wav` and inserted with
  `status = kept`, `qa_status = needs_review`, `qa_json = {"passed": false, "issues":
  ["Recorded audio kept as is. Confirm."], "duration_ratio": 1.0}`. Narration pieces that are not
  the whole chunk are cut with `extract`. A chunk whose diarization call fails stays one narration
  segment and adds a warning. Segment indexes are assigned in source order across all pieces.
- The per-segment loop body moves into `_narrate_segment(job, project, segment, profile,
  previous_context, previous_style, persona, speed) -> tuple[str, dict]` returning the new
  `(previous_context, previous_style)`. `process_job` iterates all segments in index order,
  skipping `original` ones except to append `"\n[Recording plays: <transcript>]"` to
  `previous_context`. Continuity (`previous_style`) skips originals.
- Safety net inside `_narrate_segment`: if `english_run(transcript)`, QA is marked failed with the
  issue "Possible recorded audio (English speech in transcript). Mark as original if so." even
  when the model QA passes.
- `regenerate_segment` on an `original` segment re-cuts the clip and leaves QA as needs review.
- `set_segment_kind(segment_id, kind)`: to `original`, cut the clip from the normalised source,
  clear `faithful_en`/`narration_en`/`style_json`, set the kept status and needs-review QA; to
  `narration`, clear the pass-through fields and run `_narrate_segment` with context from the
  previous narration segment. Runs as a background task from the endpoint.
- `confirm_segment(segment_id)`: `qa_status = passed`, `status = kept` for originals; 409 otherwise.
- `assemble_project`: the gap next to an original segment on either side is `MIN_GAP_MS`;
  narration-to-narration gaps use `segment_gap_ms` as today.

### Web UI (`app/static/app.js`, `index.html`, `style.css`)

- Segment card shows a "Recorded audio" badge for originals, hides the English narration box, shows
  the clip transcript read-only and the audio player, and offers "Confirm recording" and "Treat as
  narration". Narration cards gain "Keep as recorded". The progress line appends job warnings.

## Error handling

| Case | Behaviour |
|---|---|
| Diarization model unavailable or call fails on the sample | No reference; detection off for the job; warning "Recording detection unavailable: <reason>". Job completes as narration only. Safety net still flags English runs. |
| Diarization fails on one chunk | That chunk is one narration segment; warning names the chunk. |
| Reference under 2 s | Same as no reference. |
| Original span covers the whole chunk | One original segment. |
| Narrator misidentified (everything "other") | Sinhala-script check keeps narration; English-only misfires would reach the reviewer as originals needing confirmation, never silently. |
| Old jobs and rows | `kind` reads as narration; assembly and regeneration unchanged. |
| Toggle to narration fails mid-way | Segment marked failed with the error, as `regenerate_segment` does today. |

## Testing

Unit tests, no paid calls:

- `tests/test_recordings.py`: `has_foreign_script`/`is_english` (Sinhala, Devanagari, Latin, empty),
  `choose_reference` (dominant speaker, trim to 10 s, under 2 s → None), `original_spans` on the real
  probe spans from the Gilgo chunk (one recording 13.75–36.64 s), plus: Sinhala span closes a
  recording, mislabelled Latin "Okay." inside stays inside, gap over 10 s splits, under 1.5 s dropped,
  pad and clamp; `cut_plan` (leading/trailing silence absorbed, narration kept only when it overlaps
  narrator speech, whole chunk original, no originals); `english_run`; `clip_text`.
- `tests/test_ai.py`: `diarize` sends the model, `diarized_json`, `chunking_strategy`, the
  narrator name, and a `data:audio/wav;base64,` reference; omits reference fields when none.
- `tests/test_audio.py`: `extract` and `extract_levelled` durations and mono 24 kHz output.
- `tests/test_pipeline.py` with a fake diarizer: a 65 s source whose fake spans mark 20–30 s as
  another English speaker yields narration / original / narration segments in order; the original
  segment has source audio, no AI calls, needs-review QA and the confirm note; the following
  narration segment's adaptation context contains "[Recording plays:"; continuity skips the
  original; gaps next to it are 300 ms; a failing diarizer yields one narration segment plus a
  warning; the English-run safety net flags a narration segment; `set_segment_kind` both ways;
  `confirm_segment`.
- `tests/test_api.py`: the two new endpoints and `detect_recordings` in the job config.
- `tests/test_database.py`: `kind` and `warnings_json` columns are added to an old database.

One paid check, approved with the feature: run the reference selection and one chunk diarization on
the Gilgo Beach source and report the spans, to confirm the 911 call is found where the transcript
says it is. No tuning of thresholds without a listen.

## Out of scope, noted for later

- Editing detection boundaries by hand on the review page (splitting a segment at a time).
- Clips the narrator talks over.
- Non-English evidence audio (the Sinhala-script check would need replacing by a pure speaker rule).

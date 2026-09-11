# Natural pauses — design

Date: 2026-09-06
Status: approved by the project owner in conversation; implementation plan to follow.
Builds on: `main` after the natural-narration and preserve-recordings merges.

## Goal

The breaks in the English narration must sound like a person telling a story: clause breaks short,
sentence ends settled, paragraph shifts longer, held beats longer still, none of them metronomic,
and no stray choppiness inside a phrase. Today the voice model chooses every pause length itself,
and the same instruction yields different pauses on different runs. Measured on the 5 September
export the owner preferred against the newest output: the new voice has almost twice the spread of
pause lengths, more 0.7–1.5 s holds, two gaps over 1.5 s, and more sub-300 ms micro-breaks. Three
causes: a numeric pause instruction derived from the source with a 300 ms detection floor (which
overstates the narrator's real 200–300 ms habit), paragraph breaks and ellipses in the adapted script
that the voice turns into long holds, and assembly gaps added on top of the silence the voice
already leaves at segment edges.

After this change the pipeline shapes every pause itself, from human-like targets, and checks the
result. The voice model still produces the speech; it no longer decides the silences.

## Decisions taken with the owner

| Question | Decision |
|---|---|
| Approach | Shape pauses after synthesis using word-timestamp alignment; calmer direction; a pause check in QA; verified by a listening set and a real-model run. |
| Targets | Fixed storyteller bands, nudged shorter for a continuous narrator and longer for a measured one. Not a copy of the Sinhala pause lengths. |
| Time | Not a constraint. Correctness is. |

## How shaping works

1. **Synthesize as today** to `NNNN_rNN_raw.wav`.
2. **Find the pauses** with FFmpeg `silencedetect` at −40 dB, minimum 80 ms. These edges are precise.
3. **Find the words** with one transcription of the raw audio using `whisper-1`, `verbose_json`,
   `timestamp_granularities=["word"]`, `language="en"` (new `AIClient.word_timestamps`). A probe on a
   real segment put every pause within about 50 ms of a word boundary.
4. **Tie each pause to the script.** Script tokens are the whitespace-separated words of
   `narration_en`, each carrying the break class that follows it: `clause` after `,` `;` `:`, `dash`
   after `—` or `–` (or a token that is only a dash), `beat` after `…` or `...`, `sentence` after `.`
   `!` `?`, `paragraph` when a sentence end is followed by a blank line, otherwise `none`. Spoken
   words are aligned to script tokens with `difflib.SequenceMatcher` over normalised forms
   (lower-case, alphanumerics only), which tolerates the transcriber writing "1st" as "1". Word
   timestamps drift into a silence from both sides, so for each pause the word before it is the last
   spoken word that starts before the silence and does not run past its end; that word's script
   token names the pause's class. A pause with no preceding word or whose word is
   unmatched is `unknown` and is left at its current length, never shortened. Abbreviations
   (`Mr.`, `a.m.`, initials) are not sentence ends. A word ending in a hyphen is a dash break.
   Shaping refuses (raw audio kept) when fewer than three quarters of the spoken words align.
5. **Choose a target** for each pause: a uniform random draw inside the class band, scaled by the
   narrator's pace, using a random generator seeded by the segment id so a re-run is reproducible.
   A `none` pause is capped at `UNPUNCTUATED_MAX_MS` (kept if already shorter). Bands in ms:

   | Class | Band |
   |---|---|
   | clause | 150–280 |
   | dash | 250–400 |
   | sentence | 420–650 |
   | paragraph | 850–1200 |
   | beat | 1200–1600 |

   Pace scale: `continuous` 0.85, `moderate` 1.0, `measured` 1.15 (from the speaking profile's
   derived pace). `UNPUNCTUATED_MAX_MS = 200`. Leading and trailing silence are trimmed to
   `EDGE_SILENCE_MS = 60` so the assembly gap is the only gap at a join.
6. **Rebuild the audio** in one FFmpeg call: `atrim`/`asetpts` for every speech run with a 5 ms fade
   at each end so a join never clicks, `aevalsrc=0` for every gap, `concat`. Speech is never
   time-stretched or level-changed. Overlapping pauses or an unreadable duration are errors, never
   silently dropped audio.
7. **Measure the result** (`pause_profile`): count, median, 90th percentile, maximum, pauses per
   minute; `shape` also returns the per-class counts. QA flags a segment when any pause exceeds the
   scaled `beat` band by more than 200 ms, with the numbers in the issue text. (Unpunctuated pauses
   are always capped, so they never need a flag.)

If word timestamps cannot be obtained (API error, unsupported model), shaping falls back to
punctuation-order alignment when the count of pauses of at least 250 ms equals the count of script
breaks; otherwise the raw audio is used unchanged and a job warning names the segment.

## Direction and script changes

- `app/direction.py` `_character`: drop the sentence quoting the measured mean and longest pause;
  keep the description and the derived labels. Pauses are now the pipeline's job.
- `app/ai.py` adaptation prompt: a paragraph break "only at a real shift in the story, at most two
  per passage"; the ellipsis "at most once per passage".

## Units

### `app/pauses.py` (new)

- Constants: `BANDS`, `PACE_SCALE`, `UNPUNCTUATED_MAX_MS`, `EDGE_SILENCE_MS`, `SILENCE_DB = -40`,
  `SILENCE_MIN_S = 0.08`.
- `script_tokens(script) -> list[tuple[str, str]]`: `(normalised_word, break_class_after)`.
- `normalise(word) -> str`.
- `align(tokens, spoken) -> dict[int, int]`: spoken index → script index for matched words.
- `detect_pauses(path, ffmpeg, ffprobe) -> list[tuple[float, float]]` (seconds), leading/trailing
  included.
- `classify(pauses, spoken, mapping, tokens) -> list[tuple[float, float, str]]` and
  `classify_by_order(pauses, tokens, duration_s)` (the fallback; `None` when counts differ).
- `choose_targets(classified, pace, rng, duration_s) -> list[tuple[float, float, int]]` (target ms).
- `rebuild(path, plan, duration_s, destination, ffmpeg) -> None`.
- `pause_profile(path, ffmpeg, ffprobe) -> dict` and `profile_issues(result) -> list[str]` (takes
  the dict `shape` returns).
- `shape(path, script, spoken, pace, destination, seed, ffmpeg, ffprobe) -> dict` (method, pace,
  per-class counts, plan, and the profile of the shaped file); raises `AlignmentError`.

### `app/ai.py`

- `AIClient.word_timestamps(audio_path, model) -> list[dict]` with keys `word`, `start`, `end`.

### `app/config.py`, `app/schemas.py`, `app/main.py`

- `ALIGN_MODEL` (default `whisper-1`), `SHAPE_PAUSES` (default on); `ProcessRequest.align_model`,
  `ProcessRequest.shape_pauses`; job-config keys `align_model`, `shape_pauses`.

### `app/pipeline.py`

- `_narrate_segment`: synthesize to `NNNN_rNN_raw.wav`; when shaping is on, obtain word timestamps
  (tracked call, stage `alignment`), shape to `NNNN_rNN.wav`, store the pause profile inside
  `qa_json` under `pauses`, and let `_qa_verdict` add `profile_issues`. When shaping is off or fails,
  the raw file is renamed to `NNNN_rNN.wav`; on failure the raw file is kept and copied to
  `NNNN_rNN.wav`, and a warning names the segment.
- `regenerate_segment` (tts stage) shapes the same way.
- The speaking profile's derived pace drives the scale; a missing profile means `moderate`.

### Scripts

- `scripts/pause_listening_set.py <job_id> <segment_index...>`: for each narration segment, copies the
  raw and shaped files to `data/pause_listening/`, writes a pause table for both, and writes a
  pointer to the owner's reference file. No API calls.

## Error handling

| Case | Behaviour |
|---|---|
| Word-timestamp call fails | Punctuation-order fallback if counts match; else raw audio, warning. Job continues. |
| No pauses detected | With word timestamps: rebuilt with edge trims only, profile recorded. Without: the order fallback refuses when the script has breaks, so the raw audio is kept with a warning. |
| Alignment leaves a pause unmapped | Class `unknown`: left at its current length. |
| Rebuild or profile fails in FFmpeg | Raw audio kept and copied to the final name; warning names the segment. The paid synthesis is never discarded. |
| Shaping off (`SHAPE_PAUSES=0`) | Raw file used; no alignment call. |
| Old rows | Unaffected; regeneration of an old segment shapes it. |

## Testing

- `tests/test_pauses.py`: tokenising and classes (comma, dash, ellipsis, sentence, paragraph,
  none, numbers); normalisation; alignment with a mismatched token ("1st" vs "1") and an
  inserted/dropped word; classification under timestamp drift on both sides; targets inside scaled bands and
  reproducible by seed; `none` capped and never lengthened; edge trimming; `rebuild` on a synthetic
  file of tone bursts with known gaps, asserting the output gaps by silence detection and the
  speech runs unchanged in length; `pause_profile` and `profile_issues` thresholds.
- `tests/test_ai.py`: `word_timestamps` sends `whisper-1`, `verbose_json`, word granularity, and
  maps the response.
- `tests/test_pipeline.py`: shaping runs after synthesis with the fake's word timestamps; the raw
  file is kept; `qa_json` carries `pauses`; failure of the timestamp call warns and keeps raw; the
  switch turns it off; regeneration shapes.
- `tests/test_direction.py` and `tests/test_ai.py`: the numeric pause sentence is gone; the prompt
  limits paragraph breaks and ellipses.
- Paid checks: the listening set on two real segments (raw vs shaped, with tables) delivered to the
  owner first; then `scripts/end_to_end_check.py` on the 130 s slice.

## Out of scope

- Stress or rhythm inside a phrase (the voice model's prosody).
- Changing the voice or persona.

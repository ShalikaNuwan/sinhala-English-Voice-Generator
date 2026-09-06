# Sinhala → English Voice Studio

A focused Phase 1 implementation of the supplied development specification. It converts long-form Sinhala narration into reviewable English narration through a segment-based pipeline:

`upload → validate → normalize → analyse delivery → find recordings → segment → Sinhala transcript → faithful translation → narration adaptation → TTS → QA → human review → assembly`

The MVP is deliberately simple. It uses FastAPI, SQLite, local files, FastAPI background tasks, FFmpeg, and the OpenAI API. PostgreSQL, Redis, S3, LangGraph, authentication, and custom-voice enrollment are appropriate production upgrades, but are not required to prove the core workflow.

## What is included

- WAV, MP3, M4A, MP4, WebM, OGG, and FLAC upload and validation
- mono 24 kHz normalization and silence-aware 20–60 second segmentation
- separate Sinhala transcription, faithful translation, and spoken-English adaptation stages; the
  adaptation writes a performance script (short sentences, held beats, spoken dates) and labels each
  passage with a story beat and a director's note
- configurable glossary, narrator profile, models, and standard TTS voice
- speaking-pattern matching: the source narrator's pace, pause habits, and emotional tone are
  measured once per project and used to direct the English narration
- embedded recordings (911 calls, interrogation tape, news clips) are detected by speaker
  diarization against a narrator reference clip, kept from the source with loudness alignment,
  and confirmed by the reviewer instead of being translated and re-voiced
- segment-level model-call audit records with model, prompt version, input hash, and timestamps
- text fidelity and duration-ratio QA
- side-by-side transcript/script editing, segment-only regeneration, previews, and job progress
- lossless WAV segments, assembled with real pauses between passages, exported as final WAV/MP3 plus
  Sinhala transcript and English script JSON
- original media preservation and resumable persisted stage outputs

## Requirements

- Python 3.11+
- FFmpeg and ffprobe on `PATH`
- an OpenAI API key with access to the configured models

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Set `OPENAI_API_KEY` in your shell (or load `.env` with your preferred secret manager), then run:

```bash
export OPENAI_API_KEY='your-key'
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The app does not automatically read `.env`, which avoids adding another dependency. Never commit the key.

## Configuration

All model names are environment variables so deployment is not tied to a changing alias:

| Variable | Default |
|---|---|
| `STT_MODEL` | `gpt-transcribe` |
| `TEXT_MODEL` | `gpt-5.6-terra` |
| `QA_MODEL` | `gpt-5.6-terra` |
| `TTS_MODEL` | `gpt-4o-mini-tts` |
| `TTS_VOICE` | `cedar` |
| `TTS_SPEED` | `1.0` |
| `DIARIZE_MODEL` | `gpt-4o-transcribe-diarize` |
| `DETECT_RECORDINGS` | `1` |
| `ALIGN_MODEL` | `whisper-1` |
| `SHAPE_PAUSES` | `1` |
| `AUDIO_MODEL` | `gpt-audio` |
| `DATA_DIR` | `./data` |
| `MAX_AUDIO_MINUTES` | `120` |
| `MAX_UPLOAD_MB` | `500` |

For the first real evaluation, use a 2–5 minute representative recording with a manually corrected Sinhala transcript and English reference. The specification recommends validating quality on that sample before processing full 30-minute recordings.

## API overview

- `POST /api/projects`
- `POST /api/projects/{id}/upload`
- `POST /api/projects/{id}/process`  (optional `voice`, `speed`, `audio_model`, and per-stage model overrides)
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/segments`
- `PATCH /api/segments/{id}/transcript`
- `PATCH /api/segments/{id}/script`
- `POST /api/segments/{id}/regenerate`
- `PATCH /api/segments/{id}/kind`  (`original` keeps the segment as recorded audio; `narration` voices it)
- `POST /api/segments/{id}/confirm`  (accept a kept recording)
- `POST /api/projects/{id}/assemble`
- `GET /api/projects/{id}/export`

Interactive API documentation is available at `/docs`.

## Tests

```bash
pytest
```

The tests cover segmentation, the voice-direction brief, gap and speed rules, silence insertion with real
FFmpeg, continuity through processing and regeneration, and a real upload/validation lifecycle using a
generated WAV file. They do not make paid API calls.

## Speaking-pattern matching

At the start of a job the normalized source is analysed once and the result is stored on the project:

- **measured** with ffmpeg: pause count, mean and longest pause, pauses per minute, speech ratio, loudness range
- **described** by `AUDIO_MODEL` from a 45-second excerpt: energy, emotional tone, use of pauses and emphasis
- **derived** labels: pace, pause style, dynamics

Those become the narrator character in every TTS instruction, while the per-segment adaptation style still
supplies local emotion and emphasis. Leading silence is excluded so dead air before the first word does not
skew the averages. If the audio model call fails the job continues on the measurements alone.

The profile is computed once and reused, so regenerating a segment keeps the same narration character.
Thresholds live at the top of `app/speaking_profile.py` and are meant to be tuned once you have listened
to real output.

## Narration direction

Every TTS call is directed with one brief, built in `app/direction.py` from four parts:

1. **Persona.** A fixed intimate true-crime storyteller: one real person telling one listener about a
   case that actually happened. A project can replace it by putting a `persona` string in its
   `narrator_profile` when the project is created. Non-string values are ignored.
2. **Voice character.** The measured and described speaking profile of the original narrator.
3. **How to speak.** Rules aimed at the usual synthetic tells: talk rather than read, end statements
   low, breathe before long sentences, get quieter rather than louder for intensity while staying
   fully intelligible, say names and dates carefully, report quoted speech rather than act it, sound
   like someone talking rather than presenting.
4. **This passage and continuity.** The beat, emotion, director's note, and emphasis from the
   adaptation stage, plus the beat and emotion the previous passage ended on so the voice does not
   reset at segment boundaries. Regenerating a segment looks up its predecessor for the same reason.

Pace is driven through the brief. `TTS_SPEED` (or `speed` on the process request) is passed to the
API and defaults to `1.0`. The brief no longer quotes pause lengths: pauses are shaped after synthesis
(see below).

At assembly, the gap between two segments is the adaptation's `pause_after_ms` plus the next
`pause_before_ms`; when both are zero it falls back to the source narrator's mean pause, and it is
clamped between 300 ms and the source narrator's longest measured pause. Without a speaking profile the
fallback is 600 ms and the ceiling 2500 ms.

Segments are generated as WAV so the final mix is only encoded once. WAV segment files are roughly
ten times larger than the MP3 segments earlier versions produced.

Projects processed before this change keep their old scripts, voice, and pause values. To hear the
new narration on an existing project, run processing again; regenerating only the TTS stage of a
segment keeps its old script and old pause values.

`scripts/listening_test.py` voices real segments from an existing job the old way and the new way and
asks the audio model to say which sounds more human and why. It spends API credit; use it when tuning
the persona or rules.

## Embedded recordings

Creator videos often play real evidence audio: a 911 call, an interrogation, a news clip. That audio
must not be translated or re-voiced. Once per project the pipeline diarizes the delivery sample,
takes the dominant speaker's longest span (2–10 s) as the narrator reference, and stores it in the
speaking profile. Every chunk is then diarized with that reference. A recording starts at speech by
someone other than the narrator in a Latin-only script and continues, across its own silences,
until the narrator speaks again in Sinhala. Recordings under 1.5 s are ignored; kept ones are padded
150 ms into silence.

Each recording becomes its own segment of kind `original`. It skips transcription, translation,
adaptation, and TTS; its audio is cut from the normalised source, brought to the narration's
loudness with a static gain (measured first, so clips under three seconds are handled correctly),
and given a 5 ms fade at each end; the narration that follows is told what was heard so it can
refer to it. Original segments arrive on the review page as needs review with "Recorded audio kept
as is. Confirm." Confirm them, or choose "Treat as narration" to voice one. A narration segment can
be kept with "Keep as recorded". At assembly, the gap next to a recording is the 300 ms minimum. A
recording that runs across a chunk boundary becomes two adjacent original segments joined with no gap.

If diarization is unavailable the job continues as narration only and says so in the progress line.
Independently, any narration segment whose transcript contains eight or more consecutive English
words is flagged for review as a possible recording. `DETECT_RECORDINGS=0` (or `detect_recordings`
on the process request) turns detection off. The narrator must be silent while a clip plays;
speech over a clip is not separated.

`scripts/end_to_end_check.py` runs the real pipeline on a slice of an existing project's source with
the configured models and prints every segment's kind and text. It spends API credit.

## Natural pauses

The voice model decides how the words are spoken; the pipeline decides the silences. After each
narration segment is voiced, the raw file is kept as `NNNN_rNN_raw.wav` and the pauses are detected
with FFmpeg. One `whisper-1` call with word timestamps ties every pause to its place in the script,
and the script's punctuation says what kind of break it is: a clause break after a comma, a dash, a
sentence end, a paragraph shift after a blank line, or a held beat after an ellipsis. Each kind gets
a length drawn from a band tuned for an intimate English storyteller (clause 150–280 ms, dash
250–400, sentence 420–650, paragraph 850–1200, beat 1200–1600), scaled shorter for a continuous
narrator and longer for a measured one, with a little variation so nothing is metronomic. A break the
script did not ask for is capped at 200 ms. Leading and trailing silence are trimmed to 60 ms so the
assembly gap is the only gap at a join. Speech itself is never stretched or re-levelled. The bands
live at the top of `app/pauses.py`.

QA records the pause profile of every shaped segment (count, median, 90th percentile, longest, per
minute) and flags one whose longest pause exceeds the natural ceiling or that still has unpunctuated
breaks over 200 ms. A pause the pipeline cannot tie to the script is left at its length, never
shortened. If word timestamps fail, clear pauses are matched in order to the script's breaks; if that
does not line up either, the raw voice is kept and the job says so. `SHAPE_PAUSES=0` (or
`shape_pauses` on the process request) turns shaping off.

`scripts/pause_listening_set.py <job_id> <segment_index...>` copies the raw and shaped files for
chosen segments to `data/pause_listening/` with a pause table for each, for listening.

## Production boundaries

Before a public or multi-user deployment, add authentication and project-level authorization, private object storage with signed URLs, a managed queue, retention/deletion controls, encrypted backups, request tracing, and rate limiting.

Custom Voice is intentionally not enrolled by this MVP. It should only be added after API eligibility is confirmed and explicit creator consent plus voice ownership are enforced. Standard OpenAI TTS remains the safe default. Clearly disclose to listeners that the English narration is AI-generated.

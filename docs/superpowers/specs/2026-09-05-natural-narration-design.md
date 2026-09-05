# Natural true-crime narration — design

Date: 2026-09-05
Status: approved by the project owner in conversation; implementation plan to follow.

## Goal

The English narration must sound like a real person telling one listener about a murder that actually happened, and it must keep that character for the whole video. Today the narration is produced by OpenAI `gpt-4o-mini-tts` with the `onyx` voice and a one-line direction per segment such as "This passage: tense in tone, pace moderate." Each segment is voiced in isolation and the segments are butted together with no silence between them. The result is recognisably synthetic: the tone resets at every segment boundary, sentences lift at the end, and there is no breath or held beat.

Non-goals: switching TTS providers, custom voice enrolment, removing the AI-generated disclosure that the specification requires. No text-to-speech model is guaranteed undetectable; the goal is the most human delivery the current OpenAI voices allow.

## Decisions taken with the owner

| Question | Decision |
|---|---|
| Narrator style | Intimate storyteller: close-mic, unhurried, talking to one listener. Matches the source, which addresses its audience directly. |
| Voice | `cedar` (OpenAI's recommended natural male voice). `onyx` remains selectable. |
| Approach | Rich direction, performance script, segment-to-segment continuity, real pauses in assembly. Stay on OpenAI. |
| Listening test | Approved. A few cents of API credit to voice two real segments before and after the change. |

## Architecture

```
faithful_en ──► adapt (performance script + beat + delivery + emotion + pauses)
                    │
                    ▼
        style_json on the segment
                    │
   speaking_profile ┼── previous segment's style
                    ▼
        direction.build_instructions() ──► gpt-4o-mini-tts (cedar, wav)
                    │
                    ▼
        segment WAV ──► QA ──► assemble with gaps ──► final_en.wav / .mp3
```

Four units change. Each can be understood and tested on its own.

### 1. Performance script — `AIClient.adapt()` and `NarrationAdaptation`

The adaptation prompt is rewritten to produce a script meant to be spoken, not read:

- short sentences, one idea each, contractions;
- an ellipsis (`…`) where a human would hold a beat, a dash (`—`) for a change of thought or an afterthought;
- a blank line before a shift in the story;
- dates, times, and numbers written the way they are said aloud ("May 1st, 2010", "4:51 in the morning");
- quoted speech from 911 calls and witnesses kept as plain spoken lines;
- no factual change. The existing QA stage still compares the script with the faithful translation.

The stage receives the tail of the previous segment's narration (last 600 characters) so it knows where it is in the story. Regeneration passes the previous segment's stored narration.

`NarrationAdaptation` gains two fields and keeps the rest:

| Field | Type | Meaning |
|---|---|---|
| `beat` | `Literal["setup","build","reveal","aftermath","reflection"]`, default `build` | Where this passage sits in the story arc. |
| `delivery` | `str`, default `""` | One sentence of direction for a voice actor, e.g. "Lower your voice as the call cuts out and let the last line hang." |

`pace`, `emphasis`, `pause_before_ms`, `pause_after_ms`, `emotion` are unchanged.

### 2. Voice direction — new module `app/direction.py`

`narration_instructions()` moves out of `app/ai.py` into `app/direction.py` and becomes `build_instructions(style, profile, previous_style=None, persona=None) -> str`. The brief has four parts, in this order:

1. **Persona.** A fixed paragraph describing the intimate true-crime storyteller. Default constant `DEFAULT_PERSONA`. A project can override it by setting `persona` in its existing `narrator_profile` dictionary. No new UI.
2. **Voice character.** The described and measured speaking profile of the original narrator, as today (description, pace, pause style, dynamics, mean and longest pause).
3. **How to speak.** Fixed rules that target the usual AI tells:
   - talk, do not read; slight unevenness in rhythm;
   - sentences end low and settled, never lifted or sing-song;
   - breathe before a long sentence; an ellipsis hangs for a beat; a dash is a change of thought;
   - stay quiet and close to the microphone; intensity comes from getting quieter and slower, not louder;
   - names, dates, and numbers are said carefully, as if the listener should remember them;
   - quoted 911 and witness speech is delivered as restrained reportage, not acted out;
   - never sound like an announcer, a voice assistant, or an advert; no brightness, no polish, no smile in the voice.
4. **This passage and continuity.** Beat, emotion, delivery note, emphasis, pace. If `previous_style` is given: the previous passage's beat and emotion, with an instruction to carry that mood into the first line and not reset to neutral.

Missing pieces are omitted, never rendered as `None`. With no profile and no previous style the brief is persona, rules, and this passage.

`speaking_speed(settings) -> float` returns the configured `TTS_SPEED` clamped to the API's 0.25–4.0 range. Pace is driven through the direction text; `TTS_SPEED` defaults to `1.0` and the listening test decides whether a lower value helps or sounds stretched. The value is recorded in the job configuration like the other settings.

### 3. Synthesis — `AIClient.synthesize()`

Signature becomes `synthesize(text, model, voice, style, output_path, profile=None, previous_style=None, persona=None, speed=1.0)`. It calls `build_instructions`, passes `speed`, and requests `response_format="wav"` so the final mix is not encoded twice. Generated segment files are named `NNNN_rNN.wav`. The final MP3 export is unchanged.

### 4. Pipeline and assembly — `Pipeline`, `AudioService.assemble()`

- `process_job` keeps the previous segment's style dictionary across the loop and passes it to `synthesize`. The first segment has none.
- `regenerate_segment` looks up the segment with `segment_index - 1` in the same job and passes its style, so a re-voiced segment still matches its neighbours.
- `assemble_project` computes one gap per boundary and passes `gaps_ms` to `AudioService.assemble()`, which inserts mono 24 kHz silence of that length between the two segments before the existing concat and loudness normalisation.

Gap rule, in a pure function `direction.segment_gap_ms(previous_style, next_style, profile) -> int`:

```
gap = previous.pause_after_ms + next.pause_before_ms
if gap == 0: gap = profile.measured.mean_pause_ms (or 600 if no profile)
clamp gap to [300, max(profile.measured.longest_pause_ms, 300)] (upper bound 2500 if no profile)
```

- Default voice becomes `cedar` in `Settings`, `.env.example`, README, and the web form's voice field.
- `PROMPT_VERSION` becomes `2026-09-mvp2` so audit records distinguish runs made with the new prompts.

## Error handling

- No speaking profile: brief omits the voice-character section; gap falls back to 600 ms.
- No previous style (first segment, or regeneration of segment 1): brief omits the continuity section.
- `TTS_SPEED` outside 0.25–4.0: clamped, not rejected.
- Silence insertion failure in ffmpeg: raises `AudioError` exactly as a failed concat does today; the assemble endpoint returns 409 with the message.
- Adaptation model omits `beat` or `delivery`: defaults apply; the brief still builds.

## Testing

All unit tests run against the fake client. No paid calls.

- `tests/test_direction.py`: brief contains persona, profile, rules, beat, delivery, emphasis; continuity appears only when a previous style is given; no `None` in output; persona override is used; gap rule covers the zero, clamp-low, clamp-high, and no-profile cases; speed clamping.
- `tests/test_ai.py`: `synthesize` sends `speed`, `response_format="wav"`, and instructions containing the persona; `adapt` sends the previous narration tail.
- `tests/test_pipeline.py`: second segment's synthesis receives the first segment's style; regeneration of segment 2 receives segment 1's style; generated paths end in `.wav`; assembled duration equals the sum of segment durations plus gaps within 100 ms.
- `tests/test_audio.py`: `assemble` with gaps produces the expected duration; with no gaps behaves as before.

Listening test (paid, approved): a scratch script voices segment 1 (calm intro) and segment 2 (911 call) of the Gilgo Beach project twice, once with the old one-line direction on `onyx` and once with the new brief on `cedar`. The four files are written to `data/listening_test/` with descriptive names. The brief's wording is tuned on what is heard, and the final wording is what ships.

## Out of scope, noted for later

- ElevenLabs or another provider behind the specification's `SpeechProvider` abstraction.
- A low-level room-tone bed under the whole mix to mask segment joins.
- Surfacing `beat` and `delivery` in the segment card of the web UI.

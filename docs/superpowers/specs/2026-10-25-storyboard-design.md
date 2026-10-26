# Storyboard-driven documentary video: design plan

Date: 2026-10-25. Status: proposal for the owner's review; nothing built yet.

## Why

The current pipeline (`scripts/video/`) already groups sentences into beats, but it still plans the film in
narrow windows. The model sees 30 sentences at a time with 6 sentences either side. It never sees the whole
story, so it cannot know:

- where the chapters are;
- which moment is the turning point;
- who the recurring people and places are;
- how tension should rise and fall.

Colour and effects work at the wrong level as well. The whole film gets one cold grade and one grain. Effects
come from rules (a dip to black on every whoosh), not from what is happening in the story. Finally, nothing can
be reviewed until a 90-minute render is finished.

The goal is a film that is planned like an editor plans one: understand the whole story first, write a
storyboard, review it, and then build the video, colour and effects from that storyboard.

## The flow

The flow runs in seven passes:

1. **Transcript.** whisper-1 timing is matched to the known script, giving sentences with exact times.
2. **Story bible.** One read of the whole script produces the people, places, timeline, chapters and
   tension curve.
3. **Storyboard.** Chapter by chapter, the script is broken into scenes and then beats, using both the bible
   and the full text of the chapter.
4. **Look and effects.** Each scene gets a grade, pace and effects budget from its mood, era and intensity.
5. **Review gate.** A storyboard page and a low-resolution animatic are produced. The owner approves or edits.
6. **Sourcing.** Pictures and clips are found for each beat's intent, and a vision check confirms each one
   fits.
7. **Render.** The final render, followed by automatic critics. The rendered frames are compared against the
   storyboard, and anything that fails is fixed and re-rendered.

### Pass 0: Transcript (exists, extend)

`align_script.py` and `sentences.py` already produce every sentence with its start and end time. Three
measurements are added from the audio:

- **Pace:** words per minute for each sentence.
- **Pauses:** long silences, which are natural section breaks.
- **Energy:** narration loudness, which shows where the read leans in.

The baked-in effect times from `events_mapped.json` are kept.

### Pass 1: Story bible (new, `story_bible.py`)

One long-context call reads the entire script. A 92-minute script is about 15,000 words, roughly 20,000
tokens, so it fits. The call returns a structured bible:

- **People:** name, role, the era they appear in, and how often they appear. Each person is linked to the
  library photographs that really show them.
- **Places:** each with its era (for example, the Summerdale house in 1978, or Waterloo, Iowa in 1968).
- **Timeline:** real events with dates, so every beat knows what year it is in.
- **Chapters:** sentence ranges, the purpose of each chapter (cold open, background, first crime,
  investigation, arrest, trial, aftermath), its tone, and an intensity score from 0 to 10.
- **Motifs:** images that recur, so they can be reused on purpose.
- **Sensitivity rules:** victims are shown with dignity, no remains are ever shown, and nothing is
  sensationalised.

A second model reviews the bible and challenges anything it disagrees with: gpt-6-astra, acting as creative
director.

### Pass 2: Storyboard (new, `storyboard.py`)

This pass runs chapter by chapter. Each call is given:

- the bible;
- the full text of the chapter;
- the last scene of the previous chapter.

#### Scenes

Each call returns scenes of 30 seconds to 3 minutes. A scene has:

- a setting;
- the story time;
- a purpose (what the viewer must understand);
- a mood;
- a visual idea, for example "the suburb in winter, 1978: snowy streets, then the house".

#### Beats

Inside each scene, the call returns beats of 3 to 20 seconds. Beats are decided with the whole scene in view,
not sentence by sentence. A beat has:

- **The basics:** its sentence range and times, and its visual intent, meaning what the viewer should see and
  feel.
- **Subject:** a reference to a person, place or object in the bible.
- **Shot type:** establishing, portrait, document, evidence close-up, map, board, period scene, or film clip.
- **Continuity:** for example, "same portrait of X as scene 3", so a person keeps the same face.
- **Motion and transitions:** a camera move, a transition, and the reason for that transition.
- **Extras:** an optional overlay and the effect cues.

Code checks the result. Every sentence must be covered with no gaps, beats must keep to their length
limits, and every subject must exist in the bible.

### Pass 2b: Quotes in people's own words (new, `quotes.py`)

Narration scripts quote people directly. When the narration speaks a quote, the video shows the person's exact
words on screen, inside quotation marks. A sample of the Gacy script has 16 sentences that contain quotation
marks, and they are not all speech. Each one is sorted into one of four kinds:

| Kind | Gacy example | On screen |
|---|---|---|
| `spoken` | Gacy: “I didn’t do any of this, and I don’t know anything about it.” | quote card |
| `spoken`, attributed elsewhere | “That boy is dead.” (the speaker is named in a nearby sentence) | quote card, speaker resolved from context |
| `written` | the ring engraved “JAS”; the photograph signed “To John Gacy, best wishes.” | exhibit treatment: the text as an engraving, handwriting or typed document, beside the object's photo if the library has one |
| `term` | his father calling him “sissy” | no card; the word stays in the narration and subtitles only |

#### Finding quotes (code first, model second)

1. **Code finds the quoted spans.** It reads the script's own quotation marks, including a quote that runs
   across more than one sentence (Rob: “Wait here for a little while. …”).
2. **The model classifies each span.** It decides spoken, written or term, and names the speaker or writer,
   their role, and the source when the narration gives one (a police statement, testimony, an interview, a
   letter). It does this with the story bible and the nearby sentences in view, so a quote with no "he said"
   in its own sentence still gets the right speaker.
3. **Code verifies every quote word for word against the script.** The card can only ever show words the
   narration actually says. This keeps the owner's rule of no invented lines.

#### Timing and sync

The card appears exactly when the quote starts, not when its sentence starts. Its start and end come from the
quote's own words in `words.json`.

- **Short quotes** (one or two lines) appear whole.
- **Longer quotes** reveal line by line as they are spoken.
- **Quotes over about 25 words** become pages of two lines each.

The card holds for at least 1.5 s after the last word, and it leaves before the narration moves on to a new
idea.

#### Design

- **Card layout.** Large crimson quotation marks and the words set in a clean serif. An attribution line goes
  underneath, for example "— John Wayne Gacy, to detectives, December 1978", but only with details the
  narration gives.
- **Background.** If the library has a real photograph of the speaker, it sits dimmed and blurred on one side
  of the frame, and the text sits on the other. If not, the card uses a paper or statement texture in the
  scene's look.
- **Written words.** These use the exhibit style instead: an engraving on the ring, a handwritten inscription,
  or a typed line on a document.

#### Rules

- **Only direct quotes.** Paraphrase never gets quotation marks.
- **Spacing.** At most one card in any 20 s, unless two quotes are a direct exchange (a question and its
  answer).
- **Sensitive quotes.** A quote that is sexual, threatening or graphic is flagged on the storyboard, for
  example the threat Gacy makes at 47:39 in the v2 cut. The owner decides there whether it appears as a card
  or only in the narration. The default is narration only.
- **Victims and families.** Their words use the memorial look, with no effects.
- **Subtitles.** The sidecar subtitles carry on as normal. The card is the designed version of the same words.

### Pass 3: Look and effects (new, `looks.py`)

A small vocabulary of looks is chosen per scene, not per shot, and blended across scene boundaries:

| Look | When | Grade | Pace and effects |
|---|---|---|---|
| Warm memory | childhood, normal life, the 1950s to 1970s | faded warm film, soft highlights, light halation | slow dissolves, gentle grain |
| Cold dread | tension building, a suspicion | cool shadows, desaturated, crushed blacks | slow push-ins, a heavier vignette |
| Investigation | evidence, police work, records | neutral-cool and clear, with crimson accents | exhibit and board devices, faster cuts on the beat |
| Revelation | the turning point | high contrast, near monochrome with crimson | hard cuts, one flash, a brief shake, a dip to black on the hit |
| Courtroom and aftermath | trial, sentencing, legacy | muted and steady | long holds, slow dissolves |
| Memorial | victims' portraits | soft black and white, no colour cast | long holds, no effects at all |

The grades are built in ffmpeg with `curves`, `colorbalance` and `eq`, with a `.cube` LUT optional. A scene's
intensity score sets the strength of grain, vignette, flicker and shake.

Effects have budgets. For example, there is at most one shake in each high-intensity scene, and there are no
effects on victims.

Effects attach to two things:

- **Story moments:** reveals and chapter ends.
- **Audio:** the baked-in whooshes and hits.

### Pass 4: Review gate (new)

#### The storyboard page

A local HTML storyboard has one row per beat:

- the time;
- the sentences it covers;
- the intent;
- any quote, with its speaker, its kind, and a flag if it is sensitive (the owner chooses card or narration only);
- a thumbnail of the chosen picture;
- a swatch of the look;
- the effect cues.

Chapters and scenes are shown as headers. The page also includes a tension-curve chart.

#### The animatic

A fast 480p animatic plays the narration with the chosen pictures and no heavy effects. It takes minutes
rather than hours.

#### Editing and approval

The owner can do any of the following, in the page or in `storyboard.json`:

- reorder beats;
- swap pictures;
- change a scene's look;
- lock a beat so later passes do not change it.

Only after approval does the full render run.

### Pass 5: Sourcing driven by the storyboard (extend `harvest.py`)

The storyboard produces a needs list, for example "Des Plaines street, winter 1978, colour", rather than
searching per sentence. Search order:

1. the library;
2. targeted searches of the free archives;
3. a period image generated as the last resort, never of a real person.

A vision check confirms that each picture matches its beat's intent and era. Anything showing a modern car in
a 1978 scene, for example, is rejected. A gap report lists every beat that still lacks a fitting picture.

### Pass 6: Render and critics (extend `assemble.py`)

The render works from the storyboard:

- each scene's grade, with crossfaded grades at scene boundaries;
- each beat's camera, transitions and effect cues.

After the render, automatic critics check three things:

- **Story fit:** gpt-6-astra compares sampled frames from each scene against the storyboard's intent.
- **Continuity:** the same person keeps the same face, the era stays consistent, and no picture repeats too
  soon.
- **Rhythm:** the distribution of shot lengths matches each scene's pace.

Failures are fixed and only those sections are re-rendered.

## Data model (`storyboard.json`)

```text
film      title, duration, audio, bible (people, places, timeline, motifs, rules)
chapters  id, title, first/last sentence, start/end, purpose, tone, intensity, era, look
scenes    id, chapter, first/last sentence, start/end, setting, story_time, purpose, mood, visual_idea,
          look {preset, strength}, pace {target_shot_s}, effects_budget, locked
beats     id, scene, first/last sentence, start/end, intent, subjects[], shot_type, choices[], chosen,
          camera, transition_in {kind, reason}, overlay, effects[{kind, at, reason}], quote_ids[], locked
quotes    id, kind (spoken | written | term), text (verbatim from the script), speaker, role, source,
          first_word/last_word (indices into words.json), start/end, pages[], sensitive, show (card | narration),
          style (card | exhibit | memorial)
```

## Models

| Job | Model |
|---|---|
| Word timing | whisper-1 |
| Story bible, storyboard, sourcing checks | gpt-6-sol |
| Creative-director review of the bible, the storyboard and the rendered scenes | gpt-6-astra |
| Picture descriptions and vision checks | gpt-6-sol (vision) |
| Music and effect cue audition, for episodes without baked-in sound | gpt-audio-1.5 |
| Period images, last resort | gpt-image-2.5-flare |

Code still enforces the hard rules that models must not decide:

- coverage, with no gaps in the timeline;
- free licences only;
- no remains shown;
- reuse limits;
- sync to the exact frame.

## Phases

1. **Story bible and storyboard.** Build `story_bible.py`, `storyboard.py`, `storyboard.json` and the HTML
   page. Add `quotes.py` (find, classify and verify quotes) and show quotes on the page. Test on the Gacy v2
   audio and compare against the current film.
2. **Looks, effects and animatic.** Build the looks and effects vocabulary, per-scene grades in
   `assemble.py`, the 480p animatic mode, and the quote cards: the quote card, the written-text exhibit, and
   line-by-line reveal timed to the words.
3. **Sourcing.** Build the needs list, the intent and era vision checks, and the gap report.
4. **Critics and final render.** Add the critics, re-render only the failed sections, and do the final
   render.
5. **Wrap-up.** Write tests for each pass and update the README.

## Decisions for the owner

1. The review gate: an HTML storyboard plus a 480p animatic (recommended), or the storyboard page only.
2. How far looks may vary between scenes: subtle (recommended for documentary) or bold.
3. Sound for future episodes: keep the baked-in audio from the narration and edit, or let the storyboard
   place music cues and stingers per chapter.
4. The first test: rebuild the Gacy v2 film through the new flow and compare it with the current cut.
5. Sensitive quotes: narration only by default, with the owner able to switch any one to a card on the
   storyboard (recommended); or always a card.

## Addendum (2026-10-25): scenario-based handling

### Why

The first samples showed that a passage of one kind is badly served by generic stand-ins. The clearest case
is the victims' register (66:59–68:38 in the Gacy v2 cut). It is a roll call: about one name, age and date
every 4 seconds. Under it ran a shopping street, a bus and a river.

### The approach

The storyboard labels each scene with a scenario type from a fixed catalogue. Each type has a handler that
builds that scene's beats with its own recipe. The catalogue describes true-crime narration in general, not
one case, so the same engine serves future episodes.

| Scenario | Signal | Recipe |
|---|---|---|
| roll_call | names with ages and dates in a list | one card per person, cut when the name is spoken; the real portrait if the library has one, else a typographic memorial card; unidentified victims get a plain card; memorial look, no effects |
| timeline | dates in sequence | animated timeline, each date lit as it is said |
| statistic | weighty numbers | a number card, with only the narration's words |
| quote | someone's own words | quote card or exhibit slip (built) |
| conversation | an exchange between named people | alternating quote cards, or a typed statement-transcript look |
| introduction | a person named for the first time | profile card (exists) |
| evidence | an object, document or clue | exhibit or highlight (exists) |
| geography | where places are | animated map (exists) |
| legal | charges, verdict, sentence | courtroom look and a document showing the charge or verdict |
| atmosphere | general narration | period stand-ins (the default) |

### Rules

- **Extraction is checked by code.** The model extracts each entry: name, age, date. Code verifies every
  field verbatim against the narration and times it to the spoken word (words.json).
- **Names are spelled correctly on screen.** The script's transcription spellings are corrected to the story
  bible's spelling of the person when a name matches one.
- **Text cards are scenario-bound.** They are allowed only in roll_call, timeline and statistic, and carry
  only stated facts. The owner must confirm this exception to the no-full-screen-text rule.
- **Handlers are ordinary.** A handler emits ordinary storyboard beats, so rendering, looks, checks and
  critics are unchanged.

### Build order

1. The scenario label and the handler framework, then roll_call, then re-render sample 5 for comparison.
2. Timeline and statistic.
3. Conversation and legal.

## Addendum 2 (2026-10-25): the relevance ladder

### What went wrong

In samples 1 (Rob's last shift) and 2 (Gacy's earlier life), most pictures did not make sense. There were
three causes:

1. **The library lacks the story's places.** The free period pool is almost all 1970s Chicago South Side
   photographs from DOCUMERICA. Des Plaines, Waterloo and Las Vegas have nothing, so "Nisson Pharmacy in
   Des Plaines" got Lake Meadows Shopping Center three times in two minutes.
2. **Every beat must cut to a new picture.** Beats change about every 8 s, and only 2–3 relevant photographs
   exist per passage.
3. **The check let loose relations through.** "partly" passed.

### The ladder

Every beat declares the link between its picture and its words. The rungs are tried from the top, and a
picture with no allowed link is rejected:

1. **subject:** the named person, place or object itself.
2. **place-era:** the same town or area and the same years (verified by caption and vision; another city
   does not qualify).
3. **action:** the action or object the sentence is about (a cash register, a parked car in snow at night).
4. **reframe:** another framing of a relevant photograph already in the passage: a face or detail located
   by vision, cut as its own shot.
5. **hold:** the last relevant picture continues; same-subject beats merge up to about 25 s.
6. **device:** a scenario device (map, timeline, name card, quote card).
7. **generated detail (optional, owner's call):** an illustrative detail with no faces and no identifiable
   real place, labelled on screen as an illustration.

### Engine changes

- **Declared and verified links.** A `link` field (type + reason) on each beat; check_pictures verifies the
  link type, not just a loose fit.
- **Reframing.** A vision locate pass creates crops (face, detail) of relevant photographs as extra shots.
- **Holds.** Merging same-subject beats replaces filler cuts.
- **Targeted sourcing.** Sourcing comes from the bible's places and objects, with eras. New free archives
  are proposed: Library of Congress, US National Archives, Flickr Commons (no known restrictions) and
  Internet Archive period footage, keeping public-domain and no-known-restrictions material only.

### Build order (with the scenario engine)

1. The scenario framework and roll_call, with a new sample 5.
2. The ladder: links, reframing and holds, with new samples 1 and 2.
3. New archives and targeted sourcing.


## Addendum 3: key-fact labels (owner-approved 2026-10-25)

A physical clue the narration stresses (the blue jacket, the receipt, the class ring) gets a short label on
screen: the narration's own words in letter-spaced small capitals with a short crimson rule, over a blurred
and darkened copy of the shot on screen, for about 3 s (up to 6 s rather than leave a sliver of shot).
`key_facts.py`: a model lists the clues; code keeps only labels of one to three words that are spoken in
the sentence cited, are not a person or place in the story bible and contain nothing gruesome, and places
them on sentences that mention the object, at least 60 s apart. Labels never interrupt a quote card or the
register. Each label keeps a copy of the shot it was cut from, so re-running restores that shot first.

## Addendum 4: one engine for any narration (2026-10-25)

The pipeline had the Gacy case written into it: the harvest's Commons category, Wikipedia articles,
starter names and skip list, Chicago-area period categories, the captioning and footage prompts, a
"chicago 1970s street" fallback, the evidence board's centre and the sound mix's reveal phrases. All of
it now comes from `<dir>/case.json` (`case.py`): a model proposes the profile from the script; code keeps
only Wikipedia articles and Commons categories that exist and finds the period categories on Commons
("<place> in the <decade>", "<year> in <place>", DOCUMERICA photographs of each state). Gacy's own
profile reproduces the old constants exactly (and keeps its reveal phrases), so its films are unchanged.
The sound mix's reveal drones otherwise come from the storyboard (revelation scenes, intensity 9+).

`documentary.py prepare <dir> --audio <wav> --script <txt>` runs the whole chain to a review cut;
`documentary.py render <dir>` makes the film. Tested end to end on a second case written for the test.

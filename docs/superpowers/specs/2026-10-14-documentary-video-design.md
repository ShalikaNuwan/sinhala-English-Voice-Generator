# Documentary video from a finished narration

The owner has a finished English voice track (ElevenLabs, 98.6 min for the Gacy episode) and the
script that produced it. This builds the picture: a true-crime documentary cut, timed to the
narration, assembled with ffmpeg from images that are licensed for commercial use.

## Why licence filtering is not optional

The channel is monetised and the subjects are real people in real criminal cases. Press photographs
of the victims, the scenes and the trials are owned by news agencies and police departments.
Downloading them from a search engine invites copyright claims and demonetisation. So every image
has to come from a source that states its licence, and the licence has to permit commercial use:
public domain, CC0, or CC-BY with attribution. CC-BY-SA is avoided (share-alike would reach the
whole video); CC-BY-NC and CC-BY-ND are rejected outright. Every file used is recorded in
`credits.md` with its title, author, licence and source URL.

Real people are depicted only from genuine archival images. No generated likeness of a victim or a
killer: a fabricated photograph of a murdered teenager presented as part of a documentary is both
a serious harm to the family and the kind of thing that removes a channel.

## Stages

1. **Align** (`scripts/video/align_script.py`, built). Chunks the audio, reads word times with
   whisper, matches them back onto the known script with a sequence matcher, and writes
   `lines.json` (one entry per script line with `start`, `end`, `text`) plus `script.srt`.
2. **Plan**. Groups the aligned lines into shots of roughly 8-14 s and decides what is on screen
   for each. Writes `shots.json`. This is the checkpoint the owner reads before anything is fetched.
3. **Source** (`scripts/video/sources.py`). Fills each shot's `asset`, `licence` and `attribution`
   from Wikimedia Commons, Openverse, or image generation. Writes `credits.md`.
4. **Draw** (`scripts/video/graphics.py`). Renders the shots that are not photographs: title cards,
   date cards, timelines, maps, relationship charts.
5. **Assemble** (`scripts/video/assemble.py`). Ken Burns motion on stills, hard cuts between shots,
   the narration laid under, subtitles burned in or shipped alongside.

## The contract: shots.json

```json
{
  "audio": "/abs/path/ElevenLabs_Cacy.wav",
  "duration_s": 5916.53,
  "width": 1920, "height": 1080, "fps": 30,
  "subtitles": "burn" | "sidecar" | "none",
  "shots": [
    {
      "index": 1,
      "start": 0.0,
      "end": 9.2,
      "kind": "archival" | "stock" | "generated" | "card" | "chart" | "map",
      "query": "John Wayne Gacy 1978 mugshot",
      "prompt": "A snowbound suburban pharmacy at night, 1978, documentary still",
      "text": "December 11th, 1978",
      "subtext": "Des Plaines, Illinois",
      "motion": "zoom_in" | "zoom_out" | "pan_left" | "pan_right" | "still",
      "asset": "assets/0001.jpg",
      "licence": "Public domain",
      "attribution": "Des Plaines Police Department",
      "source_url": "https://commons.wikimedia.org/wiki/File:..."
    }
  ]
}
```

Rules every stage relies on: shots are in playing order, `start`/`end` never overlap and never
leave a gap, the last `end` equals `duration_s`. `kind` decides which stage fills `asset`:
`archival` and `stock` are searched, `generated` is generated, and `card`, `chart` and `map` are
drawn. A shot whose sourcing fails keeps `asset: null`, and assembly must survive that by holding
the previous picture rather than failing the render.

## Environment

ffmpeg 4.2.2: `zoompan`, `concat`, `drawtext`, `scale` and libx264 are present; `xfade` and
`subtitles` are **not**, so transitions are hard cuts and burned-in subtitles go through `drawtext`.
The build is not allowed to upgrade ffmpeg: the voice pipeline's loudness behaviour was tuned
against this exact version. Pillow 12.3 is installed. Wikimedia Commons and Openverse need no API
key; Pexels and Pixabay would need free keys the owner has not supplied, so they are optional extras.
Image generation uses `gpt-image-2.5` on the owner's OpenAI key.

## Revision 2, 2026-10-14: real pictures, factual overlays, editor-grade finish

The owner watched the first preview and asked for four changes. First, no full-screen text and no
invented lines: the planner had written titles like "THE MAN NEXT DOOR" and "The man no one
suspected", held on black for twelve seconds, about 55 of the first 137 seconds. Second, a palette
suited to a crime documentary. Third, rely on actual pictures from the internet rather than
generated ones. Fourth, the cut should look like a professional editor made it.

The owner has twice chosen real photographs over licence-clean substitutes after the claim risk was
explained. So copyrighted news photographs are in scope. Each image carries a `rights` value of
`free` (public domain, CC0, CC BY, CC BY-SA), `fair_use` (copyrighted, used under a documentary
fair-use claim), `generated` or `original`. `credits.md` groups them under those headings so the
owner can see at a glance which pictures could draw a claim. One exclusion is absolute: no image
showing human remains or bodies, whatever its source.

### Stages, revised

1. Align (unchanged).
2. **Harvest** (`scripts/video/harvest.py`). Builds `library/` and `library.json` for the episode
   before any planning. Sources, in order of preference: Commons case categories (Category:John
   Wayne Gacy and subcategories, all free licences now accepted with attribution kept); Commons and
   Openverse searches for every person, place and institution the script names; period photographs
   of the setting (DOCUMERICA and other 1970s Chicago categories on Commons); images on the relevant
   English Wikipedia articles, including non-free files (`fair_use`); and news photographs found by
   web search for the key subjects (`fair_use`). Duplicates are removed by URL and perceptual hash.
3. **Caption** (`scripts/video/caption.py`). A vision model describes every library image and fills
   `caption`, `subjects`, `places`, `era`, `colour`, `kind`, `graphic` and `quality`. Graphic
   images are kept on disk but never planned.
4. **Plan v2**. The planner sees the narration and the catalogue, and picks a library image for each
   shot by content. A search query is the fallback, and generation is the last resort (target
   under one shot in ten). Text appears only as factual overlays.
5. **Edit v2** (`scripts/video/assemble.py`, `scripts/video/graphics.py`), rendered with ffmpeg 9.0.2
   from the isolated conda env `/opt/miniconda3/envs/video/bin/` (override with `VIDEO_FFMPEG`,
   `VIDEO_FFPROBE`). The voice pipeline's ffmpeg 4.2.2 is untouched.

### library.json

```json
{"items": [{
  "id": "c0001", "file": "library/c0001.jpg",
  "source": "commons" | "wikipedia" | "openverse" | "web" | "generated",
  "title": "John Wayne Gacy 1978 mugshot", "source_url": "https://...",
  "licence": "Public domain" | "CC BY-SA 4.0" | "Copyrighted",
  "attribution": "Des Plaines Police Department",
  "rights": "free" | "fair_use" | "generated",
  "width": 1505, "height": 2000,
  "caption": "Black-and-white police booking photograph of John Wayne Gacy, front view, 1978.",
  "subjects": ["John Wayne Gacy"], "places": ["Des Plaines, Illinois"], "era": "1978",
  "colour": "bw" | "colour", "kind": "photo" | "document" | "newspaper" | "diagram" | "map",
  "graphic": false, "quality": 4
}]}
```

### shots.json v2 (per shot; fields not listed keep their v1 meaning)

```json
{
  "index": 1, "start": 0.0, "end": 9.2, "narration": "...",
  "kind": "archival" | "period" | "stock" | "generated" | "chart" | "map",
  "library_id": "c0001" | null,
  "asset": "library/c0001.jpg",
  "treatment": "fill" | "print" | "pan" | null,
  "motion": "zoom_in" | "zoom_out" | "pan_left" | "pan_right" | "pan_up" | "pan_down" | "still",
  "transition_in": "dissolve" | "black" | "cut",
  "overlay": {"style": "stamp" | "name", "text": "December 11, 1978", "subtext": "Des Plaines, Illinois"} | null,
  "colour": "bw" | "colour" | null,
  "licence": "...", "attribution": "...", "source_url": "...", "rights": "free" | "fair_use" | "generated" | "original"
}
```

`treatment` null means assembly decides. A picture close to 16:9 is `fill` with Ken Burns motion. A
portrait, square or very wide picture is `print`: centred over a blurred, darkened copy of itself
that fills the frame, with a thin border and a soft shadow. A `document` or `newspaper` is `pan`.
`transition_in` is `dissolve` by default (0.6 s), `black` at scene changes (a dip through black of
about 1 s), and `cut` where the planner wants impact. A dissolve begins at the planned cut point, so
picture and narration stay in sync: each clip is rendered with the transition length added to its
tail.

### Look

Palette: ground #0A0A0A, text #F2F2F2, accent crimson #9B111E, secondary grey #8C8C8C. Date and place
stamps use a typewriter face (American Typewriter, falling back to Courier New). Names use a bold
sans (Helvetica Neue or Arial Bold). Every overlay sits lower-left with a short crimson rule. It
fades in over 0.4 s, holds about 3 s, and fades out over 0.3 s. The grade is cold, desaturated,
with crushed blacks; black-and-white photographs stay black-and-white with a faint cool tone; light
film grain sits over everything. Charts and maps are rare and short (5 s at most) and use the same
palette. There is no full-screen text, and no line that is not a fact from the narration.

## Revision 3, 2026-10-14: moving footage, motion graphics, sound

The owner asked for short video clips among the photographs and a cut that holds a viewer. Three
additions, each its own module, joined by the shot list.

**Footage** (`scripts/video/footage.py`, catalogue `clips.json`, files under `clips/`). Real
footage first: public-domain and CC films from the Internet Archive (Prelinger Archives and other
collections), Wikimedia Commons video, and the US National Archives. Wanted: 1970s Chicago and
suburban streets, winter and snow, police cars and stations, courts, prisons, night roads, offices
and shops of the period. YouTube is not a source: downloading breaks its terms, and broadcast
footage is matched by Content ID. Where nothing real exists, `sora-2` generates short atmospheric
clips. They carry no person's likeness, at most twenty per film, and are marked `rights: "generated"`.
Each clip entry:

```json
{"id": "v0001", "file": "clips/v0001.mp4", "source": "archive.org" | "commons" | "nara" | "generated",
 "title": "...", "source_url": "...", "licence": "Public domain", "attribution": "...", "rights": "free" | "generated",
 "duration_s": 312.4, "width": 640, "height": 480, "fps": 29.97,
 "segments": [{"in": 41.0, "out": 49.5, "caption": "Snow falls on a city street at night, cars passing, 1970s.",
               "era": "1970s", "quality": 3, "graphic": false}]}
```

A clip is described per usable segment, because a public-domain film runs for minutes and only a few
seconds of it suit this story.

**Shots, v3.** A shot may run a clip segment instead of a still: `media: "clip"`, `clip_id`,
`clip_in` (seconds into the source). Assembly fills the frame, grades it like the stills, keeps the
overlays and transitions, and speeds it by at most 10% when the shot is slightly longer than the
segment. A shot never holds a frozen clip. `media: "motion"` is a rendered motion-graphics clip
(below). Everything else is `media: "photo"`, the default. Clips should run about once every 45 to
90 seconds, more often where the story moves fast.

**Motion graphics** (`scripts/video/motion.py`): an evidence board, with real library photographs
pinned to a dark board, joined by red string and labelled, filmed by a slow camera move; an animated
map of the Chicago area, where pins drop in order as each place is named; and date stamps that type
themselves out. All of them use the crime palette.

**Sound** (`scripts/video/sound.py`). A low music bed, ducked automatically under the narration, and
sparse effects: a camera shutter when a new case photograph appears, typewriter keys under a date
stamp, a low whoosh into a dip to black. The narration is always the loudest thing. The final master
is -14 LUFS integrated with true peak at or below -1 dBTP, as YouTube expects. Music and effects come
from sources licensed for commercial use, with attribution recorded in `credits.md`, or are
generated by ElevenLabs when the owner's key allows it.

## Revision 4, 2026-10-14: told as an investigation

The owner wants the film to feel like someone explaining an investigation: showing the evidence as
the narrator talks about it, not just running pictures behind a voice. Five devices, each a rendered
full-frame clip (`media: "motion"`), so assembly treats them like any other motion shot.

- **Profile** (`investigation.profile_card`). When the narration introduces a person whose photograph
  is in the library: the photograph as a print on the left, and on the right a short case-file
  record typed on line by line (NAME, AGE, ROLE, and a date where relevant). Every field must be
  stated in the narration of that shot or the one before. One profile per person, ever.
- **Exhibit** (`investigation.exhibit`). An evidence object (the house diagram, the business card,
  a receipt, a newspaper page, a painting) shown as a tagged exhibit on a dark desk, with a slow
  push. The tag text is factual: what it is and where or when it comes from.
- **Highlight** (`investigation.highlight`). When the narration points at a detail inside a picture
  (an address, a name in a newspaper, a room on the diagram), the camera pushes toward it and a red
  circle or box draws on around it. The detail's box comes from a vision model
  (`investigation.locate`), and a second vision check confirms the crop shows that detail before it
  is used. No confirmation, no highlight.
- **Board** (`motion.evidence_board`). It returns at turning points and grows: each return pins the
  people, places and objects established so far, joined by red string.
- **Map** (`motion.map_reveal`). Pins drop in as places enter the story.

Text rules carry over. Nothing appears on screen that the narration does not state, apart from
structural labels ("CASE FILE", "EXHIBIT", field names) that carry no claim.

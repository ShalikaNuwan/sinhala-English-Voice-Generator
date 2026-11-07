# Documentary animations

Everything specific to one film lives in `data/video/<film>/`, which Git ignores. The engine and tools here work for any film.

## An animation

An animation is a JSON spec in `data/video/<film>/explainers/`. `scripts/video/explainer.py` draws it. See `examples/`:

| Example | Shows |
|---|---|
| `map_with_photos.json` | A real map with photos, tags and pins |
| `map_route_and_card.json` | Routes along real streets, a card that fills in line by line |
| `floor_plan_incident.json` | A floor plan with the camera moving between rooms |
| `countdown_ring.json` | A counter with its number and unit inside the ring |

- `start` and `end` are film seconds and match the shot boundaries. All other times count from `start`.
- `theme: "light"` gives the linen board; leave it out for the dark board.
- `base` is a `map` (with `bbox` and `zoom`) or a `plan` (units 0-100 by 0-56).
- `camera` is a list of `[t, x, y, zoom]`.

Rules the films follow: every route connects at both ends; a name label goes only on that person's picture; no remains, wounds, weapons or crime scenes; victims appear only in real portraits.

## The tools

Run each with `FILM=<folder>`, for example `FILM=bundy .venv/bin/python scripts/video/tools/stills_tool.py ...`.

| Tool | Does |
|---|---|
| `stills_tool.py` | Check stills from a spec before rendering |
| `apply_specs.py` | Render specs and point shots at them |
| `smooth_joins.py` | Dissolve where an animation meets a different picture |
| `render_parts.py` | Render the whole film at 1080p in parts |
| `patch_mid.py` | Re-render one section and splice it into the 720p film |
| `make_subs.py`, `sync_check.py` | Build translated subtitles and check their timing |
| `chapters_click.py` | Transcribe the film from each chapter time (needs faster-whisper) |
| `cut_short.py` | Cut a vertical Short from the film without cropping |

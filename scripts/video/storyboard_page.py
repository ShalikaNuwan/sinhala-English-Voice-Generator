"""The storyboard as a page to review before anything is rendered.

    .venv/bin/python scripts/video/storyboard_page.py <video_dir>     # -> <video_dir>/storyboard.html

Reads `storyboard.json` (storyboard.py). One self-contained local HTML page: the tension curve over the
film, then each chapter and scene with its purpose, mood and look, and one row per beat with its time,
a thumbnail of the chosen picture, the intent, the narration it covers, the shot, transition and
effects, and any quote card. Every quote is listed with how it will be shown, sensitive ones flagged,
and the creative director's notes on the narration close the page. Thumbnails go in
`storyboard_thumbs/` beside it.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

FFMPEG = "/opt/miniconda3/envs/video/bin/ffmpeg"
LOOK_COLOURS = {
    "warm_memory": ("#c9a27a", "Warm memory"), "cold_dread": ("#5b7a99", "Cold dread"),
    "investigation": ("#9aa3a8", "Investigation"), "revelation": ("#b3121f", "Revelation"),
    "courtroom": ("#7d756c", "Courtroom"), "memorial": ("#d6d6d6", "Memorial"),
}
DEVICE_KINDS = {"board", "map", "profile", "exhibit", "highlight", "chart"}


def clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def thumb_source(beat: dict, library: dict[str, dict]) -> tuple[str, float | None] | None:
    """The file a beat's thumbnail comes from, and a time in it for a clip."""
    if beat.get("media") == "clip" and beat.get("asset"):
        return beat["asset"], float(beat.get("clip_in") or 0) + 0.5
    if beat.get("asset") and beat.get("media") == "photo":
        return beat["asset"], None
    for key in ("subject_id",):
        item = library.get(beat.get(key) or "")
        if item:
            return item["file"], None
    pins = beat.get("pins") or []
    if pins and pins[0].get("library_id") in library:
        return library[pins[0]["library_id"]]["file"], None
    return None


def make_thumb(video_dir: Path, beat: dict, library: dict[str, dict], out_dir: Path) -> str | None:
    found = thumb_source(beat, library)
    if not found:
        return None
    src, at = found
    tag = hashlib.sha1(f"{src}|{at}".encode()).hexdigest()[:8]     # a new picture gets a new thumbnail
    out = out_dir / f"{beat['index']:04d}_{tag}.jpg"
    if not out.exists():
        path = video_dir / src
        try:
            if at is not None:
                subprocess.run([FFMPEG, "-v", "error", "-y", "-ss", f"{at:.2f}", "-i", str(path), "-frames:v", "1",
                                "-vf", "scale=320:-2", str(out)], check=True, timeout=60)
            else:
                img = Image.open(path).convert("RGB")
                img.thumbnail((320, 220))
                img.save(out, quality=82)
        except Exception:  # noqa: BLE001 - a missing thumbnail must not stop the page
            return None
    return f"{out_dir.name}/{out.name}"


def tension_svg(storyboard: dict, width: int = 1000, height: int = 120) -> str:
    """Scene intensity over the film as a stepped line, chapters marked, each step in its look's colour."""
    duration = max(1.0, storyboard["duration_s"])
    x = lambda t: t / duration * width
    y = lambda v: height - 12 - v / 10 * (height - 24)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="curve" role="img" aria-label="Tension over the film">']
    for c in storyboard["bible"]["chapters"]:
        parts.append(f'<line x1="{x(c["start"]):.1f}" x2="{x(c["start"]):.1f}" y1="0" y2="{height}" class="chapter-line"/>')
    for s in storyboard["scenes"]:
        colour = LOOK_COLOURS.get(s["look"], ("#888", ""))[0]
        parts.append(f'<rect x="{x(s["start"]):.1f}" y="{y(s["intensity"]):.1f}" width="{max(1.0, x(s["end"]) - x(s["start"])):.1f}" '
                     f'height="{height - 12 - y(s["intensity"]):.1f}" fill="{colour}" opacity="0.85">'
                     f'<title>{html.escape(s["title"])} ({clock(s["start"])}), intensity {s["intensity"]}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def quote_block(q: dict) -> str:
    flag = ' <span class="flag">sensitive</span>' if q.get("sensitive") else ""
    shown = {"card": "shown as a card", "narration": "narration only"}.get(q.get("show"), q.get("show", ""))
    if q.get("style") == "exhibit":
        shown = "shown as an exhibit (written words)"
    who = html.escape(q.get("speaker") or "unknown")
    extra = " · ".join(html.escape(x) for x in (q.get("setting"), q.get("date")) if x)
    held = f' <span class="muted">({html.escape(q["held_back"])})</span>' if q.get("held_back") else ""
    return (f'<div class="quote {q.get("show")}"><span class="qmark">“</span>{html.escape(q["text"])}<span class="qmark">”</span>'
            f'<div class="qmeta">{q["id"]} · {clock(q["start"])} · {html.escape(q.get("kind", ""))} · {who}'
            f'{" · " + extra if extra else ""} · {shown}{flag}{held}</div></div>')


def render(storyboard: dict, thumbs: dict[int, str | None]) -> str:
    beats = {b["index"]: b for b in storyboard["beats"]}
    quotes = {q["id"]: q for q in storyboard.get("quotes", [])}
    chapters = storyboard["bible"]["chapters"]
    cards = sum(len(b.get("quote_ids") or []) for b in beats.values())
    out = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Storyboard</title><style>
:root{{--bg:#0d0d0f;--panel:#17171b;--line:#2a2a31;--text:#e9e7e4;--muted:#9a978f;--crimson:#c8102e}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:24px 16px 80px}}h1{{font-size:28px;margin:0 0 4px}}
.muted{{color:var(--muted)}}.curve{{width:100%;height:120px;background:var(--panel);border:1px solid var(--line);border-radius:8px;margin:16px 0}}
.chapter-line{{stroke:#444;stroke-width:1}}.legend span{{display:inline-flex;align-items:center;gap:6px;margin:0 14px 6px 0;font-size:13px}}
.sw{{width:14px;height:14px;border-radius:3px;display:inline-block}}
.chapter{{margin-top:40px;border-top:2px solid var(--crimson);padding-top:14px}}.chapter h2{{margin:0;font-size:22px}}
.scene{{margin:22px 0 8px;padding:12px 14px;background:var(--panel);border-left:5px solid #888;border-radius:6px}}
.scene h3{{margin:0 0 4px;font-size:17px}}.scene p{{margin:2px 0;font-size:14px}}
.beat{{display:grid;grid-template-columns:70px 200px 1fr;gap:14px;padding:12px 0;border-bottom:1px solid var(--line)}}
.beat img,.nothumb{{width:200px;height:120px;object-fit:cover;border-radius:4px;background:#222}}
.nothumb{{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}}
.time{{font-variant-numeric:tabular-nums;color:var(--muted)}}.intent{{font-weight:600}}.narr{{color:#cfccc6;font-size:14px;margin:4px 0}}
.tags span{{display:inline-block;font-size:12px;padding:1px 7px;margin:2px 4px 0 0;border:1px solid var(--line);border-radius:10px;color:var(--muted)}}
.tags .fx{{border-color:var(--crimson);color:#ff8a95}}.tags .warn{{border-color:#b8860b;color:#f0c674}}.tags .ok{{border-color:#4f7a4f;color:#9fd39f}}.quote{{margin:8px 0;padding:10px 12px;background:#1f1c1c;border-radius:6px;font-family:Georgia,serif}}
.quote.narration{{opacity:.7}}.qmark{{color:var(--crimson);font-size:22px;line-height:0}}.qmeta{{font:12px -apple-system,Helvetica,sans-serif;color:var(--muted);margin-top:4px}}
.flag{{background:var(--crimson);color:#fff;border-radius:3px;padding:0 5px}}
@media (max-width:700px){{.beat{{grid-template-columns:1fr}}.beat img,.nothumb{{width:100%;height:auto;aspect-ratio:16/10}}}}
</style></head><body><main>
<h1>Storyboard</h1>
<div class="muted">{clock(storyboard['duration_s'])} · {len(chapters)} chapters · {len(storyboard['scenes'])} scenes · {len(beats)} beats · {cards} quote cards</div>
<p>{html.escape(storyboard['bible'].get('logline', ''))}</p>
{tension_svg(storyboard)}
<div class="legend">""" + "".join(f'<span><i class="sw" style="background:{c}"></i>{name}</span>' for c, name in LOOK_COLOURS.values()) + "</div>"]
    scenes_by_chapter: dict[int, list[dict]] = {}
    for s in storyboard["scenes"]:
        scenes_by_chapter.setdefault(s["chapter"], []).append(s)
    for c in chapters:
        out.append(f'<section class="chapter"><h2>{c["id"]}. {html.escape(c["title"])}</h2>'
                   f'<div class="muted">{clock(c["start"])}–{clock(c["end"])} · {html.escape(c["era"])} · '
                   f'{LOOK_COLOURS.get(c["look"], ("", c["look"]))[1]} · intensity {c["intensity"]}</div>'
                   f'<p>{html.escape(c["purpose"])} <span class="muted">Tone: {html.escape(c["tone"])}</span></p>')
        for s in scenes_by_chapter.get(c["id"], []):
            colour = LOOK_COLOURS.get(s["look"], ("#888", ""))[0]
            out.append(f'<div class="scene" style="border-left-color:{colour}"><h3>{html.escape(s["title"])} '
                       f'<span class="muted">{clock(s["start"])}–{clock(s["end"])}</span></h3>'
                       f'<p><b>Setting:</b> {html.escape(s["setting"])} · <b>When:</b> {html.escape(s["story_time"])} · '
                       f'<b>Mood:</b> {html.escape(s["mood"])} · <b>Look:</b> {LOOK_COLOURS.get(s["look"], ("", s["look"]))[1]}'
                       f' · <b>Intensity:</b> {s["intensity"]}'
                       + (f' · <b>Scenario:</b> {html.escape(s["scenario"])}' if s.get("scenario") else "")
                       + f'</p><p><b>Purpose:</b> {html.escape(s["purpose"])}</p>'
                       f'<p><b>Visual idea:</b> {html.escape(s["visual_idea"])}</p></div>')
            for index in s["beats"]:
                b = beats[index]
                img = thumbs.get(index)
                pic = f'<img src="{img}" alt="" loading="lazy">' if img else f'<div class="nothumb">{html.escape(b["kind"])}</div>'
                tags = list(dict.fromkeys([b.get("shot_type") or "", b["kind"], b.get("motion") or "",
                                           f'in: {b.get("transition_in")}']))
                if b.get("overlay"):
                    tags.append("label: " + (b["overlay"].get("text") or ""))
                if b.get("continuity"):
                    tags.append("continuity: " + b["continuity"])
                fx = "".join(f'<span class="fx">{html.escape(e)}</span>' for e in b.get("effects") or [])
                check = b.get("check") or {}
                if check.get("replaced"):
                    fx += f'<span class="ok">picture swapped after check (was {html.escape(check["replaced"])})</span>'
                elif check and check.get("fits") != "yes":
                    note = check.get("fits", "") + ("" if check.get("era_ok", True) else ", wrong era")
                    fx += (f'<span class="warn" title="{html.escape(check.get("problem") or "")}">check: '
                           f'{html.escape(note)}</span>')
                qs = "".join(quote_block(quotes[q]) for q in b.get("quote_ids") or [] if q in quotes)
                out.append(f'<div class="beat"><div class="time">{clock(b["start"])}<br>{b["end"] - b["start"]:.1f}s</div>{pic}'
                           f'<div><div class="intent">{html.escape(b.get("intent") or b.get("reason") or "")}</div>'
                           f'<div class="narr">{html.escape(b["narration"])}</div>'
                           f'<div class="tags">' + "".join(f"<span>{html.escape(t)}</span>" for t in tags if t) + fx + f"</div>{qs}</div></div>")
        out.append("</section>")
    if quotes:
        out.append('<section class="chapter"><h2>All quotes</h2><p class="muted">To show a narration-only quote as a card, '
                   'set its "show" to "card" in storyboard.json (quotes list) and re-run the page.</p>'
                   + "".join(quote_block(q) for q in quotes.values()) + "</section>")
    notes = (storyboard["bible"].get("review") or {}).get("narration_notes") or []
    if notes:
        out.append('<section class="chapter"><h2>Notes on the narration</h2><p class="muted">From the creative-director '
                   'review. The film is cut to the recorded audio; these are for a future edit.</p><ul>'
                   + "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul></section>")
    out.append("</main></body></html>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    args = parser.parse_args()
    video_dir = Path(args.video_dir)
    storyboard = json.loads((video_dir / "storyboard.json").read_text(encoding="utf-8"))
    from storyboard import load_library
    library, _ = load_library(video_dir, free_only=False)
    out_dir = video_dir / "storyboard_thumbs"
    out_dir.mkdir(exist_ok=True)
    thumbs = {b["index"]: make_thumb(video_dir, b, library, out_dir) for b in storyboard["beats"]}
    (video_dir / "storyboard.html").write_text(render(storyboard, thumbs), encoding="utf-8")
    print(f"wrote {video_dir / 'storyboard.html'} ({sum(1 for t in thumbs.values() if t)} thumbnails)")


if __name__ == "__main__":
    main()

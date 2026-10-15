"""Run the picture stages in order: fill the gaps the library left, draw the rare chart, assemble.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/build.py <video_dir> [options]

      --shots N        only the first N shots, for a preview
      --from N         start at shot N
      --subtitles X    burn | sidecar | none   (default: whatever shots.json says)
      --out PATH       output file (default: <video_dir>/cut.mp4, or preview.mp4 for a slice)

Shots the planner matched to a library photograph are used as they are. A shot with no library match
is searched for (a real photograph first); a shot the planner marked `generated` is generated, as the
last resort. Every stage resumes: nothing on disk is fetched, generated, drawn or rendered twice.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assemble  # noqa: E402
import graphics  # noqa: E402
import investigation  # noqa: E402
import motion  # noqa: E402
import sources  # noqa: E402

DRAWN = {"chart", "map"}
SEARCHED = {"archival", "period", "stock"}


def chosen_shots(plan: dict, start: int, count: int | None) -> list[dict]:
    """The shots this run works on. Times are left alone: assembly does its own offsetting."""
    shots = [s for s in plan["shots"] if s["index"] >= start]
    if count:
        shots = shots[:count]
    if not shots:
        sys.exit("no shots in that range")
    return shots


def draw(shots: list[dict], video_dir: Path, width: int, height: int) -> int:
    """Render each chart and map that is not already on disk."""
    folder = video_dir / "assets"
    folder.mkdir(parents=True, exist_ok=True)
    drawn = 0
    for shot in shots:
        if shot["kind"] not in DRAWN:
            continue
        path = folder / f"{shot['index']:04d}_{shot['kind']}.jpg"
        shot.update({"asset": f"assets/{path.name}", "licence": "Original artwork",
                     "attribution": "Drawn for this film", "rights": "original"})
        if path.exists() and path.stat().st_size:
            continue
        try:
            graphics.save(graphics.render_shot(shot, width, height), path)
            drawn += 1
        except Exception as exc:  # noqa: BLE001 - one unusable chart must not stop the film
            shot.update({"asset": None, "error": f"draw failed: {str(exc)[:120]}"})
    return drawn


def _digest(value) -> str:
    import hashlib
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:8]


def render_motion(shots: list[dict], video_dir: Path, library: dict[str, dict]) -> int:
    """Render every board, map, profile, exhibit and highlight that is not already on disk.

    A highlight whose detail the vision check cannot confirm runs as a plain photograph of its
    subject instead: an unconfirmed circle would point the viewer at the wrong thing.
    """
    folder = video_dir / "assets" / "motion"
    folder.mkdir(parents=True, exist_ok=True)
    exhibit_number = 0
    made = 0
    for shot in shots:
        if shot.get("media") != "motion":
            continue
        seconds = round(shot["end"] - shot["start"], 3)
        subject = library.get(shot.get("subject_id") or "")
        photo = video_dir / subject["file"] if subject else None
        kind = shot["kind"]
        if kind == "exhibit":
            exhibit_number += 1
        params = {k: shot.get(k) for k in ("kind", "subject_id", "fields", "tag", "detail", "label", "pins", "links", "points")}
        out = folder / f"{shot['index']:04d}_{kind}_{_digest([params, seconds, exhibit_number])}.mp4"
        try:
            if not out.exists():
                if kind == "board":
                    pins = [{"image": str(video_dir / library[p["library_id"]]["file"]), "label": p.get("label", ""),
                             "sub": p.get("sub", "")} for p in shot.get("pins") or []]
                    links = [(l["a"], l["b"]) for l in shot.get("links") or []]
                    motion.evidence_board(pins, links, out, seconds)
                elif kind == "map":
                    motion.map_reveal(shot.get("points") or [], out, seconds)
                elif kind == "profile":
                    fields = [(f["label"], f["value"]) for f in shot.get("fields") or []]
                    investigation.profile_card(photo, fields, out, seconds)
                elif kind == "exhibit":
                    investigation.exhibit(photo, shot.get("tag") or "", out, seconds, number=exhibit_number)
                elif kind == "highlight":
                    box = investigation.locate(photo, shot.get("detail") or "")
                    if box is None:
                        raise LookupError(f"could not confirm {shot.get('detail')!r} in the picture")
                    investigation.highlight(photo, box, out, seconds, label=shot.get("label") or None)
                made += 1
            shot["asset"] = str(out.relative_to(video_dir))
        except Exception as exc:  # noqa: BLE001 - a device that fails becomes its photograph
            print(f"  shot #{shot['index']} {kind} skipped: {str(exc)[:120]}", flush=True)
            if subject:
                shot.update({"media": "photo", "kind": "archival", "asset": subject["file"],
                             "colour": subject.get("colour"), "kind_hint": subject.get("kind")})
            else:
                shot.update({"media": "photo", "kind": "period", "asset": None, "query": "chicago 1970s street"})
    return made


def fill_gaps(shots: list[dict], video_dir: Path) -> list[dict]:
    """Search for, or generate, only the shots the library did not cover."""
    gaps = [s for s in shots if not s.get("asset") and s.get("media", "photo") == "photo"
            and (s["kind"] in SEARCHED or s["kind"] == "generated")]
    if not gaps:
        return shots
    # The sourcing stage knows two searched kinds; a period picture is searched like stock.
    staged = []
    for shot in gaps:
        copy = dict(shot)
        if copy["kind"] == "period":
            copy["kind"] = "stock"
        if copy["kind"] in ("archival", "stock") and not copy.get("query"):
            copy["query"] = "chicago 1970s street"
        staged.append(copy)
    filled = {s["index"]: s for s in sources.fill_shots(staged, video_dir / "assets")}
    out = []
    for shot in shots:
        done = filled.get(shot["index"])
        if done is None:
            out.append(shot)
            continue
        merged = dict(shot)
        for key in ("asset", "licence", "attribution", "source_url", "error"):
            if key in done:
                merged[key] = done[key]
        merged["rights"] = "generated" if shot["kind"] == "generated" else ("free" if done.get("asset") else None)
        out.append(merged)
    return out


RIGHTS_HEADINGS = (
    ("free", "Freely licensed (public domain, CC0, CC BY, CC BY-SA)"),
    ("fair_use", "Copyrighted, used under a documentary fair-use claim"),
    ("generated", "Generated for this film"),
    ("original", "Drawn for this film"),
)


def credits(shots: list[dict]) -> str:
    """Every picture in the cut, once, grouped by how the owner may use it."""
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for shot in shots:
        if not shot.get("asset"):
            continue
        groups[shot.get("rights") or "free"].setdefault(shot["asset"], shot)
    out = ["# Picture credits", ""]
    for key, heading in RIGHTS_HEADINGS:
        if not groups.get(key):
            continue
        out += [f"## {heading} ({len(groups[key])})", ""]
        for asset, shot in sorted(groups[key].items()):
            parts = [f"`{asset}`", shot.get("attribution") or "Unknown", shot.get("licence") or ""]
            if shot.get("source_url"):
                parts.append(shot["source_url"])
            out.append("- " + " — ".join(p for p in parts if p))
        out.append("")
    return "\n".join(out)


FINAL_MBPS = 11.5   # the full-quality master with grain, measured on the demos
PEAK_FACTOR = 2.1   # the joined sections and the final file sit on disk together for a while


def space_needed(seconds: float) -> float:
    """Bytes the assembly needs free: the master twice over near the end, plus a margin."""
    return seconds * FINAL_MBPS * 1e6 / 8 * PEAK_FACTOR + 2e9


def wait_for_space(video_dir: Path, seconds: float, poll_s: int = 60) -> None:
    """Hold before assembly until the disk can take it, rather than fail halfway through the film."""
    import shutil
    import time
    need = space_needed(seconds)
    while (free := shutil.disk_usage(video_dir).free) < need:
        print(f"  waiting for disk space: {free / 1e9:.1f} GB free, {need / 1e9:.1f} GB needed", flush=True)
        time.sleep(poll_s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--from", dest="start", type=int, default=1)
    parser.add_argument("--subtitles", default=None, choices=["burn", "sidecar", "none"])
    parser.add_argument("--out", default="")
    parser.add_argument("--sound", action="store_true", help="mix music and effects under the narration")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    plan = json.loads((video_dir / "shots.json").read_text(encoding="utf-8"))
    preview = args.shots is not None or args.start > 1
    shots = chosen_shots(plan, args.start, args.shots)
    span = shots[-1]["end"] - shots[0]["start"]
    print(f"{len(shots)} shots, {span / 60:.1f} min"
          + (f" (from shot {args.start} at {shots[0]['start'] / 60:.1f} min)" if preview else ""))

    from_library = sum(1 for s in shots if s.get("library_id"))
    print(f"  {from_library} shots use a library photograph or clip")
    library = {i["id"]: i for i in json.loads((video_dir / "library.json").read_text(encoding="utf-8"))["items"]}
    print(f"  rendered {render_motion(shots, video_dir, library)} motion shots")
    print(f"  drew {draw([s for s in shots if s.get('media', 'photo') == 'photo'], video_dir, plan['width'], plan['height'])} charts")
    shots = fill_gaps(shots, video_dir)
    missing = [s["index"] for s in shots if not s.get("asset")]
    print(f"  {len(shots) - len(missing)}/{len(shots)} shots have a picture"
          + (f"; held over: {missing[:12]}" if missing else ""))

    # The whole plan is written back with this run's shots filled, so previews and a full pass share
    # one file and neither loses the other's work.
    filled_path = video_dir / "shots.filled.json"
    merged = {s["index"]: s for s in plan["shots"]}
    if filled_path.exists():
        merged.update({s["index"]: s for s in json.loads(filled_path.read_text(encoding="utf-8"))["shots"]
                       if s.get("asset")})
    merged.update({s["index"]: s for s in shots})
    plan["shots"] = [merged[i] for i in sorted(merged)]
    sound_credits = ""
    if args.sound:
        import subprocess
        span = shots[-1]["end"]
        cmd = [sys.executable, str(Path(__file__).with_name("sound.py")), str(video_dir)]
        if preview:
            cmd += ["--preview", f"{span:.2f}"]
        print("mixing sound", flush=True)
        subprocess.run(cmd, check=True)
        # Absolute: assembly resolves relative paths against the shot list's own folder.
        plan["audio"] = str((video_dir / "sound" / ("mix_preview.wav" if preview else "mix.wav")).resolve())
        import sound
        sound_credits = sound.credits_markdown(json.loads((video_dir / "sound" / "credits.json").read_text(encoding="utf-8")))
    filled_path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    (video_dir / "credits.md").write_text(credits(plan["shots"]) + ("\n" + sound_credits if sound_credits else ""),
                                          encoding="utf-8")

    out = Path(args.out) if args.out else video_dir / ("preview.mp4" if preview else "cut.mp4")
    wait_for_space(video_dir, span)
    print(f"assembling to {out}")
    assemble.render(filled_path, out, subtitles=args.subtitles or plan.get("subtitles"),
                    start_index=shots[0]["index"], limit=len(shots), keep_intermediates=args.keep_intermediates)
    print(f"done: {out}")


if __name__ == "__main__":
    main()

"""Re-render one middle part of the finished 720p film and splice it between the existing film on either side.
Both ends must be part boundaries from render_720_parts.py (hard cuts, where the film has a keyframe).
usage: patch_mid.py <first index of the part> <first index of the next part>"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, NAME  # noqa: E402
import assemble  # noqa: E402
import build  # noqa: E402
import plan_edit  # noqa: E402

FIRST, NEXT = int(sys.argv[1]), int(sys.argv[2])
OUT = D / f"{NAME}_documentary_720p.mp4"
NEW = D / "work" / "parts" / "patched_720p.mp4"
MID = D / "work" / "parts" / f"mid_{FIRST:04d}_{NEXT - 1}.mp4"
MID.parent.mkdir(parents=True, exist_ok=True)

plan_edit.use_case(D)
plan = json.loads((D / "shots.json").read_text(encoding="utf-8"))
shots = plan["shots"]
library = {i["id"]: i for i in json.loads((D / "library.json").read_text(encoding="utf-8"))["items"]}
print("rendered", build.render_motion(shots, D, library), "motion shots", flush=True)
build.draw([s for s in shots if s.get("media", "photo") == "photo"], D, 1920, 1080)
shots = build.fill_gaps(shots, D)
plan["shots"] = shots
if (D / "words.json").exists():
    build.cue_name_cards(plan["shots"], json.loads((D / "words.json").read_text(encoding="utf-8")))
(D / "credits.md").write_text(build.credits(plan["shots"]), encoding="utf-8")
plan["width"], plan["height"] = 1280, 720
path = D / "shots.720p.json"
path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
for position, shot in enumerate(shots, start=1):
    shot.setdefault("index", position)
total = float(shots[-1]["end"])


def keyframe(index):
    shot = next(s for s in shots if s["index"] == index)
    assert (shot.get("transition_in") or "dissolve").lower() == "cut", f"#{index} must be a hard cut"
    cut = round(float(shot["start"]) * 30) / 30
    probe = subprocess.run([assemble.FFPROBE, "-v", "error", "-select_streams", "v", "-read_intervals",
                            f"{cut - 3:.2f}%{cut + 3:.2f}", "-show_entries", "packet=pts_time,dts_time,flags", "-of", "csv=p=0",
                            str(OUT)], capture_output=True, text=True, check=True).stdout
    keys = [tuple(map(float, l.split(",")[:2])) for l in probe.splitlines() if ",K" in l]
    key = min(keys, key=lambda k: abs(k[0] - cut))
    assert abs(key[0] - cut) < 0.3, f"no keyframe at {cut:.3f} (nearest {key[0]:.3f})"
    return key


k1 = keyframe(FIRST)[1] - 0.01  # the concat demuxer compares an outpoint with DTS; the margin beats ffprobe rounding
k2 = keyframe(NEXT)[0] if NEXT else None
print(f"splicing from keyframe dts {k1:.6f} to {k2}", flush=True)
assemble.render(path, MID, subtitles="none", start_index=FIRST,
                limit=sum(1 for s in shots if s["index"] >= FIRST and (not NEXT or s["index"] < NEXT)))
listing = MID.parent / "patch_concat.txt"
tail = f"file '{OUT}'\ninpoint {k2:.6f}\n" if NEXT else ""
listing.write_text(f"file '{OUT}'\noutpoint {k1:.6f}\nfile '{MID}'\n" + tail, encoding="utf-8")
audio = assemble.resolve_asset(D, plan.get("audio"))
assemble.run([assemble.FFMPEG, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
              "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
              # exact 30 fps, keeping each frame's display offset (PTS-DTS) so the B-frames stay in order
              "-bsf:v", "setts=dts=(N-2)/(30*TB):pts=PTS-DTS+(N-2)/(30*TB)", "-movflags", "+faststart", str(NEW)])
assemble.check_duration(NEW, total)
NEW.replace(OUT)
MID.unlink(missing_ok=True)
listing.unlink(missing_ok=True)
print("done:", OUT, flush=True)

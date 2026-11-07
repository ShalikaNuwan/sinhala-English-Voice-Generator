"""The whole film at 1920x1080, full quality, rendered in N parts so the disk only ever holds about the film plus
one part. Parts split on a hard cut inside one animation (seamless), each part is a normal assemble.render slice
(resumable), then the parts are joined with -c:v copy under the narration at exact 30 fps, and each part is deleted
as soon as the join has read past it.  usage: FILM=<name> render_parts.py [N]"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, NAME  # noqa: E402
import assemble  # noqa: E402
import build  # noqa: E402
import plan_edit  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT = D / f"{NAME}_final_1080p.mp4"
PARTS = D / "work" / "parts1080"
PARTS.mkdir(parents=True, exist_ok=True)

plan_edit.use_case(D)
plan = json.loads((D / "shots.json").read_text(encoding="utf-8"))
shots = plan["shots"]
library = {i["id"]: i for i in json.loads((D / "library.json").read_text(encoding="utf-8"))["items"]}
print("rendered", build.render_motion(shots, D, library), "motion shots", flush=True)
build.draw([s for s in shots if s.get("media", "photo") == "photo"], D, 1920, 1080)
shots = build.fill_gaps(shots, D)
missing = [s["index"] for s in shots if not s.get("asset")]
print(len(shots), "shots;", len(missing), "without a picture", missing[:20], flush=True)
plan["shots"] = shots
if (D / "words.json").exists():
    build.cue_name_cards(plan["shots"], json.loads((D / "words.json").read_text(encoding="utf-8")))
plan["width"], plan["height"] = 1920, 1080
path = D / "shots.1080p.json"
path.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")

for position, shot in enumerate(shots, start=1):
    shot.setdefault("index", position)
total = float(shots[-1]["end"])
# split points: a shot that hard-cuts from the same animation, nearest each k/N of the running time
cuts = [i for i in range(1, len(shots))
        if (shots[i].get("transition_in") or "dissolve").lower() == "cut"
        and shots[i].get("media") == "motion" and shots[i - 1].get("media") == "motion"
        and shots[i].get("explainer") and shots[i].get("explainer") == shots[i - 1].get("explainer")]
splits = sorted({min(cuts, key=lambda i: abs(float(shots[i]["start"]) - total * k / N)) for k in range(1, N)})
bounds = [0] + splits + [len(shots)]
ranges = [(shots[a]["index"], b - a, float(shots[a]["start"]), float(shots[b - 1]["end"]))
          for a, b in zip(bounds, bounds[1:])]
print("parts:", [(r[0], r[1], round(r[2]), round(r[3])) for r in ranges], flush=True)

parts = []
for n, (first, count, start, end) in enumerate(ranges, start=1):
    part = PARTS / f"part_{n:02d}_{first:04d}_{count}.mp4"
    parts.append((part, end))
    if part.exists() and abs(assemble.probe_duration(part) - (end - start)) < 0.2:
        print("part", n, "already rendered", flush=True)
        continue
    while shutil.disk_usage(D).free < 3.5e9:
        print("WAIT disk low before part", n, flush=True)
        time.sleep(120)
    print(f"part {n}/{len(ranges)}: shots #{first}+{count} ({start:.1f}-{end:.1f}s) {time.strftime('%H:%M')}", flush=True)
    assemble.render(path, part, subtitles="none", start_index=first, limit=count)

# join under the narration; free each part once the join has read past it
listing = PARTS / "concat.txt"
listing.write_text(assemble.concat_list(p for p, _ in parts), encoding="utf-8")
audio = assemble.resolve_asset(D, plan.get("audio"))
argv = [assemble.FFMPEG, "-y", "-v", "error", "-progress", "pipe:1", "-nostats",
        "-f", "concat", "-safe", "0", "-i", str(listing), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        # exact 30 fps, keeping each frame's display offset (PTS-DTS) so the B-frames stay in order
        "-bsf:v", "setts=dts=(N-2)/(30*TB):pts=PTS-DTS+(N-2)/(30*TB)", "-movflags", "+faststart", str(OUT)]
print("joining", len(parts), "parts into", OUT, flush=True)
proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
freed = 0
for line in proc.stdout:
    if line.startswith("out_time_us="):
        try:
            t = int(line.split("=")[1]) / 1e6
        except ValueError:
            continue
        while freed < len(parts) - 1 and t > parts[freed][1] + 10:
            parts[freed][0].unlink(missing_ok=True)
            print(f"  joined past part {freed + 1}; deleted it", flush=True)
            freed += 1
if proc.wait() != 0:
    raise SystemExit("join failed (parts not yet deleted are kept; deleted ones re-render on a re-run)")
for p, _ in parts:
    p.unlink(missing_ok=True)
listing.unlink(missing_ok=True)
assemble.check_duration(OUT, total)
print("done:", OUT, flush=True)

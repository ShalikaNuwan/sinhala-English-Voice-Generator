"""Cut a YouTube Short from a finished film: the whole 16:9 frame at full width on a 1080x1920 canvas, nothing cropped.
A plain background suits a dark animation; blur suits one that changes colour. Audio and picture may start apart:
--hold-to shows the first clean frame while the narrator's first word plays, past a dissolve from the previous scene.
usage: FILM=<name> cut_short.py <start> <end> <out.mp4> [--hold-to SECONDS] [--colour 0C1016]   (film seconds)"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, FFMPEG, NAME  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("start", type=float)
ap.add_argument("end", type=float)
ap.add_argument("out")
ap.add_argument("--hold-to", type=float)
ap.add_argument("--colour")
a = ap.parse_args()
film = D / f"{NAME}_final_1080p.mp4"
total = a.end - a.start
picture = a.hold_to or a.start
hold = picture - a.start
head = f"[0:v]tpad=start_duration={hold:.2f}:start_mode=clone," if hold > 0 else "[0:v]"
if a.colour:
    video = f"{head}scale=1080:608:flags=lanczos,pad=1080:1920:0:656:color=0x{a.colour}"
else:
    video = (f"{head}split=2[f][b];[b]scale=-2:1920:flags=fast_bilinear,crop=1080:1920,gblur=sigma=45,"
             "eq=brightness=-0.14:saturation=0.75[bg];[f]scale=1080:608:flags=lanczos[fg];[bg][fg]overlay=0:656")
graph = (f"{video},fade=t=in:st=0:d=0.3,fade=t=out:st={total - 0.5:.2f}:d=0.5,format=yuv420p[v];"
         f"[1:a]afade=t=in:d=0.04,afade=t=out:st={total - 0.15:.2f}:d=0.15[a]")
subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", str(picture), "-t", f"{a.end - picture:.2f}", "-i", str(film),
                "-ss", str(a.start), "-t", f"{total:.2f}", "-i", str(film), "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", a.out], check=True)
print("done", a.out)

"""Render approved explainer/callback specs and point their shots at them.
   FILM=<name> .venv/bin/python apply_specs.py <spec.json>:<shot,shot,...> [...]"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import film  # noqa: E402,F401
import shot_tool as t
import build, explainer, motion
lib = {i["id"]: i for i in t.library_items()}
for arg in sys.argv[1:]:
    rel, shots = arg.split(":")
    shots = [int(x) for x in shots.split(",")]
    spec = json.loads((t.D / rel).read_text())
    kind = "callback" if "highlights" in spec else "explainer"
    out = t.D / "assets" / "motion" / f"{kind}_{build._digest(spec)}.mp4"
    seconds = round(spec["end"] - spec["start"], 3)
    t0 = time.time()
    if not out.exists():
        if kind == "explainer":
            explainer.render(spec, t.D, out, seconds)
        else:
            b = spec["board"]
            motion.board_callback([dict(p, image=str(t.D / lib[p["library_id"]]["file"])) for p in b["pins"]],
                                  dict(b["subject"], image=str(t.D / lib[b["subject"]["library_id"]]["file"])),
                                  out, seconds, spec["path"], spec["highlights"])
    print(rel, "rendered", round(time.time() - t0), "s", flush=True)
    with t.locked():
        storyboard = t.review.engine()
        sb = t.review.load_storyboard(t.D)
        prev = None
        for n in sorted(shots):
            beat = t.review.find_beat(sb, n)
            assert beat["start"] >= spec["start"] - 0.05 and beat["end"] <= spec["end"] + 0.05, (n, rel)
            for key in ("clip_id", "clip_out", "crop", "fields", "board", "explainer", "callback"):
                beat.pop(key, None)
            beat.update({"kind": kind, "media": "motion", kind: rel, "library_id": None, "asset": None,
                         "overlay": None, "detail": "", "motion": "still", "effects": [], "colour": None,
                         "rights": "original", "licence": "Original artwork", "attribution": "Drawn for this film",
                         "source_url": None, "owner": {"choice": f"{kind} {Path(rel).stem}", "at": time.time()}})
            if prev == n - 1:
                beat["transition_in"] = "cut"
            prev = n
        storyboard.save(t.D, sb)
        for n in sorted(shots):
            clip = t.regen.render_preview(t.D, n)
            t.review.set_mark(t.D, n, None)
            t.regen.note(t.D, n, state="done", step="Done", clip=clip, check=None, error="",
                         explanation=f"(Claude) {kind} animation {Path(rel).stem} in place of the earlier picture")
            print(" ", n, clip, flush=True)

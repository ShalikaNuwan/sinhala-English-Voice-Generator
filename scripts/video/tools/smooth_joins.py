"""Where an animation meets a shot with a real picture (either way), or two different animations meet, the
incoming shot dissolves instead of cutting. Shots inside one continuous animation keep their cuts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import shot_tool as t
ANIM = ("explainer", "callback", "reveal")
def key(b):
    k = b.get("kind")
    return (k, b.get(k) if k in ("explainer", "callback") else ("opening" if k == "reveal" else None)) if k in ANIM else None
with t.locked():
    storyboard = t.review.engine()
    sb = t.review.load_storyboard(t.D)
    beats = sb["beats"]
    changed = []
    for prev, cur in zip(beats, beats[1:]):
        a, b = key(prev), key(cur)
        if (a or b) and a != b and (cur.get("transition_in") or "cut") == "cut":
            cur["transition_in"] = "dissolve"
            changed.append(cur["index"])
    storyboard.save(t.D, sb)
print("dissolves set on", len(changed), "shots:", changed)

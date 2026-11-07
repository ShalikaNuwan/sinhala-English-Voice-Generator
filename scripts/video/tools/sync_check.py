"""Every sentence, every language: is the right text on screen when the narrator says it, and is every word there?"""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, NAME  # noqa: E402

sentences = json.loads((D / "sentences.json").read_text(encoding="utf-8"))


def sec(x):
    h, m, s = x.strip().replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def squash(t):
    return re.sub(r"\s+", "", t)


for lang, name in (("de", "de"), ("fr", "fr"), ("pt", "pt-BR"), ("es", "es")):
    text = {}
    for f in glob.glob(str(D / f"work/subs/{lang}_[12]_*.tsv")):
        for row in open(f, encoding="utf-8"):
            if row.strip():
                i, t = row.rstrip("\n").split("\t", 1)
                text[int(i)] = " ".join(t.split())
    cues = []
    for block in (D / f"{NAME}_final_1080p.{name}.srt").read_text(encoding="utf-8").strip().split("\n\n"):
        rows = block.split("\n")
        a, b = rows[1].split(" --> ")
        cues.append((sec(a), sec(b), " ".join(rows[2:])))
    same = (D / f"{NAME}_final_720p.{name}.srt").read_text(encoding="utf-8") == (D / f"{NAME}_final_1080p.{name}.srt").read_text(encoding="utf-8")
    # every word of the translation survives, in order
    whole = squash("".join(c[2] for c in cues))
    source = squash("".join(text[s["index"]] for s in sentences))
    # exact: walk the characters, so each sentence lands on the cue that really holds its first letter
    bounds, at = [], 0
    for a, b, t in cues:
        n = len(squash(t))
        bounds.append((at, at + n, a, b))
        at += n
    offs, k, gaps = 0, 0, []
    for s in sentences:
        while bounds[k][1] <= offs:
            k += 1
        lo, hi, a, b = bounds[k]
        shown_at = a + (b - a) * (offs - lo) / max(hi - lo, 1)
        gaps.append((round(shown_at - s["start"], 2), s["index"], round(a - s["start"], 2)))
        offs += len(squash(text[s["index"]]))
    d = sorted(g[0] for g in gaps)
    n = len(d)
    late = [g for g in gaps if g[2] > 0.3]
    early = sorted(gaps, key=lambda g: g[2])[:3]
    far = [g for g in gaps if abs(g[0]) > 1.5]
    print(f"  text vs voice: median {d[n // 2]:+.2f}s, 1%..99% {d[n // 100]:+.2f}..{d[-n // 100]:+.2f}s, worst {d[0]:+.2f}/{d[-1]:+.2f}s, more than 1.5s off: {len(far)} {far[:5]}; cue starts after voice >0.3s: {len(late)}; earliest cue starts {early}", flush=True)
    wrong, early, late = [], [0], [0]
    shown = sum(b - a for a, b, _ in cues)
    print(f"{name}: cues {len(cues)} text complete {whole == source} 720p copy same {same} sentences not located {len(wrong)} {wrong[:8]} "
          f"worst early {max(early or [0]):.2f}s worst late {max(late or [0]):.2f}s on screen {shown / 60:.0f} min", flush=True)

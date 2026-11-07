"""Build one .srt per language from the translated sentences: at most two lines of 42 characters, timed on the
narration's sentence timing, a long sentence split at its punctuation with the time shared by length.
translations in work/subs/<lang>_<part>_<piece>.tsv, one "index<TAB>text" per line.
usage: FILM=<name> make_subs.py de fr pt es"""
import glob
import json
import math
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from film import D, NAME  # noqa: E402

LINE, CUE, GAP, MIN_S, MAX_S, READ = 42, 84, 0.08, 1.0, 7.0, 16.0
TITLES = ("Dr.", "Mrs.", "Mr.", "Ms.", "Sgt.", "Lt.", "Det.")
EARLY = 0.9
NAMES = {"de": "de", "fr": "fr", "pt": "pt-BR", "es": "es"}
sentences = json.loads((D / "sentences.json").read_text(encoding="utf-8"))
FILM_END = sentences[-1]["end"] - 0.02


def load(lang):
    out = {}
    for f in sorted(glob.glob(str(D / f"work/subs/{lang}_[12]_*.tsv"))):
        for n, row in enumerate(open(f, encoding="utf-8"), 1):
            row = row.rstrip("\n")
            if not row.strip():
                continue
            index, text = row.split("\t", 1)
            assert int(index) not in out, (f, n, index)
            out[int(index)] = " ".join(text.split())
    return out


def split_point(text, target):
    """Where to break near `target`: after punctuation if one is close, else at a space."""
    best, score = None, 1e9
    for m in re.finditer(r" ", text):
        i = m.start()
        cost = abs(i - target)
        before = text[i - 1]
        if before in ".!?;:":
            cost -= 18
        elif before in ",—–":
            cost -= 12
        if cost < score:
            best, score = i, cost
    return best


def pieces(text):
    if len(text) <= CUE:
        return [text]
    n = math.ceil(len(text) / (CUE - 6))
    i = split_point(text, len(text) / n)
    if i is None:
        return [text]
    return pieces(text[:i].strip()) + pieces(text[i:].strip())


def two_lines(text):
    if len(text) <= LINE:
        return text
    spaces = [m.start() for m in re.finditer(" ", text)]
    fit = [i for i in spaces if i <= LINE and len(text) - i - 1 <= LINE]
    if not fit:
        i = split_point(text, len(text) / 2)
        return text if i is None else text[:i].strip() + "\n" + text[i:].strip()
    # a balanced break, a little better after punctuation
    i = min(fit, key=lambda k: abs(k - len(text) / 2) - (6 if text[k - 1] in ".,;:!?" else 0))
    return text[:i].strip() + "\n" + text[i:].strip()


def stamp(x):
    ms = round(x * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def build(lang):
    text = load(lang)
    missing = [s["index"] for s in sentences if s["index"] not in text]
    assert not missing, (lang, len(missing), missing[:10])
    # a sentence too short to read joins the next one
    units, i = [], 0
    while i < len(sentences):
        s = sentences[i]
        unit = {"text": text[s["index"]], "start": s["start"], "end": max(s.get("spoken_end", s["end"]), s["start"] + 0.2)}
        while i + 1 < len(sentences) and (unit["end"] - unit["start"] < 1.6 or sentences[i]["text"].endswith(TITLES)):
            nxt = sentences[i + 1]
            joined = unit["text"] + " " + text[nxt["index"]]
            if not sentences[i]["text"].endswith(TITLES) and (len(joined) > CUE or nxt["start"] - unit["end"] > 0.6):
                break
            unit["text"], unit["end"] = joined, max(nxt.get("spoken_end", nxt["end"]), nxt["start"] + 0.2)
            i += 1
        units.append(unit)
        i += 1
    cues = []
    for u in units:
        parts = pieces(u["text"])
        total, at = sum(len(p) for p in parts), u["start"]
        for p in parts:
            span = (u["end"] - u["start"]) * len(p) / total
            cues.append({"start": at, "end": at + span, "text": p})
            at += span
    for a, b in zip(cues, cues[1:] + [None]):
        limit = (b["start"] - GAP) if b else FILM_END
        want = max(a["end"] + 0.25, a["start"] + MIN_S, a["start"] + len(a["text"]) / READ)
        a["end"] = min(want, a["start"] + MAX_S, limit)
    # a cue too fast to read borrows spare time from the cue before it, up to EARLY seconds ahead of the voice
    for _ in range(3):
        for a, b in zip(cues, cues[1:]):
            need = len(b["text"]) / READ - (b["end"] - b["start"])
            spare = (a["end"] - a["start"]) - max(len(a["text"]) / READ, MIN_S)
            if need > 0 and spare > 0:
                move = min(need, spare, EARLY - b.get("early", 0))
                if move > 0:
                    b["start"] -= move
                    b["early"] = b.get("early", 0) + move
                    a["end"] = min(a["end"], b["start"] - GAP)
    name = NAMES[lang]
    body = "\n".join(f"{n}\n{stamp(c['start'])} --> {stamp(c['end'])}\n{two_lines(c['text'])}\n" for n, c in enumerate(cues, 1))
    out = D / f"{NAME}_final_1080p.{name}.srt"
    out.write_text(body, encoding="utf-8")
    shutil.copyfile(out, D / f"{NAME}_final_720p.{name}.srt")
    return text, cues


def check(lang, text, cues):
    en = {s["index"]: s["text"] for s in sentences}
    same = [i for i, t in text.items() if t == en[i] and len(t) > 25]
    ratio = [(i, round(len(t) / len(en[i]), 2)) for i, t in text.items() if len(en[i]) > 30 and not 0.65 < len(t) / len(en[i]) < 1.75]
    years = [i for i, t in text.items() if set(re.findall(r"\b(?:19|20)\d\d\b", en[i])) - set(re.findall(r"\b(?:19|20)\d\d\b", t))]
    over = sum(1 for a, b in zip(cues, cues[1:]) if a["end"] > b["start"])
    order = sum(1 for a, b in zip(cues, cues[1:]) if b["start"] < a["start"])
    fast = [c for c in cues if len(c["text"]) / max(c["end"] - c["start"], 0.1) > 21]
    long_lines = sum(1 for c in cues if any(len(l) > LINE + 4 for l in two_lines(c["text"]).split("\n")))
    print(f"{lang}: sentences {len(text)} cues {len(cues)} overlaps {over} order {order} long lines {long_lines} "
          f"over 21 cps {len(fast)} ({100 * len(fast) / len(cues):.1f}%) same-as-English {len(same)} length outliers {len(ratio)} "
          f"years differ {len(years)}", flush=True)
    print("  same", same[:8], "ratio", ratio[:8], "years", years[:8], flush=True)


for lang in sys.argv[1:]:
    text, cues = build(lang)
    check(lang, text, cues)

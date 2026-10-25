"""Time every sentence of the script against the audio, and carry old event times onto a re-cut audio.

    .venv/bin/python scripts/video/sentences.py <video_dir> <script.txt> [--old-words W --old-events E]

Reads `words.json` (from align_script.py) in `video_dir`, splits the script into sentences and gives
each the time of its first and last spoken word. Writes `sentences.json`.

When the audio is a re-cut of an earlier mix (pauses shortened, nothing reordered), the earlier
mix's effects are baked into it at moved times. Given the earlier alignment's `words.json` and the
mix's `.events.json`, each event is carried across by interpolating between words matched in both
alignments, and written to `events_mapped.json`, so the new cut can land its pictures on them.
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_script import BREAK_TAG, normalise  # noqa: E402

# A sentence ends at . ! ? or … , with any closing quote or bracket kept on it, before a space and the
# start of the next sentence (a capital, a digit, or an opening quote).
SENTENCE_END = re.compile(r"[.!?…][\"”’)\]]*(?=\s+[\"“‘(\[]?[A-Z0-9])")


def _split(line: str) -> list[str]:
    parts, start = [], 0
    for match in SENTENCE_END.finditer(line):
        parts.append(line[start:match.end()].strip())
        start = match.end()
    parts.append(line[start:].strip())
    return [part for part in parts if part]


def script_sentences(text: str) -> list[str]:
    """The script's sentences in order, break tags removed, each line split at sentence ends."""
    out: list[str] = []
    for line in BREAK_TAG.sub("", text).splitlines():
        if line.strip():
            out += _split(line.strip())
    return out


def _tokens(sentences: list[str]) -> tuple[list[str], list[int]]:
    tokens, owner = [], []
    for index, sentence in enumerate(sentences):
        for raw in sentence.split():
            token = normalise(raw)
            if token:
                tokens.append(token)
                owner.append(index)
    return tokens, owner


def _matches(a: list[str], b: list[str]) -> dict[int, int]:
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pairs: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for step in range(block.size):
            pairs[block.a + step] = block.b + step
    return pairs


def time_sentences(sentences: list[str], words: list[dict], duration: float) -> list[dict]:
    """Each sentence's start and end from its matched words; an unheard one borrows its neighbours' gap.

    Sentences butt against each other: a sentence runs from its first word to the next one's first
    word, so the pictures cut on speech, not halfway through a pause.
    """
    tokens, owner = _tokens(sentences)
    heard = [normalise(w["word"]) for w in words]
    pairs = _matches(tokens, heard)
    out = []
    for index, sentence in enumerate(sentences):
        spots = [pairs[p] for p, o in enumerate(owner) if o == index and p in pairs]
        total = sum(1 for o in owner if o == index)
        out.append({
            "index": index, "text": sentence,
            "start": words[min(spots)]["start"] if spots else None,
            "spoken_end": words[max(spots)]["end"] if spots else None,
            "matched": round(len(spots) / total, 2) if total else 0.0,
        })
    # Unheard sentences take the time between the neighbours that were heard.
    for index, entry in enumerate(out):
        if entry["start"] is None:
            before = next((out[i]["spoken_end"] for i in range(index - 1, -1, -1) if out[i]["spoken_end"] is not None), 0.0)
            entry["start"], entry["spoken_end"] = before, before
    # Starts must rise; a mismatch that puts one earlier than its predecessor is held at it.
    for earlier, later in zip(out, out[1:]):
        later["start"] = max(later["start"], earlier["start"])
    out[0]["start"] = 0.0
    for earlier, later in zip(out, out[1:]):
        earlier["end"] = later["start"]
    out[-1]["end"] = duration
    for entry in out:
        entry["start"], entry["end"] = round(entry["start"], 3), round(entry["end"], 3)
    return out


def time_map(old_words: list[dict], new_words: list[dict]) -> list[tuple[float, float]]:
    """(old time, new time) for every word heard in both, in order: a map from one cut to the other."""
    old = [normalise(w["word"]) for w in old_words]
    new = [normalise(w["word"]) for w in new_words]
    pairs = _matches(old, new)
    points = sorted((old_words[i]["start"], new_words[j]["start"]) for i, j in pairs.items())
    clean: list[tuple[float, float]] = []
    for old_t, new_t in points:  # keep the map monotonic: a matched word earlier in the new cut is noise
        if not clean or (old_t > clean[-1][0] and new_t >= clean[-1][1]):
            clean.append((old_t, new_t))
    return clean


def map_time(t: float, points: list[tuple[float, float]], max_gap: float = 30.0) -> float | None:
    """Carry a time across by interpolating between the matched words around it.

    Inside a pause that was shortened, the time is placed proportionally. Returns None when the
    nearest matched words are too far apart to trust.
    """
    if not points:
        return None
    olds = [p[0] for p in points]
    k = bisect.bisect_left(olds, t)
    if k == 0:
        return max(0.0, points[0][1] - (points[0][0] - t))
    if k == len(points):
        return points[-1][1] + (t - points[-1][0])
    (o1, n1), (o2, n2) = points[k - 1], points[k]
    if o2 - o1 > max_gap:
        return None
    share = (t - o1) / (o2 - o1) if o2 > o1 else 0.0
    return n1 + share * (n2 - n1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("script")
    parser.add_argument("--old-words", default="")
    parser.add_argument("--old-events", default="")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    words = json.loads((video_dir / "words.json").read_text(encoding="utf-8"))
    lines = json.loads((video_dir / "lines.json").read_text(encoding="utf-8"))
    duration = max(line["end"] for line in lines)
    duration = max(duration, words[-1]["end"] if words else 0.0)
    sentences = time_sentences(script_sentences(Path(args.script).read_text(encoding="utf-8")), words, duration)
    (video_dir / "sentences.json").write_text(json.dumps(sentences, ensure_ascii=False, indent=1), encoding="utf-8")
    low = sum(1 for s in sentences if s["matched"] < 0.5)
    lengths = sorted(s["end"] - s["start"] for s in sentences)
    print(f"{len(sentences)} sentences; median {lengths[len(lengths) // 2]:.1f}s; {low} heard poorly")

    if args.old_words and args.old_events:
        points = time_map(json.loads(Path(args.old_words).read_text(encoding="utf-8")), words)
        events = json.loads(Path(args.old_events).read_text(encoding="utf-8"))
        mapped = []
        for event in events:
            new_t = map_time(float(event["t"]), points)
            if new_t is not None:
                mapped.append(dict(event, t_old=event["t"], t=round(new_t, 3)))
        (video_dir / "events_mapped.json").write_text(json.dumps(mapped, indent=1), encoding="utf-8")
        print(f"{len(mapped)} of {len(events)} effects carried onto the new cut; "
              f"{len(points)} matched words in the time map")


if __name__ == "__main__":
    main()

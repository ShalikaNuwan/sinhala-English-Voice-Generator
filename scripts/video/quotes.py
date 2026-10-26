"""People's own words: find the quotes in the narration, decide how each is shown, and time it to the word.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/quotes.py <video_dir>

Reads `sentences.json`, `words.json` (align stage) and `bible.json` (story_bible.py).

Code finds every span between quotation marks in the script, including a quote that runs over more than
one sentence, and times it to the words actually spoken, so a quote card can appear exactly when the
quote begins rather than when its sentence does. A model then classifies each span by its id only: is
it speech, written words (an engraving, an inscription, a document), or just a term or nickname; who
said or wrote it, in what setting; and is it sexual, threatening or graphic. The model never returns
the quote's text, so the words shown on screen are always the script's own.

Code then decides how each quote is shown: speech as a quote card (sensitive speech stays in the
narration unless the owner switches it on in the storyboard), written words as an exhibit, terms not at
all; at most one card in any 20 seconds unless two quotes are a direct exchange. Writes `quotes.json`.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from align_script import normalise  # noqa: E402
from app.config import settings  # noqa: E402

OPEN, CLOSE, STRAIGHT = "“", "”", '"'
MAX_SENTENCES = 4          # a quote still open after this many sentences is closed where its first sentence ends
LINE_CHARS = 40            # a card line holds about this many characters
PAGE_LINES = 3             # at most this many lines on screen at once
CARD_GAP_S = 20.0          # at most one card in this long, unless two quotes answer each other
HOLD_S = 1.5               # a card stays this long after its last word


# ------------------------------------------------------------------------------------ finding

def find_spans(sentences: list[dict]) -> list[dict]:
    """Every quoted span in reading order: where it opens and closes (sentence, character), its text,
    and the words before it in its first sentence (which usually name the speaker)."""
    spans: list[dict] = []
    open_at: tuple[int, int] | None = None
    for s in sentences:
        text, i = s["text"], s["index"]
        if open_at and i - open_at[0] >= MAX_SENTENCES:
            first = sentences[open_at[0]]
            spans.append(_span(sentences, open_at, (open_at[0], len(first["text"])), closed=False))
            open_at = None
        for pos, ch in enumerate(text):
            opening = ch == OPEN or (ch == STRAIGHT and open_at is None)
            closing = ch == CLOSE or (ch == STRAIGHT and open_at is not None)
            if opening and open_at is None:
                open_at = (i, pos + 1)
            elif closing and open_at is not None:
                spans.append(_span(sentences, open_at, (i, pos), closed=True))
                open_at = None
    if open_at:
        first = sentences[open_at[0]]
        spans.append(_span(sentences, open_at, (open_at[0], len(first["text"])), closed=False))
    for n, span in enumerate(spans, start=1):
        span["id"] = f"q{n:03d}"
    return [s for s in spans if s["text"]]


def _span(sentences: list[dict], start: tuple[int, int], end: tuple[int, int], closed: bool) -> dict:
    (a, pa), (b, pb) = start, end
    if a == b:
        text = sentences[a]["text"][pa:pb]
    else:
        parts = [sentences[a]["text"][pa:]] + [sentences[k]["text"] for k in range(a + 1, b)] + [sentences[b]["text"][:pb]]
        text = " ".join(parts)
    return {"first": a, "last": b, "open_char": pa, "close_char": pb, "text": text.strip(), "closed": closed,
            "lead": sentences[a]["text"][:max(0, pa - 1)].strip()}


# ------------------------------------------------------------------------------------- timing

def script_tokens(sentences: list[dict]) -> list[dict]:
    """Every word of the script with the sentence and character it starts at."""
    tokens = []
    for s in sentences:
        for m in re.finditer(r"\S+", s["text"]):
            norm = normalise(m.group())
            if norm:
                tokens.append({"sentence": s["index"], "char": m.start(), "end_char": m.end(), "norm": norm})
    return tokens


def token_times(tokens: list[dict], words: list[dict]) -> list[tuple[float, float] | None]:
    """The spoken start and end of each script word that whisper heard; None where it did not."""
    heard = [normalise(w["word"]) for w in words]
    matcher = difflib.SequenceMatcher(None, [t["norm"] for t in tokens], heard, autojunk=False)
    times: list[tuple[float, float] | None] = [None] * len(tokens)
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            w = words[block.b + k]
            times[block.a + k] = (float(w["start"]), float(w["end"]))
    return times


def inside(token: dict, span: dict) -> bool:
    key = (token["sentence"], token["char"])
    return (span["first"], span["open_char"] - 1) <= key and (token["sentence"], token["end_char"]) <= (span["last"], span["close_char"] + 1)


def _greedy(words: list[str], width: int) -> list[str]:
    out, line = [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


def lines_of(text: str, width: int | None = None) -> list[str]:
    """Break a quote into card lines at word boundaries, balanced so no line is left with a stray word:
    as many lines as a plain wrap needs, but each about the same length."""
    width = width or LINE_CHARS
    words = text.split()
    plain = _greedy(words, width)
    if len(plain) < 2:
        return plain
    even = max(max(len(w) for w in words), -(-len(text) // len(plain)))
    for target in range(even, width + 1):
        balanced = _greedy(words, target)
        if len(balanced) <= len(plain):
            return balanced
    return plain


def pages_of(count: int, per_page: int | None = None) -> list[list[int]]:
    """Line indices grouped into pages of about equal size, never more than `per_page`."""
    per_page = per_page or PAGE_LINES
    if count <= 0:
        return []
    n = -(-count // per_page)
    size, extra = divmod(count, n)
    out, start = [], 0
    for k in range(n):
        length = size + (1 if k < extra else 0)
        out.append(list(range(start, start + length)))
        start += length
    return out


def time_span(span: dict, tokens: list[dict], times: list, sentences: list[dict]) -> None:
    """Give a span its spoken start and end, and each card line the time its first word is said."""
    idx = [k for k, t in enumerate(tokens) if inside(t, span)]
    heard = [times[k] for k in idx if times[k]]
    first_s, last_s = sentences[span["first"]], sentences[span["last"]]
    span["start"] = round(heard[0][0], 3) if heard else first_s["start"]
    span["end"] = round(heard[-1][1], 3) if heard else last_s["end"]
    span["word_count"] = len(idx)
    span["heard"] = round(len(heard) / len(idx), 2) if idx else 0.0
    # Lines take the time of their first word; an unheard one is spaced evenly between its neighbours.
    lines = lines_of(span["text"])
    starts, cursor = [], 0
    for line in lines:
        k = idx[cursor] if cursor < len(idx) else None
        starts.append(times[k][0] if k is not None and times[k] else None)
        cursor += len(line.split())
    known = [(n, t) for n, t in enumerate(starts) if t is not None]
    for n, t in enumerate(starts):
        if t is None:
            before = max(((m, x) for m, x in known if m < n), default=(-1, span["start"]))
            after = min(((m, x) for m, x in known if m > n), default=(len(lines), span["end"]))
            starts[n] = before[1] + (after[1] - before[1]) * (n - before[0]) / max(1, after[0] - before[0])
    starts[0] = span["start"]
    span["lines"] = [{"text": line, "at": round(max(span["start"], t), 3)} for line, t in zip(lines, starts)]
    span["pages"] = pages_of(len(lines))


# ------------------------------------------------------------------------------------- the model

class QuoteCall(BaseModel):
    id: str = Field(description="The quote id given, copied exactly.")
    kind: Literal["spoken", "written", "term"] = Field(
        description="spoken: words someone said. written: words written, engraved, printed or signed. "
                    "term: a nickname, label or single word in quotation marks, not a statement.")
    speaker: str = Field(default="", description="Who said or wrote it, full name as the story bible has it; empty if unknown.")
    speaker_group: Literal["perpetrator", "victim", "family", "police", "legal", "witness", "other", "unknown"] = "unknown"
    setting: str = Field(default="", description="Where or to whom, ONLY as the narration states it, e.g. 'to detectives'. Empty if not stated.")
    date: str = Field(default="", description="When, ONLY as the narration states it. Empty if not stated.")
    sensitive: bool = Field(description="True if the words are sexual, threatening, graphic or degrading.")
    reason: str = Field(default="", description="Half a sentence on the call.")


class QuoteCalls(BaseModel):
    quotes: list[QuoteCall]


SYSTEM = """You classify the quotations in the narration of a true-crime documentary, so the film can
show people's own words on screen. For each quote id: is it speech (spoken), written words (written:
an engraving, an inscription, a signature, a letter, a document), or just a term or nickname in
quotation marks (term)? Who said or wrote it, using the people of the story bible and the sentences
around it (a quote often names its speaker in a nearby sentence, not its own)? Give the setting and date
only when the narration states them. Mark sensitive anything sexual, threatening, graphic or
degrading. Never rewrite or repeat the quote."""


def classify(client, model: str, spans: list[dict], sentences: list[dict], bible: dict) -> dict[str, QuoteCall]:
    people = "\n".join(f"  {p['name']}: {p['role']}" + (" (victim)" if p.get("victim") else "") for p in bible.get("people", []))
    blocks = []
    for span in spans:
        a, b = max(0, span["first"] - 3), min(len(sentences) - 1, span["last"] + 3)
        context = " ".join(f"({k}) {sentences[k]['text']}" for k in range(a, b + 1))
        blocks.append(f"[{span['id']}] quote: “{span['text']}”\n  in context: {context}")
    user = f"People in the story:\n{people}\n\nQuotes:\n\n" + "\n\n".join(blocks)
    response = client.responses.parse(model=model, input=[{"role": "system", "content": SYSTEM},
                                                          {"role": "user", "content": user}], text_format=QuoteCalls)
    return {q.id: q for q in response.output_parsed.quotes}


# ------------------------------------------------------------------------------------- the rules

def decide(spans: list[dict], calls: dict[str, QuoteCall]) -> None:
    """How each quote is shown, decided in code from the model's classification."""
    last_card: dict | None = None
    for span in spans:
        call = calls.get(span["id"])
        span.update(call.model_dump(exclude={"id"}) if call else
                    {"kind": "term", "speaker": "", "speaker_group": "unknown", "setting": "", "date": "",
                     "sensitive": False, "reason": "not classified"})
        if span["kind"] == "term":
            span["show"], span["style"] = "narration", None
            continue
        if span["kind"] == "written":
            span["show"], span["style"] = "card", "exhibit"
        else:
            span["style"] = "memorial" if span["speaker_group"] in ("victim", "family") else "card"
            span["show"] = "narration" if span["sensitive"] else "card"
        if span["show"] != "card":
            continue
        if last_card and span["start"] - last_card["end"] < CARD_GAP_S and not exchange(last_card, span):
            span["show"], span["held_back"] = "narration", "within 20 s of another card"
            continue
        last_card = span


def exchange(a: dict, b: dict) -> bool:
    """Two spoken quotes from different people within a few seconds: a question and its answer."""
    return (a.get("kind") == b.get("kind") == "spoken" and a.get("speaker") != b.get("speaker")
            and b["start"] - a["end"] < 8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir")
    parser.add_argument("--model", default=settings.text_model)
    parser.add_argument("--retime", action="store_true",
                        help="re-time and re-wrap the saved quotes.json without calling the model")
    args = parser.parse_args()

    from openai import OpenAI

    video_dir = Path(args.video_dir)
    sentences = json.loads((video_dir / "sentences.json").read_text(encoding="utf-8"))
    words = json.loads((video_dir / "words.json").read_text(encoding="utf-8"))
    tokens = script_tokens(sentences)
    times = token_times(tokens, words)
    if args.retime:
        saved = json.loads((video_dir / "quotes.json").read_text(encoding="utf-8"))
        for span in saved:
            time_span(span, tokens, times, sentences)
        (video_dir / "quotes.json").write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"re-timed {len(saved)} quotes")
        return
    bible = json.loads((video_dir / "bible.json").read_text(encoding="utf-8"))
    spans = find_spans(sentences)
    for span in spans:
        time_span(span, tokens, times, sentences)
    print(f"{len(spans)} quoted spans; classifying with {args.model}")
    calls = classify(OpenAI(api_key=settings.openai_api_key), args.model, spans, sentences, bible) if spans else {}
    decide(spans, calls)
    (video_dir / "quotes.json").write_text(json.dumps(spans, ensure_ascii=False, indent=1), encoding="utf-8")
    for s in spans:
        m, sec = divmod(int(s["start"]), 60)
        flag = " SENSITIVE" if s.get("sensitive") else ""
        print(f"  {s['id']} {m:3d}:{sec:02d} {s['kind']:<7} {s['show']:<9} {s.get('speaker') or '?':<24}{flag}  “{s['text'][:60]}”")


if __name__ == "__main__":
    main()

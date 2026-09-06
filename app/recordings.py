"""Finds evidence audio embedded in a creator recording so it can be kept instead of re-voiced.

Pure functions over diarization spans: dictionaries with `speaker`, `start`, `end` (seconds) and
`text`. When a narrator reference clip was supplied, the narrator's spans are labelled NARRATOR.
The narrator speaks Sinhala; embedded recordings are English. The diarizer sometimes writes Sinhala
in another Indic script and sometimes mislabels a short English word inside a recording as the
narrator, so the rules below lean on script, not on the speaker label alone.
"""

from __future__ import annotations

import re


NARRATOR = "narrator"

# The narrator reference clip cut from the delivery sample; the API accepts 2-10 s.
REFERENCE_MIN_S = 2.0
REFERENCE_MAX_S = 10.0
# A recording stays open across its own silences; only narrator speech or an implausible gap ends it.
MAX_INTERNAL_GAP_MS = 10_000
# Anything shorter is a stray label, not a recording worth keeping.
MIN_SPAN_MS = 1500
# Room into the surrounding silence so a cut never clips a word.
PAD_MS = 150

Piece = tuple[int, int, str]

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_PUNCTUATION = ".,!?;:\"'“”‘’()[]—–-"


def has_foreign_script(text: str) -> bool:
    """True when the text has letters outside the Latin range: the narrator speaking Sinhala."""
    return any(ch.isalpha() and ord(ch) >= 0x0250 for ch in text or "")


def _ms(seconds: float) -> int:
    return round(float(seconds) * 1000)


def choose_reference(spans: list[dict]) -> tuple[float, float] | None:
    """The dominant speaker's longest span, trimmed to what the API accepts as a reference."""
    totals: dict[str, float] = {}
    for item in spans:
        speaker = item.get("speaker") or "?"
        totals[speaker] = totals.get(speaker, 0.0) + max(0.0, float(item["end"]) - float(item["start"]))
    if not totals:
        return None
    dominant = max(totals, key=totals.get)
    longest = max(
        (item for item in spans if (item.get("speaker") or "?") == dominant),
        key=lambda item: float(item["end"]) - float(item["start"]),
    )
    start, end = float(longest["start"]), float(longest["end"])
    if end - start < REFERENCE_MIN_S:
        return None
    return start, min(end, start + REFERENCE_MAX_S)


def narration_spans(spans: list[dict]) -> list[tuple[int, int]]:
    """Where the narrator is actually heard, in ms relative to the chunk."""
    return sorted(
        (_ms(item["start"]), _ms(item["end"]))
        for item in spans
        if has_foreign_script(item.get("text") or "")
    )


def original_spans(spans: list[dict], chunk_ms: int) -> list[tuple[int, int]]:
    """Embedded recordings in ms relative to the chunk.

    A recording opens at a Latin-only or empty span by someone other than the narrator and stays
    open across silences and across any Latin-only spans, however they are labelled, until the
    narrator speaks (non-Latin letters) or the gap exceeds MAX_INTERNAL_GAP_MS. The review gate is
    the guard against a diarizer that writes Sinhala in Latin letters.
    """
    found: list[list[int]] = []
    current: list[int] | None = None

    def close() -> None:
        nonlocal current
        if current is not None and current[1] - current[0] >= MIN_SPAN_MS:
            found.append(current)
        current = None

    for item in sorted(spans, key=lambda item: float(item["start"])):
        start, end = _ms(item["start"]), _ms(item["end"])
        end = max(start, end)
        if has_foreign_script(item.get("text") or ""):
            close()
            continue
        other = item.get("speaker") != NARRATOR
        if current is None:
            if other:
                current = [start, end]
            continue
        if start - current[1] > MAX_INTERNAL_GAP_MS:
            close()
            if other:
                current = [start, end]
            continue
        current[1] = max(current[1], end)
    close()
    return [(max(0, start - PAD_MS), min(chunk_ms, end + PAD_MS)) for start, end in found]


def cut_plan(
    chunk_start_ms: int,
    chunk_end_ms: int,
    originals: list[tuple[int, int]],
    narration: list[tuple[int, int]],
) -> list[Piece]:
    """Split a chunk into ordered narration and original pieces, in absolute source time.

    Narration pieces survive only where the narrator is heard; silence next to a recording belongs
    to the recording. Chunk bounds are absolute source ms; `originals` and `narration` are
    chunk-relative ms; the result is absolute.
    """
    length = chunk_end_ms - chunk_start_ms
    if not originals:
        return [(chunk_start_ms, chunk_end_ms, "narration")]

    def has_narration(start: int, end: int) -> bool:
        return any(s < end and e > start for s, e in narration)

    pieces: list[list] = []
    cursor = 0
    for start, end in sorted(originals):
        start, end = max(0, start), min(length, end)
        if end <= start:
            continue
        if start > cursor:
            pieces.append([cursor, start, "narration"])
        pieces.append([start, end, "original"])
        cursor = max(cursor, end)
    if cursor < length:
        pieces.append([cursor, length, "narration"])

    merged: list[list] = []
    for start, end, kind in pieces:
        if kind == "narration" and not has_narration(start, end):
            if merged and merged[-1][2] == "original":
                merged[-1][1] = max(merged[-1][1], end)  # trailing silence joins the recording before it
                continue
            kind = "original"  # leading silence joins the recording after it
        if kind == "original" and merged and merged[-1][2] == "original":
            merged[-1][1] = max(merged[-1][1], end)
            continue
        merged.append([start, end, kind])
    return [(chunk_start_ms + s, chunk_start_ms + e, k) for s, e, k in merged]


def clip_text(spans: list[dict], start_ms: int, end_ms: int) -> str:
    """What is said inside a recording, for the reviewer."""
    parts: list[str] = []
    for item in sorted(spans, key=lambda item: float(item["start"])):
        inside = _ms(item["end"]) > start_ms and _ms(item["start"]) < end_ms
        text = (item.get("text") or "").strip()
        if inside and text and not has_foreign_script(text):
            parts.append(text)
    return " ".join(parts)


def english_run(text: str, minimum_words: int = 8) -> bool:
    """True when a transcript contains a run of English words long enough to be a recording."""
    run = 0
    for token in (text or "").split():
        word = token.strip(_PUNCTUATION)
        if has_foreign_script(word):
            run = 0
        elif _LATIN_WORD.fullmatch(word):
            run += 1
            if run >= minimum_words:
                return True
    return False

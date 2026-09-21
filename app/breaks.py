"""Turns the narrator's own silences into break marks for a script voiced elsewhere.

The diarizer hears the Sinhala as phrases with times. Phrases that follow one another with almost
no gap are one flowing run; a real pause ends a run. Each English line renders one run (or a merged
pair), so the silence after the run becomes the break after the line, written the way ElevenLabs
Multilingual v2 reads it: <break time="0.8s" />. Pure functions; the pipeline stores, the export
writes.
"""

from __future__ import annotations


# Phrases closer than this are one flowing run.
FLOW_GAP_MS = 150
# An English sentence end carries more weight than a Sinhala run-on; give it at least this much.
SENTENCE_MIN_MS = 400
# ElevenLabs' ceiling for one break tag.
BREAK_MAX_MS = 3000
# The ceiling when the diarizer heard too little of a segment to trust its silences.
GUARDED_MAX_MS = 1000
# ElevenLabs v2 already pauses about this long at a comma; a smaller mark only spends the tag budget.
MIN_MARK_MS = 300
# For segments without phrase data.
DEFAULT_SENTENCE_MS = 500
DEFAULT_PARAGRAPH_MS = 900
# Under ElevenLabs v2's 10,000-character generation limit, and few enough marks to stay stable.
PART_MAX_CHARS = 9000
PART_MAX_BREAKS = 40

_CLOSERS = "\"'”’)]"


def runs(phrases: list[dict], segment_end_ms: int, next_phrase_start_ms: int | None = None) -> list[dict]:
    """Group phrases into flowing runs and measure the silence after each."""
    ordered = sorted((p for p in phrases if (p.get("text") or "").strip()), key=lambda p: p["start_ms"])
    result: list[dict] = []
    for item in ordered:
        text = item["text"].strip()
        if result and item["start_ms"] - result[-1]["end_ms"] < FLOW_GAP_MS:
            result[-1]["end_ms"] = max(result[-1]["end_ms"], item["end_ms"])
            result[-1]["text"] = f"{result[-1]['text']} {text}"
        else:
            result.append({"start_ms": item["start_ms"], "end_ms": item["end_ms"], "text": text, "gap_after_ms": 0})
    for earlier, later in zip(result, result[1:]):
        earlier["gap_after_ms"] = max(0, later["start_ms"] - earlier["end_ms"])
    if result:
        tail_to = next_phrase_start_ms if next_phrase_start_ms is not None else segment_end_ms
        result[-1]["gap_after_ms"] = max(0, tail_to - result[-1]["end_ms"])
    return result


def coverage(phrases: list[dict], start_ms: int, end_ms: int) -> float:
    """How much of the narrator's stretch was actually heard as speech, 0 to 1.

    The stretch runs from the first phrase to the last, clamped to the segment, so silence before
    the narrator starts (after a recording, say) or after they finish does not count against them;
    only holes between phrases do. An overlap is counted once. No phrases, or a segment with no
    length, gives 0.0 and 1.0 respectively.
    """
    if end_ms - start_ms <= 0:
        return 1.0
    ordered = sorted(phrases, key=lambda item: item["start_ms"])
    if not ordered:
        return 0.0
    first = max(start_ms, min(item["start_ms"] for item in ordered))
    last = min(end_ms, max(item["end_ms"] for item in ordered))
    length = last - first
    if length <= 0:
        return 1.0
    heard = 0
    reached = first
    for phrase in ordered:
        begin, finish = max(first, phrase["start_ms"], reached), min(last, phrase["end_ms"])
        if finish > begin:
            heard += finish - begin
            reached = finish
    return heard / length


def ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip(_CLOSERS).endswith((".", "!", "?", "…"))


def break_seconds(gap_ms: int, line_text: str, cap_ms: int = BREAK_MAX_MS) -> float | None:
    """The break to write after a line, from the measured silence; None when punctuation is enough."""
    if ends_sentence(line_text):
        gap_ms = max(gap_ms, SENTENCE_MIN_MS)
    if gap_ms < MIN_MARK_MS:
        return None
    # Python rounds half to even on exact quarter-seconds (250 ms -> 0.2); the 50 ms is inaudible.
    return round(min(gap_ms, cap_ms) / 1000, 1)


def mark(seconds: float) -> str:
    return f'<break time="{seconds:.1f}s" />'


def covers_runs(lines: list[dict], run_count: int) -> bool:
    """Every run rendered exactly once, in order."""
    if any(not line.get("runs") for line in lines):
        return False
    used = [number for line in lines for number in line.get("runs", [])]
    return used == list(range(1, run_count + 1))


def attach_breaks(lines: list[dict], run_list: list[dict], cap_ms: int = BREAK_MAX_MS) -> list[dict]:
    """Give each line the times and trailing silence of the runs it covers, and the break to write.

    The measured silence is always kept as it was; only the mark written from it obeys `cap_ms`.
    """
    result = []
    for line in lines:
        covered = [run_list[number - 1] for number in line.get("runs", []) if 1 <= number <= len(run_list)]
        covered.sort(key=lambda run: run["start_ms"])
        gap = covered[-1]["gap_after_ms"] if covered else 0
        text = (line.get("text") or "").strip()
        result.append({
            "text": text,
            "runs": list(line.get("runs", [])),
            "start_ms": covered[0]["start_ms"] if covered else None,
            "end_ms": covered[-1]["end_ms"] if covered else None,
            "gap_after_ms": gap,
            "break_s": break_seconds(gap, text, cap_ms),
        })
    return result


def default_lines(narration_text: str, start_ms: int, end_ms: int) -> list[dict]:
    """Lines for a segment without phrase data: one per paragraph, default breaks, time shared by length."""
    paragraphs = [p.strip() for p in (narration_text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    total = sum(len(p) for p in paragraphs) or 1
    cursor = start_ms
    result = []
    for position, text in enumerate(paragraphs):
        span = round((end_ms - start_ms) * len(text) / total)
        gap = DEFAULT_SENTENCE_MS if position == len(paragraphs) - 1 else DEFAULT_PARAGRAPH_MS
        result.append({"text": text, "runs": [], "start_ms": cursor, "end_ms": cursor + span, "gap_after_ms": gap, "break_s": break_seconds(gap, text)})
        cursor += span
    result[-1]["end_ms"] = end_ms
    return result


def render(lines: list[dict]) -> str:
    """Script text with a mark after every line that has a break."""
    parts = []
    for line in lines:
        text = " ".join(line["text"].replace("<", "").replace(">", "").split())
        parts.append(f"{text} {mark(line['break_s'])}\n" if line.get("break_s") else f"{text}\n")
    return "".join(parts)


def split_parts(lines: list[dict]) -> list[list[dict]]:
    """Split lines into generation-sized parts by character count and by number of marks."""
    parts: list[list[dict]] = []
    current: list[dict] = []
    chars = marks = 0
    for line in lines:
        has_mark = 1 if line.get("break_s") else 0
        cost = len(line["text"]) + 1 + (22 if has_mark else 0)
        if current and (chars + cost > PART_MAX_CHARS or marks + has_mark > PART_MAX_BREAKS):
            parts.append(current)
            current, chars, marks = [], 0, 0
        current.append(line)
        chars += cost
        marks += has_mark
    if current:
        parts.append(current)
    return parts

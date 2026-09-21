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


def ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip(_CLOSERS).endswith((".", "!", "?", "…"))


def break_seconds(gap_ms: int, line_text: str) -> float | None:
    """The break to write after a line, from the measured silence; None when punctuation is enough."""
    if gap_ms < FLOW_GAP_MS:
        return None
    if ends_sentence(line_text):
        gap_ms = max(gap_ms, SENTENCE_MIN_MS)
    return round(min(gap_ms, BREAK_MAX_MS) / 1000, 1)


def mark(seconds: float) -> str:
    return f'<break time="{seconds:.1f}s" />'


def covers_runs(lines: list[dict], run_count: int) -> bool:
    """Every run rendered exactly once, in order."""
    used = [number for line in lines for number in line.get("runs", [])]
    return used == list(range(1, run_count + 1))


def attach_breaks(lines: list[dict], run_list: list[dict]) -> list[dict]:
    """Give each line the times and trailing silence of the runs it covers, and the break to write."""
    result = []
    for line in lines:
        covered = [run_list[number - 1] for number in line.get("runs", []) if 1 <= number <= len(run_list)]
        gap = covered[-1]["gap_after_ms"] if covered else 0
        text = (line.get("text") or "").strip()
        result.append({
            "text": text,
            "runs": list(line.get("runs", [])),
            "start_ms": covered[0]["start_ms"] if covered else None,
            "end_ms": covered[-1]["end_ms"] if covered else None,
            "gap_after_ms": gap,
            "break_s": break_seconds(gap, text),
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
    if result:
        result[-1]["end_ms"] = end_ms
    return result


def render(lines: list[dict]) -> str:
    """Script text with a mark after every line that has a break."""
    return "".join(
        f"{line['text']} {mark(line['break_s'])}\n" if line.get("break_s") else f"{line['text']}\n"
        for line in lines
    )


def split_parts(lines: list[dict]) -> list[list[dict]]:
    """Split lines into generation-sized parts by character count and by number of marks."""
    parts: list[list[dict]] = []
    current: list[dict] = []
    chars = marks = 0
    for line in lines:
        has_mark = 1 if line.get("break_s") else 0
        cost = len(line["text"]) + 1 + (20 if has_mark else 0)
        if current and (chars + cost > PART_MAX_CHARS or marks + has_mark > PART_MAX_BREAKS):
            parts.append(current)
            current, chars, marks = [], 0, 0
        current.append(line)
        chars += cost
        marks += has_mark
    if current:
        parts.append(current)
    return parts

"""Mends sentences that a segment cut split in two, in the English only.

Segments are cut at silences, and a cut sometimes falls inside a sentence: its start ends one
segment and its end begins the next. A model reads the two sides of each cut and names the smaller
half to move, separately for the narration and the faithful English, because the adaptation may
already have carried a fragment across in one text and not the other. A move is mechanical: the
fragment must be an exact run of whole words at the edge it comes from. Often the cut instead left a
broken sentence at the end of one segment that the next one finishes or repeats; a merge replaces the
two with one sentence, which may only use their words (no new name, number or other word of four
letters or more). The audio, its times and the Sinhala transcript never change.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Literal

from pydantic import BaseModel, Field

from . import breaks, recordings
from .database import Database, utc_now


REVOICE_NOTE = "Text changed by the sentence fix; regenerate the voice"
# How much faithful English the model sees on each side of a cut.
FAITHFUL_EXCERPT_CHARS = 400
EDGE_LINES = 2

Direction = Literal["back", "forward", "merge", ""]
Place = Literal["a", "b", ""]
# The two texts a move can change, as the segment columns that hold them.
TEXTS = (("narration", "narration_en"), ("faithful", "faithful_en"))


class SplitDecision(BaseModel):
    narration_direction: Direction = Field(
        default="",
        description="back: move the start of segment B onto the end of A; forward: move the end of A onto the start "
                    "of B; merge: A ends with a broken sentence that B finishes or repeats, so the two become one "
                    "sentence; empty when the narration is not split at this cut.",
    )
    fragment_narration: str = Field(
        default="",
        description="back/forward: the narration words to move, copied exactly. merge: the one joined sentence. "
                    "Empty when not split.",
    )
    a_tail_narration: str = Field(default="", description="merge only: the broken sentence at the very end of A, copied exactly.")
    b_head_narration: str = Field(default="", description="merge only: the words at the very start of B that finish or repeat it, copied exactly.")
    place_narration: Place = Field(default="", description="merge only: a or b, the segment the joined sentence goes in.")
    faithful_direction: Direction = Field(default="", description="The same choice for the faithful English, decided on its own.")
    fragment_faithful: str = Field(default="", description="The faithful English words to move, or the joined sentence. Empty when not split.")
    a_tail_faithful: str = Field(default="", description="merge only: the faithful English broken sentence at the end of A, copied exactly.")
    b_head_faithful: str = Field(default="", description="merge only: the faithful English words at the start of B that finish or repeat it, copied exactly.")
    place_faithful: Place = Field(default="", description="merge only: a or b, for the faithful English.")
    reason: str = Field(default="", description="One short sentence on why.")

    @property
    def split(self) -> bool:
        return bool(self.narration_direction or self.faithful_direction)


def pairs(segments: list[dict]) -> list[tuple[dict, dict]]:
    """Neighbouring narration segments that both have English; a recording between them breaks the pair."""
    ordered = sorted(segments, key=lambda item: item["segment_index"])

    def eligible(item: dict) -> bool:
        return item.get("kind") == "narration" and bool((item.get("narration_en") or "").strip()) and bool((item.get("faithful_en") or "").strip())

    return [(a, b) for a, b in zip(ordered, ordered[1:]) if eligible(a) and eligible(b)]


def _tail(text: str, limit: int) -> str:
    """The last `limit` characters or fewer, starting on a whole word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[-limit:]
    if not text[-limit - 1].isspace() and len(cut.split(None, 1)) > 1:
        cut = cut.split(None, 1)[1]  # the first word was cut in half
    return cut.strip()


def _head(text: str, limit: int) -> str:
    """The first `limit` characters or fewer, ending on a whole word."""
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if not text[limit].isspace() and len(cut.rsplit(None, 1)) > 1:
        cut = cut.rsplit(None, 1)[0]  # the last word was cut in half
    return cut.strip()


def _paragraphs(item: dict) -> list[str]:
    if item.get("lines"):
        return [line["text"] for line in item["lines"]]
    return [p.strip() for p in (item.get("narration_en") or "").split("\n\n") if p.strip()]


def excerpts(a: dict, b: dict) -> tuple[str, str, str, str]:
    """What the model sees of a cut: the last lines and faithful tail of A, the first lines and faithful head of B."""
    return (
        "\n\n".join(_paragraphs(a)[-EDGE_LINES:]),
        _tail(a.get("faithful_en") or "", FAITHFUL_EXCERPT_CHARS),
        "\n\n".join(_paragraphs(b)[:EDGE_LINES]),
        _head(b.get("faithful_en") or "", FAITHFUL_EXCERPT_CHARS),
    )


def _words(text: str) -> list[str]:
    return text.split()


def _take_front(text: str, count: int) -> tuple[str, str]:
    """The first `count` words (whitespace collapsed) and the rest of the text with its own spacing."""
    matches = list(re.finditer(r"\S+", text))
    moved = " ".join(match.group() for match in matches[:count])
    return moved, text[matches[count - 1].end():].strip()


def _take_back(text: str, count: int) -> tuple[str, str]:
    """The text without its last `count` words, keeping its own spacing, and those words."""
    matches = list(re.finditer(r"\S+", text))
    moved = " ".join(match.group() for match in matches[-count:])
    return text[:matches[-count].start()].strip(), moved


def _text(item: dict, column: str) -> str:
    """A segment's text in one column; the narration of a segment with lines is its lines."""
    if column == "narration_en" and item.get("lines"):
        return "\n\n".join(line["text"] for line in item["lines"])
    return item.get(column) or ""


def _ends_sentence(text: str, following: str) -> bool:
    """Whether a text stops at a sentence end before `following`. Closing quotes and brackets don't
    count. The translation marks a sentence the cut broke off with a trailing "..." or "[incomplete]",
    or with "..." opening the next segment, so none of those is an end."""
    if following.lstrip().startswith(("...", "…")):
        return False
    stripped = text.rstrip()
    if re.search(r"\[[^\]]*incomplete[^\]]*\]\W*$", stripped, re.IGNORECASE):
        return False
    stripped = stripped.rstrip("\"'”’)]")
    if stripped.endswith(("...", "…")):
        return False
    return stripped.endswith((".", "!", "?"))


def _check(name: str, a: dict, b: dict, column: str, direction: str, fragment: str) -> str | None:
    """Why a move of this text is refused, or None when the fragment is an exact whole-word edge."""
    wanted = _words(fragment)
    if direction not in ("back", "forward"):
        return f"{name}: no direction given for {fragment.strip()!r}"
    if not wanted:
        return f"{name}: the fragment is empty"
    if _ends_sentence(_text(a, column), _text(b, column)):
        return f"{name}: segment {a['segment_index']} already ends a sentence"
    source = b if direction == "back" else a
    have = _words(_text(source, column))
    edge = have[:len(wanted)] if direction == "back" else have[-len(wanted):]
    where = f"{'start' if direction == 'back' else 'end'} of segment {source['segment_index']}"
    if edge != wanted:
        return f"{name}: {' '.join(wanted)!r} is not the {where}"
    if len(wanted) >= len(have):
        return f"{name}: the fragment is all of segment {source['segment_index']}"
    return None


def _rebreak(line: dict, text: str) -> dict:
    """An edge line with new text: the break it earns is worked out again from its measured silence, as `relines` does."""
    cap = breaks.GUARDED_MAX_MS if line.get("coverage", 1.0) < recordings.COVERAGE_MIN else breaks.BREAK_MAX_MS
    return dict(line, text=text, break_s=breaks.break_seconds(line.get("gap_after_ms") or 0, text, cap_ms=cap))


def _absorb(line: dict, emptied: list[dict], front: bool) -> dict:
    """`line` taking over the runs and the outer times of the edge lines emptied beside it."""
    # emptied[0] is the outermost line: the segment's first at the front, its last at the back.
    if front:
        runs = [run for item in emptied for run in item.get("runs", [])] + list(line.get("runs", []))
        merged = dict(line, runs=runs, start_ms=emptied[0].get("start_ms", line.get("start_ms")))
    else:
        runs = list(line.get("runs", [])) + [run for item in reversed(emptied) for run in item.get("runs", [])]
        merged = dict(line, runs=runs, end_ms=emptied[0].get("end_ms", line.get("end_ms")),
                      gap_after_ms=emptied[0].get("gap_after_ms", line.get("gap_after_ms")))
    return _rebreak(merged, merged["text"])


def _cut_lines(lines: list[dict], front: bool, count: int) -> tuple[list[dict], str]:
    """Take `count` words off the front or the back of the lines; a line left empty goes, with its break.

    Its spoken runs stay in the segment: they join the line that becomes the new edge, so the lines
    still cover every run and the edge line spans the audio the emptied one did.
    """
    remaining_lines = list(lines)
    taken: list[str] = []
    emptied: list[dict] = []
    remaining = count
    while remaining:
        edge = remaining_lines[0] if front else remaining_lines[-1]
        size = len(_words(edge["text"]))
        if size <= remaining:
            taken.append(" ".join(_words(edge["text"])))
            emptied.append(remaining_lines.pop(0 if front else -1))
            remaining -= size
            if not remaining and emptied and remaining_lines:
                remaining_lines[0 if front else -1] = _absorb(remaining_lines[0 if front else -1], emptied, front)
            continue
        if emptied:
            edge = _absorb(edge, emptied, front)
        if front:
            moved, rest = _take_front(edge["text"], remaining)
            remaining_lines[0] = _rebreak(edge, rest)
        else:
            rest, moved = _take_back(edge["text"], remaining)
            remaining_lines[-1] = _rebreak(edge, rest)
        taken.append(moved)
        remaining = 0
    return remaining_lines, " ".join(taken if front else reversed(taken))


def _cut(item: dict, column: str, front: bool, count: int) -> tuple[str, list[dict] | None, str]:
    """The source side of a move: its new text, its new lines (None for a text without them) and the words taken."""
    if column == "narration_en" and item.get("lines"):
        lines, moved = _cut_lines(item["lines"], front, count)
        if front and lines:
            lines[0] = dict(lines[0], text=_capitalised(lines[0]["text"]))
        return "\n\n".join(line["text"] for line in lines), lines, moved
    text = _text(item, column)
    if front:
        moved, rest = _take_front(text, count)
        return _capitalised(rest), None, moved
    rest, moved = _take_back(text, count)
    return rest, None, moved


def _capitalised(text: str) -> str:
    """A text whose start was cut away begins a sentence now, so its first letter is a capital."""
    return text[:1].upper() + text[1:] if text[:1].islower() else text


def _join(item: dict, column: str, moved: str, at_end: bool) -> tuple[str, list[dict] | None]:
    """The receiving side of a move: the words join its edge line, or its text, with a space."""
    if column == "narration_en" and item.get("lines"):
        lines = list(item["lines"])
        if at_end:
            lines[-1] = _rebreak(lines[-1], f"{lines[-1]['text'].rstrip()} {moved}")
        else:
            lines[0] = _rebreak(lines[0], f"{moved} {lines[0]['text'].lstrip()}")
        return "\n\n".join(line["text"] for line in lines), lines
    text = _text(item, column)
    return (f"{text.rstrip()} {moved}" if at_end else f"{moved} {text.lstrip()}"), None


def _plain(text: str) -> list[str]:
    """The words of a text for comparing meaning: lower case, apostrophes and punctuation gone."""
    text = re.sub(r"['\u2019]s\b", "", text.lower())  # "Liz's" is the word "Liz"
    return re.findall(r"[a-z0-9]+", text.replace("\u2019", "").replace("'", ""))


def _check_merge(name: str, a: dict, b: dict, column: str, a_tail: str, b_head: str, sentence: str, place: str) -> str | None:
    """Why a merge of this text is refused, or None. The broken half and its echo must be exact edges,
    both segments must keep some text, and the joined sentence may only use their words: every word of
    four letters or more, and every number, has to come from the two halves it replaces."""
    tail, head = _words(a_tail), _words(b_head)
    if not tail or not head or not sentence.strip():
        return f"{name}: a merge needs the broken half, the words that finish it and the joined sentence"
    if place not in ("a", "b"):
        return f"{name}: a merge needs a place, a or b"
    a_words, b_words = _words(_text(a, column)), _words(_text(b, column))
    if a_words[-len(tail):] != tail:
        return f"{name}: {a_tail.strip()!r} is not the end of segment {a['segment_index']}"
    if b_words[:len(head)] != head:
        return f"{name}: {b_head.strip()!r} is not the start of segment {b['segment_index']}"
    emptied = b if place == "a" and len(head) >= len(b_words) else a if place == "b" and len(tail) >= len(a_words) else None
    if emptied is not None:
        return f"{name}: the merge would leave segment {emptied['segment_index']} empty"
    known = set(_plain(a_tail) + _plain(b_head))

    def familiar(word: str) -> bool:
        # The same word with another ending ("introduce" for "introduced") is not a new word.
        return word in known or any(len(other) >= 4 and (other.startswith(word) or word.startswith(other)) for other in known)

    new = [word for word in _plain(sentence) if (len(word) >= 4 or any(ch.isdigit() for ch in word)) and not familiar(word)]
    if new:
        return f"{name}: the joined sentence has words not in the words it replaces: {', '.join(sorted(set(new)))}"
    return None


def _replace_edge(item: dict, column: str, front: bool, count: int, sentence: str) -> tuple[str, list[dict] | None]:
    """Put `sentence` in place of the first (front) or last `count` words. With lines, the innermost line
    the words reach keeps its other words and takes the sentence; lines wholly inside the words go, and
    their runs join it."""
    if column == "narration_en" and item.get("lines"):
        lines = list(item["lines"])
        order = range(len(lines)) if front else range(len(lines) - 1, -1, -1)
        remaining = count
        for position in order:
            size = len(_words(lines[position]["text"]))
            if size >= remaining:
                break
            remaining -= size
        text = lines[position]["text"]
        if front:
            _, rest = _take_front(text, remaining)
            joined = f"{sentence} {rest}".strip()
            emptied = lines[:position]
        else:
            rest, _ = _take_back(text, remaining)
            joined = f"{rest} {sentence}".strip()
            emptied = list(reversed(lines[position + 1:]))
        line = _absorb(lines[position], emptied, front) if emptied else lines[position]
        line = _rebreak(line, joined)
        kept = [line] + lines[position + 1:] if front else lines[:position] + [line]
        return "\n\n".join(item["text"] for item in kept), kept
    text = _text(item, column)
    if front:
        _, rest = _take_front(text, count)
        return f"{sentence} {rest}".strip(), None
    rest, _ = _take_back(text, count)
    return f"{rest} {sentence}".strip(), None


def apply_move(a: dict, b: dict, decision: SplitDecision) -> tuple[dict, dict, list[str]]:
    """The changed columns of each segment after the decision's moves, and why any move was refused.

    Pure. Each text is checked and moved on its own: a refused narration move still lets a valid
    faithful move through, and the other way round. Lines change only with the narration.
    """
    a_fields: dict = {}
    b_fields: dict = {}
    problems: list[str] = []
    for name, column in TEXTS:
        direction = getattr(decision, f"{name}_direction")
        fragment = getattr(decision, f"fragment_{name}")
        if not direction and not fragment.strip():
            continue
        if direction == "merge":
            a_tail, b_head = getattr(decision, f"a_tail_{name}"), getattr(decision, f"b_head_{name}")
            place = getattr(decision, f"place_{name}")
            problem = _check_merge(name, a, b, column, a_tail, b_head, fragment, place)
            if problem:
                problems.append(problem)
                continue
            if place == "a":
                a_text, a_lines = _replace_edge(a, column, False, len(_words(a_tail)), fragment.strip())
                b_text, b_lines, _ = _cut(b, column, True, len(_words(b_head)))
            else:
                a_text, a_lines, _ = _cut(a, column, False, len(_words(a_tail)))
                b_text, b_lines = _replace_edge(b, column, True, len(_words(b_head)), fragment.strip())
            a_fields[column], b_fields[column] = a_text, b_text
            if name == "narration":
                a_fields["lines"], b_fields["lines"] = a_lines or [], b_lines or []
            continue
        problem = _check(name, a, b, column, direction, fragment)
        if problem:
            problems.append(problem)
            continue
        count = len(_words(fragment))
        back = direction == "back"
        source, target, source_fields, target_fields = (b, a, b_fields, a_fields) if back else (a, b, a_fields, b_fields)
        source_text, source_lines, moved = _cut(source, column, back, count)
        target_text, target_lines = _join(target, column, moved, back)
        source_fields[column], target_fields[column] = source_text, target_text
        if name == "narration":
            source_fields["lines"] = source_lines or []
            target_fields["lines"] = target_lines or []
    return a_fields, b_fields, problems


def _notes(a: dict, b: dict, decision: SplitDecision, moved: list[str]) -> tuple[list[str], list[str]]:
    """What each side of the cut says about the moves, naming the text that moved."""
    by_direction: dict[str, list[str]] = {}
    for name in moved:
        by_direction.setdefault(getattr(decision, f"{name}_direction"), []).append(name)
    a_notes, b_notes = [], []
    for direction, names in by_direction.items():
        texts = " and ".join(names)
        if direction == "merge":
            for place in sorted({getattr(decision, f"place_{name}") for name in names}):
                into = [name for name in names if getattr(decision, f"place_{name}") == place]
                joined = " and ".join(into)
                if place == "a":
                    a_notes.append(f"Sentence joined across the cut with segment {b['segment_index']} ({joined})")
                    b_notes.append(f"Sentence joined across the cut into segment {a['segment_index']} ({joined})")
                else:
                    a_notes.append(f"Sentence joined across the cut into segment {b['segment_index']} ({joined})")
                    b_notes.append(f"Sentence joined across the cut with segment {a['segment_index']} ({joined})")
        elif direction == "back":
            a_notes.append(f"End of sentence moved in from segment {b['segment_index']} ({texts})")
            b_notes.append(f"End of sentence moved out to segment {a['segment_index']} ({texts})")
        else:
            a_notes.append(f"Start of sentence moved out to segment {b['segment_index']} ({texts})")
            b_notes.append(f"Start of sentence moved in from segment {a['segment_index']} ({texts})")
    return a_notes, b_notes


def _store(db: Database, segment: dict, fields: dict, notes: list[str]) -> None:
    """Write one segment's new text and notes; voiced audio no longer matches a changed narration."""
    stored = dict(segment.get("sentence_fix") or {})
    stored["notes"] = list(stored.get("notes") or []) + notes
    stored["at"] = utc_now()
    columns = {"sentence_fix_json": json.dumps(stored), "updated_at": utc_now()}
    if "narration_en" in fields:
        columns["narration_en"] = fields["narration_en"]
        columns["lines_json"] = json.dumps(fields["lines"])
    if "faithful_en" in fields:
        columns["faithful_en"] = fields["faithful_en"]
    if "narration_en" in fields and segment.get("tts_audio_path"):
        qa = dict(segment.get("qa") or {})
        qa["passed"] = False
        qa["issues"] = [issue for issue in qa.get("issues", []) if issue != REVOICE_NOTE] + [REVOICE_NOTE]
        columns |= {"qa_json": json.dumps(qa), "qa_status": "needs_review", "status": "needs_review"}
    assignments = ", ".join(f"{column}=?" for column in columns)
    db.execute(f"UPDATE segments SET {assignments} WHERE id=?", (*columns.values(), segment["id"]))


def looks_broken(a: dict, b: dict) -> bool:
    """Whether either text at this cut still reads as a broken sentence: A stops short of a sentence end
    (or ends on a break mark), or B starts in the middle of one (a break mark or a small letter)."""
    for _name, column in TEXTS:
        a_text, b_text = _text(a, column), _text(b, column).lstrip()
        if not _ends_sentence(a_text, b_text) or b_text[:1].islower():
            return True
    return False


def fix_job(
    db: Database,
    job_id: str,
    decide: Callable[[dict, dict], SplitDecision],
    warn: Callable[[str], None],
    workers: int = 6,
    on_progress: Callable[[int, int], None] | None = None,
    passes: int = 3,
) -> list[int]:
    """Mend every cut of a job's narration and return the indices of the segments that changed.

    Each pass decides its cuts in parallel on the text as it stood before the pass. The moves then
    apply in segment order on the current text, so two moves on the two ends of one segment both land,
    and a move that would empty a segment is refused. The first pass asks about every cut; a model
    sometimes misses a broken one, so later passes ask again about the cuts that still look broken,
    until a pass changes nothing. Refusals and failed decisions become warnings, each said once.
    """
    changed: set[int] = set()
    said: set[str] = set()

    def once(message: str) -> None:
        if message not in said:
            said.add(message)
            warn(message)

    for number in range(passes):
        segments = db.all("SELECT * FROM segments WHERE job_id=? ORDER BY segment_index", (job_id,))
        pair_list = pairs(segments)
        if number:
            pair_list = [(a, b) for a, b in pair_list if looks_broken(a, b)]
        if not pair_list:
            break
        mended = _one_pass(db, pair_list, segments, decide, once, workers, on_progress)
        changed |= mended
        if number and not mended:
            break
    return sorted(changed)


def _one_pass(
    db: Database,
    pair_list: list[tuple[dict, dict]],
    segments: list[dict],
    decide: Callable[[dict, dict], SplitDecision],
    warn: Callable[[str], None],
    workers: int,
    on_progress: Callable[[int, int], None] | None,
) -> set[int]:
    decisions: dict[int, SplitDecision | Exception] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(pair_list)))) as pool:
        futures = {pool.submit(decide, a, b): position for position, (a, b) in enumerate(pair_list)}
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                decisions[futures[future]] = future.result()
            except Exception as exc:  # noqa: BLE001 - one cut's model failure leaves that cut as it was
                decisions[futures[future]] = exc
            if on_progress:
                on_progress(done, len(pair_list))

    current = {segment["segment_index"]: segment for segment in segments}
    changed: set[int] = set()
    for position, (first, second) in enumerate(pair_list):
        decision = decisions[position]
        label = f"segments {first['segment_index']} and {second['segment_index']}"
        if isinstance(decision, Exception):
            warn(f"Sentence fix skipped {label}: {str(decision)[:200]}")
            continue
        a, b = current[first["segment_index"]], current[second["segment_index"]]
        a_fields, b_fields, problems = apply_move(a, b, decision)
        for problem in problems:
            warn(f"Sentence fix refused between {label}: {problem}")
        moved = [name for name, column in TEXTS if column in a_fields]
        if not moved:
            continue
        a_notes, b_notes = _notes(a, b, decision, moved)
        for segment, fields, notes in ((a, a_fields, a_notes), (b, b_fields, b_notes)):
            _store(db, segment, fields, notes)
            current[segment["segment_index"]] = db.one("SELECT * FROM segments WHERE id=?", (segment["id"],))
            changed.add(segment["segment_index"])
    return changed

"""Splits a reviewed project into narration parts around the kept recordings, ready to voice elsewhere.

Every run of narration between two kept recordings is one part, carrying its lines and the break
after each line; a part is split further under the generation limits. Pure functions; the pipeline
writes the files.
"""

from __future__ import annotations

from . import breaks


def plan_parts(segments: list[dict]) -> list[dict]:
    """Alternating narration parts (with lines) and recordings, in playing order."""
    parts: list[dict] = []
    current: dict | None = None
    for segment in sorted(segments, key=lambda item: item["segment_index"]):
        if segment.get("kind") == "original":
            if current:
                parts.append(current)
                current = None
            parts.append({
                "kind": "recording",
                "segment_index": segment["segment_index"],
                "start_ms": segment["start_ms"],
                "end_ms": segment["end_ms"],
                "audio_path": segment.get("tts_audio_path"),
                "transcript": segment.get("transcript_si") or "",
            })
            continue
        lines = list(segment.get("lines") or []) or breaks.default_lines(segment.get("narration_en") or "", segment["start_ms"], segment["end_ms"])
        if not lines:
            continue
        if current is None:
            current = {"kind": "narration", "segments": [], "lines": [], "start_ms": segment["start_ms"], "end_ms": segment["end_ms"]}
        current["segments"].append(segment["segment_index"])
        current["lines"].extend(dict(line, segment_index=segment["segment_index"]) for line in lines)
        current["end_ms"] = segment["end_ms"]
    if current:
        parts.append(current)
    return parts


def expand(parts: list[dict]) -> list[dict]:
    """Split narration parts under the generation limits and number every piece in playing order."""
    pieces: list[dict] = []
    for part in parts:
        if part["kind"] == "recording":
            pieces.append(dict(part, number=len(pieces) + 1))
            continue
        for chunk in breaks.split_parts(part["lines"]):
            timed = [line for line in chunk if line.get("start_ms") is not None]
            pieces.append({
                "kind": "narration",
                "number": len(pieces) + 1,
                "segments": sorted({line["segment_index"] for line in chunk if "segment_index" in line}) or part["segments"],
                "lines": chunk,
                "start_ms": timed[0]["start_ms"] if timed else part["start_ms"],
                "end_ms": timed[-1]["end_ms"] if timed else part["end_ms"],
            })
    if pieces and pieces[-1]["kind"] == "narration" and pieces[-1]["lines"]:
        # Nothing follows the programme's last line, so a trailing mark would only append silence.
        last = pieces[-1]["lines"]
        last[-1] = dict(last[-1], break_s=None)
    return pieces


def file_name(piece: dict) -> str:
    if piece["kind"] == "recording":
        return f"{piece['number']:02d}_recording_segment{piece['segment_index']:03d}.wav"
    return f"{piece['number']:02d}_narration.txt"


def render(piece: dict) -> str:
    return breaks.render(piece["lines"])


def index_lines(pieces: list[dict]) -> list[str]:
    """A human-readable order sheet, one line per file."""
    lines = ["# Script parts, in playing order", "",
             "Voice each narration file on its own (ElevenLabs Multilingual v2 reads the break marks), then place",
             "the recording files between the voiced parts. Target is the length of the original passage; a mark at",
             "the end of a narration file is the narrator's pause before the recording that follows it.", ""]
    for piece in pieces:
        span = f"{piece['start_ms'] / 1000:.1f}-{piece['end_ms'] / 1000:.1f} s"
        duration = (piece["end_ms"] - piece["start_ms"]) / 1000
        if piece["kind"] == "recording":
            lines.append(f"- `{file_name(piece)}`: kept recording, segment {piece['segment_index']}, source {span}, duration {duration:.1f} s")
        else:
            segs = ", ".join(str(index) for index in piece["segments"])
            chars = len(render(piece))
            marks = sum(1 for line in piece["lines"] if line.get("break_s"))
            lines.append(f"- `{file_name(piece)}`: narration, segments {segs}, source {span}, target {duration:.1f} s, {chars} characters, {marks} break mark{'s' if marks != 1 else ''}")
    return lines

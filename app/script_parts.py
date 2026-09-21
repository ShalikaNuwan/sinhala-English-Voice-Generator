"""Splits a reviewed project into narration parts around the kept recordings.

When the voice is made outside the pipeline (for example in the ElevenLabs web app), the script is
handed over as ordered parts: every run of narration between two kept recordings is one part, so
each part can be voiced on its own and the real recording dropped in between. Pure functions only;
the pipeline writes the files.
"""

from __future__ import annotations


def plan_parts(segments: list[dict]) -> list[dict]:
    """Group ordered segments into alternating narration parts and recordings.

    A narration part carries the adapted scripts of its segments joined by blank lines, the segment
    indexes it covers, and the source time span. A recording carries its segment index, source span,
    and the path of the kept clip.
    """
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
        text = (segment.get("narration_en") or "").strip()
        if not text:
            continue
        if current is None:
            current = {"kind": "narration", "segments": [], "start_ms": segment["start_ms"], "end_ms": segment["end_ms"], "text": ""}
        current["segments"].append(segment["segment_index"])
        current["end_ms"] = segment["end_ms"]
        current["text"] = f"{current['text']}\n\n{text}" if current["text"] else text
    if current:
        parts.append(current)
    return parts


def index_lines(parts: list[dict], files: list[str]) -> list[str]:
    """A human-readable order sheet for the parts, one line per file."""
    lines = ["# Script parts, in playing order", "",
             "Voice each narration part on its own, then place the recording files between them.", ""]
    for part, name in zip(parts, files):
        span = f"{part['start_ms'] / 1000:.1f}-{part['end_ms'] / 1000:.1f} s"
        if part["kind"] == "recording":
            lines.append(f"- `{name}`: kept recording, segment {part['segment_index']}, source {span}")
        else:
            segs = ", ".join(str(index) for index in part["segments"])
            lines.append(f"- `{name}`: narration, segments {segs}, source {span}, {len(part['text'])} characters")
    return lines

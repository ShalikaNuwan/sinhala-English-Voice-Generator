"""Pairs a Realtime session's voice-detection events into speech turns.

The transcription session reports every moment speech starts and stops, and transcribes each turn
separately; the three kinds of event arrive interleaved and pair by item id. A stop is reported once
the configured silence has elapsed, so the speech itself ended that much earlier. Pure functions; the
network lives in `AIClient.speech_turns`.
"""

from __future__ import annotations


def pair_events(events: list[dict], silence_ms: int, duration_ms: int) -> list[dict]:
    """Turn start, stop and transcript events into `[{start_ms, end_ms, text}]`, sorted by start.

    A turn still open when the audio ran out closes at its end; events without a start of their own,
    and turns with no length, are dropped.
    """
    starts: dict[str, int] = {}
    stops: dict[str, int] = {}
    texts: dict[str, str] = {}
    for event in events:
        item_id = event.get("item_id")
        if not item_id:
            continue
        kind = event.get("type")
        if kind == "start":
            starts[item_id] = int(event.get("ms") or 0)
        elif kind == "stop":
            stops[item_id] = int(event.get("ms") or 0)
        elif kind == "text":
            texts[item_id] = (event.get("text") or "").strip()
    paired = []
    for item_id, start_ms in starts.items():
        end_ms = max(start_ms, stops[item_id] - silence_ms) if item_id in stops else duration_ms
        if end_ms <= start_ms:
            continue
        paired.append({"start_ms": start_ms, "end_ms": end_ms, "text": texts.get(item_id, "")})
    return sorted(paired, key=lambda turn: turn["start_ms"])

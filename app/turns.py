"""Pairs a Realtime session's voice-detection events into speech turns.

The transcription session reports every moment speech starts and stops, and transcribes each turn
separately; the three kinds of event arrive interleaved and pair by item id. A stop is reported once
the configured silence has elapsed, so the speech itself ended that much earlier. Pure functions; the
network lives in `AIClient.speech_turns`.
"""

from __future__ import annotations

# A phrase this short, touching a piece edge, is the tail or head of a word the chunk cut through.
FRAGMENT_MS = 500
EDGE_MS = 50


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


def drop_edge_fragments(phrases: list[dict], start_ms: int, end_ms: int) -> list[dict]:
    """Drop a short phrase that touches the piece's start or end.

    Chunk edges fall wherever the planner put them, so a piece can begin or end inside a word. The
    voice detection reports that scrap as its own turn; kept, it would be a run of its own, and the
    adaptation would have to put a line on it. Dropping it lets the neighbouring run's silence reach
    the edge, which is what the narrator's pause really did.
    """
    kept = list(phrases)
    if kept and kept[0]["start_ms"] - start_ms <= EDGE_MS and kept[0]["end_ms"] - kept[0]["start_ms"] < FRAGMENT_MS:
        kept = kept[1:]
    if kept and end_ms - kept[-1]["end_ms"] <= EDGE_MS and kept[-1]["end_ms"] - kept[-1]["start_ms"] < FRAGMENT_MS:
        kept = kept[:-1]
    return kept

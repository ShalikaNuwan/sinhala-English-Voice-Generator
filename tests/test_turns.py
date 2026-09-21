from app import turns


def start(ms: int, item: str) -> dict:
    return {"type": "start", "ms": ms, "item_id": item, "text": None}


def stop(ms: int, item: str) -> dict:
    return {"type": "stop", "ms": ms, "item_id": item, "text": None}


def text(value: str, item: str) -> dict:
    return {"type": "text", "ms": None, "item_id": item, "text": value}


def test_a_start_and_its_stop_become_one_turn():
    events = [start(64, "a"), stop(9600, "a"), text("පළවෙනි", "a")]

    assert turns.pair_events(events, 200, 40_000) == [{"start_ms": 64, "end_ms": 9400, "text": "පළවෙනි"}]


def test_the_stop_time_has_the_configured_silence_taken_off_it():
    # The stop is reported once the silence has elapsed, so the speech ended silence_ms earlier.
    paired = turns.pair_events([start(1000, "a"), stop(5000, "a")], 800, 40_000)

    assert paired == [{"start_ms": 1000, "end_ms": 4200, "text": ""}]


def test_a_stop_closer_than_the_silence_never_ends_before_it_started():
    assert turns.pair_events([start(1000, "a"), stop(1100, "a")], 500, 40_000) == []


def test_a_turn_still_open_at_the_end_closes_with_the_audio():
    events = [start(1000, "a"), stop(2200, "a"), start(3000, "b"), text("අන්තිම", "b")]

    assert turns.pair_events(events, 200, 12_000) == [
        {"start_ms": 1000, "end_ms": 2000, "text": ""},
        {"start_ms": 3000, "end_ms": 12_000, "text": "අන්තිම"},
    ]


def test_transcripts_arriving_out_of_order_still_reach_their_own_turn():
    events = [
        start(0, "a"), stop(1200, "a"),
        start(2000, "b"), stop(3200, "b"),
        text("දෙවැනි", "b"), text("පළවෙනි", "a"),
    ]

    assert [turn["text"] for turn in turns.pair_events(events, 200, 9000)] == ["පළවෙනි", "දෙවැනි"]


def test_a_turn_without_a_transcript_keeps_an_empty_text():
    assert turns.pair_events([start(0, "a"), stop(1200, "a")], 200, 9000)[0]["text"] == ""


def test_a_stop_or_a_transcript_without_a_start_is_ignored():
    events = [stop(500, "ghost"), text("කවුද", "ghost"), start(1000, "a"), stop(2200, "a")]

    assert turns.pair_events(events, 200, 9000) == [{"start_ms": 1000, "end_ms": 2000, "text": ""}]


def test_turns_come_back_sorted_by_start_whatever_order_they_arrived_in():
    events = [start(5000, "b"), start(1000, "a"), stop(2200, "a"), stop(6200, "b")]

    assert [turn["start_ms"] for turn in turns.pair_events(events, 200, 9000)] == [1000, 5000]

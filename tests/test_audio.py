from app.audio import AudioService


def test_short_audio_is_one_segment():
    assert AudioService.plan_segments(35_000, [12_000, 25_000]) == [(0, 35_000)]


def test_prefers_silence_near_target():
    assert AudioService.plan_segments(95_000, [22_000, 39_000, 58_000, 80_000]) == [
        (0, 39_000),
        (39_000, 95_000),
    ]


def test_falls_back_to_safe_fixed_boundaries():
    segments = AudioService.plan_segments(130_000, [])
    assert segments == [(0, 45_000), (45_000, 90_000), (90_000, 130_000)]
    assert all(end > start for start, end in segments)


def test_avoids_tiny_tail_for_continuous_one_minute_audio():
    assert AudioService.plan_segments(64_740, []) == [(0, 64_740)]

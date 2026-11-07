import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import explainer  # noqa: E402


def test_counter_text_stays_inside_the_ring():
    for final, unit in (("7", "minutes"), ("1989", "years without an answer"), ("9", "days after she disappeared"), ("30", "")):
        item = {"start": 0, "end": int(final), "final": final, "unit": unit, "size": 0.1}
        r, fnt, ufnt, base, lines = explainer.counter_layout(item, 1080)
        assert fnt.getlength(final) < 2 * r
        for line, y in lines:
            assert abs(y) < r
            half = (r * r - y * y) ** 0.5
            assert ufnt.getlength(line) <= 2 * half

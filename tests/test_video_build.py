from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import build  # noqa: E402


def test_library_shots_are_left_alone_and_only_gaps_are_sourced(monkeypatch, tmp_path):
    monkeypatch.setattr(build.plan_edit, "FALLBACK_QUERY", "chicago 1970s street")   # the case's fallback search
    seen = []

    def fake_fill(shots, assets_dir, on_progress=None):
        seen.extend((s["index"], s["kind"], s["query"]) for s in shots)
        return [dict(s, asset=f"assets/{s['index']:04d}.jpg", licence="CC0 1.0", attribution="x") for s in shots]

    monkeypatch.setattr(build.sources, "fill_shots", fake_fill)
    shots = [
        {"index": 1, "kind": "archival", "library_id": "c1", "asset": "library/c1.jpg", "rights": "fair_use", "query": ""},
        {"index": 2, "kind": "period", "asset": None, "query": "chicago winter street"},
        {"index": 3, "kind": "archival", "asset": None, "query": ""},
    ]
    out = build.fill_gaps(shots, tmp_path)

    # A period picture is searched as stock, and a gap with no query still gets a search.
    assert seen == [(2, "stock", "chicago winter street"), (3, "archival", "chicago 1970s street")]
    assert out[0]["asset"] == "library/c1.jpg" and out[0]["rights"] == "fair_use"
    assert out[1]["kind"] == "period" and out[1]["asset"] == "assets/0002.jpg" and out[1]["rights"] == "free"


def test_credits_group_each_picture_once_by_rights():
    shots = [
        {"asset": "library/c1.jpg", "rights": "free", "attribution": "Des Plaines Police", "licence": "Public domain", "source_url": "u1"},
        {"asset": "library/c1.jpg", "rights": "free", "attribution": "Des Plaines Police", "licence": "Public domain", "source_url": "u1"},
        {"asset": "library/n1.jpg", "rights": "fair_use", "attribution": "Chicago Tribune", "licence": "Copyrighted", "source_url": "u2"},
        {"asset": None, "rights": None},
    ]
    text = build.credits(shots)
    assert text.count("library/c1.jpg") == 1
    assert "## Freely licensed" in text and "## Copyrighted, used under a documentary fair-use claim (1)" in text
    assert text.index("Freely licensed") < text.index("fair-use claim")


def test_the_full_film_asks_for_room_for_the_master_twice_over():
    # 98.6 minutes at full quality: about 8.5 GB of master, held twice near the end, plus 2 GB.
    need = build.space_needed(98.6 * 60)
    assert 19e9 < need < 21e9

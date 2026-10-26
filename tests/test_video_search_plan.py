from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import search_plan  # noqa: E402


def test_searches_merge_into_the_needs_list_by_query_and_remember_their_scenes():
    existing = [{"query": "1978 Des Plaines snowy street", "beats": [15]}]
    searches = [search_plan.Search(scene=3, query="1978 des plaines snowy street", kind="place"),
                search_plan.Search(scene=3, query="1970s pharmacy interior", kind="setting"),
                search_plan.Search(scene=4, query="1970s pharmacy interior", kind="setting"),
                search_plan.Search(scene=5, query="  ", kind="object")]
    merged = {n["query"].lower(): n for n in search_plan.merge_needs(existing, searches)}
    assert set(merged) == {"1978 des plaines snowy street", "1970s pharmacy interior"}
    assert merged["1978 des plaines snowy street"]["beats"] == [15] and merged["1978 des plaines snowy street"]["scenes"] == [3]
    assert merged["1970s pharmacy interior"]["scenes"] == [3, 4]

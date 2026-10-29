import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import case  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "gacy_case.json"


def test_the_gacy_profile_loads_and_names_the_case():
    profile = case.CaseProfile(**json.loads(FIXTURE.read_text()))
    assert profile.subject == "John Wayne Gacy" and profile.keyword == "Gacy"
    assert case.brief(profile).startswith("the John Wayne Gacy murders (Chicago")


def test_period_categories_are_found_as_commons_names_them_and_only_real_ones_kept():
    results = {
        "Seattle 1970s": ["Category:Fire Station No. 25, Seattle (1970s building)", "Category:Seattle in the 1970s"],
        "Salt Lake City 1970s": ["Category:1970 in Salt Lake City", "Category:1976 in Salt Lake City",
                                 "Category:1920s books from Salt Lake City"],
    }
    real = {"Category:DOCUMERICA photographs of Utah"}
    cats = case.period_categories(["Seattle", "Salt Lake City"], ["Utah", "Washington"], "1970s", list(range(1974, 1990)),
                                  search=lambda q: results.get(q, []), exists=lambda titles: {t for t in titles if t in real})
    assert [(c.category, c.depth) for c in cats] == [
        ("Category:Seattle in the 1970s", 1), ("Category:DOCUMERICA photographs of Utah", 0),
        ("Category:1976 in Salt Lake City", 0)]


def test_a_missing_case_profile_says_how_to_make_one(tmp_path):
    try:
        case.load(tmp_path)
    except SystemExit as exc:
        assert "case.py" in str(exc)
    else:
        raise AssertionError("expected SystemExit")

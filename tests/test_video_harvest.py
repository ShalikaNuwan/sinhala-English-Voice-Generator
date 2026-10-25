from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from PIL import Image, ImageDraw

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "video" / "harvest.py"


def _load():
    """scripts/ is not a package, so the module is loaded from its path."""
    spec = importlib.util.spec_from_file_location("video_harvest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


harvest = _load()


def picture(size=(640, 480), shade=0):
    """A small test picture with structure, so its hash is not all ones or zeros."""
    image = Image.new("L", size, 255)
    draw = ImageDraw.Draw(image)
    w, h = size
    draw.rectangle([0, 0, w // 2, h // 2], fill=shade)
    draw.ellipse([w // 2, h // 2, w - 1, h - 1], fill=shade)
    return image


# --- perceptual hash ------------------------------------------------------------------------

def test_average_hash_is_64_bits_and_scale_invariant():
    small = harvest.average_hash(picture((320, 240)))
    large = harvest.average_hash(picture((1600, 1200)))
    assert 0 <= small < 2 ** 64
    assert harvest.hamming(small, large) <= 2


def test_average_hash_tells_different_pictures_apart():
    one = picture()
    other = Image.new("L", (640, 480), 255)
    ImageDraw.Draw(other).rectangle([320, 0, 639, 479], fill=0)
    assert harvest.hamming(harvest.average_hash(one), harvest.average_hash(other)) > 5


def test_hamming_counts_differing_bits():
    assert harvest.hamming(0b1011, 0b0001) == 2
    assert harvest.hamming(0, 0) == 0
    assert harvest.hamming(2 ** 64 - 1, 0) == 64


# --- dedupe --------------------------------------------------------------------------------

def item(id_, url, hash_, width, height, rights="free"):
    return {"id": id_, "url": url, "hash": hash_, "width": width, "height": height, "rights": rights}


def test_dedupe_by_url_keeps_first():
    kept, dropped = harvest.dedupe([
        item("c0001", "https://x/a.jpg", 1, 800, 600),
        item("o0001", "https://x/a.jpg", 2 ** 40, 800, 600),
    ])
    assert [i["id"] for i in kept] == ["c0001"]
    assert dropped == [("o0001", "c0001")]


def test_dedupe_by_hash_keeps_the_larger_file():
    kept, dropped = harvest.dedupe([
        item("c0001", "https://x/a.jpg", 0b111, 600, 400),
        item("o0001", "https://y/b.jpg", 0b100, 2400, 1600),   # 2 bits away: same picture
        item("o0002", "https://y/c.jpg", 2 ** 64 - 1, 800, 600),  # far away: different picture
    ])
    assert sorted(i["id"] for i in kept) == ["o0001", "o0002"]
    assert dropped == [("c0001", "o0001")]


def test_dedupe_threshold_is_five_bits():
    near = item("o0001", "https://y/b.jpg", 0b11111, 2400, 1600)       # 5 bits from 0
    far = item("o0002", "https://y/c.jpg", 0b111111 << 10, 3000, 2000)  # 6 bits from 0
    kept, _ = harvest.dedupe([item("c0001", "https://x/a.jpg", 0, 600, 400), near, far])
    assert sorted(i["id"] for i in kept) == ["o0001", "o0002"]


def test_dedupe_prefers_a_free_copy_over_a_larger_fair_use_one():
    kept, dropped = harvest.dedupe([
        item("c0001", "https://x/a.jpg", 0, 600, 800, "free"),
        item("n0001", "https://news/a.jpg", 1, 1200, 1600, "fair_use"),
    ])
    assert [i["id"] for i in kept] == ["c0001"]
    assert dropped == [("n0001", "c0001")]


def test_dedupe_ignores_items_without_a_hash():
    kept, _ = harvest.dedupe([
        item("c0001", "https://x/a.jpg", None, 600, 400),
        item("c0002", "https://x/b.jpg", None, 600, 400),
    ])
    assert len(kept) == 2


# --- size floor ----------------------------------------------------------------------------

def test_size_floor_uses_the_short_side():
    assert harvest.big_enough({"width": 1920, "height": 480})
    assert not harvest.big_enough({"width": 1920, "height": 479})
    assert not harvest.big_enough({"width": 300, "height": 2000})


def test_size_floor_exempts_the_case_categories():
    assert harvest.big_enough({"width": 200, "height": 150, "case": True})
    assert not harvest.big_enough({"width": 200, "height": 150, "case": False})


# --- og:image extraction -------------------------------------------------------------------

BASE = "https://news.example.com/2018/12/gacy-house.html"


def test_page_image_prefers_og_image():
    html = """<html><head>
      <meta name="twitter:image" content="https://cdn.example.com/tw.jpg">
      <meta property="og:image" content="https://cdn.example.com/og.jpg">
    </head><body><article><img src="/big.jpg" width="2000" height="1000"></article></body></html>"""
    assert harvest.page_image(html, BASE) == "https://cdn.example.com/og.jpg"


def test_page_image_falls_back_to_twitter_image_and_resolves_relative_urls():
    html = """<head><meta content="/img/tw.jpg" name="twitter:image:src"></head><body></body>"""
    assert harvest.page_image(html, BASE) == "https://news.example.com/img/tw.jpg"


def test_page_image_falls_back_to_the_largest_article_image():
    html = """<body>
      <header><img src="/logo.png" width="3000" height="3000"></header>
      <article>
        <img src="/small.jpg" width="200" height="100">
        <img src="https://cdn.example.com/large.jpg" width="1200" height="800">
        <img data-src="/lazy.jpg" width="900" height="600">
      </article></body>"""
    assert harvest.page_image(html, BASE) == "https://cdn.example.com/large.jpg"


def test_page_image_skips_site_wide_share_pictures():
    html = """<head>
      <meta property="og:image" content="https://media.example.com/2024/08/News-Stream-live-thumbnail.png">
      <meta name="twitter:image" content="https://media.example.com/site-logo.png">
    </head><body><article><img src="/photos/gacy-house.jpg" width="1200" height="800"></article></body>"""
    assert harvest.page_image(html, BASE) == "https://news.example.com/photos/gacy-house.jpg"


def test_page_image_returns_none_when_there_is_nothing():
    assert harvest.page_image("<html><body><p>No pictures.</p></body></html>", BASE) is None
    assert harvest.page_image("", BASE) is None


# --- Commons category walking -------------------------------------------------------------

CANNED = {
    ("Category:John Wayne Gacy", None): {
        "continue": {"cmcontinue": "page2", "continue": "-||"},
        "query": {"categorymembers": [
            {"ns": 6, "title": "File:John Wayne Gacy 1978 mugshot.jpg"},
            {"ns": 14, "title": "Category:Victims of John Wayne Gacy"},
        ]},
    },
    ("Category:John Wayne Gacy", "page2"): {
        "query": {"categorymembers": [
            {"ns": 6, "title": "File:John Gacy Pogo December 1976 Martin Zielinski.jpg"},
        ]},
    },
    ("Category:Victims of John Wayne Gacy", None): {
        "query": {"categorymembers": [
            {"ns": 6, "title": "File:Robert Piest.jpg"},
            {"ns": 6, "title": "File:John Wayne Gacy 1978 mugshot.jpg"},  # in two categories
            {"ns": 14, "title": "Category:Robert Piest"},
            {"ns": 14, "title": "Category:John Wayne Gacy"},  # a cycle
        ]},
    },
    ("Category:Robert Piest", None): {
        "query": {"categorymembers": [{"ns": 6, "title": "File:Piest relatives at Gacy trial.png"}]},
    },
}


def canned_fetch(calls):
    def fetch(url, params):
        calls.append(params)
        return CANNED[(params["cmtitle"], params.get("cmcontinue"))]
    return fetch


def test_walk_category_follows_subcategories_continuation_and_cycles():
    calls = []
    files = harvest.walk_category("Category:John Wayne Gacy", fetch=canned_fetch(calls))
    assert files == [
        "File:John Wayne Gacy 1978 mugshot.jpg",
        "File:John Gacy Pogo December 1976 Martin Zielinski.jpg",
        "File:Robert Piest.jpg",
        "File:Piest relatives at Gacy trial.png",
    ]
    assert len(calls) == 4  # each category page read once, the cycle is not followed


def test_walk_category_respects_depth():
    files = harvest.walk_category("Category:John Wayne Gacy", fetch=canned_fetch([]), max_depth=0)
    assert files == [
        "File:John Wayne Gacy 1978 mugshot.jpg",
        "File:John Gacy Pogo December 1976 Martin Zielinski.jpg",
    ]


def imageinfo_page(title, licence, artist, non_free=False, repo="local"):
    meta = {"LicenseShortName": {"value": licence}, "Artist": {"value": artist}}
    if non_free:
        meta["NonFree"] = {"value": "true"}
    return {
        "title": title,
        "imagerepository": repo,
        "imageinfo": [{
            "url": "https://upload.wikimedia.org/wikipedia/commons/a/ab/X.jpg",
            "width": 3000, "height": 2000,
            "thumburl": "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/X.jpg/1920px-X.jpg",
            "thumbwidth": 1920, "thumbheight": 1280,
            "extmetadata": meta,
        }],
    }


def test_commons_item_keeps_every_licence_and_reduces_the_artist():
    page = imageinfo_page("File:John Wayne Gacy 1978 mugshot.jpg", "CC BY-SA 4.0",
                          '<a href="//x">Des&#160;Plaines <b>Police</b></a>')
    found = harvest.commons_item(page, found_by="case:Category:John Wayne Gacy", case=True)
    assert found["licence"] == "CC BY-SA 4.0"
    assert found["attribution"] == "Des Plaines Police"
    assert found["rights"] == "free"
    assert found["source"] == "commons"
    assert found["title"] == "John Wayne Gacy 1978 mugshot"
    assert found["url"].endswith("1920px-X.jpg")
    assert (found["width"], found["height"]) == (1920, 1280)
    assert found["source_url"] == "https://commons.wikimedia.org/wiki/File:John_Wayne_Gacy_1978_mugshot.jpg"
    assert found["case"] is True


def test_commons_item_reports_the_real_size_of_an_unscaled_original():
    page = imageinfo_page("File:Robert Piest.jpg", "Public domain", "Chicago Tribune")
    info = page["imageinfo"][0]
    info.update({"width": 640, "height": 800, "thumbwidth": 1920, "thumbheight": 2400,
                 "thumburl": info["url"] + "?utm_content=thumbnail_unscaled"})
    found = harvest.commons_item(page, found_by="case", case=True)
    assert (found["width"], found["height"]) == (640, 800)


def test_commons_item_skips_non_images():
    page = imageinfo_page("File:Trial transcript.pdf", "Public domain", "Court")
    assert harvest.commons_item(page, found_by="x") is None


# --- rights -------------------------------------------------------------------------------

def test_rights_per_source():
    assert harvest.rights_for("commons", "CC BY-SA 4.0") == "free"
    assert harvest.rights_for("openverse", "CC BY 2.0") == "free"
    assert harvest.rights_for("wikipedia", "Public domain") == "free"
    assert harvest.rights_for("wikipedia", "Fair use", non_free=True) == "fair_use"
    assert harvest.rights_for("web", "Copyrighted") == "fair_use"
    assert harvest.rights_for("generated", "Generated") == "generated"


def test_wikipedia_non_free_file_is_fair_use_with_its_wikipedia_page():
    page = imageinfo_page("File:John Wayne Gacy mugshot.jpg", "Fair use", "Des Plaines PD", non_free=True)
    found = harvest.wikipedia_item(page, found_by="wikipedia:John Wayne Gacy")
    assert found["rights"] == "fair_use"
    assert found["licence"] == "Copyrighted"
    assert found["source"] == "wikipedia"
    assert found["source_url"] == "https://en.wikipedia.org/wiki/File:John_Wayne_Gacy_mugshot.jpg"


def test_wikipedia_shared_file_stays_free_and_points_at_commons():
    page = imageinfo_page("File:Des Plaines River.jpg", "CC BY 3.0", "Someone", repo="shared")
    found = harvest.wikipedia_item(page, found_by="wikipedia:Des Plaines River")
    assert found["rights"] == "free"
    assert found["licence"] == "CC BY 3.0"
    assert found["source_url"] == "https://commons.wikimedia.org/wiki/File:Des_Plaines_River.jpg"


def test_web_item_is_fair_use_and_credits_the_outlet():
    found = harvest.web_item({"subject": "Summerdale house", "page_url": "https://www.chicagotribune.com/x"},
                             "https://cdn.example.com/og.jpg", title="Gacy house search")
    assert found["rights"] == "fair_use"
    assert found["licence"] == "Copyrighted"
    assert found["attribution"] == "chicagotribune.com"
    assert found["source"] == "web"
    assert found["source_url"] == "https://www.chicagotribune.com/x"


# --- relevance and ids ---------------------------------------------------------------------

def test_relevance_needs_the_surname_for_people():
    assert harvest.relevant("John Wayne Gacy", "File:John Wayne Gacy 1978 mugshot.jpg", person=True)
    assert not harvest.relevant("John Wayne Gacy", "John Wayne in Stagecoach", person=True)
    assert harvest.relevant("Cook County Jail", "Cook County Jail, Chicago, 2010", person=False)
    assert harvest.relevant("Rosalynn Carter", "Rosalynn Carter official portrait", person=True)
    assert not harvest.relevant("Cook County Jail", "Sunset over a lake", person=False)


def test_a_private_person_needs_a_gacy_context():
    assert not harvest.relevant("Michael Rossi", "Michael Rossi at the 2012 Giro", person=True)
    assert harvest.relevant("John Butkovich", "John Butkovich John Gacy Chicago Tribune 1980", person=True)


def test_next_id_is_zero_padded_per_prefix():
    taken = {"c0001", "c0002", "o0001"}
    assert harvest.next_id("c", taken) == "c0003"
    assert harvest.next_id("n", taken) == "n0001"


def test_host_gate_holds_waiters_through_a_back_off(monkeypatch):
    clock = {"t": 100.0}
    monkeypatch.setattr(harvest.time, "monotonic", lambda: clock["t"])
    monkeypatch.setattr(harvest.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    gate = harvest.HostGate(spacing=1.0)
    gate.wait()
    gate.back_off(600)
    gate.wait()
    assert clock["t"] >= 700.0


def test_entities_merge_with_the_starter_list_and_drop_broad_names():
    found = [
        {"name": "John Wayne Gacy", "kind": "person", "note": "the killer", "query": "John Wayne Gacy"},
        {"name": "Angela", "kind": "person", "note": "a first name only", "query": "Angela"},
        {"name": "Chicago", "kind": "place", "note": "city", "query": "Chicago"},
        {"name": "Body 28", "kind": "object", "note": "", "query": "Body 28"},
    ]
    names = [e["name"] for e in harvest.merge_entities(found)]
    assert names.count("John Wayne Gacy") == 1
    assert "Angela" not in names and "Chicago" not in names and "Body 28" not in names
    assert "Nisson Pharmacy" in names and "Rosalynn Carter" in names


def test_search_query_per_kind():
    q = harvest.search_query
    assert q({"name": "Cook County Jail", "kind": "institution", "query": "Cook County Jail John Wayne Gacy"}) == "Cook County Jail"
    assert q({"name": "Milwaukee Avenue", "kind": "place", "query": "Milwaukee Avenue Illinois John Wayne Gacy arrest"}) == "Milwaukee Avenue Illinois"
    assert q({"name": "Gregory John Godzik", "kind": "person", "query": "Gregory John Godzik"}) == "Gregory John Godzik Gacy"
    assert q({"name": "Rosalynn Carter", "kind": "person", "query": "Rosalynn Carter"}) == "Rosalynn Carter"
    assert q({"name": "Pogo the Clown", "kind": "object", "query": "John Wayne Gacy Pogo the Clown photograph"}) == "John Wayne Gacy Pogo the Clown photograph"

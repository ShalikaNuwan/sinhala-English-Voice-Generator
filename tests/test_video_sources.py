from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "video" / "sources.py"


def _load():
    """scripts/ is not a package, so the module is loaded from its path."""
    spec = importlib.util.spec_from_file_location("video_sources", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sources = _load()


# --- canned API payloads, shaped like the real responses -------------------------------------

def commons_page(pageid, index, title, licence, artist, name="Gacy.jpg"):
    return {
        "pageid": pageid,
        "ns": 6,
        "title": title,
        "index": index,
        "imagerepository": "local",
        "imageinfo": [
            {
                "size": 184203,
                "width": 2400,
                "height": 3000,
                "thumburl": f"https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/{name}/1920px-{name}",
                "thumbwidth": 1920,
                "thumbheight": 2400,
                "url": f"https://upload.wikimedia.org/wikipedia/commons/a/ab/{name}",
                "descriptionurl": "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_"),
                "extmetadata": {
                    "DateTime": {"value": "2015-05-02 11:22:33", "source": "commons-desc-page"},
                    "LicenseShortName": {"value": licence, "source": "commons-desc-page"},
                    "Artist": {"value": artist, "source": "commons-desc-page"},
                    "UsageTerms": {"value": licence, "source": "commons-desc-page"},
                },
            }
        ],
    }


ANCHOR_ARTIST = '<a href="https://www.dppd.gov" class="external text">Des Plaines Police Department</a>'
DIV_ARTIST = (
    '<div class="fn value">\n'
    '<a href="//commons.wikimedia.org/wiki/User:Aphotographer" title="User:Aphotographer">'
    "Ann&#160;Photographer</a>\n"
    "</div>"
)


def commons_payload():
    # Keyed by pageid and deliberately out of search order: "index" carries the ranking.
    return {
        "batchcomplete": "",
        "continue": {"gsroffset": 8, "continue": "gsroffset||"},
        "query": {
            "pages": {
                "6789": commons_page(6789, 2, "File:Gacy house 1978.jpg", "CC BY 2.0", DIV_ARTIST, "House.jpg"),
                "12345": commons_page(12345, 1, "File:John Wayne Gacy mugshot.jpg", "Public domain", ANCHOR_ARTIST),
                "999": commons_page(999, 3, "File:Share alike.jpg", "CC BY-SA 4.0", ANCHOR_ARTIST, "Sa.jpg"),
                "1000": commons_page(1000, 4, "File:Non commercial.jpg", "CC BY-NC 2.0", "", "Nc.jpg"),
                "1001": commons_page(1001, 5, "File:No artist.jpg", "CC0", "", "Cc0.jpg"),
            }
        },
    }


def openverse_result(rid, title, licence, version, creator, url):
    return {
        "id": rid,
        "title": title,
        "indexed_on": "2024-01-02T03:04:05Z",
        "foreign_landing_url": f"https://www.flickr.com/photos/someone/{rid}",
        "url": url,
        "creator": creator,
        "creator_url": "https://www.flickr.com/photos/someone",
        "license": licence,
        "license_version": version,
        "license_url": f"https://creativecommons.org/licenses/{licence}/{version}/",
        "provider": "flickr",
        "source": "flickr",
        "category": "photograph",
        "filesize": 482211,
        "filetype": "jpg",
        "width": 1600,
        "height": 1067,
        "attribution": "\"Snowy street\" by Someone is licensed under CC BY 2.0.",
    }


def openverse_payload():
    return {
        "result_count": 3,
        "page_count": 1,
        "page_size": 8,
        "page": 1,
        "results": [
            openverse_result("a1", "Snowy street", "by", "2.0", "Jane Doe",
                             "https://live.staticflickr.com/65535/a1_b.jpg"),
            openverse_result("a2", "Suburb at night", "by-nc-sa", "4.0", "Somebody",
                             "https://live.staticflickr.com/65535/a2_b.jpg"),
            openverse_result("a3", "Street lamp", "cc0", "1.0", "",
                             "https://live.staticflickr.com/65535/a3_b.png"),
        ],
    }


@pytest.fixture
def fake_api(monkeypatch):
    """Answer both APIs from canned payloads and record what was asked for."""
    calls = []

    def fetch_json(url, params):
        calls.append((url, dict(params)))
        if "commons" in url:
            return commons_payload()
        return openverse_payload()

    monkeypatch.setattr(sources, "fetch_json", fetch_json)
    return calls


@pytest.fixture
def fake_download(monkeypatch):
    """Write a stand-in file instead of fetching bytes."""
    got = []

    def download(url, destination):
        got.append((url, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\xff\xd8pretend-jpeg")

    monkeypatch.setattr(sources, "download", download)
    return got


# --- the licence gate ------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name",
    [
        "Public domain",
        "public domain",
        "Public Domain Mark 1.0",
        "PDM 1.0",
        "pdm",
        "PD-US-expired",
        "CC0",
        "CC0 1.0",
        "cc0",
        "CC BY 2.0",
        "CC BY 4.0",
        "CC-BY-3.0",
        "cc by-4.0",
        "by",
        "Attribution 3.0 Unported",
    ],
)
def test_commercial_licence_accepts_commercial_terms(name):
    assert sources.commercial_licence(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "CC BY-SA 2.0",
        "CC BY-SA 4.0",
        "CC-BY-SA-3.0",
        "ccbysa20",
        "by-sa",
        "CC BY-NC 2.0",
        "CC BY-NC-SA 4.0",
        "by-nc-nd",
        "CC BY-ND 4.0",
        "Attribution-ShareAlike 3.0",
        "Attribution-NonCommercial 2.0",
    ],
)
def test_commercial_licence_rejects_restricted_terms(name):
    assert sources.commercial_licence(name) is False


def test_commercial_licence_separates_by_from_by_sa():
    assert sources.commercial_licence("CC BY 2.0") is True
    assert sources.commercial_licence("CC BY-SA 2.0") is False


@pytest.mark.parametrize(
    "name",
    ["", "   ", None, "fair use", "Fair use", "GFDL", "GPLv3", "No restrictions",
     "Copyrighted, used with permission", "by permission of the author"],
)
def test_commercial_licence_rejects_unknown_and_empty(name):
    assert sources.commercial_licence(name) is False


# --- Commons ---------------------------------------------------------------------------------

def test_commons_search_keeps_commercial_results_in_search_order(fake_api):
    results = sources.search_commons("John Wayne Gacy", limit=8)

    assert [r["licence"] for r in results] == ["Public domain", "CC BY 2.0", "CC0"]
    assert results[0]["title"] == "File:John Wayne Gacy mugshot.jpg"
    assert results[0]["url"].endswith("1920px-Gacy.jpg")
    assert (results[0]["width"], results[0]["height"]) == (1920, 2400)
    assert results[0]["source_url"] == "https://commons.wikimedia.org/wiki/File:John_Wayne_Gacy_mugshot.jpg"


def test_commons_search_asks_the_api_for_licensed_bitmaps(fake_api):
    sources.search_commons("snow 1978", limit=5)

    url, params = fake_api[0]
    assert url == sources.COMMONS_API
    assert params["gsrsearch"] == "snow 1978 filetype:bitmap"
    assert params["gsrnamespace"] == 6
    assert params["gsrlimit"] == 5
    assert params["iiprop"] == "url|extmetadata|size"
    assert params["iiurlwidth"] == 1920
    assert params["format"] == "json"
    assert "github.com/ShalikaNuwan" in sources.USER_AGENT   # Wikimedia requires a contact in the User-Agent


def test_commons_search_reduces_html_artist_to_a_readable_name(fake_api):
    results = sources.search_commons("John Wayne Gacy")

    assert results[0]["attribution"] == "Des Plaines Police Department"
    assert results[1]["attribution"] == "Ann Photographer"  # multi-line <div class="fn value">
    assert results[2]["attribution"] == "Unknown"  # nothing survived stripping


def test_commons_search_drops_a_non_commercial_result(fake_api):
    titles = [r["title"] for r in sources.search_commons("anything")]

    assert "File:Share alike.jpg" not in titles
    assert "File:Non commercial.jpg" not in titles


def test_commons_search_survives_an_empty_response(monkeypatch):
    monkeypatch.setattr(sources, "fetch_json", lambda url, params: {"batchcomplete": ""})

    assert sources.search_commons("nothing here") == []


# --- Openverse -------------------------------------------------------------------------------

def test_openverse_search_parses_results(fake_api):
    results = sources.search_openverse("snowy suburban street at night", limit=8)

    assert [r["licence"] for r in results] == ["CC BY 2.0", "CC0 1.0"]
    assert results[0]["title"] == "Snowy street"
    assert results[0]["url"] == "https://live.staticflickr.com/65535/a1_b.jpg"
    assert results[0]["attribution"] == "Jane Doe"
    assert results[0]["source_url"] == "https://www.flickr.com/photos/someone/a1"
    assert (results[0]["width"], results[0]["height"]) == (1600, 1067)
    assert results[1]["attribution"] == "Unknown"


def test_openverse_search_filters_licences_server_side(fake_api):
    sources.search_openverse("snow", limit=4)

    url, params = fake_api[0]
    assert url == sources.OPENVERSE_API
    assert params == {"q": "snow", "page_size": 4, "license": "cc0,by,pdm"}


def test_openverse_search_rejects_a_non_commercial_result(fake_api):
    results = sources.search_openverse("snow")

    assert all("nc" not in r["licence"].lower() for r in results)
    assert [r["title"] for r in results] == ["Snowy street", "Street lamp"]


# --- the likeness guard ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "prompt",
    [
        "A snowbound suburban pharmacy at night, 1978, documentary still",
        "An empty courtroom bench, morning light, 35mm film grain",
        "A crawlspace under a bungalow, lit by a single bulb, no people",
        "Aerial view of Des Plaines, Illinois in winter, documentary photo",
    ],
)
def test_names_a_real_person_allows_atmospheric_prompts(prompt):
    assert sources.names_a_real_person(prompt) is False


@pytest.mark.parametrize(
    "prompt",
    [
        "A portrait of John Wayne Gacy in a clown suit",
        "Police mugshot of Robert Piest, 1978",
        "photo of Jeffrey Dahmer at his trial",
        "A realistic likeness of Rose West",
        "Close-up of the face of John Gacy, documentary still",
    ],
)
def test_names_a_real_person_rejects_a_likeness_of_a_named_person(prompt):
    assert sources.names_a_real_person(prompt) is True


def test_generate_image_refuses_a_named_likeness(tmp_path):
    with pytest.raises(ValueError, match="real person"):
        sources.generate_image("A portrait of John Wayne Gacy", tmp_path / "x.png")

    assert not (tmp_path / "x.png").exists()


# --- download --------------------------------------------------------------------------------

def test_download_writes_the_fetched_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "fetch_bytes", lambda url: b"\xff\xd8jpeg-bytes")

    sources.download("https://upload.wikimedia.org/x.jpg", tmp_path / "assets" / "0001.jpg")

    assert (tmp_path / "assets" / "0001.jpg").read_bytes() == b"\xff\xd8jpeg-bytes"


def test_download_skips_a_file_that_is_already_there(tmp_path):
    cached = tmp_path / "0001.jpg"
    cached.write_bytes(b"already here")

    sources.download("https://example.invalid/none.jpg", cached)  # no network: returns early

    assert cached.read_bytes() == b"already here"


# --- fill_shots ------------------------------------------------------------------------------

def shot(index, kind, **extra):
    base = {"index": index, "start": float(index), "end": float(index) + 9.0, "kind": kind,
            "motion": "zoom_in", "asset": None}
    base.update(extra)
    return base


def test_fill_shots_fills_an_archival_shot(tmp_path, fake_api, fake_download):
    progress = []
    shots = [shot(1, "archival", query="John Wayne Gacy 1978 mugshot")]

    filled = sources.fill_shots(shots, tmp_path / "assets", on_progress=lambda *a: progress.append(a))

    assert filled[0]["asset"] == "assets/0001.jpg"
    assert filled[0]["licence"] == "Public domain"
    assert filled[0]["attribution"] == "Des Plaines Police Department"
    assert filled[0]["source_url"].startswith("https://commons.wikimedia.org/wiki/File:")
    assert "error" not in filled[0]
    assert (tmp_path / "assets" / "0001.jpg").read_bytes()
    assert shots[0]["asset"] is None  # the input is left alone
    assert len(progress) == 1


def test_fill_shots_searches_commons_for_archival_and_openverse_for_stock(tmp_path, fake_api, fake_download):
    shots = [shot(1, "archival", query="gacy"), shot(2, "stock", query="snow")]

    sources.fill_shots(shots, tmp_path / "assets")

    asked = {url: params for url, params in fake_api}
    assert asked[sources.COMMONS_API]["gsrsearch"] == "gacy filetype:bitmap"
    assert asked[sources.OPENVERSE_API]["q"] == "snow"
    assert len(fake_api) == 2  # neither shot needed the fallback source


def test_fill_shots_falls_back_to_the_other_source(tmp_path, monkeypatch, fake_download):
    def fetch_json(url, params):
        if "commons" in url:
            return {"query": {"pages": {}}}
        return openverse_payload()

    monkeypatch.setattr(sources, "fetch_json", fetch_json)
    filled = sources.fill_shots([shot(1, "archival", query="gacy")], tmp_path / "assets")

    assert filled[0]["licence"] == "CC BY 2.0"
    assert filled[0]["attribution"] == "Jane Doe"


def test_fill_shots_leaves_drawn_shots_untouched(tmp_path, fake_api, fake_download):
    shots = [shot(1, "card", text="December 11th, 1978"), shot(2, "chart"), shot(3, "map")]

    filled = sources.fill_shots(shots, tmp_path / "assets")

    assert [s["asset"] for s in filled] == [None, None, None]
    assert not any("error" in s for s in filled)
    assert fake_download == []
    assert fake_api == []


def test_fill_shots_records_an_error_instead_of_raising_when_nothing_is_found(tmp_path, monkeypatch, fake_download):
    monkeypatch.setattr(sources, "fetch_json", lambda url, params: {"query": {"pages": {}}, "results": []})

    filled = sources.fill_shots([shot(1, "archival", query="nothing at all"), shot(2, "card")], tmp_path / "assets")

    assert filled[0]["asset"] is None
    assert filled[0]["error"]
    assert filled[1].get("error") is None


def test_fill_shots_records_an_error_when_a_search_blows_up(tmp_path, monkeypatch, fake_download):
    def boom(url, params):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(sources, "fetch_json", boom)
    filled = sources.fill_shots([shot(1, "stock", query="snow")], tmp_path / "assets")

    assert filled[0]["asset"] is None
    assert "connection reset" in filled[0]["error"]


def test_fill_shots_does_not_reuse_an_image_for_neighbouring_shots(tmp_path, fake_api, fake_download):
    shots = [shot(i, "archival", query="gacy") for i in range(1, 4)]

    filled = sources.fill_shots(shots, tmp_path / "assets")

    picked = [s["source_url"] for s in filled]
    assert len(set(picked)) == 3
    assert [s["asset"] for s in filled] == ["assets/0001.jpg", "assets/0002.jpg", "assets/0003.jpg"]


def test_fill_shots_repeats_the_longest_unused_image_when_there_is_no_choice(tmp_path, fake_api, fake_download):
    shots = [shot(i, "archival", query="gacy") for i in range(1, 16)]

    filled = sources.fill_shots(shots, tmp_path / "assets")

    # Only three licensed results exist, so they cycle: a repeat is always the oldest picture.
    picked = [s["source_url"] for s in filled]
    assert picked[:3] == picked[3:6] == picked[6:9]
    assert len(set(picked[:3])) == 3
    assert all(s["asset"] for s in filled)


def test_fill_shots_generates_the_generated_shots(tmp_path, fake_api, fake_download, monkeypatch):
    seen = []

    def generate_image(prompt, destination, size="1536x1024"):
        seen.append((prompt, destination))
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x89PNG")

    monkeypatch.setattr(sources, "generate_image", generate_image)
    shots = [shot(1, "generated", prompt="A snowbound suburban pharmacy at night, 1978")]

    filled = sources.fill_shots(shots, tmp_path / "assets")

    assert filled[0]["asset"] == "assets/0001.png"
    assert filled[0]["licence"] == "Generated"
    assert filled[0]["attribution"] == "Generated with OpenAI gpt-image-2.5"
    assert seen[0][0].startswith("A snowbound")


def test_fill_shots_records_an_error_when_a_generated_shot_is_refused(tmp_path, fake_api, fake_download):
    filled = sources.fill_shots(
        [shot(1, "generated", prompt="A portrait of John Wayne Gacy")], tmp_path / "assets"
    )

    assert filled[0]["asset"] is None
    assert "real person" in filled[0]["error"]


def test_fill_shots_records_an_error_for_a_generated_shot_with_no_prompt(tmp_path, fake_api, fake_download):
    filled = sources.fill_shots([shot(1, "generated")], tmp_path / "assets")

    assert filled[0]["asset"] is None
    assert filled[0]["error"] == "no prompt"


def test_fill_shots_keeps_the_shot_order_it_was_given(tmp_path, fake_api, fake_download):
    shots = [shot(1, "archival", query="a"), shot(2, "card"), shot(3, "stock", query="b"),
             shot(4, "archival", query="c")]

    filled = sources.fill_shots(shots, tmp_path / "assets")

    assert [s["index"] for s in filled] == [1, 2, 3, 4]
    assert [s["kind"] for s in filled] == ["archival", "card", "stock", "archival"]


# --- credits ---------------------------------------------------------------------------------

def test_credits_markdown_lists_each_asset_once_under_its_heading():
    shots = [
        shot(1, "archival", asset="assets/0001.jpg", licence="Public domain",
             attribution="Des Plaines Police Department",
             source_url="https://commons.wikimedia.org/wiki/File:Gacy.jpg"),
        shot(2, "archival", asset="assets/0002.jpg", licence="Public domain",
             attribution="Des Plaines Police Department",
             source_url="https://commons.wikimedia.org/wiki/File:Gacy.jpg"),  # same work, used twice
        shot(3, "stock", asset="assets/0003.jpg", licence="CC BY 2.0", attribution="Jane Doe",
             source_url="https://www.flickr.com/photos/someone/a1"),
        shot(4, "generated", asset="assets/0004.png", licence="Generated",
             attribution="Generated with OpenAI gpt-image-2.5", prompt="A snowbound pharmacy"),
        shot(5, "card", text="1978"),
        shot(6, "archival", asset=None, error="no licensed image found"),
    ]

    body = sources.credits_markdown(shots)

    assert body.count("https://commons.wikimedia.org/wiki/File:Gacy.jpg") == 1
    assert "## Archival" in body and "## Stock" in body and "## Generated" in body
    assert body.index("## Archival") < body.index("## Stock") < body.index("## Generated")
    assert "Des Plaines Police Department" in body
    assert "CC BY 2.0" in body
    assert "https://www.flickr.com/photos/someone/a1" in body
    assert "A snowbound pharmacy" in body
    assert "1978" not in body  # the card is drawn, not sourced
    assert "no licensed image found" not in body


def test_credits_markdown_handles_a_video_with_nothing_sourced_yet():
    body = sources.credits_markdown([shot(1, "card", text="1978")])

    assert "## Archival" not in body
    assert body.strip()


def test_credits_markdown_is_writable_next_to_shots_json(tmp_path, fake_api, fake_download):
    filled = sources.fill_shots([shot(1, "archival", query="gacy")], tmp_path / "assets")
    (tmp_path / "credits.md").write_text(sources.credits_markdown(filled), encoding="utf-8")
    (tmp_path / "shots.json").write_text(json.dumps({"shots": filled}), encoding="utf-8")

    assert "Public domain" in (tmp_path / "credits.md").read_text(encoding="utf-8")
    assert (tmp_path / filled[0]["asset"]).exists()  # asset paths are relative to shots.json


def test_shorten_keeps_the_picture_words_and_drops_the_filler():
    assert sources.shorten("snowy suburban street at night") == "snowy suburban street"
    assert sources.shorten("car waiting snowy street night") == "car waiting snowy"
    # Nothing to gain when the query is already short, or is only filler.
    assert sources.shorten("snowy street night") == ""
    assert sources.shorten("empty room") == ""
    assert sources.shorten("in the of") == ""


def test_a_search_that_finds_nothing_is_retried_with_the_shorter_query(monkeypatch):
    seen = []

    def fake_openverse(query, limit=8):
        seen.append(query)
        return [{"url": "https://x/a.jpg", "licence": "CC0 1.0", "attribution": "x", "source_url": "u"}] \
            if query == "snowy suburban street" else []

    monkeypatch.setattr(sources, "search_openverse", fake_openverse)
    monkeypatch.setattr(sources, "search_commons", lambda query, limit=8: [])

    results, failures = sources.search_shot({"kind": "stock", "query": "snowy suburban street at night"})

    assert [r["url"] for r in results] == ["https://x/a.jpg"]
    assert seen == ["snowy suburban street at night", "snowy suburban street"]
    assert any("retrying as" in f for f in failures)

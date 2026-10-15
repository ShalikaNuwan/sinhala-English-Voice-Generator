from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
from pathlib import Path

from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "video" / "caption.py"


def _load():
    """scripts/ is not a package, so the module is loaded from its path."""
    spec = importlib.util.spec_from_file_location("video_caption", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


caption = _load()


HARVESTED = {
    "id": "c0001", "file": "library/c0001.jpg", "source": "commons",
    "title": "John Wayne Gacy 1978 mugshot", "source_url": "https://commons.wikimedia.org/wiki/File:X.jpg",
    "licence": "Public domain", "attribution": "Des Plaines Police Department", "rights": "free",
    "width": 1505, "height": 2000,
}

RESULT = {
    "caption": "Black-and-white police booking photograph of John Wayne Gacy, front view, 1978.",
    "subjects": ["John Wayne Gacy"], "places": ["Des Plaines, Illinois"], "era": "1978",
    "colour": "bw", "kind": "photo", "graphic": False, "quality": 4,
}


def test_merge_fills_caption_fields():
    merged = caption.merge_caption(HARVESTED, RESULT)
    for key, value in RESULT.items():
        assert merged[key] == value
    assert HARVESTED.get("caption") is None  # the input is not mutated


def test_merge_never_clobbers_harvest_fields():
    hostile = dict(RESULT, id="x9999", file="elsewhere.jpg", rights="generated", licence="CC0",
                   title="Something else", width=1, source_url="https://evil")
    merged = caption.merge_caption(HARVESTED, hostile)
    for key, value in HARVESTED.items():
        assert merged[key] == value
    assert merged["caption"] == RESULT["caption"]


def test_pending_skips_items_already_captioned():
    done = caption.merge_caption(HARVESTED, RESULT)
    fresh = dict(HARVESTED, id="c0002", file="library/c0002.jpg")
    blank = dict(HARVESTED, id="c0003", file="library/c0003.jpg", caption="")
    assert [i["id"] for i in caption.pending([done, fresh, blank])] == ["c0002", "c0003"]


def test_pending_respects_redo():
    done = caption.merge_caption(HARVESTED, RESULT)
    assert [i["id"] for i in caption.pending([done], redo=True)] == ["c0001"]


def test_data_url_downscales_to_about_1024(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (3000, 1500), (40, 40, 40)).save(path)
    url = caption.image_data_url(path, max_side=1024)
    assert url.startswith("data:image/jpeg;base64,")
    decoded = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(decoded.size) == 1024


def test_write_atomic_replaces_the_file(tmp_path):
    path = tmp_path / "library.json"
    path.write_text("old", encoding="utf-8")
    caption.write_atomic(path, {"items": [HARVESTED]})
    assert json.loads(path.read_text(encoding="utf-8"))["items"][0]["id"] == "c0001"
    assert not list(tmp_path.glob("*.tmp"))


def test_context_names_title_source_and_url():
    text = caption.context_text(HARVESTED)
    assert "John Wayne Gacy 1978 mugshot" in text
    assert "commons" in text
    assert HARVESTED["source_url"] in text

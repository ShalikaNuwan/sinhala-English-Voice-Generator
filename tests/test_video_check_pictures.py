from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import check_pictures  # noqa: E402


def test_a_picture_stays_unless_it_misleads_is_out_of_era_or_careless():
    ok = {"fits": "partly", "misleads": False, "era_ok": True, "care_ok": True}
    assert check_pictures.passes(ok)
    assert check_pictures.passes(dict(ok, fits="no"))           # an unrelated stand-in stays, as a wish
    assert not check_pictures.passes(dict(ok, misleads=True))
    assert not check_pictures.passes(dict(ok, era_ok=False))
    assert not check_pictures.passes(dict(ok, care_ok=False))
    assert not check_pictures.passes(dict(ok, link_ok=False))       # a Chicago street is not Des Plaines
    assert not check_pictures.passes(None)


def test_a_replacement_may_not_already_run_within_a_minute():
    beats = [{"start": 0, "library_id": "a"}, {"start": 30, "library_id": "x"}, {"start": 100, "library_id": "b"}]
    assert check_pictures.reused_near(beats, 1, "a") and not check_pictures.reused_near(beats, 1, "b")


def test_a_check_is_cached_by_picture_and_beat_so_a_changed_intent_is_checked_again():
    beat = {"intent": "Show the house.", "narration": "The house on Summerdale."}
    assert check_pictures.check_key(beat, "c1") == check_pictures.check_key(dict(beat), "c1")
    assert check_pictures.check_key(beat, "c1") != check_pictures.check_key(beat, "c2")
    assert check_pictures.check_key(beat, "c1") != check_pictures.check_key(dict(beat, intent="Show the street."), "c1")


def test_another_choice_is_tried_as_the_beat_would_run_it_a_clip_with_its_range():
    beat = {"index": 3, "library_id": "c1", "asset": "library/c1.jpg", "media": "photo", "narration": "n"}
    clip = {"id": "v1#0", "file": "clips/v1.mp4", "media": "clip", "in": 4.0, "out": 9.0}
    trial = check_pictures.candidate_beat(beat, clip)
    assert (trial["library_id"], trial["media"], trial["clip_in"], trial["clip_out"]) == ("v1#0", "clip", 4.0, 9.0)
    assert beat["library_id"] == "c1"                      # the beat itself is untouched


def test_a_photograph_is_sent_small(tmp_path):
    Image.new("RGB", (2000, 1500), (90, 90, 90)).save(tmp_path / "big.jpg")
    data = check_pictures.picture_bytes(tmp_path, {"asset": "big.jpg", "media": "photo"})
    import io
    assert data and max(Image.open(io.BytesIO(data)).size) <= 512
    assert check_pictures.picture_bytes(tmp_path, {"asset": None}) is None


def test_a_clip_too_short_for_the_beat_is_never_offered_as_a_replacement():
    beat = {"start": 10.0, "end": 23.8}
    assert not check_pictures.long_enough({"media": "clip", "length": 9.5}, beat)
    assert check_pictures.long_enough({"media": "clip", "length": 15.0}, beat)
    assert check_pictures.long_enough({"media": "photo"}, beat)

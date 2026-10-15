"""Investigation devices: layout, typing, camera, stroke, boxes and the locate cache are pure and
tested without rendering; locate's two vision calls run against a fake client; each device renders
once at a small size and is checked with ffprobe."""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "video"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"video_{name}", ROOT / "scripts" / "video" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"video_{name}"] = module
    spec.loader.exec_module(module)
    return module


inv = load("investigation")
FFPROBE = str(Path(inv.ffmpeg_path()).with_name("ffprobe"))
W, H = inv.WIDTH, inv.HEIGHT


def photo(path: Path, size, colour=(120, 110, 100), mono=False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    draw = ImageDraw.Draw(image)
    for x in range(0, size[0], 9):
        draw.line((x, 0, x, size[1]), fill=(200, 60, 40) if not mono else (210, 210, 210))
    draw.ellipse((size[0] * 0.3, size[1] * 0.3, size[0] * 0.7, size[1] * 0.7), fill=(30, 30, 30))
    image.save(path)
    return path


def probe(path: Path) -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate,pix_fmt,color_space",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


FIELDS = [("NAME", "Robert Piest"), ("AGE", "15"),
          ("LAST SEEN", "December 11, 1978, Nisson Pharmacy, Des Plaines")]


# --- profile layout ---------------------------------------------------------------

@pytest.mark.parametrize("size", [(510, 754), (1505, 2000), (181, 255), (1600, 1200), (3000, 900), (60, 60)])
def test_profile_photo_and_panel_never_overlap_and_stay_in_frame(size):
    layout = inv.profile_layout(size, FIELDS)
    p = layout["photo"]
    shadowed = (p["footprint"][0], p["footprint"][1] - p["pin_r"],
                p["footprint"][2] + H * 0.02, p["footprint"][3] + H * 0.03)
    assert not inv.overlaps(shadowed, layout["panel"])
    assert shadowed[2] < layout["panel"][0]
    assert inv.inside(shadowed, W, H)
    assert inv.inside(layout["panel"], W, H)
    # The print stays in the left third (a little over, for the tilt), the panel on the right.
    assert p["footprint"][2] <= W * 0.42
    assert layout["panel"][0] >= W * 0.40


def test_profile_text_sits_inside_the_panel():
    layout = inv.profile_layout((510, 754), FIELDS)
    x0, y0, x1, y1 = layout["panel"]
    for field in layout["fields"]:
        lx, ly = field["label_xy"]
        vx, vy = field["value_xy"]
        assert x0 < lx < x1 and y0 < ly < y1
        bottom = vy + len(field["lines"]) * field["step"]
        assert y0 < vy and bottom <= y1
        for line in field["lines"]:
            assert vx + field["font"].getlength(line) <= x1


def test_a_small_photo_is_not_blown_up_past_the_cap():
    layout = inv.profile_layout((181, 255), FIELDS)
    on_screen = layout["photo"]["scale"] * (1 + inv.PROFILE_DRIFT)
    assert on_screen <= inv.MAX_UPSCALE + 1e-9
    big = inv.profile_layout((1505, 2000), FIELDS)
    assert layout["photo"]["h"] < big["photo"]["h"]  # shown smaller, not stretched to the full size


def test_a_long_profile_shrinks_to_fit():
    many = [(f"FIELD {i}", "A rather long value that will need to wrap onto a second line") for i in range(7)]
    layout = inv.profile_layout((510, 754), many)
    assert inv.inside(layout["panel"], W, H)
    assert layout["panel"][3] - layout["panel"][1] <= H * 0.82


def test_profile_needs_fields():
    with pytest.raises(ValueError):
        inv.profile_layout((100, 100), [])


def test_profile_pin_sits_on_the_print_border_not_the_face():
    layout = inv.profile_layout((1505, 2000), FIELDS)
    p = layout["photo"]
    top = p["cy"] - p["h"] / 2
    assert abs(p["pin"][1] - top) < p["h"] * 0.03


# --- exhibit layout -----------------------------------------------------------------

def _area(r):
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])


def _intersection(a, b):
    return (max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))


@pytest.mark.parametrize("size", [(496, 237), (1200, 628), (510, 754), (2400, 3000), (181, 255), (4000, 600)])
def test_exhibit_object_and_tag_inside_frame_and_tag_barely_covers_object(size):
    tag = inv.tag_layout("Diagram of the house, 8213 W. Summerdale, Kokomo Tribune, Dec 1978", 3)
    layout = inv.exhibit_layout(size, (tag["w"], tag["h"]))
    o, t = layout["object"], layout["tag"]
    assert inv.inside(o["footprint"], W, H)
    assert inv.inside(t["footprint"], W, H)
    shared = _area(_intersection(o["footprint"], t["footprint"]))
    assert shared <= 0.25 * _area(t["footprint"])
    assert o["scale"] * (1 + inv.EXHIBIT_PUSH) <= inv.MAX_UPSCALE + 1e-9


def test_exhibit_tag_goes_below_a_wide_object_and_beside_a_tall_one():
    tag = inv.tag_layout("Business card", None)
    wide = inv.exhibit_layout((1200, 600), (tag["w"], tag["h"]))
    tall = inv.exhibit_layout((600, 900), (tag["w"], tag["h"]))
    assert wide["tag"]["cy"] > wide["object"]["footprint"][3] - 1
    assert tall["tag"]["cx"] > tall["object"]["footprint"][2] - 1


def test_exhibit_push_never_loses_the_tag():
    tag = inv.tag_layout("Business card, P.D.M. Contractors", 2)
    for size in [(496, 237), (1200, 628), (510, 754)]:
        layout = inv.exhibit_layout(size, (tag["w"], tag["h"]))
        start, end = inv.exhibit_camera(layout)
        assert start == (0.5, 0.5)
        view = inv.view_box(end[0] * W, end[1] * H, 1 + inv.EXHIBIT_PUSH, (W, H))
        t = layout["tag"]["footprint"]
        assert view[0] <= t[0] and t[2] <= view[2] and view[1] <= t[1] and t[3] <= view[3]


def test_tag_text_wraps_inside_the_tag():
    tag = inv.tag_layout("Diagram of the house, 8213 W. Summerdale Avenue, Kokomo Tribune, December 1978", 1)
    head_font, text_font = tag["fonts"]
    for line in tag["lines"]:
        assert tag["text_x"] + text_font.getlength(line) <= tag["w"]
    assert tag["text_y"] + len(tag["lines"]) * tag["step"] <= tag["h"]
    assert tag["number"] == "No. 1"
    assert inv.tag_layout("x", None)["number"] is None


# --- typing ------------------------------------------------------------------------

def test_typing_plan_is_sequential_and_done_by_the_end():
    lengths = [12, 2, 25, 21]
    plan = inv.typing_plan(lengths, 0.45, 4.5, [0, 0.42, 0.42, 0.18])
    assert plan[-1][1] <= 4.5 + 1e-9
    for (a0, a1), (b0, b1) in zip(plan, plan[1:]):
        assert a0 < a1 <= b0 < b1


def test_characters_revealed_per_frame():
    lengths = [12, 2, 25]
    plan = inv.typing_plan(lengths, 0.4, 4.0)
    previous = [0] * len(lengths)
    revealed = []
    for n in range(0, 150):
        counts = inv.typed_counts(n / 30, plan, lengths)
        assert all(c >= p for c, p in zip(counts, previous))  # never untypes
        assert all(0 <= c <= m for c, m in zip(counts, lengths))
        # A later line only starts once the one above is complete.
        for i in range(1, len(lengths)):
            if counts[i]:
                assert counts[i - 1] == lengths[i - 1]
        revealed.append(sum(counts) - sum(previous))
        previous = counts
    assert previous == lengths
    assert max(revealed) <= 2  # about 13 characters a second at 30 fps: one, at most two, a frame
    assert inv.typed_counts(0.0, plan, lengths) == [0, 0, 0]


def test_short_text_in_a_long_window_types_at_a_readable_pace_and_finishes_early():
    plan = inv.typing_plan([5], 0.5, 20.0)
    t0, t1 = plan[0]
    assert t0 == 0.5
    assert t1 - t0 == pytest.approx(5 / 13.0)


def test_many_characters_in_a_short_window_speed_up_rather_than_overrun():
    plan = inv.typing_plan([200], 0.0, 2.0)
    assert plan[0][1] == pytest.approx(2.0)


def test_typing_line_points_at_the_line_being_typed():
    plan = inv.typing_plan([4, 4], 0.0, 2.0, [0, 0.5])
    assert inv.typing_line(0.01, plan, [4, 4]) == 0
    assert inv.typing_line(plan[1][0] + 0.01, plan, [4, 4]) == 1
    assert inv.typing_line(5.0, plan, [4, 4]) is None


def test_profile_lines_pause_longer_between_fields():
    layout = inv.profile_layout((510, 754), FIELDS)
    lengths, pauses = inv.profile_lines(layout)
    assert sum(lengths) == sum(len(line) for f in layout["fields"] for line in f["lines"])
    assert pauses[0] == 0 and pauses[1] > 0.3


# --- highlight camera and stroke -------------------------------------------------------

BOX_PX = (900.0, 520.0, 1010.0, 600.0)


def test_highlight_camera_starts_on_the_whole_print():
    cx, cy, zoom = inv.highlight_camera(0.0, BOX_PX)
    assert (cx, cy, zoom) == pytest.approx((W / 2, H / 2, 1.0))


def test_highlight_camera_ends_with_the_box_centred_at_the_target_size():
    camera = inv.highlight_camera(1.0, BOX_PX)
    centre = inv.to_screen(((BOX_PX[0] + BOX_PX[2]) / 2, (BOX_PX[1] + BOX_PX[3]) / 2), camera)
    assert centre == pytest.approx((W / 2, H / 2), abs=1e-6)
    top = inv.to_screen((BOX_PX[0], BOX_PX[1]), camera)
    bottom = inv.to_screen((BOX_PX[2], BOX_PX[3]), camera)
    assert (bottom[1] - top[1]) / H == pytest.approx(inv.HIGHLIGHT_TARGET_H)


def test_a_wide_box_is_held_to_the_width_cap_instead():
    wide = (400.0, 500.0, 1400.0, 560.0)
    camera = inv.highlight_camera(1.0, wide)
    a, b = inv.to_screen(wide[:2], camera), inv.to_screen(wide[2:], camera)
    assert (b[0] - a[0]) / W == pytest.approx(inv.HIGHLIGHT_MAX_W)
    assert (b[1] - a[1]) / H < inv.HIGHLIGHT_TARGET_H


def test_highlight_camera_zooms_steadily_and_the_box_never_leaves_the_frame():
    zooms = []
    for i in range(101):
        camera = inv.highlight_camera(i / 100, BOX_PX)
        zooms.append(camera[2])
        x, y = inv.to_screen(((BOX_PX[0] + BOX_PX[2]) / 2, (BOX_PX[1] + BOX_PX[3]) / 2), camera)
        assert 0 <= x <= W and 0 <= y <= H
    assert all(b >= a - 1e-12 for a, b in zip(zooms, zooms[1:]))


def test_print_rect_matches_assembly_print_rules():
    small = inv.print_rect((496, 237))
    assert (small[2] - small[0]) == pytest.approx(496 * inv.MAX_UPSCALE)
    big = inv.print_rect((4000, 3000))
    assert (big[3] - big[1]) == pytest.approx(H * inv.PRINT_FIT)
    assert (big[0] + big[2]) / 2 == pytest.approx(W / 2)


def test_stroke_progress_runs_zero_to_one():
    assert inv.stroke_progress(0.9, 1.0, 0.8) == 0
    assert inv.stroke_progress(1.0, 1.0, 0.8) == 0
    assert inv.stroke_progress(1.8, 1.0, 0.8) == 1
    assert inv.stroke_progress(5.0, 1.0, 0.8) == 1
    values = [inv.stroke_progress(1.0 + i * 0.01, 1.0, 0.8) for i in range(81)]
    assert all(b >= a for a, b in zip(values, values[1:]))
    assert inv.stroke_progress(1.4, 1.0, 0.8) == pytest.approx(0.5)


def test_ellipse_stroke_rings_the_box_and_closes_past_its_start():
    points = inv.stroke_points("ellipse", seed=3)
    assert all(math.hypot(x, y) > 1.15 for x, y, _ in points)  # clear of the detail itself
    assert all(math.hypot(x, y) < 1.8 for x, y, _ in points)   # but close round it
    angles = [math.atan2(y, x) for x, y, _ in points]
    turned = sum((b - a + math.pi) % (2 * math.pi) - math.pi for a, b in zip(angles, angles[1:]))
    assert abs(turned) > 2 * math.pi  # a little more than one full turn: the pen overlaps its start
    widths = [w for _, _, w in points]
    assert widths[0] < max(widths) and widths[-1] < max(widths)  # tapered at both ends


def test_box_stroke_stays_outside_the_box():
    for x, y, _ in inv.stroke_points("box", seed=5):
        assert max(abs(x), abs(y)) >= 1.0


def test_stroke_is_hand_drawn_not_a_perfect_ellipse_and_seeded():
    a = inv.stroke_points("ellipse", seed=1)
    b = inv.stroke_points("ellipse", seed=1)
    c = inv.stroke_points("ellipse", seed=2)
    assert a == b and a != c
    radii = [math.hypot(x / 1.3, y / 1.36) for x, y, _ in a]
    assert max(radii) - min(radii) > 0.03


def test_partial_stroke_follows_length():
    points = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (2.0, 0.0, 1.0)]
    assert inv.partial_stroke(points, 0) == []
    half = inv.partial_stroke(points, 0.5)
    assert half[-1][:2] == pytest.approx((1.0, 0.0))
    quarter = inv.partial_stroke(points, 0.25)
    assert quarter[-1][:2] == pytest.approx((0.5, 0.0))
    assert inv.partial_stroke(points, 1.0) == points


def test_label_goes_right_of_the_ring_else_left_else_below():
    plate = (200, 50)
    right = inv.label_spot((800, 400, 1100, 700), plate)
    assert right[0] > 1100
    left = inv.label_spot((1500, 400, 1800, 700), plate)
    assert left[0] + plate[0] < 1500
    below = inv.label_spot((100, 200, 1820, 600), plate)
    assert below[1] > 600
    for spot in (right, left, below):
        assert 0 <= spot[0] and spot[0] + plate[0] <= W and 0 <= spot[1] and spot[1] + plate[1] <= H


# --- boxes ---------------------------------------------------------------------------

def test_clamp_box_orders_clamps_and_widens():
    assert inv.clamp_box((0.6, 0.7, 0.2, 0.1)) == pytest.approx((0.2, 0.1, 0.6, 0.7))
    assert inv.clamp_box((-0.2, 0.1, 1.4, 0.9)) == pytest.approx((0.0, 0.1, 1.0, 0.9))
    x0, y0, x1, y1 = inv.clamp_box((0.5, 0.5, 0.505, 0.9))
    assert x1 - x0 == pytest.approx(inv.HIGHLIGHT_MIN_BOX)
    assert (x0 + x1) / 2 == pytest.approx(0.5025)
    x0, _, x1, _ = inv.clamp_box((0.999, 0.1, 1.0, 0.5))
    assert x1 <= 1.0 and x1 - x0 == pytest.approx(inv.HIGHLIGHT_MIN_BOX)


@pytest.mark.parametrize("bad", [None, (0.1, 0.2, 0.3), (0.1, float("nan"), 0.3, 0.4)])
def test_clamp_box_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        inv.clamp_box(bad)


def test_normalise_box_accepts_fractions_dicts_and_pixels():
    assert inv.normalise_box([0.1, 0.2, 0.5, 0.6]) == (0.1, 0.2, 0.5, 0.6)
    assert inv.normalise_box({"x0": 0.5, "y0": 0.6, "x1": 0.1, "y1": 0.2, "found": True}) == (0.1, 0.2, 0.5, 0.6)
    assert inv.normalise_box([50, 20, 150, 120], image_size=(200, 200)) == (0.25, 0.1, 0.75, 0.6)
    assert inv.normalise_box([0.0, 0.0, 1.004, 0.5]) == (0.0, 0.0, 1.0, 0.5)  # rounding over 1 is clamped


@pytest.mark.parametrize("bad", [
    None, [0.1, 0.2, 0.3], {"x0": 0.1}, ["a", 0, 1, 1], [0.2, 0.2, 0.2, 0.8],
    [0.1, 0.1, 0.105, 0.105], [50, 20, 150, 120], [-0.5, 0.1, 0.5, 0.5], [0.1, float("inf"), 0.5, 0.5],
])
def test_normalise_box_rejects_malformed(bad):
    assert inv.normalise_box(bad) is None


def test_verify_crop_adds_a_margin_and_stays_inside(tmp_path):
    image = Image.new("RGB", (400, 200))
    crop = inv.verify_crop(image, (0.25, 0.25, 0.75, 0.75), margin=0.2)
    assert crop.size == (280, 140)  # the 200x100 box plus 20% of itself on every side
    edge = inv.verify_crop(image, (0.0, 0.0, 0.1, 0.1))
    assert edge.size[0] <= 400 and edge.size[1] <= 200


# --- locate cache and fake client -------------------------------------------------------

class FakeResponses:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return SimpleNamespace(output_parsed=None)
        return SimpleNamespace(output_parsed=kwargs["text_format"](**answer))


class FakeClient:
    def __init__(self, *answers):
        self.responses = FakeResponses(answers)


FOUND = {"found": True, "x0": 0.40, "y0": 0.50, "x1": 0.55, "y1": 0.76, "reason": "labelled"}
YES = {"shows": True, "reason": "The crop shows the crawl space label."}
NO = {"shows": False, "reason": "This is the driveway."}


def test_cache_key_depends_on_file_content_and_description(tmp_path):
    a = photo(tmp_path / "a.png", (40, 30))
    b = photo(tmp_path / "b.png", (40, 30), colour=(10, 10, 10))
    same = tmp_path / "copy.png"
    same.write_bytes(a.read_bytes())
    assert inv.cache_key(a, "the crawl space") == inv.cache_key(same, "The  crawl space ")
    assert inv.cache_key(a, "the crawl space") != inv.cache_key(b, "the crawl space")
    assert inv.cache_key(a, "the crawl space") != inv.cache_key(a, "the garage")


def test_cache_writes_keep_other_entries(tmp_path):
    cache = tmp_path / "sub" / "locate_cache.json"
    inv.write_cache(cache, "k1", {"box": [0.1, 0.1, 0.2, 0.2]})
    inv.write_cache(cache, "k2", {"box": None})
    data = json.loads(cache.read_text())
    assert set(data) == {"k1", "k2"}
    assert not list(cache.parent.glob("*.tmp"))
    (tmp_path / "broken.json").write_text("{not json")
    assert inv.read_cache(tmp_path / "broken.json") == {}
    assert inv.read_cache(tmp_path / "missing.json") == {}


def test_locate_returns_the_box_on_yes_and_caches_it(tmp_path):
    picture = photo(tmp_path / "diagram.png", (496, 237), mono=True)
    cache = tmp_path / "locate_cache.json"
    client = FakeClient(FOUND, YES)
    box = inv.locate(picture, "the crawl space", client=client, model="fake", cache_path=cache)
    assert box == (0.4, 0.5, 0.55, 0.76)
    assert len(client.responses.calls) == 2
    first, second = client.responses.calls
    assert first["model"] == "fake"
    content = first["input"][0]["content"]
    assert content[1]["type"] == "input_image" and content[1]["image_url"].startswith("data:image/jpeg;base64,")
    assert "crawl space" in content[0]["text"]
    assert "crawl space" in second["input"][0]["content"][0]["text"]
    entry = inv.cached_entry(picture, "the crawl space", cache)
    assert entry["verified"] is True and entry["box"] == [0.4, 0.5, 0.55, 0.76]

    # A rerun costs nothing.
    silent = FakeClient()
    assert inv.locate(picture, "the crawl space", client=silent, model="fake", cache_path=cache) == box
    assert silent.responses.calls == []


def test_locate_returns_none_on_no_and_caches_the_no(tmp_path):
    picture = photo(tmp_path / "diagram.png", (496, 237))
    cache = tmp_path / "locate_cache.json"
    client = FakeClient(FOUND, NO)
    assert inv.locate(picture, "the crawl space", client=client, model="fake", cache_path=cache) is None
    entry = inv.cached_entry(picture, "the crawl space", cache)
    assert entry["box"] is None and entry["verified"] is False
    assert entry["candidate"] == [0.4, 0.5, 0.55, 0.76]
    assert "driveway" in entry["reason"]
    silent = FakeClient()
    assert inv.locate(picture, "the crawl space", client=silent, model="fake", cache_path=cache) is None
    assert silent.responses.calls == []


@pytest.mark.parametrize("first", [
    None,                                                     # no structured output at all
    {"found": False, "x0": 0, "y0": 0, "x1": 0, "y1": 0},     # model says it is not there
    {"found": True, "x0": 0.3, "y0": 0.3, "x1": 0.3, "y1": 0.6},  # degenerate box
    {"found": True, "x0": 30, "y0": 20, "x1": 9000, "y1": 50},    # pixels past the image
])
def test_locate_returns_none_on_a_malformed_or_empty_first_answer(tmp_path, first):
    picture = photo(tmp_path / "p.png", (300, 200))
    client = FakeClient(first)
    assert inv.locate(picture, "a door", client=client, model="fake", cache_path=tmp_path / "c.json") is None
    assert len(client.responses.calls) == 1  # no verification without a box


def test_locate_returns_none_on_a_malformed_verdict(tmp_path):
    picture = photo(tmp_path / "p.png", (300, 200))
    client = FakeClient(FOUND, None)
    assert inv.locate(picture, "a door", client=client, model="fake", cache_path=tmp_path / "c.json") is None


def test_locate_does_not_cache_api_errors(tmp_path):
    picture = photo(tmp_path / "p.png", (300, 200))
    cache = tmp_path / "c.json"
    assert inv.locate(picture, "a door", client=FakeClient(RuntimeError("503")), model="fake", cache_path=cache) is None
    assert inv.read_cache(cache) == {}
    assert inv.locate(picture, "a door", client=FakeClient(FOUND, RuntimeError("timeout")), model="fake",
                      cache_path=cache) is None
    assert inv.read_cache(cache) == {}
    # Next time it asks again, and succeeds.
    assert inv.locate(picture, "a door", client=FakeClient(FOUND, YES), model="fake", cache_path=cache) is not None


def test_locate_sends_small_scans_enlarged(tmp_path):
    picture = photo(tmp_path / "p.png", (181, 255))
    client = FakeClient(FOUND, YES)
    inv.locate(picture, "a face", client=client, model="fake", cache_path=tmp_path / "c.json")
    import base64
    import io
    url = client.responses.calls[0]["input"][0]["content"][1]["image_url"]
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(sent.size) >= inv.LOCATE_MIN_SIDE


def test_locate_needs_a_description(tmp_path):
    with pytest.raises(ValueError):
        inv.locate(photo(tmp_path / "p.png", (30, 30)), "  ", client=FakeClient(), cache_path=tmp_path / "c.json")


# --- grading ----------------------------------------------------------------------------

def test_grade_keeps_black_and_white_neutral_and_colour_coloured():
    grey = Image.new("RGB", (64, 64), (128, 128, 128))
    out = inv.grade(grey).getpixel((5, 5))
    assert max(out) - min(out) <= 6 and out[2] >= out[0]  # neutral, the faintest cool tone
    sepia = Image.new("RGB", (64, 64), (150, 132, 108))
    out = inv.grade(sepia).getpixel((5, 5))
    assert max(out) - min(out) <= 6  # a toned scan becomes black-and-white
    red = Image.new("RGB", (64, 64), (200, 40, 40))
    out = inv.grade(red).getpixel((5, 5))
    assert out[0] > out[1] + 60  # colour stays colour
    card = Image.new("RGB", (64, 64), (225, 222, 215))  # a near-grey card with a band of red ink
    ImageDraw.Draw(card).rectangle((10, 30, 50, 36), fill=(200, 40, 35))
    graded = inv.grade(card).getpixel((30, 33))
    assert graded[0] > graded[1] + 60


# --- small real renders -----------------------------------------------------------------

def _check(path: Path, seconds: float, width: int, height: int):
    stream = probe(path)
    assert int(stream["width"]) == width and int(stream["height"]) == height
    assert int(stream["nb_read_frames"]) == round(seconds * 30)
    assert stream["r_frame_rate"] == "30/1"
    assert stream["pix_fmt"] == "yuv420p"


def test_profile_card_renders_exact_frames(tmp_path):
    out = inv.profile_card(photo(tmp_path / "p.png", (181, 255)), FIELDS, tmp_path / "profile.mp4", 1.27,
                           width=480, height=270)
    _check(out, 1.27, 480, 270)
    assert not list(tmp_path.glob("*.part.mp4"))


def test_exhibit_renders_exact_frames(tmp_path):
    out = inv.exhibit(photo(tmp_path / "e.png", (496, 237), colour=(235, 235, 235)), "Diagram of the house",
                      tmp_path / "exhibit.mp4", 1.0, number=4, width=480, height=270)
    _check(out, 1.0, 480, 270)


def test_highlight_renders_exact_frames(tmp_path):
    out = inv.highlight(photo(tmp_path / "h.png", (496, 237)), (0.4, 0.5, 0.55, 0.76), tmp_path / "hl.mp4", 1.5,
                        label="Crawl space", width=480, height=270)
    _check(out, 1.5, 480, 270)
    box = inv.highlight(photo(tmp_path / "h2.png", (300, 400)), (0.1, 0.1, 0.3, 0.2), tmp_path / "hl2.mp4", 1.0,
                        shape="box", width=480, height=270)
    _check(box, 1.0, 480, 270)


def test_devices_refuse_nonsense(tmp_path):
    p = photo(tmp_path / "p.png", (100, 100))
    with pytest.raises(ValueError):
        inv.profile_card(p, FIELDS, tmp_path / "x.mp4", 0)
    with pytest.raises(ValueError):
        inv.exhibit(p, "  ", tmp_path / "x.mp4", 2)
    with pytest.raises(ValueError):
        inv.highlight(p, (0.1, 0.1, 0.2, 0.2), tmp_path / "x.mp4", 2, shape="star")
    assert not (tmp_path / "x.mp4").exists()

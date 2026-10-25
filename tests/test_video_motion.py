"""Motion graphics: layout, camera and timing are pure and tested without rendering, plus small renders."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "video"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(f"video_{name}", ROOT / "scripts" / "video" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"video_{name}"] = module
    spec.loader.exec_module(module)
    return module


motion = load("motion")
graphics = load("graphics")
FFPROBE = str(Path(motion.ffmpeg_path()).with_name("ffprobe"))


def probe(path: Path, entries: str, stream: str = "v:0") -> dict:
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", stream, "-show_entries", entries,
                          "-of", "default=nw=1", str(path)], capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in out.stdout.strip().splitlines() if "=" in line)


def photo(path: Path, size, colour=(120, 110, 100)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    for x in range(0, size[0], 7):
        for y in range(size[1]):
            image.putpixel((x, y), (200, 190, 180))
    image.save(path)
    return path


# --- evidence board: layout ---------------------------------------------------

ASPECTS = [0.75, 0.68, 2.09, 1.91, 1.26, 1.0, 0.8, 1.5]
BOARD = (3840, 2160)


@pytest.mark.parametrize("count", range(1, 9))
def test_board_prints_never_overlap_more_than_slightly_and_stay_on_the_board(count):
    labels = [(f"Label number {i}", "a typed line beneath") for i in range(count)]
    items = motion.board_layout(ASPECTS[:count], labels, BOARD)
    prints = [motion.footprint(item) for item in items]

    for i, a in enumerate(prints):
        assert a[0] >= 0 and a[1] >= 0 and a[2] <= BOARD[0] and a[3] <= BOARD[1], f"pin {i} leaves the board"
        for b in prints[i + 1:]:
            assert motion.overlap_fraction(a, b) <= motion.BOARD_OVERLAP + 1e-9


def test_the_subject_is_the_largest_print_in_the_middle():
    items = motion.board_layout([0.75, 0.75, 0.75, 0.75, 0.75], [("a", None)] * 5, BOARD)

    assert items[0]["h"] > max(item["h"] for item in items[1:])
    assert abs(items[0]["cx"] - BOARD[0] / 2) < BOARD[0] * 0.05


def test_every_print_is_tilted_a_little_and_never_much():
    items = motion.board_layout(ASPECTS[:6], [("x", None)] * 6, BOARD)

    for item in items:
        assert motion.BOARD_TILT[0] <= abs(item["angle"]) <= motion.BOARD_TILT[1]


def test_the_layout_is_the_same_every_time_for_the_same_seed():
    labels = [("a", "b")] * 5
    assert motion.board_layout(ASPECTS[:5], labels, BOARD) == motion.board_layout(ASPECTS[:5], labels, BOARD)


def test_a_pin_sits_in_the_top_of_its_print():
    item = motion.board_layout([0.75], [("a", None)], BOARD)[0]
    x, y = item["pin"]

    assert abs(x - item["cx"]) < item["w"] * 0.1
    assert item["cy"] - item["h"] / 2 < y < item["cy"] - item["h"] * 0.4


def test_links_name_pins_by_index_or_label():
    pins = [{"label": "John Wayne Gacy"}, {"label": "Robert Piest"}, {"label": "Des Plaines Police"}]

    assert motion.resolve_links([("John Wayne Gacy", "robert piest"), (2, 0)], pins) == [(0, 1), (2, 0)]
    with pytest.raises(ValueError):
        motion.resolve_links([("John Wayne Gacy", "Nobody")], pins)
    with pytest.raises(ValueError):
        motion.resolve_links([(0, 0)], pins)


def test_strings_draw_one_at_a_time_and_finish_well_before_the_end():
    schedule = motion.string_schedule(4, 10.0)

    assert schedule[0][0] > 0.5
    for (start, end), (next_start, _) in zip(schedule, schedule[1:]):
        assert start < end <= next_start
    assert schedule[-1][1] <= 10.0 - 1.0


def test_the_board_camera_pushes_in_once_and_keeps_every_print_in_frame():
    labels = [("John Wayne Gacy", "Arrested"), ("Robert Piest", "15"), ("8213 W. Summerdale", "x"),
              ("P.D.M. Contractors", "y"), ("Des Plaines Police", "z")]
    items = motion.board_layout([0.75, 0.68, 2.09, 1.91, 1.26], labels, BOARD)
    keys = motion.board_keyframes(items, BOARD, 10.0)

    assert keys[0][3] == 1.0 and 1.0 < keys[-1][3] <= motion.BOARD_PUSH_MAX
    zooms = [motion.camera_at(t / 10, keys)[2] for t in range(101)]
    assert all(b >= a - 1e-9 for a, b in zip(zooms, zooms[1:])), "the push never pulls back"
    for t in (0.0, 5.0, 10.0):
        x0, y0, x1, y1 = motion.view_box(*motion.camera_at(t, keys), BOARD)
        for fx0, fy0, fx1, fy1 in map(motion.footprint, items):
            assert x0 <= fx0 and y0 <= fy0 and fx1 <= x1 and fy1 <= y1


def test_the_camera_eases_at_both_ends_and_hits_its_keyframes():
    keys = [(0.0, 100.0, 100.0, 1.0), (10.0, 300.0, 200.0, 1.2)]

    assert motion.camera_at(0.0, keys) == (100.0, 100.0, 1.0)
    assert motion.camera_at(10.0, keys) == pytest.approx((300.0, 200.0, 1.2))
    start_step = motion.camera_at(0.1, keys)[0] - motion.camera_at(0.0, keys)[0]
    middle_step = motion.camera_at(5.1, keys)[0] - motion.camera_at(5.0, keys)[0]
    assert start_step < middle_step / 10


def test_a_view_never_leaves_the_canvas():
    assert motion.view_box(0, 0, 1.5, (3840, 2160)) == (0, 0, 2560, 1440)
    x0, y0, x1, y1 = motion.view_box(3840, 2160, 2.0, (3840, 2160))
    assert (x1, y1) == (3840, 2160)


# --- map: pin timing ------------------------------------------------------------

def test_map_pins_land_in_order_evenly_spaced_and_the_finished_map_is_held():
    plan = motion.map_schedule(4, 8.0)
    lands = [p["land"] for p in plan]

    assert lands == sorted(lands)
    gaps = {round(b - a, 6) for a, b in zip(lands, lands[1:])}
    assert len(gaps) == 1
    assert lands[-1] <= 8.0 - 1.6 + 1e-9
    for step in plan:
        assert step["drop_from"] == pytest.approx(step["land"] - motion.PIN_DROP_S)
        assert step["label_from"] > step["land"]


def test_the_route_into_a_pin_draws_before_it_lands_and_the_first_pin_has_none():
    plan = motion.map_schedule(3, 6.0)

    assert plan[0]["route_from"] is None
    for previous, step in zip(plan, plan[1:]):
        assert previous["land"] <= step["route_from"] < step["land"]


def test_a_long_map_shot_does_not_wait_forever_between_pins():
    plan = motion.map_schedule(3, 30.0)

    assert plan[1]["land"] - plan[0]["land"] <= 1.8 + 1e-9


def test_one_point_drops_after_the_lead_in():
    plan = motion.map_schedule(1, 4.0)

    assert len(plan) == 1 and plan[0]["drop_from"] == pytest.approx(0.7)


def test_close_map_labels_never_collide_and_never_sit_on_a_pin():
    size = (3840, 2160)
    points = [{"x": 0.30, "y": 0.38}, {"x": 0.33, "y": 0.42}, {"x": 0.52, "y": 0.47}, {"x": 0.40, "y": 0.70}]
    spots = [motion.map_place(p, size) for p in points]
    widths, h, radius = [400, 560, 640, 600], 60, 20
    corners = motion.label_spots(spots, widths, h, radius, (0, 0) + size)
    rects = [(x, y, x + w, y + h) for (x, y), w in zip(corners, widths)]
    pins = [(x - radius, y - radius, x + radius, y + radius) for x, y in spots]

    for i, rect in enumerate(rects):
        for other in rects[i + 1:] + pins:
            assert motion.overlap_fraction(rect, other) == 0


# --- typed stamp: what shows when --------------------------------------------------

def test_the_stamp_types_the_date_then_the_place_with_a_solid_caret_while_typing():
    plan = motion.typing_plan(17, 21, finish_by=2.5)

    assert motion.typed_state(0.0, plan)[:2] == (0, 0)
    date, place, caret = motion.typed_state(plan["type_from"] + 5.5 / plan["date_cps"], plan)
    assert (date, place, caret) == (5, 0, "date")
    date, place, caret = motion.typed_state(plan["place_from"] + 3.5 / plan["place_cps"], plan)
    assert (date, place, caret) == (17, 3, "place")
    assert motion.typed_state(plan["done"] + 0.01, plan)[:2] == (17, 21)


def test_the_caret_blinks_before_typing_and_after_it():
    plan = motion.typing_plan(17, 21, finish_by=2.5)

    assert motion.typed_state(0.1, plan)[2] == "date" and motion.typed_state(0.52, plan)[2] is None
    after = plan["done"]
    assert motion.typed_state(after + 0.2, plan)[2] == "place"
    assert motion.typed_state(after + 0.7, plan)[2] is None
    assert motion.typed_state(after + 1.2, plan)[2] == "place"


def test_every_frame_shows_at_least_as_much_as_the_one_before():
    plan = motion.typing_plan(17, 21, finish_by=2.5)
    states = [motion.typed_state(n / 30, plan) for n in range(150)]

    for (d1, p1, _), (d2, p2, _) in zip(states, states[1:]):
        assert d2 >= d1 and p2 >= p1
    assert states[-1][:2] == (17, 21)


def test_long_text_types_faster_to_be_done_on_time():
    plan = motion.typing_plan(60, 60, finish_by=1.5)

    assert plan["done"] == pytest.approx(1.5)
    assert plan["date_cps"] > motion.DATE_CPS


def test_a_stamp_without_a_place_ends_on_the_date():
    plan = motion.typing_plan(17, 0, finish_by=2.5)

    assert motion.typed_state(plan["done"] + 0.1, plan) == (17, 0, "date")


def test_keystrokes_land_once_per_character_in_order():
    plan = motion.typing_plan(17, 21, finish_by=2.5)
    keys = motion.keystrokes(plan)

    assert len(keys) == 38 and keys == sorted(keys)
    assert motion.typed_state(keys[0] + 0.001, plan)[0] == 1


def test_the_stamp_text_length_counts_what_is_typed():
    date, place = graphics.stamp_text_length("December 11, 1978", "Des Plaines, Illinois")

    assert (date, place) == (17, 21)


# --- small real renders -------------------------------------------------------------

def test_an_evidence_board_renders_at_a_small_size(tmp_path):
    pins = [{"image": photo(tmp_path / "a.jpg", (300, 400)), "label": "John Wayne Gacy", "sub": "Arrested"},
            {"image": photo(tmp_path / "b.png", (500, 250), (60, 70, 80)), "label": "8213 W. Summerdale"},
            {"image": photo(tmp_path / "c.jpg", (400, 300)), "label": "Des Plaines Police"}]
    out = motion.evidence_board(pins, [(0, 1), ("Des Plaines Police", "John Wayne Gacy")],
                                tmp_path / "board.mp4", seconds=3.0, width=320, height=180, fps=15)

    info = probe(out, "stream=width,height,nb_frames,pix_fmt,color_range")
    assert (info["width"], info["height"], info["nb_frames"]) == ("320", "180", "45")
    assert info["pix_fmt"] == "yuv420p" and info["color_range"] == "tv"
    assert not (tmp_path / "board.part.mp4").exists()


def test_a_map_renders_at_a_small_size(tmp_path):
    out = motion.map_reveal([{"name": "Des Plaines", "x": 0.3, "y": 0.38},
                             {"name": "Nisson Pharmacy", "x": 0.33, "y": 0.42}],
                            tmp_path / "map.mp4", seconds=2.0, title="Cook County", width=320, height=180, fps=15)

    info = probe(out, "stream=width,height,nb_frames")
    assert (info["width"], info["height"], info["nb_frames"]) == ("320", "180", "30")


def test_a_typed_stamp_is_a_transparent_mov(tmp_path):
    out = motion.typed_stamp("December 11, 1978", "Des Plaines, Illinois", tmp_path / "stamp.mov",
                             seconds=2.0, width=320, height=180, fps=15)

    info = probe(out, "stream=codec_name,pix_fmt,nb_frames")
    assert info == {"codec_name": "qtrle", "pix_fmt": "argb", "nb_frames": "30"}


def test_renderers_refuse_nonsense_lengths(tmp_path):
    with pytest.raises(ValueError):
        motion.evidence_board([{"image": photo(tmp_path / "a.jpg", (30, 40))}], [], tmp_path / "x.mp4", seconds=1.0)
    with pytest.raises(ValueError):
        motion.map_reveal([], tmp_path / "y.mp4")

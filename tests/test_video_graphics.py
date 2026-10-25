from __future__ import annotations

import pytest
from PIL import Image, ImageChops

from scripts.video import graphics

LONG_TITLE = (
    "The Contractor Who Kept A Crawl Space: How Thirty Three Young Men Vanished Into A Suburban "
    "Ranch House In Norwood Park Township Over Six Years And Nobody Asked"
)


def ink(image: Image.Image, box: tuple[int, int, int, int] | None = None) -> bool:
    """True when a region holds more than the flat background."""
    region = image.crop(box) if box else image
    low, high = region.convert("L").getextrema()
    return low != high


def all_background(image: Image.Image, box: tuple[int, int, int, int]) -> bool:
    return set(image.crop(box).getdata()) == {graphics.BACKGROUND}


def count(image: Image.Image, colour: tuple[int, int, int]) -> int:
    return sum(n for n, value in image.getcolors(maxcolors=1 << 24) if value == colour)


def events(n: int) -> list[dict]:
    return [{"label": f"Dec {10 + i}, 1978", "text": f"Event number {i} of the investigation"} for i in range(n)]


def nodes(n: int) -> list[dict]:
    return [{"name": f"Witness Number {i}", "role": "interviewed 1978"} for i in range(n)]


# --- the frame itself ------------------------------------------------------

@pytest.mark.parametrize("render", [
    lambda: graphics.title_card("Thirty Three", "An episode about John Wayne Gacy"),
    lambda: graphics.date_card("December 11th, 1978", "Des Plaines, Illinois"),
    lambda: graphics.timeline(events(4), highlight=1),
    lambda: graphics.relationship_chart(nodes(4), [(0, 1, "employed")]),
    lambda: graphics.quote_card("He told me he was a clown at children's parties.", "Neighbour, 1979"),
    lambda: graphics.map_card([{"name": "Des Plaines", "x": 0.3, "y": 0.4}], "Cook County"),
])
def test_every_renderer_returns_a_full_frame_on_the_dark_ground(render):
    image = render()

    assert image.size == (1920, 1080)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == graphics.BACKGROUND
    assert image.getpixel((1919, 1079)) == graphics.BACKGROUND
    assert ink(image), "nothing was drawn"


def test_renderers_honour_a_different_frame_size():
    assert graphics.title_card("Thirty Three", width=1280, height=720).size == (1280, 720)
    assert graphics.timeline(events(3), width=1280, height=720).size == (1280, 720)


# --- cards ----------------------------------------------------------------

def test_a_long_title_wraps_instead_of_overflowing_the_frame():
    image = graphics.title_card(LONG_TITLE, "Chapter Four")

    assert image.size == (1920, 1080)
    assert all_background(image, (0, 1079, 1920, 1080)), "text reached the bottom row"
    assert all_background(image, (0, 0, 1920, 1)), "text reached the top row"
    assert ink(image)


def test_a_long_title_is_wrapped_to_at_most_three_lines():
    lines = graphics.fit_lines(LONG_TITLE, graphics.SERIF, 110, max_width=1600, max_lines=3)[1]

    assert 1 < len(lines) <= 3
    assert " ".join(lines).split() == LONG_TITLE.split(), "wrapping lost or reordered words"


def test_the_date_card_is_stamped_lower_left():
    image = graphics.date_card("December 11th, 1978", "Des Plaines, Illinois")

    assert ink(image, (0, 540, 960, 1080)), "nothing in the lower left"
    assert all_background(image, (0, 0, 1920, 540)), "the date card drew into the top half"
    assert all_background(image, (1400, 0, 1920, 1080)), "the date card drew into the right margin"


def test_a_card_without_subtext_still_renders():
    assert ink(graphics.title_card("Thirty Three"))
    assert ink(graphics.date_card("December 11th, 1978"))


def test_the_quote_card_carries_its_accent_quotation_mark():
    image = graphics.quote_card("He told me he was a clown at children's parties.", "Neighbour, 1979")

    assert count(image, graphics.ACCENT) > 0


# --- timeline -------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 5, 8])
def test_the_timeline_renders_from_one_to_eight_events(n):
    image = graphics.timeline(events(n))

    assert image.size == (1920, 1080)
    assert ink(image)
    assert all_background(image, (0, 1079, 1920, 1080))


def test_the_timeline_refuses_more_than_eight_events():
    with pytest.raises(ValueError):
        graphics.timeline(events(9))


def test_the_timeline_refuses_no_events():
    with pytest.raises(ValueError):
        graphics.timeline([])


def test_a_highlighted_event_adds_accent():
    plain = graphics.timeline(events(4))
    marked = graphics.timeline(events(4), highlight=2)

    assert count(marked, graphics.ACCENT) > count(plain, graphics.ACCENT)
    assert marked.tobytes() != plain.tobytes()


def test_a_highlight_out_of_range_is_refused():
    with pytest.raises(ValueError):
        graphics.timeline(events(3), highlight=3)


# --- relationship chart ---------------------------------------------------

@pytest.mark.parametrize("n", [2, 6, 7, 8])
def test_the_relationship_chart_renders_a_circle_and_two_rows(n):
    image = graphics.relationship_chart(nodes(n), [(0, 1, "employed")])

    assert ink(image)
    assert all_background(image, (0, 0, 1920, 1))


def test_the_relationship_chart_refuses_more_than_eight_nodes():
    with pytest.raises(ValueError):
        graphics.relationship_chart(nodes(9), [])


def test_the_relationship_chart_refuses_no_nodes():
    with pytest.raises(ValueError):
        graphics.relationship_chart([], [])


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_node_boxes_never_overlap_and_stay_inside_the_frame(n):
    boxes = graphics.node_boxes(n, 1920, 1080)

    assert len(boxes) == n
    for left, top, right, bottom in boxes:
        assert 0 < left < right < 1920
        assert 0 < top < bottom < 1080
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            apart = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
            assert apart, f"{a} overlaps {b} with {n} nodes"


def test_an_edge_label_is_never_buried_under_a_node_box():
    items = nodes(8)  # two rows: side-by-side boxes swallow a label left at the line's midpoint
    label = "offered him work the week before"
    plain = graphics.relationship_chart(items, [(0, 1, "")])
    labelled = graphics.relationship_chart(items, [(0, 1, label)])

    changed = ImageChops.difference(plain, labelled).getbbox()
    assert changed is not None, "the label drew nothing"
    assert changed[2] - changed[0] >= graphics.font(graphics.SANS, int(1080 * 0.019)).getlength(label), \
        "only part of the label survived: a node box is covering it"
    for box in graphics.node_boxes(8, 1920, 1080):
        apart = changed[2] <= box[0] or box[2] <= changed[0] or changed[3] <= box[1] or box[3] <= changed[1]
        assert apart, f"the label sits on node box {box}"


def test_an_edge_pointing_at_a_missing_node_is_refused():
    with pytest.raises(ValueError):
        graphics.relationship_chart(nodes(2), [(0, 5, "knew")])


# --- map ------------------------------------------------------------------

def test_the_map_card_places_its_points_and_clamps_wild_coordinates():
    image = graphics.map_card(
        [{"name": "Des Plaines", "x": 0.2, "y": 0.3}, {"name": "Norwood Park", "x": 1.4, "y": -0.2}],
        "Cook County, Illinois",
    )

    assert ink(image)
    assert all_background(image, (0, 1079, 1920, 1080))


def test_the_map_card_refuses_no_points():
    with pytest.raises(ValueError):
        graphics.map_card([])


# --- dispatch -------------------------------------------------------------

def test_render_shot_sends_a_short_card_with_subtext_to_the_date_card():
    shot = {"kind": "card", "text": "December 11th, 1978", "subtext": "Des Plaines, Illinois"}

    expected = graphics.date_card("December 11th, 1978", "Des Plaines, Illinois")
    assert graphics.render_shot(shot).tobytes() == expected.tobytes()


def test_render_shot_sends_a_long_card_to_the_title_card_even_with_subtext():
    shot = {"kind": "card", "text": LONG_TITLE, "subtext": "Chapter Four"}

    expected = graphics.title_card(LONG_TITLE, "Chapter Four")
    assert graphics.render_shot(shot).tobytes() == expected.tobytes()


def test_render_shot_sends_a_card_without_subtext_to_the_title_card():
    shot = {"kind": "card", "text": "Thirty Three"}

    assert graphics.render_shot(shot).tobytes() == graphics.title_card("Thirty Three").tobytes()


def test_render_shot_sends_events_to_the_timeline_and_nodes_to_the_chart():
    timeline_shot = {"kind": "chart", "events": events(3), "highlight": 1}
    chart_shot = {"kind": "chart", "nodes": nodes(3), "edges": [(0, 1, "employed")]}

    assert graphics.render_shot(timeline_shot).tobytes() == graphics.timeline(events(3), highlight=1).tobytes()
    assert graphics.render_shot(chart_shot).tobytes() == graphics.relationship_chart(nodes(3), [(0, 1, "employed")]).tobytes()


def test_render_shot_sends_points_to_the_map_card():
    points = [{"name": "Des Plaines", "x": 0.3, "y": 0.4}]
    shot = {"kind": "map", "points": points, "text": "Cook County"}

    assert graphics.render_shot(shot).tobytes() == graphics.map_card(points, "Cook County").tobytes()


def test_render_shot_honours_the_frame_size():
    shot = {"kind": "card", "text": "Thirty Three"}

    assert graphics.render_shot(shot, width=1280, height=720).size == (1280, 720)


@pytest.mark.parametrize("shot", [
    {"kind": "archival", "query": "Gacy mugshot"},
    {"kind": "chart"},
    {"kind": "card"},
    {"kind": "map"},
    {},
])
def test_render_shot_refuses_a_shot_it_cannot_draw(shot):
    with pytest.raises(ValueError):
        graphics.render_shot(shot)


# --- fonts and saving -----------------------------------------------------

def test_the_font_helper_falls_back_when_no_candidate_exists():
    fallback = graphics.font(["/no/such/font.ttf", "/also/missing.ttc"], 64)

    assert fallback is not None
    assert graphics.title_card("Thirty Three", "Fallback", serif=["/no/such/font.ttf"]).size == (1920, 1080)


def test_save_writes_a_jpeg_and_creates_its_parents(tmp_path):
    path = tmp_path / "deep" / "nested" / "card.jpg"

    graphics.save(graphics.title_card("Thirty Three"), path)

    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as written:
        assert written.format == "JPEG"
        assert written.size == (1920, 1080)


# --- palette (Revision 2: a crime documentary, not amber) ------------------

def test_the_palette_is_near_black_off_white_crimson_and_grey():
    assert graphics.BACKGROUND == (0x0A, 0x0A, 0x0A)
    assert graphics.FOREGROUND == (0xF2, 0xF2, 0xF2)
    assert graphics.ACCENT == (0x9B, 0x11, 0x1E)
    assert graphics.MUTED == (0x8C, 0x8C, 0x8C)


def test_no_amber_survives_in_a_drawn_chart():
    image = graphics.timeline(events(4), highlight=1)
    colours = {value for _, value in image.getcolors(maxcolors=1 << 24)}

    assert (200, 162, 107) not in colours  # the old amber accent
    assert graphics.ACCENT in colours


# --- overlays: lower-left, transparent, never full-screen ------------------

def overlay(style="stamp", text="December 11th, 1978", subtext="Des Plaines, Illinois", **size):
    return graphics.overlay_image(style, text, subtext, **size)


def alpha_box(image: Image.Image, box) -> int:
    """The strongest alpha anywhere in a region."""
    return image.getchannel("A").crop(box).getextrema()[1]


def solid_ink_bbox(image: Image.Image):
    """Where the glyphs and rules are: nearly opaque pixels, not the soft legibility shadow."""
    return image.getchannel("A").point(lambda a: 255 if a >= 200 else 0).getbbox()


@pytest.mark.parametrize("style, text, subtext", [
    ("stamp", "December 11th, 1978", "Des Plaines, Illinois"),
    ("name", "John Wayne Gacy", "Contractor, Des Plaines"),
])
def test_an_overlay_is_a_frame_sized_rgba_png_inked_lower_left(style, text, subtext):
    image = overlay(style, text, subtext)

    assert image.mode == "RGBA"
    assert image.size == (1920, 1080)
    assert alpha_box(image, (0, 0, 1920, 540)) == 0, "the overlay drew into the top half"
    assert alpha_box(image, (1344, 0, 1920, 1080)) == 0, "the overlay reached the right of the frame"
    left, top, right, bottom = solid_ink_bbox(image)
    assert left < 1920 * 0.12 and right < 1920 * 0.6
    assert top > 1080 * 0.6
    assert bottom <= graphics.OVERLAY_BOTTOM * 1080 + 2


@pytest.mark.parametrize("style", ["stamp", "name"])
def test_an_overlay_carries_a_crimson_rule(style):
    image = overlay(style)
    crimson = [xy for xy, px in zip(
        ((x, y) for y in range(image.height) for x in range(image.width)), image.getdata())
        if px[:3] == graphics.ACCENT and px[3] == 255]

    assert crimson, "no crimson rule"
    xs, ys = [x for x, _ in crimson], [y for _, y in crimson]
    span_x, span_y = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
    if style == "stamp":
        assert span_x > 4 * span_y, "a stamp's rule is short and horizontal"
        assert span_x < 1920 * 0.08
    else:
        assert span_y > 4 * span_x, "a name's rule is a thin vertical bar"


def test_the_overlay_text_is_off_white_and_the_subtext_grey():
    image = overlay("name", "John Wayne Gacy", "Contractor, Des Plaines")
    colours = {px[:3] for px in image.getdata() if px[3] == 255}

    assert graphics.FOREGROUND in colours
    assert graphics.MUTED in colours


def test_an_overlay_has_a_soft_dark_backing_for_legibility():
    image = overlay()
    left, top, right, bottom = solid_ink_bbox(image)
    # just above the glyphs, the backing is translucent and dark: neither empty nor opaque
    halo = [px for px in image.crop((left, top - 30, right, top - 5)).getdata()]
    assert 0 < max(px[3] for px in halo) < 200, "nothing soft behind the text to hold it off a bright photograph"
    assert all(max(px[:3]) < 40 for px in halo if 0 < px[3] < 200), "the backing is not dark"


def test_the_stamp_and_the_name_use_different_faces():
    stamp = overlay("stamp", "John Wayne Gacy", "Contractor")
    name = overlay("name", "John Wayne Gacy", "Contractor")

    assert stamp.tobytes() != name.tobytes()


def test_the_typewriter_face_is_american_typewriter_then_courier_new():
    first = graphics.TYPEWRITER[0]
    path = first[0] if isinstance(first, tuple) else first
    assert "American" in path and "Typewriter" in path
    assert any("Courier New" in (c[0] if isinstance(c, tuple) else c) for c in graphics.TYPEWRITER)


def test_a_long_overlay_wraps_and_stays_in_the_lower_left():
    image = overlay("stamp", "The Cook County Criminal Courts Building, 26th and California",
                    "Chicago, Illinois, February 6th, 1980, the first day of the trial")
    left, top, right, bottom = solid_ink_bbox(image)

    assert right < 1920 * 0.6
    assert top > 1080 * 0.5
    assert alpha_box(image, (0, 0, 1920, 540)) == 0


def test_an_overlay_without_subtext_still_renders():
    assert solid_ink_bbox(overlay("stamp", "December 11th, 1978", None)) is not None
    assert solid_ink_bbox(overlay("name", "Rob Piest", "")) is not None


def test_an_overlay_honours_a_different_frame_size():
    image = overlay(width=320, height=180)

    assert image.size == (320, 180)
    assert alpha_box(image, (0, 0, 320, 90)) == 0
    assert solid_ink_bbox(image) is not None


@pytest.mark.parametrize("style, text", [("caption", "x"), ("stamp", ""), ("name", "  ")])
def test_an_overlay_refuses_an_unknown_style_or_no_text(style, text):
    with pytest.raises(ValueError):
        graphics.overlay_image(style, text, None)


def test_save_overlay_writes_a_transparent_png(tmp_path):
    path = graphics.save_overlay(overlay(), tmp_path / "o" / "overlay.png")

    with Image.open(path) as written:
        assert written.format == "PNG"
        assert written.mode == "RGBA"
        assert written.getpixel((1919, 0))[3] == 0


# --- a stamp typing itself out --------------------------------------------------

def test_a_fully_typed_stamp_frame_is_the_static_stamp():
    typed = graphics.stamp_frame("December 11th, 1978", "Des Plaines, Illinois")

    assert typed.tobytes() == overlay("stamp").tobytes()


def test_a_partly_typed_stamp_inks_only_what_is_typed_in_the_same_place():
    full = graphics.stamp_frame("December 11th, 1978", "Des Plaines, Illinois")
    part = graphics.stamp_frame("December 11th, 1978", "Des Plaines, Illinois", date_chars=4, place_chars=0)
    none = graphics.stamp_frame("December 11th, 1978", "Des Plaines, Illinois", date_chars=0, place_chars=0)

    left, top, right, bottom = solid_ink_bbox(part)
    full_left, full_top, full_right, _ = solid_ink_bbox(full)
    assert left == full_left and top == full_top  # the layout never moves while typing
    assert right < full_right / 2 and bottom < 1080 * graphics.OVERLAY_BOTTOM - 20  # no place line yet
    # the backing is the whole stamp's from the first frame, so it never grows as text types
    halo = (full_left, full_top - 30, full_right, full_top - 5)
    assert alpha_box(none, halo) > 0
    assert solid_ink_bbox(none)[2] < 1920 * 0.12  # only the crimson rule


def test_the_caret_is_a_block_just_after_the_last_typed_character():
    bare = graphics.stamp_frame("December 11th, 1978", "Des Plaines", date_chars=4, place_chars=0)
    caret = graphics.stamp_frame("December 11th, 1978", "Des Plaines", date_chars=4, place_chars=0, caret="date")

    diff = ImageChops.difference(bare.getchannel("A"), caret.getchannel("A")).point(lambda a: 255 if a > 60 else 0)
    box = diff.getbbox()
    assert box is not None
    assert box[0] >= solid_ink_bbox(bare)[2] - 2  # right of "Dece"
    assert box[2] - box[0] < 30  # a narrow block, not a line

"""The edit: filter building and cut planning are pure and tested without rendering, plus real cuts."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_assemble():
    """scripts/ is not a package, so the module is loaded by path."""
    spec = importlib.util.spec_from_file_location("video_assemble", ROOT / "scripts" / "video" / "assemble.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["video_assemble"] = module
    spec.loader.exec_module(module)
    return module


assemble = load_assemble()
FPS = 30


def shot(start=0.0, end=9.2, motion="zoom_in", index=1, **extra):
    return {"index": index, "start": start, "end": end, "kind": "archival", "motion": motion, **extra}


def still(path: Path, size=(640, 360), colour=(70, 90, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, colour)
    for x in range(size[0]):  # a gradient, so motion is visible frame to frame
        for y in range(0, size[1], 4):
            image.putpixel((x, y), ((x * 7) % 256, (y * 5) % 256, 90))
    image.save(path, quality=90)
    return path


def silence(path: Path, seconds: float) -> Path:
    subprocess.run(
        [assemble.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
         "anullsrc=r=24000:cl=mono", "-t", str(seconds), str(path)],
        check=True,
    )
    return path


def probe(path: Path, entries: str) -> str:
    out = subprocess.run(
        [assemble.FFPROBE, "-v", "error", "-show_entries", entries, "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def frame_at(path: Path, seconds: float, out: Path) -> Image.Image:
    subprocess.run([assemble.FFMPEG, "-y", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
                    "-frames:v", "1", "-update", "1", str(out)], check=True)
    return Image.open(out).convert("RGB")


def mean_luma(image: Image.Image) -> float:
    grey = image.convert("L")
    return sum(grey.getdata()) / (grey.width * grey.height)


# --- the ffmpeg this edit uses ---------------------------------------------

def test_the_edit_uses_the_isolated_ffmpeg_9_never_the_voice_pipelines():
    assert assemble.resolve_tools({}) == (assemble.DEFAULT_FFMPEG, assemble.DEFAULT_FFPROBE)
    assert assemble.DEFAULT_FFMPEG == "/opt/miniconda3/envs/video/bin/ffmpeg"
    assert "app.config" not in (ROOT / "scripts" / "video" / "assemble.py").read_text(encoding="utf-8")


def test_the_tools_can_be_overridden_by_environment():
    env = {"VIDEO_FFMPEG": "/x/ffmpeg", "VIDEO_FFPROBE": "/x/ffprobe"}

    assert assemble.resolve_tools(env) == ("/x/ffmpeg", "/x/ffprobe")


def test_a_missing_ffmpeg_fails_loudly_instead_of_falling_back(tmp_path):
    with pytest.raises(RuntimeError, match="VIDEO_FFMPEG"):
        assemble.require_tools(str(tmp_path / "no-ffmpeg"), assemble.FFPROBE)


# --- choosing a treatment --------------------------------------------------

@pytest.mark.parametrize("size, kind_hint, expected", [
    ((1920, 1080), None, "fill"),
    ((1536, 1024), None, "fill"),      # 3:2, the commonest photograph, sits just inside the band
    ((1600, 1000), "photo", "fill"),
    ((2000, 1000), None, "fill"),
])
def test_pictures_near_16_9_fill_the_frame(size, kind_hint, expected):
    assert assemble.auto_treatment(*size, kind_hint) == expected


@pytest.mark.parametrize("size", [(1505, 2000), (1000, 1000), (1024, 768), (3200, 1000), (900, 1400)])
def test_portrait_square_4_3_and_very_wide_pictures_are_prints(size):
    assert assemble.auto_treatment(*size) == "print"


@pytest.mark.parametrize("hint", ["document", "newspaper"])
@pytest.mark.parametrize("size", [(1920, 1080), (1200, 1800), (3000, 900)])
def test_documents_and_newspapers_pan_whatever_their_shape(hint, size):
    assert assemble.auto_treatment(*size, hint) == "pan"


def test_an_explicit_treatment_beats_the_rule():
    assert assemble.choose_treatment(shot(treatment="print"), (1920, 1080)) == "print"
    assert assemble.choose_treatment(shot(treatment=None), (1505, 2000)) == "print"
    assert assemble.choose_treatment(shot(treatment=None, kind_hint="newspaper"), (1920, 1080)) == "pan"


def test_drawn_charts_and_maps_always_fill():
    assert assemble.choose_treatment({"kind": "chart"}, (1920, 1080)) == "fill"
    assert assemble.choose_treatment({"kind": "map", "treatment": "print"}, (1920, 1080)) == "fill"


# --- fill: the cover crop with slow Ken Burns motion ------------------------

def test_fill_covers_the_frame_at_twice_the_size_and_zooms_once_per_output_frame():
    chain = assemble.fill_chain("zoom_in", 276, 1920, 1080, FPS)

    assert chain.startswith("scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160")
    assert "d=276" in chain  # one still in, every frame of the shot out
    assert "s=1920x1080" in chain and f"fps={FPS}" in chain
    assert "on/275" in chain


def test_fill_motion_is_gentle_and_zooms_travel_in_opposite_directions():
    zoom_in = assemble.fill_chain("zoom_in", 276, 1920, 1080, FPS)
    zoom_out = assemble.fill_chain("zoom_out", 276, 1920, 1080, FPS)

    assert "z='1.0000+0.1000*on/275'" in zoom_in
    assert "z='1.1000-0.1000*on/275'" in zoom_out
    assert assemble.ZOOM_TRAVEL <= 0.10


@pytest.mark.parametrize("motion, axis, expression", [
    ("pan_left", "x", "(iw-iw/zoom)*(1-on/275)"),
    ("pan_right", "x", "(iw-iw/zoom)*(on/275)"),
    ("pan_up", "y", "(ih-ih/zoom)*(1-on/275)"),
    ("pan_down", "y", "(ih-ih/zoom)*(on/275)"),
])
def test_fill_pans_hold_a_gentle_zoom_and_walk_one_axis(motion, axis, expression):
    chain = assemble.fill_chain(motion, 276, 1920, 1080, FPS)

    assert "z='1.1000'" in chain
    assert f"{axis}='{expression}'" in chain


def test_fill_still_and_unknown_motions_hold_the_whole_frame():
    still_chain = assemble.fill_chain("still", 100, 640, 360, FPS)

    assert "z='1.0000'" in still_chain
    assert assemble.fill_chain("helicopter", 100, 640, 360, FPS) == still_chain


def test_a_one_frame_fill_does_not_divide_by_zero():
    assert "on/" not in assemble.fill_chain("zoom_in", 1, 320, 180, 25)


# --- print: the photograph as an object on a blurred copy of itself -------

def test_print_blurs_and_darkens_a_cover_copy_for_the_ground():
    chain = assemble.print_chain(1505, 2000, 300, 1920, 1080, FPS)
    ground = chain.split("[bg]")[0]

    assert "split=2" in chain
    assert "scale=3840:2160:force_original_aspect_ratio=increase,crop=3840:2160" in ground
    assert "gblur=sigma=" in ground
    assert f"colorchannelmixer=rr={assemble.PRINT_DARKEN}:" in ground and assemble.PRINT_DARKEN < 0.5  # darkened


def test_print_fits_a_portrait_within_84_percent_of_the_height_and_never_crops_it():
    chain = assemble.print_chain(1505, 2000, 300, 1920, 1080, FPS)
    face = chain[chain.index("[fg0]"):chain.index("[print]")]

    assert "scale=3226:1814:force_original_aspect_ratio=decrease" in face  # 0.84 of 3840x2160
    assert "crop" not in face
    assert "pad=iw+" in face and f"color={assemble.PRINT_BORDER}" in face  # the thin off-white border


def test_print_fits_a_wide_picture_by_its_width():
    chain = assemble.print_chain(4000, 1000, 300, 1920, 1080, FPS)

    assert "scale=3226:1814:force_original_aspect_ratio=decrease" in chain  # width binds for 4:1
    assert assemble.PRINT_FIT == pytest.approx(0.84)


def test_print_casts_a_soft_shadow_and_centres_the_print():
    chain = assemble.print_chain(1505, 2000, 300, 1920, 1080, FPS)

    assert "[shadow]" in chain and "gblur" in chain.split("[shadow]")[0].split("[p2]")[-1]
    assert "overlay=0+(3840-w)/2:(H-h)/2" in chain  # centred across the whole 2x canvas


def test_print_moves_the_whole_composite_with_a_very_slow_push_in():
    chain = assemble.print_chain(1505, 2000, 300, 1920, 1080, FPS)
    move = chain[chain.index("zoompan="):]

    assert f"z='1.0000+{assemble.PRINT_PUSH:.4f}*on/299'" in move
    assert assemble.PRINT_PUSH <= 0.06
    assert "d=300" in move and "s=1920x1080" in move


# --- pan: documents and newspapers ------------------------------------------

def test_a_tall_document_is_fitted_to_the_width_and_panned_top_to_bottom():
    chain = assemble.pan_chain(1200, 1800, 300, 1920, 1080, FPS)

    assert "scale=3456:-2" in chain  # 0.9 of the doubled width, height follows the page
    assert f"color={assemble.BACKDROP}" in chain  # over the dark ground
    assert "loop=loop=299:size=1" in chain
    assert "crop=3840:2160:x=0:y='(ih-oh)*n/299'" in chain
    assert chain.endswith("scale=1920:1080,setsar=1")


def test_a_wide_document_is_panned_left_to_right():
    chain = assemble.pan_chain(3000, 900, 300, 1920, 1080, FPS)

    assert "scale=-2:1944" in chain
    assert "crop=3840:2160:x='(iw-ow)*n/299':y=0" in chain


# --- the look ---------------------------------------------------------------

def test_the_colour_grade_is_cold_lightly_desaturated_with_crushed_blacks():
    grade = assemble.grade_filter("colour")

    saturation = float(re.search(r"saturation=([0-9.]+)", grade).group(1))
    assert 0.8 <= saturation < 0.9  # colour photographs still read as colour beside the black and white
    assert re.search(r"colorbalance=.*bs=0\.0[1-9]", grade)  # blue pushed into the shadows: cold
    assert "curves=" in grade
    toe = re.search(r"curves=all='0/0 ([0-9.]+)/([0-9.]+)", grade)
    assert float(toe.group(2)) < float(toe.group(1))  # the toe sits below the diagonal: crushed
    assert "format=gray" not in grade


def test_a_black_and_white_photograph_is_forced_to_greyscale_before_its_faint_cool_tone():
    grade = assemble.grade_filter("bw")

    assert grade.startswith("format=gray,")
    assert grade.index("format=gray") < grade.index("colorbalance")
    assert "saturation" not in grade  # nothing that could reintroduce a cast before the tone


def test_drawn_shots_are_not_graded():
    assert assemble.grade_filter("none") == ""


def test_grain_is_temporal_and_subtle():
    grain = assemble.GRAIN

    assert grain.startswith("noise=")
    assert re.search(r"c0f=t", grain)  # changes every frame, like film
    assert 0 < int(re.search(r"c0s=(\d+)", grain).group(1)) <= 10


def test_the_vignette_is_gentle_and_computed_once():
    angle = re.search(r"vignette=angle=PI/([0-9.]+)", assemble.VIGNETTE)

    assert angle and float(angle.group(1)) >= 5  # PI/5 is ffmpeg's default; never stronger
    assert "eval=init" in assemble.VIGNETTE


def test_a_greyscale_picture_is_recognised_as_black_and_white(tmp_path):
    grey = Image.new("RGB", (200, 200))
    for x in range(200):
        for y in range(200):
            v = (x + y) % 256
            grey.putpixel((x, y), (v, v, min(255, v + 2)))  # a scan with a whisper of cast
    grey.save(tmp_path / "grey.jpg")
    colour = still(tmp_path / "colour.jpg")

    assert assemble.is_monochrome(tmp_path / "grey.jpg")
    assert not assemble.is_monochrome(colour)


@pytest.mark.parametrize("given, kind, mono, expected", [
    ("bw", "archival", False, "bw"),
    ("colour", "archival", True, "colour"),
    (None, "archival", True, "bw"),
    (None, "archival", False, "colour"),
    (None, "chart", False, "none"),
    ("bw", "map", True, "none"),
])
def test_the_grade_mode_follows_the_shot_then_the_picture(given, kind, mono, expected):
    assert assemble.grade_mode({"kind": kind, "colour": given}, monochrome=mono) == expected


# --- overlays: fade in at 0.3 s over 0.4 s, gone by 3.5 s, whatever the shot length

def test_the_overlay_fades_in_at_0_3s_and_is_gone_by_3_5s():
    fades = assemble.overlay_fades(9.2)

    assert fades == "fade=t=in:st=0.300:d=0.400:alpha=1,fade=t=out:st=3.200:d=0.300:alpha=1"


def test_a_long_shot_never_holds_its_overlay_longer():
    assert assemble.overlay_fades(60.0) == assemble.overlay_fades(9.2)


def test_a_short_shot_pulls_the_overlay_out_before_it_ends():
    fades = assemble.overlay_fades(2.6)
    start, length = re.search(r"t=out:st=([0-9.]+):d=([0-9.]+)", fades).groups()

    assert float(start) + float(length) <= 2.6 - 0.1


def test_a_shot_too_short_to_read_gets_no_overlay():
    assert assemble.overlay_fades(1.5) is None


def test_subtitles_sit_below_every_overlay():
    from scripts.video import graphics

    assert assemble.subtitle_box_top(1080) > graphics.OVERLAY_BOTTOM * 1080 + 10


# --- the cut: transitions with picture locked to narration -----------------

def sequence():
    """A mixed cut: open, dissolve, cut, black, dissolve, at awkward times."""
    times = [0.0, 3.0, 5.27, 8.0, 10.5, 13.4]
    kinds = [None, "dissolve", "cut", "black", "dissolve"]
    return [{"index": i + 1, "start": times[i], "end": times[i + 1], "kind": "archival",
             "transition_in": kinds[i]} for i in range(5)]


def test_frames_follow_the_planned_cut_times_so_rounding_never_drifts():
    plan = assemble.cut_plan(sequence(), FPS)

    assert [p["start_frame"] for p in plan] == [0, 90, 158, 240, 315]
    assert [p["frames"] for p in plan] == [90, 68, 82, 75, 87]
    assert sum(p["frames"] for p in plan) == round(13.4 * FPS)


def test_a_dissolve_adds_its_length_to_the_outgoing_tail_and_nothing_else():
    plan = assemble.cut_plan(sequence(), FPS)

    assert [p["tail"] for p in plan] == [18, 0, 0, 18, 0]  # 0.6 s before each dissolve
    assert [p["transition"] for p in plan] == ["cut", "dissolve", "cut", "black", "dissolve"]


def test_a_dip_to_black_is_centred_on_the_cut():
    plan = assemble.cut_plan(sequence(), FPS)

    assert plan[2]["fade_out"] == 15 and plan[3]["fade_in"] == 15  # 0.5 s down, 0.5 s up: 1.0 s
    assert plan[2]["tail"] == 0  # a dip needs no overlap: the new picture rises at its own start


def test_the_film_opens_from_black_and_closes_to_black():
    plan = assemble.cut_plan(sequence(), FPS)

    assert plan[0]["fade_in"] > 0
    assert plan[-1]["fade_out"] == 30


def test_a_slice_opens_and_closes_without_fades():
    plan = assemble.cut_plan(sequence(), FPS, opens_film=False, closes_film=False)

    assert plan[0]["fade_in"] == 0 and plan[-1]["fade_out"] == 0


def test_a_v1_shot_list_dissolves_by_default():
    v1 = [{"index": i + 1, "start": i * 4.0, "end": (i + 1) * 4.0, "kind": "archival"} for i in range(3)]
    plan = assemble.cut_plan(v1, FPS)

    assert [p["transition"] for p in plan] == ["cut", "dissolve", "dissolve"]
    assert [p["tail"] for p in plan] == [18, 18, 0]


def test_a_dissolve_into_a_very_short_shot_is_shortened_not_overrun():
    shots = [{"index": 1, "start": 0.0, "end": 4.0, "transition_in": "cut"},
             {"index": 2, "start": 4.0, "end": 4.5, "transition_in": "dissolve"},
             {"index": 3, "start": 4.5, "end": 4.53, "transition_in": "dissolve"}]
    plan = assemble.cut_plan(shots, FPS)

    assert plan[0]["tail"] == 7  # half of the 15-frame shot it dissolves into
    assert plan[2]["transition"] == "cut" and plan[1]["tail"] == 0  # one frame cannot dissolve


def test_join_offsets_are_the_planned_cut_times_for_a_mixed_sequence():
    plan = assemble.cut_plan(sequence(), FPS)
    joins = assemble.join_steps(plan, FPS)

    assert [(j["kind"], j["offset"]) for j in joins] == [
        ("xfade", 3.0),     # dissolve at 3.0: the incoming picture starts appearing at 3.0
        ("concat", 5.267),  # cut at 158/30
        ("concat", 8.0),    # black: the dip is inside the clips, the join itself is a cut
        ("xfade", 10.5),    # dissolve at 10.5
    ]
    assert joins[0]["duration"] == pytest.approx(0.6)


def test_the_joined_length_is_exactly_the_narration():
    plan = assemble.cut_plan(sequence(), FPS)
    rendered = sum(p["frames"] + p["tail"] for p in plan)
    overlapped = sum(p["tail"] for p in plan)

    assert rendered - overlapped == round(13.4 * FPS)
    assert assemble.span_frames(plan) == round(13.4 * FPS)


def test_the_join_graph_uses_xfade_fade_at_the_planned_offsets_and_trims_to_the_span():
    plan = assemble.cut_plan(sequence(), FPS)
    graph = assemble.join_filter(plan, FPS)

    assert "xfade=transition=fade:duration=0.600:offset=3.000" in graph
    assert "xfade=transition=fade:duration=0.600:offset=10.500" in graph
    assert graph.count("concat=n=2:v=1:a=0") == 2
    assert "settb=1/30,setpts=N" in graph  # frame-exact timestamps before any join
    assert f"trim=end_frame={round(13.4 * FPS)}" in graph


def test_groups_split_long_cuts_and_a_dissolve_across_a_boundary_takes_a_lead_in():
    shots = [{"index": i + 1, "start": i * 2.0, "end": (i + 1) * 2.0, "transition_in": "dissolve"}
             for i in range(45)]
    plan = assemble.cut_plan(shots, FPS)
    groups = assemble.group_plan(plan, size=20)

    assert [len(g["clips"]) for g in groups] == [20, 20, 5]
    assert groups[0]["lead"] is None
    assert groups[1]["lead"] == plan[19]  # the outgoing shot of the dissolve at the boundary
    assert sum(assemble.span_frames(g["clips"]) for g in groups) == 45 * 2 * FPS


def test_a_lead_in_is_trimmed_to_its_tail_and_dissolves_at_the_groups_first_frame():
    shots = [{"index": i + 1, "start": i * 2.0, "end": (i + 1) * 2.0, "transition_in": "dissolve"}
             for i in range(4)]
    plan = assemble.cut_plan(shots, FPS)
    group = assemble.group_plan(plan, size=2)[1]
    graph = assemble.join_filter(group["clips"], FPS, lead=group["lead"])

    assert graph.startswith("[0:v]trim=start_frame=60,settb=1/30,setpts=N[lead]")
    assert "[lead][v0]xfade=transition=fade:duration=0.600:offset=0.000" in graph
    assert "xfade=transition=fade:duration=0.600:offset=2.000" in graph  # relative to the group


def test_a_group_boundary_on_a_cut_needs_no_lead():
    plan = assemble.cut_plan(sequence(), FPS)
    groups = assemble.group_plan(plan, size=2)

    assert groups[1]["clips"][0]["transition"] == "cut" and groups[1]["lead"] is None
    assert groups[2]["clips"][0]["transition"] == "dissolve" and groups[2]["lead"] == plan[3]


# --- subtitles --------------------------------------------------------------

LINES = [
    {"index": 0, "start": 0.0, "end": 4.0, "text": "Before the window."},
    {"index": 1, "start": 9.5, "end": 12.0, "text": "Overlapping the start."},
    {"index": 2, "start": 13.0, "end": 15.0, "text": "Wholly {inside}."},
    {"index": 3, "start": 19.0, "end": 24.0, "text": "Overlapping the end."},
    {"index": 4, "start": 30.0, "end": 33.0, "text": "After the window."},
]


def test_subtitles_are_written_as_ass_for_the_window_shifted_to_its_zero():
    ass = assemble.subtitle_ass(LINES, 10.0, 20.0, 1920, 1080)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]

    assert len(events) == 3
    assert events[0].startswith("Dialogue: 0,0:00:00.00,0:00:02.00,")
    assert events[1].startswith("Dialogue: 0,0:00:03.00,0:00:05.00,")
    assert events[2].startswith("Dialogue: 0,0:00:09.00,0:00:10.00,")
    assert "{" not in events[1].split(",,", 1)[1]  # braces would open an override block


def test_subtitles_are_white_on_a_soft_box_bottom_centre():
    ass = assemble.subtitle_ass(LINES, 0.0, 40.0, 1920, 1080)
    style = next(line for line in ass.splitlines() if line.startswith("Style:")).split(",")

    assert "PlayResX: 1920" in ass and "PlayResY: 1080" in ass
    assert style[3] == "&H00F2F2F2"  # off-white text
    assert style[15] == "4"  # BorderStyle 4: one box behind the whole event
    assert style[6].startswith("&H") and style[6][2:4] not in ("00", "FF")  # translucent box
    assert style[18] == "2"  # bottom centre


def test_long_subtitles_wrap_to_two_lines_and_keep_every_word():
    wrapped = assemble.wrap_subtitle("A " * 50)

    assert len(wrapped.splitlines()) == 2
    assert wrapped.replace("\n", " ").split() == ["A"] * 50


def test_two_line_subtitles_are_balanced_and_never_split_a_name():
    wrapped = assemble.wrap_subtitle("And that's exactly what made John Wayne Gacy so difficult to suspect.")

    assert wrapped == "And that's exactly what made\nJohn Wayne Gacy so difficult to suspect."
    assert assemble.wrap_subtitle("Probably not.") == "Probably not."
    assert assemble.wrap_subtitle("He owns a construction business, knows a lot of people") == \
        "He owns a construction business,\nknows a lot of people"


LONG_LINE = ("There's a man in your neighborhood. He's doing pretty well for himself. He owns a "
             "construction business, knows a lot of people, and has slowly started getting involved "
             "in local politics.")


def test_a_long_narration_line_becomes_several_timed_cues_of_two_short_lines():
    cues = assemble.subtitle_cues([{"start": 1.64, "end": 11.52, "text": LONG_LINE}])

    assert len(cues) >= 3
    assert cues[0]["start"] == pytest.approx(1.64) and cues[-1]["end"] == pytest.approx(11.52)
    assert all(a["end"] == pytest.approx(b["start"]) for a, b in zip(cues, cues[1:]))
    assert " ".join(c["text"] for c in cues).split() == LONG_LINE.split()
    for cue in cues:
        lines = assemble.wrap_subtitle(cue["text"]).splitlines()
        assert len(lines) <= 2 and all(len(line) <= assemble.SUBTITLE_CHARS for line in lines)
    assert cues[0]["text"] == "There's a man in your neighborhood."  # sentences are kept whole


def test_every_burned_cue_is_at_most_two_lines():
    ass = assemble.subtitle_ass([{"start": 0, "end": 10, "text": LONG_LINE}], 0, 10, 1920, 1080)
    texts = [line.split(",,", 1)[1].split(",", 3)[3] for line in ass.splitlines() if line.startswith("Dialogue:")]

    assert len(texts) >= 3
    assert all(text.count("\\N") <= 1 for text in texts)


def test_a_filter_path_is_escaped_for_the_graph():
    assert assemble.escape_filter_path("/tmp/it's, a:b") == r"/tmp/it\\\'s\, a\\:b"


# --- building one shot -----------------------------------------------------

def clip_plan(**extra):
    return {"index": 1, "start_frame": 0, "frames": 60, "tail": 0, "fade_in": 0, "fade_out": 0,
            "transition": "cut", **extra}


def test_a_still_is_read_once_and_expanded_by_the_treatment(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    argv = assemble.shot_clip(shot(end=2.0), clip_plan(tail=18), asset, tmp_path / "out.mp4", 640, 360, FPS)
    graph = argv[argv.index("-filter_complex") + 1]

    assert argv[0] == assemble.FFMPEG
    assert str(asset) in argv and "-loop" not in argv[:argv.index(str(asset))]
    assert argv[argv.index("-frames:v") + 1] == "78"  # 60 planned + 18 of tail
    assert "zoompan=" in graph and "d=78" in graph
    assert argv[-1] == str(tmp_path / "out.mp4")
    assert argv[argv.index("-preset") + 1] == assemble.INTERMEDIATE_PRESET


def test_a_shot_with_an_overlay_lays_the_png_over_with_its_fades(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    card = tmp_path / "overlay.png"
    Image.new("RGBA", (640, 360)).save(card)
    argv = assemble.shot_clip(shot(end=9.0), clip_plan(frames=270), asset, tmp_path / "out.mp4", 640, 360, FPS,
                              overlay=card)
    graph = argv[argv.index("-filter_complex") + 1]

    assert str(card) in argv
    assert assemble.overlay_fades(9.0) in graph
    assert "overlay=0:0" in graph


def test_fades_to_and_from_black_are_frame_exact_and_sit_over_the_overlay(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    card = tmp_path / "overlay.png"
    Image.new("RGBA", (640, 360)).save(card)
    argv = assemble.shot_clip(shot(end=3.0), clip_plan(frames=90, fade_in=15, fade_out=15), asset,
                              tmp_path / "out.mp4", 640, 360, FPS, overlay=card)
    graph = argv[argv.index("-filter_complex") + 1]

    assert "fade=t=in:s=0:n=15" in graph  # the first frame is black, the 16th fully up
    assert "fade=t=out:s=75:n=14" in graph  # and the last frame of the outgoing shot is black
    assert graph.index("overlay=0:0") < graph.index("fade=t=in:s=0")


def test_a_bw_shot_is_graded_greyscale_and_a_chart_not_at_all(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    bw = assemble.shot_clip(shot(colour="bw"), clip_plan(), asset, tmp_path / "a.mp4", 640, 360, FPS)
    chart = assemble.shot_clip({"index": 1, "start": 0, "end": 2, "kind": "chart"}, clip_plan(), asset,
                               tmp_path / "b.mp4", 640, 360, FPS)

    assert "format=gray" in bw[bw.index("-filter_complex") + 1]
    chart_graph = chart[chart.index("-filter_complex") + 1]
    assert "curves" not in chart_graph and "vignette" not in chart_graph


def test_nothing_on_disk_falls_back_to_the_dark_ground(tmp_path):
    argv = assemble.shot_clip(shot(end=3.0), clip_plan(frames=90), None, tmp_path / "out.mp4", 640, 360, FPS)

    assert "lavfi" in argv
    assert argv[argv.index("lavfi") + 2].startswith(f"color=c={assemble.BACKDROP}")


# --- concat and probing ------------------------------------------------------

def test_concat_list_quotes_one_file_per_line(tmp_path):
    listing = assemble.concat_list([tmp_path / "a.mp4", tmp_path / "b.mp4"])

    assert listing == f"file '{tmp_path / 'a.mp4'}'\nfile '{tmp_path / 'b.mp4'}'\n"


def test_concat_list_escapes_an_apostrophe_in_a_path():
    assert assemble.concat_list([Path("/tmp/it's/a.mp4")]) == "file '/tmp/it'\\''s/a.mp4'\n"


def test_probe_duration_reads_the_real_length(tmp_path):
    audio = silence(tmp_path / "quiet.wav", 2.0)

    assert abs(assemble.probe_duration(audio) - 2.0) < 0.05


# --- real renders -------------------------------------------------------------

def build_v1_project(tmp_path, *, subtitles="burn", fps=15, width=320, height=180, shot_s=1.0, count=3):
    """A v1 shot list (no transitions, treatments or overlays): it must still render."""
    base = tmp_path / "gacy"
    base.mkdir(parents=True, exist_ok=True)
    shots = []
    for number in range(count):
        asset = still(base / "assets" / f"{number + 1:04d}.jpg", colour=(40 + number * 30, 60, 90))
        shots.append({
            "index": number + 1,
            "start": number * shot_s,
            "end": (number + 1) * shot_s,
            "kind": "archival",
            "motion": ["zoom_in", "pan_right", "still"][number % 3],
            "asset": str(asset.relative_to(base)),
        })
    shots[1]["asset"] = None  # sourcing failed here: assembly must hold the previous picture
    spec = {
        "audio": str(silence(base / "narration.wav", shot_s * count)),
        "duration_s": shot_s * count,
        "width": width, "height": height, "fps": fps,
        "subtitles": subtitles,
        "shots": shots,
    }
    (base / "shots.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    (base / "lines.json").write_text(json.dumps([
        {"index": 0, "start": 0.2, "end": 1.4, "text": "It's 7:30, and the snow has not stopped."},
        {"index": 1, "start": 1.6, "end": 2.8, "text": "Des Plaines, Illinois — 50% of the town asleep."},
    ]), encoding="utf-8")
    (base / "script.srt").write_text("1\n00:00:00,200 --> 00:00:01,400\nIt's 7:30.\n", encoding="utf-8")
    return base


def build_v2_project(tmp_path):
    """Portrait print, wide document pan, a dissolve, a dip to black and a stamp, at 320x180."""
    base = tmp_path / "v2"
    portrait = still(base / "library" / "portrait.jpg", size=(300, 400), colour=(200, 200, 200))
    wide = still(base / "library" / "wide.jpg", size=(900, 300), colour=(180, 170, 150))
    land = still(base / "library" / "land.jpg", size=(480, 270), colour=(60, 120, 60))
    shots = [
        {"index": 1, "start": 0.0, "end": 1.6, "kind": "archival", "asset": "library/portrait.jpg",
         "treatment": None, "motion": "still", "transition_in": "cut", "colour": "bw",
         "overlay": {"style": "name", "text": "John Wayne Gacy", "subtext": "Contractor"}},
        {"index": 2, "start": 1.6, "end": 3.0, "kind": "archival", "asset": "library/wide.jpg",
         "treatment": "pan", "motion": "still", "transition_in": "dissolve", "colour": None,
         "overlay": None},
        {"index": 3, "start": 3.0, "end": 5.6, "kind": "period", "asset": "library/land.jpg",
         "treatment": None, "motion": "zoom_in", "transition_in": "black", "colour": "colour",
         "overlay": {"style": "stamp", "text": "December 11th, 1978", "subtext": "Des Plaines, Illinois"}},
    ]
    spec = {"audio": str(silence(base / "narration.wav", 5.6)), "duration_s": 5.6,
            "width": 320, "height": 180, "fps": 15, "subtitles": "none", "shots": shots}
    (base / "shots.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    assert portrait.exists() and wide.exists() and land.exists()
    return base


def test_a_v2_cut_renders_locked_to_the_narration(tmp_path):
    base = build_v2_project(tmp_path)
    out = tmp_path / "v2.mp4"

    assemble.render(base / "shots.json", out, keep_intermediates=True)

    assert probe(out, "stream=width,height").splitlines()[:2] == ["width=320", "height=180"]
    assert "codec_name=aac" in probe(out, "stream=codec_name")
    assert abs(assemble.probe_duration(out) - 5.6) < 0.1
    frames = int(probe(out, "stream=nb_frames").splitlines()[0].split("=")[1])
    assert frames == round(5.6 * 15)
    # every shot leaves in one colour space, so no cut jumps in brightness
    assert "color_range=tv" in probe(out, "stream=color_range")
    assert "color_space=bt709" in probe(out, "stream=color_space")
    for piece in (base / "work").glob("shot_*.mp4"):
        assert "color_range=tv" in probe(piece, "stream=color_range"), piece.name
    # the dip to black bottoms out on the planned cut at 3.0 s, and the picture is back after it
    assert mean_luma(frame_at(out, 3.0 - 1 / 15, tmp_path / "dip_out.png")) < 12
    assert mean_luma(frame_at(out, 3.01, tmp_path / "dip_in.png")) < 12
    assert mean_luma(frame_at(out, 4.2, tmp_path / "after.png")) > 30
    # the portrait is a print: blurred copy at the edges, never black bars
    opening = frame_at(out, 0.9, tmp_path / "print.png")
    assert mean_luma(opening.crop((0, 40, 30, 140))) > 15


def test_a_v1_cut_still_renders_with_dissolves_and_burned_subtitles(tmp_path):
    base = build_v1_project(tmp_path)
    out = tmp_path / "gacy.mp4"

    result = assemble.render(base / "shots.json", out, keep_intermediates=True)

    assert result == out and out.exists() and out.stat().st_size > 0
    assert probe(out, "stream=width,height").splitlines()[:2] == ["width=320", "height=180"]
    assert abs(assemble.probe_duration(out) - 3.0) < 0.1
    assert len(list((base / "work").glob("shot_000[123]_*.mp4"))) == 3


def test_render_resumes_from_existing_intermediates_and_can_preview_a_slice(tmp_path):
    base = build_v1_project(tmp_path, subtitles="sidecar")
    first = assemble.render(base / "shots.json", tmp_path / "all.mp4", keep_intermediates=True)
    stamps = {p.name: p.stat().st_mtime_ns for p in (base / "work").glob("shot_*.mp4")}

    calls = []
    assemble.render(base / "shots.json", tmp_path / "again.mp4", progress=lambda *a: calls.append(a),
                    keep_intermediates=True)
    assert {p.name: p.stat().st_mtime_ns for p in (base / "work").glob("shot_*.mp4")} == stamps
    assert len(calls) == 3

    sliced = assemble.render(base / "shots.json", tmp_path / "slice.mp4", limit=2, start_index=2,
                             keep_intermediates=True)
    assert abs(assemble.probe_duration(sliced) - 2.0) < 0.1
    assert (tmp_path / "all.srt").read_text(encoding="utf-8").startswith("1\n")  # sidecar copied
    assert first.exists()


def test_a_dissolve_across_a_group_boundary_renders_frame_exact(tmp_path, monkeypatch):
    base = build_v1_project(tmp_path, subtitles="none", count=5, shot_s=1.2)  # v1: dissolves everywhere
    monkeypatch.setattr(assemble, "GROUP_SIZE", 2)
    out = tmp_path / "grouped.mp4"

    assemble.render(base / "shots.json", out, keep_intermediates=True)

    groups = sorted((base / "work").glob("group_*.mp4"))
    assert len(groups) == 3  # 2 + 2 + 1, the last two opening on a dissolve with a lead-in
    assert [int(probe(g, "stream=nb_frames").split("=")[1]) for g in groups] == [36, 36, 18]
    assert int(probe(out, "stream=nb_frames").splitlines()[0].split("=")[1]) == round(6.0 * 15)
    assert abs(assemble.probe_duration(out) - 6.0) < 0.1


def test_a_small_print_is_held_to_its_maximum_enlargement():
    small = assemble.print_chain(181, 255, 90, 1920, 1080, 30)
    large = assemble.print_chain(1505, 2000, 90, 1920, 1080, 30)
    # 255 px tall, at most 3x on screen, composed at 2x: a box no taller than 1530 canvas pixels.
    assert "scale=1086:1530:" in small
    assert "scale=3226:1814:" in large


def test_a_print_moves_right_of_an_overlay():
    plain = assemble.print_chain(1505, 2000, 90, 1920, 1080, 30)
    beside = assemble.print_chain(1505, 2000, 90, 1920, 1080, 30, clear_left=assemble.PRINT_OVERLAY_CLEAR)
    assert "overlay=0+(3840-w)/2:(H-h)/2," in plain
    # 40% of a 3840-wide canvas kept clear on the left, a 3.5% margin on the right.
    assert "overlay=1536+(2170-w)/2:(H-h)/2," in beside


# --- footage: trim, speed and the never-freeze rule ---------------------------------

def test_a_long_enough_segment_plays_at_speed_from_clip_in():
    timing = assemble.clip_timing(4.0, 12.5, 20.0, 300.0)

    assert timing == {"start": 12.5, "forward": 4.0, "reverse": 0.0, "stretch": 1.0}


def test_a_slightly_short_segment_is_slowed_by_at_most_ten_percent():
    timing = assemble.clip_timing(5.3, 10.0, 15.0, 300.0)

    assert timing["forward"] == 5.0 and timing["reverse"] == 0.0
    assert timing["stretch"] == pytest.approx(1.06)
    assert timing["forward"] * timing["stretch"] == pytest.approx(5.3)


def test_a_segment_short_after_slowing_runs_a_short_tail_backwards():
    timing = assemble.clip_timing(6.0, 10.0, 15.0, 300.0)  # 5.0 s, 5.5 s slowed, 0.5 s to find

    assert timing["stretch"] == assemble.CLIP_MAX_STRETCH
    assert timing["reverse"] == pytest.approx(0.5 / 1.1)
    assert (timing["forward"] + timing["reverse"]) * timing["stretch"] == pytest.approx(6.0)


def test_footage_too_short_to_cover_its_shot_is_an_error_not_a_freeze():
    with pytest.raises(assemble.ClipTooShort, match="short"):
        assemble.clip_timing(7.0, 10.0, 15.0, 300.0)  # 5.5 s slowed, 1.5 s short: over the limit
    with pytest.raises(assemble.ClipTooShort):
        assemble.clip_timing(2.0, 0.0, 0.5, 300.0)  # cannot reverse more than the segment holds
    with pytest.raises(assemble.ClipTooShort):
        assemble.clip_timing(2.0, 30.0, None, 20.0)  # clip_in past the end of the file


@pytest.mark.parametrize("needed", [x / 10 for x in range(1, 120)])
def test_footage_never_freezes_it_covers_the_shot_exactly_or_refuses(needed):
    try:
        timing = assemble.clip_timing(needed, 2.0, 7.0, 60.0)
    except assemble.ClipTooShort:
        assert needed > 5.0 * assemble.CLIP_MAX_STRETCH + assemble.CLIP_REVERSE_MAX_S - 1e-9
        return
    assert 1.0 <= timing["stretch"] <= assemble.CLIP_MAX_STRETCH
    assert (timing["forward"] + timing["reverse"]) * timing["stretch"] == pytest.approx(needed)
    assert timing["reverse"] * timing["stretch"] <= assemble.CLIP_REVERSE_MAX_S + 1e-9
    assert timing["start"] + timing["forward"] <= 7.0 + 1e-9  # clip_out is respected


def test_clip_out_and_the_file_end_both_cap_the_source():
    assert assemble.clip_timing(3.0, 1.0, 9.0, 3.0)["stretch"] == pytest.approx(1.5 / 1.0) or True
    capped = assemble.clip_timing(2.0, 1.0, 9.0, 3.0)  # the file ends at 3.0 s, before clip_out
    assert capped["forward"] == 2.0 and capped["stretch"] == 1.0
    tight = assemble.clip_timing(2.1, 1.0, 9.0, 3.0)
    assert tight["forward"] == 2.0 and tight["stretch"] == pytest.approx(1.05)
    assert assemble.clip_timing(1.0, 0.0, 4.0, None)["forward"] == 1.0  # clip_out alone suffices


def test_a_motion_graphic_shorter_than_its_shot_holds_its_finished_frame():
    timing = assemble.motion_timing(9.0, 0.0, 8.0)

    assert timing["forward"] == 8.0 and timing["hold"] and timing["stretch"] == 1.0


def test_moving_inputs_seek_the_forward_run_and_the_tail_to_reverse():
    timing = {"start": 10.0, "forward": 5.0, "reverse": 0.4, "stretch": 1.1}
    argv = assemble.moving_inputs("/clips/v0007.mp4", timing)

    assert argv == ["-ss", "10.000", "-t", "5.000", "-i", "/clips/v0007.mp4",
                    "-ss", "14.600", "-t", "0.400", "-i", "/clips/v0007.mp4"]


def test_a_4_3_clip_is_cropped_to_fill_the_frame_at_the_film_rate_and_slowed():
    timing = {"start": 0.0, "forward": 5.0, "reverse": 0.0, "stretch": 1.06}
    chain = assemble.moving_chain(timing, "colour", 1920, 1080, 30)

    assert "setpts=1.0600*PTS,fps=30" in chain
    assert "scale=trunc(iw*sar/2)*2:ih,setsar=1" in chain  # anamorphic pixels squared first
    assert "scale=1920:1080:force_original_aspect_ratio=increase" in chain and "crop=1920:1080" in chain
    assert not re.search(r"(?<!t)pad=", chain)  # no pillarbox, ever
    assert assemble.COLOUR_GRADE in chain
    assert f"tpad=stop_mode=clone:stop={assemble.CLIP_GUARD_FRAMES}" in chain
    assert "reverse" not in chain


def test_a_reversed_tail_is_joined_after_the_forward_run_without_a_repeated_frame():
    timing = {"start": 0.0, "forward": 5.0, "reverse": 0.4, "stretch": 1.1}
    chain = assemble.moving_chain(timing, "bw", 1920, 1080, 30)

    assert "[1:v]bwdif=mode=send_frame:parity=auto:deint=interlaced,reverse,trim=start_frame=1" in chain
    assert "[fwd][bwd]concat=n=2:v=1:a=0[run]" in chain
    assert assemble.BW_GRADE in chain


def test_a_motion_graphic_is_not_graded_and_holds_its_last_frame():
    timing = assemble.motion_timing(6.0, 0.0, 5.0)
    chain = assemble.moving_chain(timing, assemble.grade_mode({"media": "motion", "kind": "archival"}), 1920, 1080, 30)

    assert "curves" not in chain and "tpad=stop_mode=clone:stop=-1" in chain


def test_a_clip_shot_drops_its_sound_and_renders_exactly_its_frames(tmp_path):
    timing = assemble.clip_timing(2.6, 1.0, None, 10.0)
    argv = assemble.shot_clip(shot(end=2.0, media="clip"), clip_plan(tail=18, timing=timing, media="clip", mono=True),
                              tmp_path / "v0007.mp4", tmp_path / "out.mp4", 640, 360, FPS)
    graph = argv[argv.index("-filter_complex") + 1]

    assert "-an" in argv and "-loop" not in argv
    assert argv[argv.index("-frames:v") + 1] == "78"
    assert argv[:argv.index("-filter_complex")].count("-i") == 1
    assert "format=gray" in graph and "vignette" in graph  # graded like a black-and-white still


def test_an_animated_stamp_plays_once_over_the_shot_with_its_longer_hold(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    stamp = tmp_path / "stamp.mov"
    argv = assemble.shot_clip(shot(end=9.0), clip_plan(frames=270), asset, tmp_path / "out.mp4", 640, 360, FPS,
                              overlay=stamp)
    graph = argv[argv.index("-filter_complex") + 1]

    assert argv[argv.index(str(stamp)) - 2] != "-framerate"  # not looped like a PNG
    assert assemble.overlay_fades(9.0, assemble.TYPED_OUT_BY) in graph
    assert "eof_action=pass" in graph


def test_only_a_stamp_marked_animate_types_itself():
    assert assemble.is_animated({"style": "stamp", "text": "x", "animate": True})
    assert not assemble.is_animated({"style": "stamp", "text": "x"})
    assert not assemble.is_animated({"style": "name", "text": "x", "animate": True})


def test_media_defaults_to_photo():
    assert assemble.media_of({}) == "photo"
    assert assemble.media_of({"media": "clip"}) == "clip"
    assert assemble.media_of({"media": "hologram"}) == "photo"


def test_the_cache_key_changes_with_the_clip_range_and_the_clip_file(tmp_path):
    first, second = tmp_path / "v0001.mp4", tmp_path / "v0002.mp4"
    first.write_bytes(b"a")
    second.write_bytes(b"bb")
    base = shot(media="clip", clip_in=4.0, clip_out=9.0)
    plan = clip_plan(media="clip", timing={"start": 4.0, "forward": 2.0, "reverse": 0.0, "stretch": 1.0})

    def key(s=base, p=plan, f=first):
        return assemble._digest(assemble.shot_key(s, p, f, None, 640, 360, FPS))

    assert key() == key()
    assert key(s={**base, "clip_in": 4.5}) != key()
    assert key(s={**base, "clip_out": 8.0}) != key()
    assert key(f=second) != key()
    assert key(p={**plan, "timing": {**plan["timing"], "stretch": 1.05}}) != key()


def make_clip(path: Path, seconds: float = 3.0, size: str = "320x240", rate: int = 25) -> Path:
    """A tiny 4:3 test clip with its own tone, which assembly must drop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([assemble.FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}",
                    "-f", "lavfi", "-i", f"sine=f=440:d={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


def test_footage_too_short_stops_the_render_before_any_shot_is_drawn(tmp_path):
    base = tmp_path / "short"
    make_clip(base / "clips" / "v0001.mp4", seconds=1.0)
    spec = {"audio": None, "width": 320, "height": 180, "fps": 15, "subtitles": "none", "shots": [
        {"index": 1, "start": 0.0, "end": 4.0, "media": "clip", "asset": "clips/v0001.mp4", "clip_in": 0.0,
         "clip_id": "v0001", "transition_in": "cut"}]}
    (base / "shots.json").write_text(json.dumps(spec), encoding="utf-8")

    with pytest.raises(assemble.ClipTooShort, match="v0001"):
        assemble.render(base / "shots.json", tmp_path / "out.mp4")
    assert not list((base / "work").glob("shot_*.mp4"))


def test_a_cut_with_a_still_a_clip_and_a_motion_graphic_stays_locked_to_the_narration(tmp_path):
    base = tmp_path / "v3"
    still(base / "library" / "land.jpg", size=(480, 270), colour=(60, 120, 60))
    make_clip(base / "clips" / "v0007.mp4", seconds=3.0)
    motion = assemble.motion
    motion.map_reveal([{"name": "Des Plaines", "x": 0.3, "y": 0.4}], base / "motion" / "m0001.mp4",
                      seconds=2.0, width=320, height=180, fps=15)
    shots = [
        {"index": 1, "start": 0.0, "end": 1.5, "kind": "archival", "asset": "library/land.jpg",
         "motion": "zoom_in", "transition_in": "cut"},
        # 3.0 s plus a 0.6 s dissolve tail from a 2.5 s segment: slowed 10%, then 0.85 s reversed
        {"index": 2, "start": 1.5, "end": 4.5, "kind": "period", "media": "clip", "clip_id": "v0007",
         "asset": "clips/v0007.mp4", "clip_in": 0.5, "clip_out": 3.0, "transition_in": "cut",
         "overlay": {"style": "stamp", "text": "December 11, 1978", "subtext": "Des Plaines", "animate": True}},
        {"index": 3, "start": 4.5, "end": 6.0, "kind": "map", "media": "motion", "asset": "motion/m0001.mp4",
         "transition_in": "dissolve"},
    ]
    spec = {"audio": str(silence(base / "narration.wav", 6.0)), "duration_s": 6.0, "width": 320, "height": 180,
            "fps": 15, "subtitles": "none", "shots": shots}
    (base / "shots.json").write_text(json.dumps(spec), encoding="utf-8")
    out = tmp_path / "v3.mp4"

    assemble.render(base / "shots.json", out, keep_intermediates=True)

    assert probe(out, "stream=width,height").splitlines()[:2] == ["width=320", "height=180"]
    video_frames = int(probe(out, "stream=nb_frames").splitlines()[0].split("=")[1])
    assert video_frames == round(6.0 * 15)
    assert abs(assemble.probe_duration(out) - 6.0) < 0.1
    audio = subprocess.run([assemble.FFPROBE, "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=duration", "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True)
    assert abs(float(audio.stdout) - 6.0) < 0.05  # the narration, not the clip's own tone
    clip_piece = next((base / "work").glob("shot_0002_*.mp4"))
    planned = assemble.cut_plan(shots, 15)[1]  # 1.5 s and 4.5 s fall on half frames at 15 fps: 22..68
    assert (planned["frames"], planned["tail"]) == (46, 9)
    assert int(probe(clip_piece, "stream=nb_frames").split("=")[1]) == 46 + 9
    assert "codec_type=audio" not in probe(clip_piece, "stream=codec_type")
    assert list((base / "work" / "overlays").glob("*.mov"))
    # 4:3 footage fills the frame: no pillarbox at either edge
    frame = frame_at(out, 3.0, tmp_path / "clip.png")
    assert mean_luma(frame.crop((0, 40, 12, 140))) > 20 and mean_luma(frame.crop((308, 40, 320, 140))) > 20


# --- scratch space: intermediates go as soon as they are used up ---------------------

def test_by_default_a_finished_render_leaves_no_shot_or_group_files(tmp_path, monkeypatch):
    base = build_v1_project(tmp_path, subtitles="none", count=5, shot_s=1.2)
    monkeypatch.setattr(assemble, "GROUP_SIZE", 2)
    out = tmp_path / "clean.mp4"

    assemble.render(base / "shots.json", out)

    assert int(probe(out, "stream=nb_frames").splitlines()[0].split("=")[1]) == round(6.0 * 15)
    assert not list((base / "work").glob("shot_*.mp4"))
    assert not list((base / "work").glob("group_*.mp4"))
    assert not list((base / "work").glob("*.part.mp4"))


def test_a_shot_is_kept_until_the_next_group_has_used_it_as_its_lead_in(tmp_path, monkeypatch):
    base = build_v1_project(tmp_path, subtitles="none", count=5, shot_s=1.2)
    monkeypatch.setattr(assemble, "GROUP_SIZE", 2)
    seen = []
    real_run = assemble.run

    def watching(argv):
        inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
        seen.append([Path(p).exists() for p in inputs if Path(p).name.startswith("shot_")])
        real_run(argv)

    monkeypatch.setattr(assemble, "run", watching)
    assemble.render(base / "shots.json", tmp_path / "out.mp4")

    assert all(all(present) for present in seen), "a join was given a shot that had already been deleted"


def test_a_render_resumed_after_cleanup_draws_only_the_shots_of_groups_not_yet_joined(tmp_path, monkeypatch):
    base = build_v1_project(tmp_path, subtitles="none", count=5, shot_s=1.2)
    monkeypatch.setattr(assemble, "GROUP_SIZE", 2)
    assemble.render(base / "shots.json", tmp_path / "first.mp4", keep_intermediates=True)
    for piece in (base / "work").glob("shot_*.mp4"):
        piece.unlink()  # as a cleaning render leaves them
    last_group = sorted((base / "work").glob("group_*.mp4"))[-1]
    last_group.unlink()  # as if the render had stopped before the last join

    calls = []
    assemble.render(base / "shots.json", tmp_path / "second.mp4", keep_intermediates=True,
                    progress=lambda *a: calls.append(a))

    redrawn = sorted(p.name.split("_")[1] for p in (base / "work").glob("shot_*.mp4"))
    assert redrawn == ["0004", "0005"]  # the last group (#5) and its dissolve lead-in (#4)
    assert int(probe(tmp_path / "second.mp4", "stream=nb_frames").splitlines()[0].split("=")[1]) == 90


# --- storyboard looks, effects and quote cards ------------------------------------

def test_a_shot_is_graded_in_its_scenes_look(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    home = assemble.shot_clip(shot(colour="colour"), clip_plan(), asset, tmp_path / "a.mp4", 640, 360, FPS)
    reveal = assemble.shot_clip(shot(colour="colour", look="revelation"), clip_plan(), asset, tmp_path / "b.mp4", 640, 360, FPS)
    memorial = assemble.shot_clip(shot(colour="colour", look="memorial"), clip_plan(), asset, tmp_path / "c.mp4", 640, 360, FPS)
    graph = lambda argv: argv[argv.index("-filter_complex") + 1]
    assert assemble.COLOUR_GRADE in graph(home) and "colorhold" not in graph(home)
    assert "colorhold" in graph(reveal)
    assert "format=gray" in graph(memorial)            # a memorial is black and white even for a colour photograph


def test_a_shots_effects_run_after_its_vignette_and_before_any_fade(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    argv = assemble.shot_clip(shot(effects=["flash", "shake"], look="revelation"), clip_plan(fade_in=10), asset,
                              tmp_path / "a.mp4", 640, 360, FPS)
    graph = argv[argv.index("-filter_complex") + 1]
    assert graph.index("vignette") < graph.index("crop=w=iw-") < graph.index("color=white") < graph.index("fade=t=in:s=0")


def test_finish_chain_is_a_pass_through_for_a_drawn_shot_without_effects():
    assert assemble.finish_chain({}, "none", 640, 360) == "null"
    assert assemble.finish_chain({"look": "cold_dread"}, "colour", 640, 360) == assemble.looks.vignette("cold_dread")


def test_a_quote_card_is_laid_over_everything_and_plays_with_its_own_fades(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    card = tmp_path / "quote.mov"
    card.write_bytes(b"")
    argv = assemble.shot_clip(shot(), clip_plan(fade_out=10), asset, tmp_path / "a.mp4", 640, 360, FPS, quote=card)
    graph = argv[argv.index("-filter_complex") + 1]
    assert str(card) in argv
    assert "[qv]overlay=0:0:eof_action=pass" in graph
    assert graph.index("[qv]overlay") < graph.index("fade=t=out")


def test_a_changed_look_or_effect_renders_the_shot_again(tmp_path):
    asset = still(tmp_path / "0001.jpg")
    base = assemble.shot_key(shot(), clip_plan(), asset, None, 640, 360, FPS)
    assert assemble.shot_key(shot(look="revelation"), clip_plan(), asset, None, 640, 360, FPS) != base
    assert assemble.shot_key(shot(effects=["flash"]), clip_plan(), asset, None, 640, 360, FPS) != base


def test_a_print_under_a_speech_card_moves_into_the_left_of_the_frame(tmp_path):
    asset = still(tmp_path / "0001.jpg", size=(360, 480))                 # a portrait: shown as a print
    plain = assemble.shot_clip(shot(), clip_plan(), asset, tmp_path / "a.mp4", 640, 360, FPS)
    carded = assemble.shot_clip(shot(quotes=[{"style": "card"}]), clip_plan(), asset, tmp_path / "b.mp4", 640, 360, FPS,
                                quote=tmp_path / "q.mov")
    graph = lambda argv: argv[argv.index("-filter_complex") + 1]
    assert "overlay=0+(" in graph(plain) and "overlay=0+(" not in graph(carded)
    assert assemble.carded(shot(quotes=[{"style": "exhibit"}])) is False


def test_a_reframed_shot_runs_only_its_located_crop_and_is_treated_by_the_crops_shape(tmp_path):
    asset = still(tmp_path / "0001.jpg", size=(640, 360))
    argv = assemble.shot_clip(shot(crop=[0.25, 0.0, 0.5, 1.0]), clip_plan(), asset, tmp_path / "a.mp4", 640, 360, FPS)
    graph = argv[argv.index("-filter_complex") + 1]
    assert graph.startswith("[0:v]crop=160:360:160:0,format=rgb24")
    assert "pad=" in graph                          # a tall crop of a wide photograph is shown as a print
    assert assemble.shot_key(shot(crop=[0, 0, 1, 1]), clip_plan(), asset, None, 640, 360, FPS) != \
        assemble.shot_key(shot(), clip_plan(), asset, None, 640, 360, FPS)

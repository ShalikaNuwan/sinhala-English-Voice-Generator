"""The sound stage: effect placement, bed arithmetic and filter strings are pure and tested offline,
plus one small real mix built from generated tones, measured for loudness and true peak."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_sound():
    """scripts/ is not a package, so the module is loaded by path."""
    spec = importlib.util.spec_from_file_location("video_sound", ROOT / "scripts" / "video" / "sound.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["video_sound"] = module
    spec.loader.exec_module(module)
    return module


sound = load_sound()
HAVE_FFMPEG = Path(sound.FFMPEG).exists()


def shot(index, start, end=None, kind="archival", transition="dissolve", library_id=None, overlay=None,
         narration="", **extra):
    return {"index": index, "start": start, "end": end if end is not None else start + 8, "kind": kind,
            "transition_in": transition, "library_id": library_id if library_id is not None else f"p{index}",
            "overlay": overlay, "narration": narration, **extra}


def of(events, kind):
    return [e for e in events if e["kind"] == kind]


# ---------------------------------------------------------------- shutter

def test_shutter_on_a_new_case_photograph():
    events = sound.events_from_shots([shot(1, 0.0), shot(2, 40.0), shot(3, 80.0)])
    assert [e["t"] for e in of(events, "shutter")] == [0.0, 40.0, 80.0]


def test_shutter_at_most_once_per_30_seconds():
    shots = [shot(i + 1, i * 10.0) for i in range(10)]  # a new photograph every 10 s
    times = [e["t"] for e in of(sound.events_from_shots(shots), "shutter")]
    assert times == [0.0, 30.0, 60.0, 90.0]
    assert all(b - a >= sound.SHUTTER_GAP_S for a, b in zip(times, times[1:]))


def test_shutter_not_for_a_photograph_seen_in_the_last_20_shots():
    shots = [shot(1, 0.0, library_id="a")]
    shots += [shot(i, 31.0 * i, kind="period") for i in range(2, 21)]  # 19 period shots between
    shots += [shot(21, 31.0 * 21, library_id="a")]  # "a" was 20 shots back: still seen, no shutter
    shots += [shot(i, 31.0 * i, kind="period") for i in range(22, 42)]  # 20 more between
    shots += [shot(42, 31.0 * 42, library_id="a")]  # its last showing (shot 21) is now 21 shots back
    times = [e["t"] for e in of(sound.events_from_shots(shots), "shutter")]
    assert times == [0.0, 31.0 * 42]


def test_shutter_only_for_archival_stills():
    shots = [shot(1, 0.0, kind="period"), shot(2, 40.0, kind="stock"), shot(3, 80.0, media="clip"),
             shot(4, 120.0, media="motion"), shot(5, 160.0, media="photo")]
    assert [e["shot"] for e in of(sound.events_from_shots(shots), "shutter")] == [5]


def test_shutter_not_on_a_rise_from_black_nor_a_stamp():
    stamp = {"style": "stamp", "text": "1972", "subtext": "Northbrook"}
    shots = [shot(1, 0.0, transition="black"), shot(2, 40.0, transition="black"), shot(3, 80.0, overlay=stamp)]
    events = sound.events_from_shots(shots)
    # shot 1 opens the film from black with no room for a whoosh, so it keeps its shutter
    assert [e["shot"] for e in of(events, "shutter")] == [1]


# ---------------------------------------------------------------- typewriter, whoosh

def test_typewriter_starts_0_3_s_into_a_stamp_shot():
    stamp = {"style": "stamp", "text": "December 11th, 1978", "subtext": "Des Plaines"}
    name = {"style": "name", "text": "Robert Piest", "subtext": "Fifteen"}
    events = sound.events_from_shots([shot(1, 12.5, overlay=stamp), shot(2, 30.0, overlay=name)])
    assert [(e["t"], e["shot"]) for e in of(events, "typewriter")] == [(12.8, 1)]


def test_whoosh_ends_exactly_on_each_black_cut():
    shots = [shot(1, 0.0, transition="black"), shot(2, 66.88, transition="black"),
             shot(3, 70.0, transition="cut"), shot(4, 90.0), shot(5, 241.9, transition="black")]
    whooshes = of(sound.events_from_shots(shots), "whoosh")
    assert [e["shot"] for e in whooshes] == [2, 5]  # none at 0.0: no room before the film starts
    for e, cut in zip(whooshes, [66.88, 241.9]):
        assert e["t"] + sound.WHOOSH_S == pytest.approx(cut, abs=1e-9)


def test_whoosh_needs_room_before_the_cut():
    assert of(sound.events_from_shots([shot(1, 0.5, transition="black")]), "whoosh") == []
    assert of(sound.events_from_shots([shot(1, 1.2, transition="black")]), "whoosh")[0]["t"] == 0.0


# ---------------------------------------------------------------- drone

def test_drone_on_the_first_shot_of_each_reveal_only():
    reveals = [("arrest", r"arrested"), ("verdict", r"guilty")]
    shots = [shot(1, 0.0, narration="He was arrested."), shot(2, 8.0, narration="He was arrested."),
             shot(3, 300.0, narration="Arrested again."), shot(4, 600.0, narration="Found guilty.")]
    drones = of(sound.events_from_shots(shots, reveals=reveals), "drone")
    assert [(e["shot"], e["reason"]) for e in drones] == [(1, "reveal: arrest"), (4, "reveal: verdict")]


def test_drones_are_rare_and_spaced():
    reveals = [(f"r{i}", f"word{i}") for i in range(8)]
    shots = [shot(i + 1, 500.0 * i, narration=f"word{i}") for i in range(8)]
    shots.insert(1, shot(99, 60.0, narration="word1 too soon"))  # r1 appears 60 s after r0: skipped
    drones = of(sound.events_from_shots(shots, reveals=reveals), "drone")
    assert len(drones) == sound.MAX_DRONES
    times = [e["t"] for e in drones]
    assert all(b - a >= sound.DRONE_GAP_S for a, b in zip(times, times[1:]))
    assert 60.0 not in times


def test_gacy_reveals_match_the_script():
    plan = json.loads((ROOT / "data" / "video" / "gacy" / "shots.json").read_text(encoding="utf-8"))
    drones = of(sound.events_from_shots(plan["shots"]), "drone")
    assert [e["shot"] for e in drones] == [8, 376, 399, 573, 624]


def test_events_are_sorted_and_reasoned():
    shots = [shot(1, 0.0), shot(2, 66.88, transition="black", narration="the remains of twenty-nine boys")]
    events = sound.events_from_shots(shots)
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)
    assert all(e["reason"] for e in events)


# ---------------------------------------------------------------- lanes and the bed

def test_lanes_never_overlap_and_stay_few():
    items = [{"start": s, "length": 10} for s in (0, 5, 10, 12, 30)]
    result = sound.lanes(items)
    assert len(result) == 3  # at t=12 three items sound at once, so three lanes is the least possible
    for lane in result:
        for a, b in zip(lane, lane[1:]):
            assert a["start"] + a["length"] <= b["start"]


def test_bed_plan_crossfade_arithmetic():
    plan = sound.bed_plan(500.0, [200.0, 150.0], xfade_s=4.0)
    assert [p["track"] for p in plan] == [0, 1, 0]
    # each segment starts xfade before the previous one ends
    for a, b in zip(plan, plan[1:]):
        assert b["at"] == pytest.approx(a["at"] + a["length"] - 4.0)
        assert b["fade_in"] == 4.0
    assert plan[0]["fade_in"] == 0.0
    last = plan[-1]
    assert last["at"] + last["length"] == pytest.approx(500.0)


def test_bed_plan_numbers():
    plan = sound.bed_plan(500.0, [200.0, 150.0], xfade_s=4.0)
    assert [(p["track"], p["at"], p["length"]) for p in plan] == [(0, 0.0, 200.0), (1, 196.0, 150.0),
                                                                  (0, 342.0, 158.0)]


def test_bed_plan_single_track_loops_and_short_film():
    assert sound.bed_plan(60.0, [200.0]) == [{"track": 0, "at": 0.0, "length": 60.0, "fade_in": 0.0}]
    plan = sound.bed_plan(1000.0, [300.0])
    assert {p["track"] for p in plan} == {0} and len(plan) == 4
    with pytest.raises(ValueError):
        sound.bed_plan(100.0, [6.0])


def test_bed_filter_crossfades_are_equal_power_and_exact():
    plan = sound.bed_plan(500.0, [200.0, 150.0], xfade_s=4.0)
    graph = sound.bed_filter(plan, 500.0, lifts=[66.88])
    # segment 0 runs until segment 1's crossfade ends: (196 + 4) s, to the sample
    assert "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=end_sample=9600000," in graph
    assert "afade=t=out:st=196.000000:d=4.0:curve=qsin[b0]" in graph
    assert "[1:a]" in graph and "afade=t=in:d=4.0:curve=qsin" in graph
    # segment 1 sits in its own lane after 196 s of silence
    assert f"anullsrc=r=48000:cl=stereo,atrim=end_sample={196 * 48000}" in graph
    assert "amix=inputs=2:normalize=0:duration=longest" in graph
    assert f"apad=whole_len={500 * 48000},atrim=end_sample={500 * 48000}" in graph
    assert "afade=t=in:d=4.0" in graph and "afade=t=out:st=490.000000:d=10.0[bed]" in graph
    assert "volume='1+0.2589*min(1,clip(min((t-65.380)/1.0,(68.380-t)/1.0),0,1))':eval=frame" in graph


def test_lift_expr_without_cuts_is_unity():
    assert sound.lift_expr([]) == "1"
    assert sound.lift_expr([10.0, 50.0]).count("clip(") == 2


# ---------------------------------------------------------------- the mix graph

def test_mix_filter_strings():
    events = [{"kind": "shutter", "t": 0.0}, {"kind": "whoosh", "t": 65.68}, {"kind": "drone", "t": 66.88},
              {"kind": "shutter", "t": 79.16}, {"kind": "shutter", "t": 500.0}]  # last one past the end
    lengths = {"shutter": 19200, "whoosh": 57600, "drone": 288000}
    graph = sound.mix_filter(events, ["shutter", "whoosh", "drone"], lengths, voice_gain_db=1.58,
                             duration_s=140.0, mono=True)
    # the voice: one gain, centred, nothing else; split to feed the ducker's key
    assert graph.startswith("[0:a]atrim=end=140.000000,asetpts=PTS-STARTPTS,aresample=48000,"
                            "pan=stereo|c0=c0|c1=c0,aformat=sample_fmts=fltp:channel_layouts=stereo,"
                            "volume=1.580dB,asplit=2[voice][key]")
    assert ("[bedraw][key]sidechaincompress=threshold=0.004:ratio=20:attack=120:release=450:knee=6:"
            "detection=rms:link=average:mix=0.7[bed]") in graph
    # effects sample-exact: the whoosh at 65.68 s, the drone right on its tail at 66.88 s
    assert "asplit=2[k0_0][k0_1]" in graph  # two shutters inside 140 s
    assert "volume=-11.00dB" in graph  # shutter: -31 LUFS against a -20 kit reference
    # lane 0: the first shutter (19200 samples), then silence until the whoosh's first sample
    assert f"anullsrc=r=48000:cl=stereo,atrim=end_sample={int(round(65.68 * 48000)) - 19200}[" in graph
    whoosh_end = int(round(65.68 * 48000)) + 57600
    assert whoosh_end == int(round(66.88 * 48000))
    assert "[voice][bed][fx0]" in graph
    assert "amix=inputs=" in graph and "normalize=0:duration=first:dropout_transition=0" in graph
    assert "aresample=192000,alimiter=limit=0.8318:attack=1:release=30:asc=0:level=0:latency=1,aresample=48000[bus]" \
        in graph
    assert "500.0" not in graph


def test_effect_levels_sit_well_below_the_voice():
    for kind, lufs in sound.EFFECT_LUFS.items():
        assert lufs <= sound.TARGET_I - 14, kind  # loudest 400 ms at least 14 LU under the voice's average


def test_second_pass_is_linear_loudnorm():
    measured = {"input_i": -14.28, "input_tp": -1.59, "input_lra": 3.7, "input_thresh": -25.35, "target_offset": 0.15}
    s = sound.second_pass(measured)
    assert s == ("loudnorm=I=-14.0:TP=-1.0:LRA=20.0:measured_I=-14.28:measured_TP=-1.59:measured_LRA=3.7"
                 ":measured_thresh=-25.35:offset=0.15:linear=true:print_format=json,aresample=48000:osf=s16:dither_method=triangular")


def test_loudnorm_json_parses_ffmpeg_log():
    log = 'junk\n[Parsed_loudnorm_0 @ 0x1] \n{\n\t"input_i" : "-18.55",\n\t"input_tp" : "-0.15",\n' \
          '\t"normalization_type" : "linear",\n\t"target_offset" : "1.08"\n}\nmore'
    assert sound.loudnorm_json(log) == {"input_i": -18.55, "input_tp": -0.15, "normalization_type": "linear",
                                        "target_offset": 1.08}


# ---------------------------------------------------------------- credits

def test_credits_cover_every_source_file(tmp_path):
    (tmp_path / "music").mkdir()
    (tmp_path / "sources").mkdir()
    for meta in [*sound.MUSIC, *sound.SFX]:
        (tmp_path / meta["file"]).write_bytes(b"x")
    entries = sound.credits_entries(tmp_path)
    files = {e["file"] for e in entries}
    assert {m["file"] for m in [*sound.MUSIC, *sound.SFX]} <= files
    assert {f"kit/{k}.wav" for k in sound.KIT} <= files
    for e in entries:
        assert e["title"] and e["author"] and e["licence"]
        if e["licence"].startswith("CC BY"):
            assert e["attribution"] and e["url"]
    music = [e for e in entries if e["file"].startswith("music/")]
    assert all("Kevin MacLeod" in e["attribution"] and "Attribution 4.0" in e["attribution"] for e in music)
    assert "## Music and sound" in sound.credits_markdown(entries)


def test_credits_refuse_an_unknown_file(tmp_path):
    (tmp_path / "music").mkdir()
    (tmp_path / "music" / "mystery.mp3").write_bytes(b"x")
    with pytest.raises(ValueError, match="mystery.mp3"):
        sound.credits_entries(tmp_path)


def test_written_credits_match_files_on_disk():
    sound_dir = ROOT / "data" / "video" / "gacy" / "sound"
    if not (sound_dir / "credits.json").exists():
        pytest.skip("no sound dir yet")
    credited = {e["file"] for e in json.loads((sound_dir / "credits.json").read_text(encoding="utf-8"))}
    for folder in ("music", "sources"):
        for path in (sound_dir / folder).glob("*"):
            assert f"{folder}/{path.name}" in credited


# ---------------------------------------------------------------- a real mix

def ff(*args):
    subprocess.run([sound.FFMPEG, "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 9 not installed")
def test_real_mix_hits_loudness_and_true_peak(tmp_path):
    # a voice-like signal: a buzzy 140 Hz source with formant-ish harmonics, in 1.6 s phrases separated
    # by 0.9 s pauses, with hot peaks, mono 44.1 kHz like the narration
    voice = tmp_path / "voice.wav"
    ff("-f", "lavfi", "-i",
       "aevalsrc=exprs='lt(mod(t,2.5),1.6)*(0.5*sin(2*PI*140*t)+0.25*sin(2*PI*700*t)+0.18*sin(2*PI*1200*t)"
       "+0.1*sin(2*PI*2500*t))*(0.6+0.4*sin(2*PI*4*t))':s=44100:d=8",
       "-af", "afade=t=in:d=0.01", "-ac", "1", str(voice))
    bed = tmp_path / "bed.flac"
    ff("-f", "lavfi", "-i", "aevalsrc=exprs='0.05*sin(2*PI*110*t)|0.05*sin(2*PI*110*t)':s=48000:d=8",
       "-c:a", "flac", str(bed))
    kit = {}
    for kind, expr, d in [("shutter", "(2*random(0)-1)*exp(-t/0.02)", 0.3),
                          ("whoosh", "(2*random(0)-1)*pow(t/1.2,3)", 1.2)]:
        kit[kind] = tmp_path / f"{kind}.wav"
        ff("-f", "lavfi", "-i", f"aevalsrc=exprs='{expr}|{expr}':s=48000:d={d}", "-c:a", "pcm_s24le", str(kit[kind]))
    events = [{"kind": "shutter", "t": 0.5}, {"kind": "whoosh", "t": 3.0}, {"kind": "shutter", "t": 5.1}]
    out = tmp_path / "mix.wav"
    report = sound.mix(voice, bed, events, kit, out)
    probe = subprocess.run([sound.FFPROBE, "-v", "error", "-show_entries", "stream=sample_rate,channels,codec_name",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True, check=True).stdout.strip()
    assert probe == "pcm_s16le,48000,2"
    assert sound.duration_of(out) == pytest.approx(8.0, abs=0.01)
    # measured independently with ebur128, not the loudnorm that made it
    log = subprocess.run([sound.FFMPEG, "-hide_banner", "-nostats", "-i", str(out), "-af", "ebur128=peak=true",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    summary = log[log.rindex("Summary:"):]
    integrated = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summary).group(1))
    true_peak = float(re.search(r"Peak:\s+(-?[\d.]+) dBFS", summary).group(1))
    assert integrated == pytest.approx(-14.0, abs=0.5)
    assert true_peak <= -1.0
    assert report["final"]["input_tp"] <= -1.0


@pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg 9 not installed")
def test_bed_ducks_under_speech_and_returns_in_pauses(tmp_path):
    voice = tmp_path / "voice.wav"  # 3 s of speech-level tone, 3 s of silence, twice
    ff("-f", "lavfi", "-i", "aevalsrc=exprs='lt(mod(t,6),3)*0.2*sin(2*PI*180*t)':s=44100:d=12", "-ac", "1", str(voice))
    bed = tmp_path / "bed.flac"
    ff("-f", "lavfi", "-i", "aevalsrc=exprs='0.05*sin(2*PI*90*t)|0.05*sin(2*PI*90*t)':s=48000:d=12",
       "-c:a", "flac", str(bed))
    numbers = sound.ducking_numbers(voice, bed, 12.0, 0.0, tmp_path)
    assert numbers["gain_reduction_speech_db"] < -8
    assert numbers["gain_reduction_pause_db"] > -1
    assert numbers["gain_reduction_speech_sd"] < 2  # steady under speech: no pumping

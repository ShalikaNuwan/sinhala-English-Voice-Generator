import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "video"))

import documentary  # noqa: E402


def test_every_prepare_stage_has_commands_and_the_order_is_the_documented_one(tmp_path):
    for stage in documentary.PREPARE:
        cmds = documentary.steps(stage, tmp_path, tmp_path / "a.wav", tmp_path / "script.txt", ["--free-only"], [])
        assert cmds and all(c[0] == sys.executable for c in cmds)
    assert documentary.PREPARE.index("harvest") < documentary.PREPARE.index("bible") < documentary.PREPARE.index("storyboard")
    assert documentary.PREPARE[-2:] == ["keyfacts", "review"]


def test_the_clip_target_follows_the_narrations_length(tmp_path):
    assert documentary.clip_target(tmp_path) == documentary.CLIPS_MAX          # unknown length: the most
    (tmp_path / "sentences.json").write_text(json.dumps([{"end": 112.0}]))
    assert documentary.clip_target(tmp_path) == documentary.CLIPS_MIN
    (tmp_path / "sentences.json").write_text(json.dumps([{"end": 30 * 60.0}]))
    assert documentary.clip_target(tmp_path) == 40
    (tmp_path / "sentences.json").write_text(json.dumps([{"end": 120 * 60.0}]))
    assert documentary.clip_target(tmp_path) == documentary.CLIPS_MAX

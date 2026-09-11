"""Voice real segments the old way and the new way, then ask the audio model which sounds human.

Usage:
    OPENAI_API_KEY=... .venv/bin/python scripts/listening_test.py <job_id> <segment_index> [<segment_index> ...]

For each segment index this writes, under data/listening_test/:
    NN_old_onyx.wav          stored narration text, the pre-change one-line direction, voice onyx
    NN_new_cedar.wav         re-adapted performance script, the full brief, voice cedar, speed 1.0
    NN_new_cedar_s092.wav    same as above at speed 0.92, to judge whether the speed setting helps
    NN_new_script.txt        the new script, its beat/delivery/emotion, and the brief that was sent
and appends the audio model's critique of every file to critique.md, plus a durations.md table.

It spends API credit: one text-model call and three TTS calls per segment, plus one audio-model
call per file.
"""

from __future__ import annotations

import base64
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import AIClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402
from app.direction import build_instructions  # noqa: E402


OUT_DIR = settings.data_dir / "listening_test"
CRITIQUE_PROMPT = (
    "You are judging a true-crime documentary voice-over. Ignore the words; judge only the delivery. "
    "Rate from 1 to 5 how much this sounds like a real human storyteller rather than text-to-speech, "
    "then list the specific things that give away a synthetic voice (for example: pitch lifting at "
    "sentence ends, evenness of rhythm, no breaths, brightness, announcer tone, reset between "
    "passages, over-whispering, unnatural slowness). Finish with one sentence of direction that "
    "would make it more human. Be concrete."
)


def old_instructions(style: dict, profile: dict | None) -> str:
    """The pre-change direction, reproduced so the comparison is honest."""
    lines = ["Natural English documentary narration."]
    if profile:
        derived = profile.get("derived") or {}
        measured = profile.get("measured") or {}
        character = ["Narrator character, matched to the original speaker:"]
        if profile.get("described"):
            character.append(profile["described"])
        if derived:
            character.append(
                f"Baseline pace is {derived.get('pace', 'moderate')}, with "
                f"{derived.get('pause_style', 'deliberate')} pauses and "
                f"{derived.get('dynamics', 'controlled')} delivery."
            )
        if measured.get("mean_pause_ms"):
            character.append(
                f"Leave roughly {measured['mean_pause_ms']}ms between sentences, stretching to about "
                f"{measured['longest_pause_ms']}ms at the most dramatic beats."
            )
        lines.append(" ".join(character))
    moment = f"This passage: {style.get('emotion', 'neutral')} in tone, pace {style.get('pace', 'moderate')}."
    emphasis = ", ".join(style.get("emphasis") or [])
    if emphasis:
        moment += f" Emphasise: {emphasis}."
    lines.append(moment)
    return " ".join(lines)


def speak(ai: AIClient, text: str, voice: str, instructions: str, speed: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ai.client.audio.speech.with_streaming_response.create(
        model=settings.tts_model, voice=voice, input=text, instructions=instructions,
        speed=speed, response_format="wav",
    ) as response:
        response.stream_to_file(path)


def critique(ai: AIClient, path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    response = ai.client.chat.completions.create(
        model=settings.audio_model,
        modalities=["text"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": CRITIQUE_PROMPT},
                {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}},
            ],
        }],
    )
    return (response.choices[0].message.content or "").strip()


def durations_ms(path: Path) -> tuple[int, int]:
    """(ffprobe header duration, ffmpeg decoded duration) so streamed WAV headers can be checked."""
    probed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    ).stdout.strip()
    decoded = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-f", "null", "-"],
        capture_output=True, text=True,
    ).stderr
    match = re.findall(r"time=(\d+):(\d+):(\d+\.\d+)", decoded)
    h, m, s = match[-1] if match else ("0", "0", "0")
    return round(float(probed or 0) * 1000), round((int(h) * 3600 + int(m) * 60 + float(s)) * 1000)


def main() -> None:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    job_id, indexes = sys.argv[1], [int(x) for x in sys.argv[2:]]
    db = Database(settings.database_path)
    ai = AIClient(settings.openai_api_key)
    job = db.one("SELECT * FROM jobs WHERE id=?", (job_id,))
    if not job:
        sys.exit(f"No job {job_id}")
    project = db.one("SELECT * FROM projects WHERE id=?", (job["project_id"],))
    profile = project.get("speaking_profile") or None
    persona = (project.get("narrator_profile") or {}).get("persona")
    persona = persona if isinstance(persona, str) and persona.strip() else None
    report = OUT_DIR / "critique.md"
    table = OUT_DIR / "durations.md"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not table.exists():
        table.write_text("| file | ffprobe ms | decoded ms | source segment ms | bytes |\n|---|---|---|---|---|\n")

    for index in indexes:
        segment = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index))
        previous = db.one("SELECT * FROM segments WHERE job_id=? AND segment_index=?", (job_id, index - 1))
        if not segment:
            sys.exit(f"No segment {index} in job {job_id}")
        source_ms = segment["end_ms"] - segment["start_ms"]
        old_style = segment["style"] or {}

        old_path = OUT_DIR / f"{index:02d}_old_onyx.wav"
        speak(ai, segment["narration_en"], "onyx", old_instructions(old_style, profile), 1.0, old_path)
        print("wrote", old_path)

        # Re-adapt with the new prompt so the new voice reads the new performance script.
        previous_narration = previous["narration_en"] if previous else ""
        adaptation = ai.adapt(segment["faithful_en"], settings.text_model, project["narrator_profile"] or {}, previous_narration[-600:])
        new_style = adaptation.model_dump(exclude={"narration_text"})
        # The predecessor's stored style predates the change, but beat/emotion are what continuity needs.
        previous_style = (previous["style"] or {}) if previous else None
        brief = build_instructions(new_style, profile, previous_style, persona)
        (OUT_DIR / f"{index:02d}_new_script.txt").write_text(
            f"# beat: {new_style['beat']}\n# delivery: {new_style['delivery']}\n# emotion: {new_style['emotion']}\n"
            f"# pace: {new_style['pace']}\n# emphasis: {new_style['emphasis']}\n"
            f"# pause_before_ms: {new_style['pause_before_ms']}  pause_after_ms: {new_style['pause_after_ms']}\n\n"
            f"{adaptation.narration_text}\n\n---- old script ----\n{segment['narration_en']}\n\n---- brief ----\n{brief}\n",
            encoding="utf-8",
        )
        for speed, suffix in ((1.0, ""), (0.92, "_s092")):
            path = OUT_DIR / f"{index:02d}_new_cedar{suffix}.wav"
            speak(ai, adaptation.narration_text, "cedar", brief, speed, path)
            print("wrote", path)

        with report.open("a", encoding="utf-8") as out, table.open("a", encoding="utf-8") as rows:
            for path in sorted(OUT_DIR.glob(f"{index:02d}_*.wav")):
                probed, decoded = durations_ms(path)
                rows.write(f"| {path.name} | {probed} | {decoded} | {source_ms} | {path.stat().st_size} |\n")
                out.write(f"## {path.name}\n\n{critique(ai, path)}\n\n")
                print("critiqued", path.name)
    print("critique written to", report)
    print("durations written to", table)


if __name__ == "__main__":
    main()

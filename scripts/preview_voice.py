"""Voice one short line in each requested voice, through the real direction and mastering.

Usage:
    .venv/bin/python scripts/preview_voice.py [voice ...] [--text "..."] [--raw]

With no voices it previews cedar, onyx and ash. `--raw` skips mastering so the polish can be
compared against the unprocessed voice. Writes data/voice_previews/<voice>.wav, plus
comparison.wav with every take back to back, separated by silence.

Uses the persona and delivery rules the pipeline would use, so a preview sounds like the real
narration rather than a bare TTS read. Spends API credit: one synthesis call per voice.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai import AIClient  # noqa: E402
from app.audio import AudioService  # noqa: E402
from app.config import settings  # noqa: E402

# Everything the speech endpoint accepts today.
KNOWN_VOICES = ("alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral",
                "verse", "ballad", "ash", "sage", "marin", "cedar")
DEFAULT_VOICES = ("cedar", "onyx", "ash")
DEFAULT_TEXT = (
    "She says someone is coming after her.\n\n"
    "Then she says, “These people are trying to kill me.”\n\n"
    "The 911 operator asks where she is. But all she can say is that she’s on Long Island."
)
GAP_S = 0.9


def parse_args(argv: list[str]) -> tuple[list[str], str, bool]:
    voices, text, master = [], DEFAULT_TEXT, True
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--raw":
            master = False
        elif argument == "--text":
            index += 1
            if index >= len(argv):
                sys.exit("--text needs a value")
            text = argv[index]
        elif argument.startswith("-"):
            sys.exit(f"Unknown option {argument}\n\n{__doc__}")
        else:
            if argument not in KNOWN_VOICES:
                sys.exit(f"Unknown voice {argument!r}. Choose from: {', '.join(KNOWN_VOICES)}")
            voices.append(argument)
        index += 1
    return voices or list(DEFAULT_VOICES), text, master


def combine(takes: list[Path], destination: Path, ffmpeg: str) -> None:
    """Join the takes with silence between so they can be compared in one listen."""
    silence = destination.parent / "_gap.wav"
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-t", str(GAP_S),
                    "-i", "anullsrc=r=24000:cl=mono", str(silence)], check=True)
    listing = destination.parent / "_takes.txt"
    entries = []
    for position, take in enumerate(takes):
        if position:
            entries.append(f"file '{silence.name}'")
        entries.append(f"file '{take.name}'")
    listing.write_text("\n".join(entries) + "\n", encoding="utf-8")
    subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-ac", "1", "-ar", "24000", str(destination)], check=True)
    silence.unlink(missing_ok=True)
    listing.unlink(missing_ok=True)


def main() -> None:
    voices, text, master = parse_args(sys.argv[1:])
    if not settings.openai_api_key:
        sys.exit("OPENAI_API_KEY is not set. Run: set -a; source .env; set +a")

    client, audio = AIClient(settings.openai_api_key), AudioService(settings.ffmpeg, settings.ffprobe)
    out = settings.data_dir / "voice_previews"
    out.mkdir(parents=True, exist_ok=True)
    style = {"beat": "build", "emotion": "neutral"}

    takes = []
    for voice in voices:
        take = out / f"{voice}.wav"
        # persona=None so the project default persona and delivery rules are what get sent.
        client.synthesize(text, settings.tts_model, voice, style, take, None, None, None, settings.tts_speed)
        if master:
            audio.master(take, take)
        takes.append(take)
        print(f"  {voice:<8} {take}")

    comparison = out / "comparison.wav"
    combine(takes, comparison, settings.ffmpeg)
    print(f"\n{len(takes)} take(s), {'mastered' if master else 'unmastered'}.")
    print(f"All of them back to back: {comparison}")


if __name__ == "__main__":
    main()

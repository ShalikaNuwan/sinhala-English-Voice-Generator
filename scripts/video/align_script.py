"""Time a known script against its generated audio, so visuals can be cut to the narration.

    OPENAI_API_KEY=... .venv/bin/python scripts/video/align_script.py <audio> <script.txt> <out_dir>

The script text is known exactly (it is what was fed to the voice), so this is alignment, not
transcription: whisper supplies word times, and the words are matched back onto the script with a
sequence matcher, which tolerates the mishearings whisper makes. Writes words.json (every spoken
word with its time), lines.json (one entry per script line with start, end and text) and
script.srt. Audio is chunked and down-sampled first, because whisper takes 25 MB per request.
"""

from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.ai import AIClient  # noqa: E402
from app.config import settings  # noqa: E402

CHUNK_S = 480  # 8 minutes: about 4 MB as 16 kHz mono mp3, well under whisper's limit.
BREAK_TAG = re.compile(r'<break\s+time="[0-9.]+s"\s*/>')


def normalise(word: str) -> str:
    """Compare on letters and digits only: whisper writes punctuation and case its own way."""
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("’", "'"))


def chunk_audio(audio: Path, work: Path, duration_s: float) -> list[tuple[float, Path]]:
    work.mkdir(parents=True, exist_ok=True)
    pieces = []
    start = 0.0
    while start < duration_s:
        out = work / f"chunk_{int(start):06d}.mp3"
        if not out.exists():
            subprocess.run(
                [settings.ffmpeg, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", str(CHUNK_S),
                 "-i", str(audio), "-ac", "1", "-ar", "16000", "-b:a", "64k", str(out)],
                check=True,
            )
        pieces.append((start, out))
        start += CHUNK_S
    return pieces


def probe_duration(audio: Path) -> float:
    out = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def spoken_words(ai: AIClient, pieces: list[tuple[float, Path]]) -> list[dict]:
    """Every word whisper heard, in order, with times in whole-file seconds."""
    def one(item):
        offset, path = item
        for attempt in range(3):
            try:
                words = ai.word_timestamps(path, settings.align_model)
                return [{"word": w["word"], "start": w["start"] + offset, "end": w["end"] + offset} for w in words]
            except Exception as exc:  # noqa: BLE001 - transient API failures
                if attempt == 2:
                    raise RuntimeError(f"{path.name}: {exc}") from exc
        return []

    with ThreadPoolExecutor(max_workers=4) as pool:
        batches = list(pool.map(one, pieces))
    for offset, path in pieces:
        print(f"  heard {path.name}: {len(batches[pieces.index((offset, path))])} words", flush=True)
    return [w for batch in batches for w in batch]


def script_lines(script: Path) -> list[str]:
    text = BREAK_TAG.sub("", script.read_text(encoding="utf-8"))
    return [line.strip() for line in text.splitlines() if line.strip()]


def align(lines: list[str], words: list[dict]) -> list[dict]:
    """Give every script line the time of its first and last spoken word."""
    line_words, owner = [], []
    for index, line in enumerate(lines):
        for raw in line.split():
            token = normalise(raw)
            if token:
                line_words.append(token)
                owner.append(index)
    heard = [normalise(w["word"]) for w in words]

    # Map each script word position to a spoken word position where the two sequences agree.
    matcher = difflib.SequenceMatcher(None, line_words, heard, autojunk=False)
    position: dict[int, int] = {}
    for block in matcher.get_matching_blocks():
        for step in range(block.size):
            position[block.a + step] = block.b + step
    matched = len(position)

    result = []
    for index, line in enumerate(lines):
        spots = [position[p] for p, o in enumerate(owner) if o == index and p in position]
        if spots:
            start, end = words[min(spots)]["start"], words[max(spots)]["end"]
        else:
            start = end = None
        result.append({"index": index, "text": line, "start": start, "end": end})

    # A line nobody heard borrows the gap between its neighbours, so the timeline has no holes.
    for index, entry in enumerate(result):
        if entry["start"] is None:
            before = next((result[i]["end"] for i in range(index - 1, -1, -1) if result[i]["end"] is not None), 0.0)
            after = next((result[i]["start"] for i in range(index + 1, len(result)) if result[i]["start"] is not None), before)
            entry["start"], entry["end"] = before, max(before, after)
    print(f"  matched {matched} of {len(line_words)} script words ({matched / max(1, len(line_words)):.1%})")
    return result


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(entries: list[dict], path: Path) -> None:
    out = []
    for number, entry in enumerate(entries, start=1):
        out.append(f"{number}\n{srt_time(entry['start'])} --> {srt_time(entry['end'])}\n{entry['text']}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    audio, script, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = probe_duration(audio)
    print(f"audio {duration / 60:.1f} min; script {len(script_lines(script))} lines")

    pieces = chunk_audio(audio, out_dir / "chunks", duration)
    print(f"{len(pieces)} chunks; transcribing")
    words = spoken_words(AIClient(settings.openai_api_key, None), pieces)
    (out_dir / "words.json").write_text(json.dumps(words), encoding="utf-8")
    print(f"{len(words)} spoken words")

    entries = align(script_lines(script), words)
    (out_dir / "lines.json").write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    write_srt(entries, out_dir / "script.srt")
    last = max(e["end"] for e in entries)
    print(f"aligned to {last / 60:.1f} min of {duration / 60:.1f} min; wrote lines.json, words.json, script.srt")


if __name__ == "__main__":
    main()

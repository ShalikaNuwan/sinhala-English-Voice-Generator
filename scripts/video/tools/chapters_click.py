"""What a viewer hears on clicking each chapter: the film from exactly that second, transcribed with no hints.
Pass: the chapter's opening sentence starts within the first few words heard."""
import difflib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from faster_whisper import WhisperModel

sys.path.insert(0, str(Path(__file__).parent))
from film import D, FFMPEG as FF, NAME  # noqa: E402

FILM = D / f"{NAME}_final_1080p.mp4"
SPAN = 12.0
sentences = json.loads((D / "sentences.json").read_text(encoding="utf-8"))
model = WhisperModel("Systran/faster-whisper-medium", device="cpu", compute_type="int8", local_files_only=True)


def plain(w):
    return re.sub(r"[^a-z0-9]", "", w.lower())


def sec(x):
    h, m, s = map(int, x.split(":"))
    return h * 3600 + m * 60 + s


good = 0
lines = (D / f"{NAME}_chapters_youtube.txt").read_text(encoding="utf-8").splitlines()
for line in lines:
    stamp, title = line.split(" ", 1)
    t = sec(stamp)
    expected = next(s for s in sentences if s["start"] >= t - 0.95)["text"]
    want = [plain(w) for w in expected.split()][:4]
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        subprocess.run([FF, "-y", "-v", "error", "-ss", str(t), "-t", str(SPAN), "-i", str(FILM), "-vn", "-ac", "1", "-ar", "16000",
                        tmp.name], check=True)
        segments, _ = model.transcribe(tmp.name, language="en", word_timestamps=True, vad_filter=False,
                                       condition_on_previous_text=False)
        heard = [(plain(w.word), float(w.start), w.word.strip()) for s in segments for w in (s.words or [])]
    tokens = [h[0] for h in heard]
    best, score = None, 0.0
    for i in range(min(len(tokens), 8)):
        r = difflib.SequenceMatcher(None, tokens[i:i + len(want)], want).ratio()
        if r > score + 1e-9:
            best, score = i, r
    ok = best is not None and best <= 3 and score >= 0.5
    good += ok
    print(f"{'PASS' if ok else 'LOOK'} {stamp} leftover words {best} match {score:.2f} first word at +{heard[best][1] if best is not None and heard else -1:.1f}s | "
          f"{title} | expected: {expected[:50]} | heard: {' '.join(h[2] for h in heard[:12])}", flush=True)
print(f"passed {good} of {len(lines)}", flush=True)

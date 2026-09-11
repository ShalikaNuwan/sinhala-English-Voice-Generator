from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class AudioError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AudioInfo:
    duration_ms: int
    codec: str
    sample_rate: int
    channels: int


# Chunks are cut with this much of their neighbours on each side so a boundary never clips a word.
CHUNK_PAD_MS = 200

# The polish a produced narration has and a raw voice file does not: light compression so the level
# stops drifting mid-sentence, a mild de-esser, and a small top-end lift for clarity. Deliberately
# gentle - it is meant to be inaudible as an effect. Loudness is applied separately, as a static gain.
MASTER_CHAIN = "acompressor=threshold=-18dB:ratio=3:attack=5:release=120,deesser=i=0.4,treble=g=2.5:f=6500"


class AudioService:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise AudioError(f"Required program is missing: {command[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "Audio processing failed").strip()
            raise AudioError(detail[-1200:]) from exc

    def probe(self, path: Path) -> AudioInfo:
        result = self._run([
            self.ffprobe,
            "-v", "error",
            "-show_entries", "format=duration:stream=codec_name,sample_rate,channels,codec_type",
            "-of", "json",
            str(path),
        ])
        payload = json.loads(result.stdout)
        streams = [s for s in payload.get("streams", []) if s.get("codec_type") == "audio"]
        if not streams:
            raise AudioError("No readable audio stream was found")
        stream = streams[0]
        duration = float(payload.get("format", {}).get("duration") or 0)
        if duration <= 0:
            raise AudioError("Audio duration could not be determined")
        return AudioInfo(
            duration_ms=round(duration * 1000),
            codec=stream.get("codec_name", "unknown"),
            sample_rate=int(stream.get("sample_rate") or 0),
            channels=int(stream.get("channels") or 0),
        )

    def normalize(self, source: Path, destination: Path) -> AudioInfo:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-y", "-v", "error", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "24000", "-sample_fmt", "s16",
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", str(destination),
        ])
        return self.probe(destination)

    def silence_boundaries(self, source: Path) -> list[int]:
        result = subprocess.run([
            self.ffmpeg, "-hide_banner", "-nostats", "-i", str(source),
            "-af", "silencedetect=noise=-36dB:d=0.45", "-f", "null", "-",
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise AudioError((result.stderr or "Silence detection failed")[-1200:])
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", result.stderr)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", result.stderr)]
        return [round(((start + end) / 2) * 1000) for start, end in zip(starts, ends)]

    @staticmethod
    def plan_segments(
        duration_ms: int,
        boundaries_ms: list[int],
        minimum_ms: int = 20_000,
        target_ms: int = 45_000,
        maximum_ms: int = 60_000,
    ) -> list[tuple[int, int]]:
        if duration_ms <= maximum_ms:
            return [(0, duration_ms)]
        boundaries = sorted({b for b in boundaries_ms if 0 < b < duration_ms})
        segments: list[tuple[int, int]] = []
        start = 0
        while duration_ms - start > maximum_ms:
            candidates = [b for b in boundaries if start + minimum_ms <= b <= start + maximum_ms]
            end = min(candidates, key=lambda b: abs(b - (start + target_ms))) if candidates else start + target_ms
            segments.append((start, end))
            start = end
        if duration_ms - start < minimum_ms and segments:
            previous_start, _ = segments.pop()
            segments.append((previous_start, duration_ms))
        else:
            segments.append((start, duration_ms))
        return segments

    def extract(self, source: Path, start_ms: int, end_ms: int, destination: Path, pad_ms: int = CHUNK_PAD_MS) -> AudioInfo:
        """Cut [start_ms, end_ms] from the source as mono 24 kHz, padded into the neighbours except at the edges."""
        info = self.probe(source)
        begin = max(0, start_ms - (pad_ms if start_ms else 0))
        finish = min(info.duration_ms, end_ms + (pad_ms if end_ms < info.duration_ms else 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-y", "-v", "error",
            "-ss", f"{begin / 1000:.3f}", "-to", f"{finish / 1000:.3f}",
            "-i", str(source), "-ac", "1", "-ar", "24000", str(destination),
        ])
        return self.probe(destination)

    def _loudness(self, source: Path, start_ms: int, end_ms: int) -> dict:
        """Measure a cut so loudness can be applied as a static gain; single-pass loudnorm is a no-op under 3 s."""
        try:
            result = subprocess.run([
                self.ffmpeg, "-hide_banner", "-nostats",
                "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}", "-i", str(source),
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-",
            ], capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise AudioError(f"Required program is missing: {self.ffmpeg}") from exc
        match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr, re.S)
        if result.returncode != 0 or not match:
            raise AudioError((result.stderr or "Loudness measurement failed")[-1200:])
        return json.loads(match.group(0))

    def extract_levelled(self, source: Path, start_ms: int, end_ms: int, destination: Path) -> AudioInfo:
        """Cut a recording out of the source, exactly, align its loudness with the narration target, and
        fade 5 ms at each end so the splice never clicks.

        loudnorm's own gain application - single-pass, and even its "linear" second pass fed the
        measured stats - needs several seconds of lookahead before it reaches the target, so on a short
        evidence clip (1-3 s) it barely moves the level at all. The gain is instead computed by hand from
        the measurement and applied as a plain static `volume` filter, which has no such minimum duration.
        """
        measured = self._loudness(source, start_ms, end_ms)
        length_s = (end_ms - start_ms) / 1000
        filters = []
        input_i = float(measured.get("input_i", "-inf"))
        if input_i > -70:  # leave silence alone
            input_tp = float(measured["input_tp"])
            gain_db = min(-16 - input_i, -1.5 - input_tp)
            filters.append(f"volume={gain_db:.2f}dB")
        filters.append("afade=t=in:st=0:d=0.005")
        filters.append(f"afade=t=out:st={max(0.0, length_s - 0.005):.3f}:d=0.005")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-y", "-v", "error",
            "-ss", f"{start_ms / 1000:.3f}", "-to", f"{end_ms / 1000:.3f}",
            "-i", str(source), "-ac", "1", "-ar", "24000",
            "-af", ",".join(filters), str(destination),
        ])
        return self.probe(destination)

    def master(self, source: Path, destination: Path) -> AudioInfo:
        """Give a voiced segment the polish of a produced narration, then bring it to the loudness target.

        Tone first, level second: the compressor changes how loud the file is, so the gain has to be
        measured from the already-compressed audio rather than from the raw voice. Loudness is applied
        as a static gain for the same reason `extract_levelled` does it - loudnorm needs several seconds
        of lookahead and a narration segment is often shorter than that. Silence is left alone so
        make-up gain never lifts a quiet passage into audible hiss.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        toned = destination.parent / f".{destination.stem}_toned.wav"
        try:
            self._run([
                self.ffmpeg, "-y", "-v", "error", "-i", str(source),
                "-ac", "1", "-ar", "24000", "-af", MASTER_CHAIN, str(toned),
            ])
            measured = self._loudness(toned, 0, self.probe(toned).duration_ms)
            input_i = float(measured.get("input_i", "-inf"))
            if input_i <= -70:  # leave silence alone
                toned.replace(destination)
                return self.probe(destination)
            gain_db = min(-16 - input_i, -1.5 - float(measured["input_tp"]))
            self._run([
                self.ffmpeg, "-y", "-v", "error", "-i", str(toned),
                "-ac", "1", "-ar", "24000", "-af", f"volume={gain_db:.2f}dB", str(destination),
            ])
        finally:
            toned.unlink(missing_ok=True)
        return self.probe(destination)

    @staticmethod
    def chunk_lead_ms(start_ms: int) -> int:
        """How far before its planned start a chunk file from `split` actually begins."""
        return CHUNK_PAD_MS if start_ms else 0

    def split(self, source: Path, destination_dir: Path) -> list[tuple[int, int, Path]]:
        info = self.probe(source)
        plan = self.plan_segments(info.duration_ms, self.silence_boundaries(source))
        destination_dir.mkdir(parents=True, exist_ok=True)
        created: list[tuple[int, int, Path]] = []
        for index, (start_ms, end_ms) in enumerate(plan, start=1):
            output = destination_dir / f"{index:04d}_source.wav"
            self.extract(source, start_ms, end_ms, output)
            created.append((start_ms, end_ms, output))
        return created

    def assemble(
        self,
        segment_paths: list[Path],
        wav_path: Path,
        mp3_path: Path,
        gaps_ms: list[int] | None = None,
    ) -> None:
        """Join the segments in order, leaving `gaps_ms[i]` of silence after segment i. A gap of 0 joins directly."""
        if not segment_paths:
            raise AudioError("No generated segments are available for assembly")
        gaps = list(gaps_ms) if gaps_ms is not None else []
        if gaps_ms is not None and len(gaps) != len(segment_paths) - 1:
            raise AudioError("Assembly needs exactly one gap between each pair of segments")
        if any(gap < 0 for gap in gaps):
            raise AudioError("Segment gaps must not be negative")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        inputs: list[str] = []
        labels: list[str] = []
        for index, path in enumerate(segment_paths):
            if index and gaps and gaps[index - 1] > 0:
                inputs.extend([
                    "-f", "lavfi", "-t", f"{gaps[index - 1] / 1000:.3f}",
                    "-i", "anullsrc=r=24000:cl=mono",
                ])
                labels.append(f"[{len(labels)}:a]")
            inputs.extend(["-i", str(path)])
            labels.append(f"[{len(labels)}:a]")
        self._run([
            self.ffmpeg, "-y", "-v", "error", *inputs,
            "-filter_complex", f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1,loudnorm=I=-16:TP=-1.5:LRA=11[out]",
            "-map", "[out]", "-ac", "1", "-ar", "24000", str(wav_path),
        ])
        self._run([
            self.ffmpeg, "-y", "-v", "error", "-i", str(wav_path),
            "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path),
        ])

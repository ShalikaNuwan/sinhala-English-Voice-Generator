"""Voicing through ElevenLabs, as an alternative to the OpenAI speech endpoint.

The two providers direct a voice very differently. OpenAI takes the whole natural-language
brief that `direction.build_instructions` writes. ElevenLabs has no equivalent field: delivery
is steered by numeric voice settings, and continuity by telling it what was said just before.
So the brief cannot be forwarded, and the caller should not pretend otherwise - what carries
over is the stability and similarity dials plus `previous_text`.

Audio comes back as raw 24 kHz mono PCM, which is exactly what the rest of the pipeline works
in, so it is wrapped in a WAV header rather than transcoded.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Callable

ENDPOINT = "https://api.elevenlabs.io/v1/text-to-speech"
# The rest of the pipeline is 24 kHz mono; asking for it directly avoids a lossy transcode.
OUTPUT_FORMAT = "pcm_24000"
SAMPLE_RATE = 24_000
TIMEOUT_S = 180

Transport = Callable[[str, dict, bytes], bytes]


class ElevenLabsError(RuntimeError):
    """Anything that stopped a segment being voiced."""


def _post(url: str, headers: dict, body: bytes) -> bytes:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.read()
    except urllib.error.HTTPError as exc:  # the body carries the reason, which the status alone does not
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise ElevenLabsError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ElevenLabsError(f"Could not reach ElevenLabs: {exc.reason}") from exc


def synthesize(
    text: str,
    output_path: Path,
    *,
    api_key: str | None,
    voice: str,
    model: str,
    stability: float,
    similarity: float,
    previous_text: str | None = None,
    transport: Transport | None = None,
) -> None:
    """Voice one segment and write it as a 24 kHz mono WAV."""
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is required to voice with ElevenLabs")

    payload: dict = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity,
            "use_speaker_boost": True,
        },
    }
    if previous_text:
        payload["previous_text"] = previous_text[-600:]

    url = f"{ENDPOINT}/{voice}?output_format={OUTPUT_FORMAT}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    send = transport or _post
    try:
        audio = send(url, headers, json.dumps(payload).encode("utf-8"))
    except ElevenLabsError:
        raise
    except Exception as exc:  # noqa: BLE001 - an injected transport may raise anything
        raise ElevenLabsError(str(exc)) from exc

    if not audio:
        raise ElevenLabsError("ElevenLabs returned no audio")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(audio)

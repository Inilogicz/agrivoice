"""
Spoken answers: text → speech (YarnGPT2 on the N-ATLaS GPU) → MP3, saved for replay.

A recording is generated once per message, language, text and speed, then
served from SPEECH_CACHE_DIR, so replays are instant and don't need the GPU.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import SpeechUnavailableError
from app.core.logging import get_logger
from app.models.base import BaseSpeechSynthesizer, SupportedLanguage

logger = get_logger(__name__)


@dataclass(frozen=True)
class SpokenAudio:
    path: Path
    cached: bool
    generation_ms: int | None = None


def encode_mp3(wav: bytes, speed: float) -> bytes:
    """Slow down/speed up without changing pitch, and encode as mono MP3."""
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-af", f"atempo={speed:.3f}",
        "-ac", "1", "-codec:a", "libmp3lame", "-b:a", "64k",
        "-f", "mp3", "pipe:1",
    ]
    result = subprocess.run(cmd, input=wav, capture_output=True, timeout=120)
    if result.returncode != 0:
        logger.error("ffmpeg could not encode speech", error=result.stderr.decode(errors="replace")[-300:])
        raise SpeechUnavailableError()
    return result.stdout


class SpeechService:
    def __init__(self, synthesizer: BaseSpeechSynthesizer) -> None:
        self._synth = synthesizer

    @property
    def model_id(self) -> str:
        return self._synth.model_id

    def _path(self, message_id: uuid.UUID, language: str, text: str, speed: float) -> Path:
        key = hashlib.sha256(f"{self._synth.model_id}|{language}|{speed:.3f}|{text}".encode()).hexdigest()[:16]
        return Path(get_settings().SPEECH_CACHE_DIR) / f"{message_id}_{language}_{key}.mp3"

    def cached(self, message_id: uuid.UUID, language: str, text: str) -> Path | None:
        path = self._path(message_id, language, text, get_settings().TTS_SPEED)
        return path if path.exists() else None

    async def audio_for(
        self, message_id: uuid.UUID, language: SupportedLanguage, text: str
    ) -> SpokenAudio:
        speed = get_settings().TTS_SPEED
        path = self._path(message_id, language, text, speed)
        if path.exists():
            return SpokenAudio(path=path, cached=True)

        t0 = time.time()
        wav = await self._synth.synthesize(text, language)
        mp3 = await asyncio.to_thread(encode_mp3, wav, speed)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so a concurrent request never serves half a file
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".part")
        with os.fdopen(fd, "wb") as f:
            f.write(mp3)
        os.replace(tmp, path)
        return SpokenAudio(path=path, cached=False, generation_ms=round((time.time() - t0) * 1000))

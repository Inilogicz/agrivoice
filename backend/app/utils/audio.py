"""
Audio utility: validation, format conversion, and preprocessing.

Whisper expects:
  - Sample rate: 16 kHz
  - Channels: mono
  - Format: float32 numpy array

We support these upload formats:
  - wav, webm, mp3, m4a, ogg
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


from app.core.config import get_settings
from app.core.exceptions import AudioProcessingError, AudioValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS = frozenset({".wav", ".webm", ".mp3", ".m4a", ".ogg"})


def validate_audio_file(file_path: str, content_type: str, file_size: int) -> None:
    """
    Validate audio file before processing.

    Raises:
        AudioValidationError: If the file fails validation.
    """
    settings = get_settings()

    # Size check
    if file_size > settings.max_audio_size_bytes:
        raise AudioValidationError(
            f"Audio file too large. Maximum size is {settings.MAX_AUDIO_SIZE_MB} MB."
        )

    # MIME type check; browsers add codec parameters ("audio/webm;codecs=opus")
    base_type = content_type.split(";", 1)[0].strip().lower()
    if base_type not in settings.ALLOWED_AUDIO_TYPES:
        raise AudioValidationError(
            f"Unsupported audio format: {content_type}. "
            f"Allowed: {', '.join(settings.ALLOWED_AUDIO_TYPES)}"
        )


def convert_to_wav_16k(input_path: str, output_path: str) -> str:
    """
    Convert audio to 16 kHz mono 16-bit WAV with ffmpeg.

    ffmpeg decodes every upload format we accept, including the WebM/Opus
    and MP4/AAC that browsers' MediaRecorder produces (libsndfile can't).

    Raises:
        AudioValidationError: If the file cannot be decoded.
    """
    settings = get_settings()
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", input_path,
        "-ac", "1", "-ar", str(settings.AUDIO_SAMPLE_RATE), "-sample_fmt", "s16",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=60)
    except FileNotFoundError as exc:
        logger.error("ffmpeg is not installed")
        raise AudioProcessingError("Audio conversion is unavailable on this server.") from exc
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", b"") or b""
        logger.warning("ffmpeg could not decode upload", error=stderr.decode(errors="replace")[-500:])
        raise AudioValidationError(
            "Could not read the audio file. Please record again or use WAV, WebM, MP3, M4A or OGG."
        ) from exc
    return output_path


def create_temp_wav(content: bytes, suffix: str = ".wav") -> str:
    """
    Write raw audio bytes to a temporary file and convert to 16 kHz WAV.

    Returns the path to the converted WAV file.
    The caller is responsible for deleting this file after use.
    """
    settings = get_settings()
    os.makedirs(settings.TEMP_AUDIO_DIR, exist_ok=True)

    # Write raw upload to temp file
    with tempfile.NamedTemporaryFile(
        dir=settings.TEMP_AUDIO_DIR,
        suffix=suffix,
        delete=False,
    ) as tmp:
        tmp.write(content)
        raw_path = tmp.name

    # Convert to 16 kHz WAV
    wav_path = raw_path.rsplit(".", 1)[0] + "_16k.wav"
    try:
        convert_to_wav_16k(raw_path, wav_path)
    finally:
        # Always clean up the original upload
        if os.path.exists(raw_path):
            os.unlink(raw_path)

    return wav_path

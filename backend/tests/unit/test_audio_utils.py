"""Unit tests for audio utilities."""
from __future__ import annotations

import pytest

from app.core.exceptions import AudioValidationError
from app.utils.audio import validate_audio_file


def test_validate_audio_file_too_large():
    with pytest.raises(AudioValidationError, match="too large"):
        validate_audio_file(
            file_path="audio.wav",
            content_type="audio/wav",
            file_size=30 * 1024 * 1024,  # 30 MB > 25 MB limit
        )


def test_validate_audio_file_unsupported_type():
    with pytest.raises(AudioValidationError, match="Unsupported audio format"):
        validate_audio_file(
            file_path="audio.mp4",
            content_type="video/mp4",
            file_size=1024,
        )


def test_validate_audio_file_valid_wav():
    # Should not raise
    validate_audio_file(
        file_path="audio.wav",
        content_type="audio/wav",
        file_size=1024 * 1024,
    )


def test_validate_audio_file_valid_webm():
    validate_audio_file(
        file_path="audio.webm",
        content_type="audio/webm",
        file_size=2 * 1024 * 1024,
    )


def test_validate_audio_file_accepts_browser_codec_parameters():
    # MediaRecorder in Chrome/Firefox reports codecs alongside the type
    validate_audio_file(
        file_path="recording.webm",
        content_type="audio/webm;codecs=opus",
        file_size=50_000,
    )


@pytest.mark.parametrize("content_type", ["audio/mp4", "audio/m4a", "audio/x-m4a", "audio/aac"])
def test_validate_audio_file_accepts_mobile_aac_labels(content_type):
    # iOS/Android (expo-audio HIGH_QUALITY) record AAC in .m4a; apps label it differently
    validate_audio_file(file_path="question.m4a", content_type=content_type, file_size=80_000)

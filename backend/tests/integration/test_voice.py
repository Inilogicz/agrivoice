"""Integration tests for the voice endpoint and language selection flow."""
from __future__ import annotations

import io
import math
import struct
import wave

import pytest
from httpx import AsyncClient

from app.models.base import BaseLanguageDetector
from app.schemas.voice import LanguageDetectionResult


def _wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate)))
            for i in range(int(seconds * sample_rate))
        )
        w.writeframes(frames)
    return buf.getvalue()


class _UndetectableLanguageDetector(BaseLanguageDetector):
    """Simulates audio that isn't confidently one of our languages."""

    async def detect(self, audio_path: str) -> LanguageDetectionResult:
        return LanguageDetectionResult(language=None, confidence=0.31, detected_label="fra")


def _audio_file() -> dict:
    return {"audio": ("question.wav", _wav_bytes(), "audio/wav")}


@pytest.mark.asyncio
async def test_list_languages(client: AsyncClient):
    response = await client.get("/api/v1/languages")
    assert response.status_code == 200
    codes = [lang["code"] for lang in response.json()["languages"]]
    assert codes == ["yo", "ha", "ig", "en-ng"]


@pytest.mark.asyncio
async def test_voice_ask_auto_detects_language(client: AsyncClient):
    response = await client.post("/api/v1/voice/ask", files=_audio_file())
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en-ng"
    assert data["language_name"] == "Nigerian English"
    assert data["language_source"] == "detected"
    assert data["conversation_id"]


@pytest.mark.asyncio
async def test_voice_ask_uses_user_selected_language(client: AsyncClient):
    response = await client.post(
        "/api/v1/voice/ask", files=_audio_file(), data={"language": "yo"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "yo"
    assert data["language_name"] == "Yoruba"
    assert data["language_source"] == "user_selected"
    assert data["asr_model"] == "MOCK/ASR-yo"


@pytest.mark.asyncio
async def test_voice_ask_rejects_unsupported_language(client: AsyncClient):
    response = await client.post(
        "/api/v1/voice/ask", files=_audio_file(), data={"language": "fr"}
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error"] == "UNSUPPORTED_LANGUAGE"


@pytest.mark.asyncio
async def test_voice_ask_asks_user_to_choose_when_not_detected(
    client: AsyncClient, mock_model_manager
):
    mock_model_manager._language_detector = _UndetectableLanguageDetector()

    response = await client.post("/api/v1/voice/ask", files=_audio_file())
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["error"] == "LANGUAGE_NOT_DETECTED"
    assert [lang["code"] for lang in detail["available_languages"]] == [
        "yo", "ha", "ig", "en-ng"
    ]

    # Resending with the user's choice succeeds
    retry = await client.post(
        "/api/v1/voice/ask", files=_audio_file(), data={"language": "ha"}
    )
    assert retry.status_code == 200
    assert retry.json()["language_source"] == "user_selected"
    assert retry.json()["language"] == "ha"


@pytest.mark.asyncio
async def test_voice_answer_includes_message_id_for_feedback(client: AsyncClient):
    data = (await client.post("/api/v1/voice/ask", files=_audio_file())).json()
    assert data["message_id"]

    feedback = await client.post(
        "/api/v1/feedback", json={"message_id": data["message_id"], "rating": "helpful"}
    )
    assert feedback.status_code == 201


@pytest.mark.asyncio
async def test_errors_use_detail_envelope(client: AsyncClient):
    resp = await client.post(
        "/api/v1/voice/ask", files={"audio": ("clip.mp4", b"xx", "video/mp4")}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == "AUDIO_VALIDATION_ERROR"

"""Spoken answers (text-to-speech) and the features endpoint."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.models.base import BaseSpeechSynthesizer

from tests.integration.test_admin import AUTH, TOKEN
from tests.integration.test_voice import _audio_file


class CountingSpeech(BaseSpeechSynthesizer):
    def __init__(self, wav: bytes) -> None:
        self.calls: list[tuple[str, str]] = []
        self._wav = wav

    @property
    def model_id(self) -> str:
        return "test/tts"

    async def synthesize(self, text: str, language: str) -> bytes:
        self.calls.append((text, language))
        return self._wav


class OfflineSpeech(BaseSpeechSynthesizer):
    @property
    def model_id(self) -> str:
        return "test/tts"

    async def synthesize(self, text: str, language: str) -> bytes:
        from app.core.exceptions import SpeechUnavailableError

        raise SpeechUnavailableError()


def _set(monkeypatch, app, key, value):
    """Dashboard-editable settings are re-applied from the store on each request."""
    monkeypatch.setattr(get_settings(), key, value)
    monkeypatch.setitem(app.state.runtime_settings._defaults, key, value)


@pytest.fixture
def speech_on(monkeypatch, tmp_path, test_app):
    _set(monkeypatch, test_app, "TTS_ENABLED", True)
    _set(monkeypatch, test_app, "TTS_SPEED", 0.85)
    monkeypatch.setattr(get_settings(), "SPEECH_CACHE_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture
async def counting(mock_model_manager):
    wav = await mock_model_manager.get_speech_synthesizer().synthesize("x", "yo")
    synth = CountingSpeech(wav)
    mock_model_manager._speech = synth
    return synth


async def _answer(client: AsyncClient, language: str = "yo") -> dict:
    return (await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": language})).json()


@pytest.mark.asyncio
async def test_features_reflect_setting(client: AsyncClient, monkeypatch, test_app):
    _set(monkeypatch, test_app, "TTS_ENABLED", False)
    assert (await client.get("/api/v1/features")).json() == {"speech": False, "translation": True}
    _set(monkeypatch, test_app, "TTS_ENABLED", True)
    assert (await client.get("/api/v1/features")).json()["speech"] is True


@pytest.mark.asyncio
async def test_answer_audio_is_mp3_and_reused(client: AsyncClient, speech_on, counting):
    answer = await _answer(client, "yo")
    url = f"/api/v1/messages/{answer['message_id']}/audio"

    first = await client.get(url)
    assert first.status_code == 200
    assert first.headers["content-type"] == "audio/mpeg"
    assert first.content[:3] == b"ID3" or first.content[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")
    assert counting.calls == [(answer["response"], "yo")]

    again = await client.get(url)
    assert again.status_code == 200 and again.content == first.content
    assert len(counting.calls) == 1  # served from the saved file
    assert len(list(speech_on.glob("*.mp3"))) == 1


@pytest.mark.asyncio
async def test_translation_audio_needs_translation_first(client: AsyncClient, speech_on, counting):
    answer = await _answer(client, "ha")
    url = f"/api/v1/messages/{answer['message_id']}/audio?language=en-ng"

    missing = await client.get(url)
    assert missing.status_code == 404
    assert missing.json()["detail"]["error"] == "TRANSLATION_NOT_FOUND"

    translated = (await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "en-ng"}
    )).json()
    ok = await client.get(url)
    assert ok.status_code == 200
    assert counting.calls[-1] == (translated["text"], "en-ng")


@pytest.mark.asyncio
async def test_speech_disabled_and_errors(client: AsyncClient, monkeypatch, speech_on, test_app):
    answer = await _answer(client, "ig")
    url = f"/api/v1/messages/{answer['message_id']}/audio"

    bad = await client.get(f"{url}?language=fr")
    assert bad.status_code == 422 and bad.json()["detail"]["error"] == "UNSUPPORTED_LANGUAGE"
    missing = await client.get("/api/v1/messages/00000000-0000-0000-0000-000000000000/audio")
    assert missing.status_code == 404 and missing.json()["detail"]["error"] == "MESSAGE_NOT_FOUND"

    _set(monkeypatch, test_app, "TTS_ENABLED", False)
    off = await client.get(url)
    assert off.status_code == 503 and off.json()["detail"]["error"] == "SPEECH_DISABLED"


@pytest.mark.asyncio
async def test_speech_offline_is_logged(client: AsyncClient, monkeypatch, speech_on, mock_model_manager):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    answer = await _answer(client, "yo")
    mock_model_manager._speech = OfflineSpeech()

    resp = await client.get(f"/api/v1/messages/{answer['message_id']}/audio")
    assert resp.status_code == 503 and resp.json()["detail"]["error"] == "SPEECH_UNAVAILABLE"
    assert not list(speech_on.glob("*.mp3"))  # nothing cached; can retry

    activity = (await client.get("/api/v1/admin/interactions?input_type=speech", headers=AUTH)).json()
    assert activity["items"][0]["error_type"] == "SPEECH_UNAVAILABLE"


@pytest.mark.asyncio
async def test_speed_setting_changes_recording(client: AsyncClient, monkeypatch, speech_on, counting, test_app):
    answer = await _answer(client, "yo")
    url = f"/api/v1/messages/{answer['message_id']}/audio"
    await client.get(url)
    _set(monkeypatch, test_app, "TTS_SPEED", 1.0)
    await client.get(url)
    assert len(counting.calls) == 2 and len(list(speech_on.glob("*.mp3"))) == 2

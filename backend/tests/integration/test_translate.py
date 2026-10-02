"""Translating answers between the four supported languages."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.models.natlas_remote import RemoteNATLaSAdapter

from tests.integration.test_admin import AUTH, TOKEN, _connect, _Job
from tests.integration.test_voice import _audio_file


async def _answer(client: AsyncClient, language: str = "ha") -> dict:
    return (await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": language})).json()


@pytest.mark.asyncio
async def test_translate_answer_and_reuse_saved_copy(client: AsyncClient):
    answer = await _answer(client, "ha")
    first = await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "en-ng"}
    )
    assert first.status_code == 200
    data = first.json()
    assert data["source_language"] == "ha" and data["target_language"] == "en-ng"
    assert data["target_language_name"] == "Nigerian English"
    assert data["text"] and data["cached"] is False

    again = (await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "en-ng"}
    )).json()
    assert again["cached"] is True and again["text"] == data["text"]

    convo = (await client.get(f"/api/v1/conversations/{answer['conversation_id']}")).json()
    assistant = next(m for m in convo["messages"] if m["role"] == "assistant")
    assert [t["language"] for t in assistant["translations"]] == ["en-ng"]


@pytest.mark.asyncio
async def test_translate_rejects_same_and_unknown_language(client: AsyncClient):
    answer = await _answer(client, "yo")
    same = await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "yo"}
    )
    assert same.status_code == 422 and same.json()["detail"]["error"] == "ALREADY_IN_LANGUAGE"

    french = await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "fr"}
    )
    assert french.status_code == 422 and french.json()["detail"]["error"] == "UNSUPPORTED_LANGUAGE"

    missing = await client.post(
        "/api/v1/translate",
        json={"message_id": "00000000-0000-0000-0000-000000000000", "target_language": "ig"},
    )
    assert missing.status_code == 404 and missing.json()["detail"]["error"] == "MESSAGE_NOT_FOUND"


@pytest.mark.asyncio
async def test_translation_prompt_and_length(client: AsyncClient, mock_model_manager, monkeypatch):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(get_settings(), "LLM_PROVIDER", "remote")
    monkeypatch.setattr(get_settings(), "DEV_USE_MOCK_MODELS", False)
    answer = await _answer(client, "en-ng")

    registry = mock_model_manager.llm_endpoint_registry
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
    mock_model_manager._llm_adapter = adapter
    await client.put("/api/v1/admin/llm-endpoint", headers=AUTH, json={"endpoint": "https://abc.gradio.live"})

    calls = []

    class Client:
        def submit(self, messages, max_new_tokens, api_name):
            calls.append((messages, max_new_tokens))
            return _Job(reply="  Lo NPK 15-15-15.  ")

    adapter._client, adapter._client_endpoint = Client(), "https://abc.gradio.live"
    resp = (await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "yo"}
    )).json()
    assert resp["text"] == "Lo NPK 15-15-15."

    (messages, max_tokens), = calls
    assert "into Yoruba" in messages[0]["content"] and "tone marks" in messages[0]["content"]
    assert messages[1]["content"] == answer["response"]
    assert max_tokens >= 600

    activity = (await client.get("/api/v1/admin/interactions?input_type=translation", headers=AUTH)).json()
    item = activity["items"][0]
    assert item["language"] == "yo" and item["prompt"] == answer["response"]
    assert item["response"] == "Lo NPK 15-15-15." and item["success"] is True


@pytest.mark.asyncio
async def test_translate_when_natlas_offline(client: AsyncClient, mock_model_manager, monkeypatch):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    answer = await _answer(client, "ig")
    registry = mock_model_manager.llm_endpoint_registry
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
    mock_model_manager._llm_adapter = adapter  # no endpoint configured

    resp = await client.post(
        "/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "en-ng"}
    )
    assert resp.status_code == 503 and resp.json()["detail"]["error"] == "LLM_UNAVAILABLE"
    # Nothing was cached, so it can be retried later
    convo = (await client.get(f"/api/v1/conversations/{answer['conversation_id']}")).json()
    assert all(not m["translations"] for m in convo["messages"])


@pytest.mark.asyncio
async def test_nigerian_language_pairs_go_through_english(client: AsyncClient, mock_model_manager, monkeypatch):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(get_settings(), "LLM_PROVIDER", "remote")
    monkeypatch.setattr(get_settings(), "DEV_USE_MOCK_MODELS", False)
    answer = await _answer(client, "ha")

    registry = mock_model_manager.llm_endpoint_registry
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
    mock_model_manager._llm_adapter = adapter
    await client.put("/api/v1/admin/llm-endpoint", headers=AUTH, json={"endpoint": "https://abc.gradio.live"})

    prompts = []

    class Client:
        def submit(self, messages, max_new_tokens, api_name):
            prompts.append(messages[0]["content"].splitlines()[1])
            reply = "English version" if "into Nigerian English" in messages[0]["content"] else "Yoruba version"
            return _Job(reply=reply)

    adapter._client, adapter._client_endpoint = Client(), "https://abc.gradio.live"
    yo = (await client.post("/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "yo"})).json()
    assert yo["text"] == "Yoruba version" and yo["source_language"] == "ha"
    assert prompts == [
        "Translate the user's message from Hausa into Nigerian English.",
        "Translate the user's message from Nigerian English into Yoruba.",
    ]

    # The English step was saved: English is now instant, and Igbo needs one call
    en = (await client.post("/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "en-ng"})).json()
    assert en["cached"] is True and en["text"] == "English version"
    await client.post("/api/v1/translate", json={"message_id": answer["message_id"], "target_language": "ig"})
    assert len(prompts) == 3 and "Nigerian English into Igbo" in prompts[-1]

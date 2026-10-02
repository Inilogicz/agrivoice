"""Integration tests for the admin N-ATLaS endpoint manager."""
from __future__ import annotations

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.models.natlas_remote import RemoteNATLaSAdapter

from tests.integration.test_voice import _audio_file

TOKEN = "test-admin-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
API = "/api/v1/admin/llm-endpoint"


class _Job:
    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def result(self, timeout=None):
        if self.error:
            raise self.error
        return self.reply


class _Client:
    def __init__(self, job):
        self.job = job

    def submit(self, *args, api_name):
        return self.job


@pytest.fixture
def admin_enabled(monkeypatch):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    monkeypatch.setattr(get_settings(), "LLM_PROVIDER", "remote")
    monkeypatch.setattr(get_settings(), "DEV_USE_MOCK_MODELS", False)


@pytest.fixture
def remote_llm(mock_model_manager):
    registry = mock_model_manager.llm_endpoint_registry
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
    mock_model_manager._llm_adapter = adapter
    return adapter


def _connect(adapter: RemoteNATLaSAdapter, endpoint: str, job: _Job) -> None:
    adapter._client, adapter._client_endpoint = _Client(job), endpoint


@pytest.mark.asyncio
async def test_admin_disabled_without_token(client: AsyncClient):
    assert (await client.get(API)).status_code == 404
    assert (await client.get("/admin")).status_code == 404


@pytest.mark.asyncio
async def test_admin_rejects_wrong_token(client: AsyncClient, admin_enabled):
    assert (await client.get(API)).status_code == 401
    bad = await client.get(API, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401
    assert (await client.get("/admin")).status_code == 200  # page itself is public


@pytest.mark.asyncio
async def test_save_and_reset_endpoint(client: AsyncClient, admin_enabled):
    start = (await client.get(API, headers=AUTH)).json()
    assert start["source"] == "none" and start["provider"] == "remote"

    saved = await client.put(API, headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    assert saved.status_code == 200
    assert saved.json()["endpoint"] == "https://abc.gradio.live"
    assert saved.json()["source"] == "dashboard"
    assert (await client.get(API, headers=AUTH)).json()["endpoint"] == "https://abc.gradio.live"

    switched = await client.put(API, headers=AUTH, json={"endpoint": "someone/natlas"})
    assert switched.json()["endpoint"] == "someone/natlas"

    reset = await client.delete(API, headers=AUTH)
    assert reset.json()["source"] == "none"


@pytest.mark.asyncio
async def test_rejects_invalid_endpoint(client: AsyncClient, admin_enabled):
    resp = await client.put(API, headers=AUTH, json={"endpoint": "not a url"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_connection_test(client: AsyncClient, admin_enabled, remote_llm):
    _connect(remote_llm, "https://abc.gradio.live", _Job(reply="OK"))
    ok = await client.post(f"{API}/test", headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    assert ok.json()["ok"] is True and ok.json()["reply"] == "OK"

    _connect(remote_llm, "https://abc.gradio.live", _Job(error=RuntimeError("session ended")))
    down = await client.post(f"{API}/test", headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    assert down.json()["ok"] is False
    assert "session ended" not in down.text  # internals stay in the logs


@pytest.mark.asyncio
async def test_voice_uses_saved_endpoint(client: AsyncClient, admin_enabled, remote_llm):
    await client.put(API, headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    _connect(remote_llm, "https://abc.gradio.live", _Job(reply="Plant early in the rains."))

    resp = await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "en-ng"})
    data = resp.json()
    assert resp.status_code == 200
    assert data["llm_available"] is True
    assert data["response"] == "Plant early in the rains."
    assert data["llm_model"] == "NCAIR1/N-ATLaS"


@pytest.mark.asyncio
async def test_voice_still_transcribes_when_natlas_offline(
    client: AsyncClient, admin_enabled, remote_llm
):
    await client.put(API, headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    _connect(remote_llm, "https://abc.gradio.live", _Job(error=RuntimeError("Kaggle stopped")))

    resp = await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "yo"})
    data = resp.json()
    assert resp.status_code == 200
    assert data["llm_available"] is False
    assert data["transcript"]
    assert "temporarily unavailable" in data["response"]


@pytest.mark.asyncio
async def test_chat_returns_503_when_natlas_offline(
    client: AsyncClient, admin_enabled, remote_llm
):
    resp = await client.post("/api/v1/chat", json={"message": "Hello", "language": "en-ng"})
    assert resp.status_code == 503
    assert resp.json()["detail"]["error"] == "LLM_UNAVAILABLE"


# ── Dashboard: activity, overview, settings ─────────────────────────────────


@pytest.fixture
def restore_tunables(monkeypatch):
    """Settings changed through the dashboard mutate the shared Settings object."""
    from app.services.runtime_settings import EDITABLE_SETTINGS

    for key in EDITABLE_SETTINGS:
        monkeypatch.setattr(get_settings(), key, getattr(get_settings(), key))


@pytest.mark.asyncio
async def test_activity_records_prompt_response_and_timings(client: AsyncClient, admin_enabled):
    await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "yo"})
    await client.post("/api/v1/chat", json={"message": "When do I plant maize?", "language": "en-ng"})

    page = (await client.get(f"{API.rsplit('/', 1)[0]}/interactions", headers=AUTH)).json()
    assert page["total"] == 2
    text, voice = page["items"]  # newest first
    assert text["input_type"] == "text" and text["prompt"] == "When do I plant maize?"
    assert text["response"] and text["success"] is True
    assert voice["input_type"] == "voice" and voice["language"] == "yo"
    assert voice["prompt"].startswith("[MOCK]") and voice["response"]
    assert voice["language_source"] == "user_selected"
    assert voice["asr_ms"] is not None and voice["llm_ms"] is not None


@pytest.mark.asyncio
async def test_activity_records_failures(client: AsyncClient, admin_enabled, mock_model_manager):
    from tests.integration.test_voice import _UndetectableLanguageDetector

    await client.post("/api/v1/voice/ask", files={"audio": ("x.mp4", b"xx", "video/mp4")})
    mock_model_manager._language_detector = _UndetectableLanguageDetector()
    await client.post("/api/v1/voice/ask", files=_audio_file())

    admin = API.rsplit("/", 1)[0]
    errors = (await client.get(f"{admin}/interactions?status=error", headers=AUTH)).json()
    assert {i["error_type"] for i in errors["items"]} == {
        "AUDIO_VALIDATION_ERROR", "LANGUAGE_NOT_DETECTED"
    }
    assert all(i["success"] is False and i["conversation_id"] is None for i in errors["items"])
    ok = (await client.get(f"{admin}/interactions?status=ok", headers=AUTH)).json()
    assert ok["total"] == 0


@pytest.mark.asyncio
async def test_activity_records_natlas_offline(client: AsyncClient, admin_enabled, remote_llm):
    await client.put(API, headers=AUTH, json={"endpoint": "https://abc.gradio.live"})
    _connect(remote_llm, "https://abc.gradio.live", _Job(error=RuntimeError("Kaggle stopped")))
    await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "ha"})

    item = (await client.get(f"{API.rsplit('/', 1)[0]}/interactions", headers=AUTH)).json()["items"][0]
    assert item["llm_available"] is False and item["error_type"] == "LLM_UNAVAILABLE"
    assert item["prompt"] and item["response"] is None

    overview = (await client.get(f"{API.rsplit('/', 1)[0]}/overview", headers=AUTH)).json()
    assert overview["llm"]["last_error"].startswith("RuntimeError")
    assert overview["stats_24h"]["llm_offline"] == 1


@pytest.mark.asyncio
async def test_overview(client: AsyncClient, admin_enabled):
    await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "ig"})
    data = (await client.get(f"{API.rsplit('/', 1)[0]}/overview", headers=AUTH)).json()
    assert data["server"]["status"] == "ok" and data["server"]["uptime_seconds"] >= 0
    assert data["stats_24h"]["requests"] == 1
    assert data["stats_24h"]["by_language"] == {"ig": 1}
    assert (await client.get(f"{API.rsplit('/', 1)[0]}/overview")).status_code == 401


@pytest.mark.asyncio
async def test_settings_update_validate_and_reset(
    client: AsyncClient, admin_enabled, restore_tunables
):
    settings_api = f"{API.rsplit('/', 1)[0]}/settings"
    listed = {s["key"]: s for s in (await client.get(settings_api, headers=AUTH)).json()["settings"]}
    assert "LLM_MAX_NEW_TOKENS" in listed and listed["LLM_MAX_NEW_TOKENS"]["overridden"] is False

    bad = await client.put(settings_api, headers=AUTH, json={"changes": {"LLM_MAX_NEW_TOKENS": 5000}})
    assert bad.status_code == 422
    unknown = await client.put(settings_api, headers=AUTH, json={"changes": {"ADMIN_TOKEN": "x"}})
    assert unknown.status_code == 422

    ok = await client.put(settings_api, headers=AUTH, json={"changes": {"LLM_MAX_NEW_TOKENS": 200}})
    assert ok.status_code == 200
    assert get_settings().LLM_MAX_NEW_TOKENS == 200  # applied live

    await client.delete(f"{settings_api}/LLM_MAX_NEW_TOKENS", headers=AUTH)
    assert get_settings().LLM_MAX_NEW_TOKENS == listed["LLM_MAX_NEW_TOKENS"]["default"]


@pytest.mark.asyncio
async def test_cors_origins_change_live(client: AsyncClient, admin_enabled, restore_tunables):
    def preflight(origin):
        return client.options(
            "/api/v1/chat",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
        )

    assert (await preflight("https://agrivoice.vercel.app")).status_code == 400
    await client.put(
        f"{API.rsplit('/', 1)[0]}/settings",
        headers=AUTH,
        json={"changes": {"CORS_ORIGINS": "http://localhost:3000, https://agrivoice.vercel.app/"}},
    )
    allowed = await preflight("https://agrivoice.vercel.app")
    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://agrivoice.vercel.app"

    bad = await client.put(
        f"{API.rsplit('/', 1)[0]}/settings",
        headers=AUTH,
        json={"changes": {"CORS_ORIGINS": "agrivoice.vercel.app/path"}},
    )
    assert bad.status_code == 422

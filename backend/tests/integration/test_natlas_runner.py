"""Start / stop / schedule N-ATLaS on Kaggle from the admin dashboard."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.core.config import get_settings
from app.models.natlas_remote import RemoteNATLaSAdapter
from app.services.natlas_runner import NatlasRunner, Schedule

from tests.integration.test_admin import AUTH, TOKEN, _connect, _Job
from tests.integration.test_voice import _audio_file

ADMIN = "/api/v1/admin"


class FakeKaggle:
    def __init__(self) -> None:
        self.pushes: list[dict] = []
        self.kaggle_status = "running"

    def push(self, folder: str) -> None:
        self.pushes.append(
            {
                "notebook": Path(folder, "notebook.ipynb").read_text(),
                "metadata": json.loads(Path(folder, "kernel-metadata.json").read_text()),
            }
        )

    def status(self, notebook_id: str) -> str:
        return self.kaggle_status


@pytest.fixture
def kaggle(test_app, monkeypatch):
    s = get_settings()
    for key, value in {
        "ADMIN_TOKEN": TOKEN,
        "LLM_PROVIDER": "remote",
        "DEV_USE_MOCK_MODELS": False,
        "KAGGLE_API_TOKEN": "kaggle-test",
        "KAGGLE_NOTEBOOK_ID": "someone/agrivoice-natlas",
        "KAGGLE_SECRETS_DATASET": "someone/agrivoice-secrets",
        "PUBLIC_BASE_URL": "https://api.example.com",
        "NATLAS_DEFAULT_RUN_HOURS": 4.0,
        "NATLAS_WEEKLY_HOURS_LIMIT": 28.0,
    }.items():
        monkeypatch.setattr(s, key, value)
    fake = FakeKaggle()
    test_app.state.natlas_runner = NatlasRunner(kaggle=fake)
    return fake


@pytest.fixture
def remote(mock_model_manager):
    registry = mock_model_manager.llm_endpoint_registry
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
    mock_model_manager._llm_adapter = adapter
    return adapter


def _run_key(fake: FakeKaggle) -> str:
    notebook = json.loads(fake.pushes[-1]["notebook"])
    config_cell = "".join(notebook["cells"][1]["source"])
    line = next(l for l in config_cell.splitlines() if l.startswith("RUN_CONFIG = "))
    return json.loads(line.removeprefix("RUN_CONFIG = "))["run_key"]


@pytest.mark.asyncio
async def test_start_requires_setup(client: AsyncClient, test_app, monkeypatch):
    monkeypatch.setattr(get_settings(), "ADMIN_TOKEN", TOKEN)
    test_app.state.natlas_runner = NatlasRunner(kaggle=FakeKaggle())
    resp = await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})
    assert resp.status_code == 409
    assert "Setup incomplete" in resp.json()["detail"]
    status = (await client.get(f"{ADMIN}/natlas", headers=AUTH)).json()
    assert status["setup"]["kaggle_api_token"] is False


@pytest.mark.asyncio
async def test_full_run_lifecycle(client: AsyncClient, kaggle, remote):
    started = await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={"hours": 2})
    assert started.status_code == 200
    assert started.json()["state"]["status"] == "starting"

    push = kaggle.pushes[-1]
    assert push["metadata"]["id"] == "someone/agrivoice-natlas"
    assert push["metadata"]["dataset_sources"] == ["someone/agrivoice-secrets"]
    assert push["metadata"]["is_private"] is True
    config = json.loads(push["notebook"])["cells"][1]["source"]
    assert any('"backend_url": "https://api.example.com"' in l for l in config)
    assert any('"run_hours": 2' in l for l in config)
    key = _run_key(kaggle)

    # Can't start twice
    again = await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})
    assert again.status_code == 409

    # Notebook comes online and registers its link
    reg = await client.post("/api/v1/natlas/register", json={"run_key": key, "endpoint": "https://abc.gradio.live"})
    assert reg.status_code == 200
    state = (await client.get(f"{ADMIN}/natlas", headers=AUTH)).json()["state"]
    assert state["status"] == "online" and state["endpoint"] == "https://abc.gradio.live"
    assert "key_hash" not in state

    # Answers flow through that link
    _connect(remote, "https://abc.gradio.live", _Job(reply="Plant after the first rains."))
    voice = (await client.post("/api/v1/voice/ask", files=_audio_file(), data={"language": "yo"})).json()
    assert voice["llm_available"] is True and voice["response"] == "Plant after the first rains."

    beat = await client.post("/api/v1/natlas/heartbeat", json={"run_key": key, "endpoint": "https://abc.gradio.live"})
    assert beat.json() == {"action": "continue"}

    # Stop from the dashboard → next heartbeat tells the notebook to stop
    stopping = await client.post(f"{ADMIN}/natlas/stop", headers=AUTH)
    assert stopping.json()["state"]["status"] == "stopping"
    beat = await client.post("/api/v1/natlas/heartbeat", json={"run_key": key, "endpoint": "https://abc.gradio.live"})
    assert beat.json() == {"action": "stop"}

    await client.post("/api/v1/natlas/stopped", json={"run_key": key})
    status = (await client.get(f"{ADMIN}/natlas", headers=AUTH)).json()
    assert status["state"]["status"] == "offline"
    assert status["recent_runs"][0]["run_id"] == state["run_id"]
    assert status["usage"]["hours_last_7_days"] >= 0

    # Dead link is forgotten, so requests fail fast
    endpoint = (await client.get(f"{ADMIN}/llm-endpoint", headers=AUTH)).json()
    assert endpoint["endpoint"] is None

    # The finished run's key no longer works
    late = await client.post("/api/v1/natlas/heartbeat", json={"run_key": key})
    assert late.json()["action"] == "stop"


@pytest.mark.asyncio
async def test_callbacks_reject_wrong_key(client: AsyncClient, kaggle):
    await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})
    bad = await client.post(
        "/api/v1/natlas/register", json={"run_key": "x" * 43, "endpoint": "https://evil.example"}
    )
    assert bad.status_code == 403
    endpoint = (await client.get(f"{ADMIN}/llm-endpoint", headers=AUTH)).json()
    assert endpoint["endpoint"] is None


@pytest.mark.asyncio
async def test_kaggle_push_failure_is_reported(client: AsyncClient, kaggle):
    def boom(folder):
        raise RuntimeError("401 Unauthorized")

    kaggle.push = boom
    resp = await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})
    assert resp.status_code == 409 and "401" in resp.json()["detail"]
    state = (await client.get(f"{ADMIN}/natlas", headers=AUTH)).json()["state"]
    assert state["status"] == "failed"
    # Can try again after a failure
    kaggle.push = FakeKaggle().push
    assert (await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})).status_code == 200


@pytest.mark.asyncio
async def test_weekly_limit(client: AsyncClient, kaggle, monkeypatch):
    monkeypatch.setattr(get_settings(), "NATLAS_WEEKLY_HOURS_LIMIT", 0.1)
    resp = await client.post(f"{ADMIN}/natlas/start", headers=AUTH, json={})
    assert resp.status_code == 409 and "Weekly limit" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_schedule_validation(client: AsyncClient, kaggle):
    bad = await client.put(f"{ADMIN}/natlas/schedule", headers=AUTH, json={"enabled": True, "days": [1], "start": "17:00", "end": "08:00"})
    assert bad.status_code == 422
    ok = await client.put(
        f"{ADMIN}/natlas/schedule",
        headers=AUTH,
        json={"enabled": True, "days": [2, 3, 4], "start": "08:00", "end": "17:00", "timezone": "Africa/Lagos"},
    )
    assert ok.status_code == 200
    assert ok.json()["schedule"]["days"] == [2, 3, 4]


def test_schedule_window():
    s = Schedule(enabled=True, timezone="Africa/Lagos", days=[3], start="08:00", end="17:00")
    # Thursday 2026-10-15 09:00 Lagos (UTC+1) = 08:00 UTC
    assert s.window(datetime(2026, 10, 15, 8, 0, tzinfo=timezone.utc)) is not None
    assert s.window(datetime(2026, 10, 15, 6, 30, tzinfo=timezone.utc)) is None  # 07:30 Lagos
    assert s.window(datetime(2026, 10, 15, 16, 0, tzinfo=timezone.utc)) is None  # 17:00 Lagos
    assert s.window(datetime(2026, 10, 16, 8, 0, tzinfo=timezone.utc)) is None   # Friday


@pytest.mark.asyncio
async def test_scheduler_starts_stops_and_respects_manual_stop(
    client: AsyncClient, kaggle, test_app, test_session, mock_model_manager, monkeypatch
):
    runner: NatlasRunner = test_app.state.natlas_runner
    registry = mock_model_manager.llm_endpoint_registry
    await runner.set_schedule(test_session, {"enabled": True, "days": list(range(7)), "start": "00:00", "end": "23:59"})

    # Inside the window, nothing running → starts
    await runner.tick(test_session, registry)
    status = await runner.describe(test_session)
    assert status["state"]["status"] == "starting" and status["state"]["trigger"] == "schedule"
    assert len(kaggle.pushes) == 1

    # Admin stops it by hand → scheduler doesn't restart it in this window
    await runner.stop(test_session, trigger="manual")
    await runner.on_stopped(test_session, registry, _run_key(kaggle))
    await runner.tick(test_session, registry)
    assert (await runner.describe(test_session))["state"]["status"] == "offline"
    assert len(kaggle.pushes) == 1

    # A scheduled run outside its window gets stopped
    state = await runner._load(test_session)
    state.last_manual_stop_at = None
    await runner._save(test_session, state)
    await runner.tick(test_session, registry)  # starts again
    assert len(kaggle.pushes) == 2
    monkeypatch.setattr(Schedule, "window", lambda self, at: None)
    await runner.tick(test_session, registry)
    assert (await runner.describe(test_session))["state"]["status"] == "stopping"


@pytest.mark.asyncio
async def test_reconcile_marks_ended_kaggle_session(client: AsyncClient, kaggle, test_app, test_session, mock_model_manager):
    runner: NatlasRunner = test_app.state.natlas_runner
    registry = mock_model_manager.llm_endpoint_registry
    await runner.start(test_session, hours=1)
    await runner.on_register(test_session, registry, _run_key(kaggle), "https://abc.gradio.live")

    state = await runner._load(test_session)
    state.started_at = time.time() - 600  # past the startup grace period
    await runner._save(test_session, state)
    kaggle.kaggle_status = "complete"
    await runner.reconcile(test_session, registry)

    state = await runner._load(test_session)
    assert state.status == "offline" and "complete" in (state.message or "")
    assert registry.current.endpoint is None


def test_schedule_date_range():
    s = Schedule(enabled=True, timezone="Africa/Lagos", days=[3, 4, 5], start="08:00", end="17:00",
                 active_from="2026-10-15", active_until="2026-10-17")
    assert s.window(datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)) is None   # Thu 1 Oct: before range
    assert s.window(datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)) is not None  # Thu 15 Oct
    assert s.window(datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)) is not None  # Sat 17 Oct
    assert s.window(datetime(2026, 10, 22, 9, 0, tzinfo=timezone.utc)) is None  # next Thu: after range


@pytest.mark.asyncio
async def test_schedule_date_range_validation(client: AsyncClient, kaggle):
    bad = await client.put(f"{ADMIN}/natlas/schedule", headers=AUTH, json={
        "enabled": True, "days": [3], "start": "08:00", "end": "17:00",
        "active_from": "2026-10-17", "active_until": "2026-10-15"})
    assert bad.status_code == 422

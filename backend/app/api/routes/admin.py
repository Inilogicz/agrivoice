"""
Admin dashboard API: server overview, activity log (every prompt/response,
including failures), runtime settings, and where N-ATLaS runs (Kaggle share
link ↔ Hugging Face Space).

Protected by `Authorization: Bearer <ADMIN_TOKEN>`. Disabled (404) when
ADMIN_TOKEN isn't set. The Kaggle notebook calls PUT automatically on start.
"""
from __future__ import annotations

import os
import platform
import secrets
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Security, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_model_manager
from app.core.config import get_settings
from app.core.exceptions import LLMUnavailableError
from app.db.database import get_db
from app.repositories.interaction_logs import InteractionLogRepository
from app.services.llm_endpoint import EndpointInfo, is_valid_endpoint
from app.services.natlas_runner import RunnerError
from app.services.runtime_settings import InvalidSettingError
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/admin", tags=["admin"])
page_router = APIRouter(include_in_schema=False)

_bearer = HTTPBearer(auto_error=False)
_PAGE = (Path(__file__).parent / "admin_page.html").read_text(encoding="utf-8")


def require_admin(credentials: HTTPAuthorizationCredentials | None = Security(_bearer)) -> None:
    configured = get_settings().ADMIN_TOKEN
    if not configured:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), configured.encode()
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class EndpointStatus(BaseModel):
    provider: Literal["local", "remote", "mock"]
    endpoint: str | None
    source: Literal["dashboard", "env", "none"]
    updated_at: datetime | None


class EndpointUpdate(BaseModel):
    endpoint: str = Field(..., min_length=3, max_length=500)


class EndpointTest(BaseModel):
    endpoint: str | None = Field(default=None, max_length=500)


class EndpointTestResult(BaseModel):
    ok: bool
    endpoint: str | None
    latency_ms: int | None = None
    reply: str | None = None
    message: str | None = None


def _status(manager: ModelManager, info: EndpointInfo) -> EndpointStatus:
    settings = get_settings()
    provider = "mock" if settings.DEV_USE_MOCK_MODELS else settings.LLM_PROVIDER
    return EndpointStatus(
        provider=provider, endpoint=info.endpoint, source=info.source, updated_at=info.updated_at
    )


@page_router.get("/admin", response_class=HTMLResponse)
async def admin_page() -> HTMLResponse:
    if not get_settings().ADMIN_TOKEN:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    return HTMLResponse(_PAGE)


@router.get("/llm-endpoint", response_model=EndpointStatus, dependencies=[Depends(require_admin)])
async def get_llm_endpoint(
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> EndpointStatus:
    info = await manager.llm_endpoint_registry.refresh(session, force=True)
    return _status(manager, info)


@router.put("/llm-endpoint", response_model=EndpointStatus, dependencies=[Depends(require_admin)])
async def set_llm_endpoint(
    body: EndpointUpdate,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> EndpointStatus:
    endpoint = body.endpoint.strip()
    if not is_valid_endpoint(endpoint):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Use a Hugging Face Space id (owner/name) or an http(s) URL.",
        )
    info = await manager.llm_endpoint_registry.save(session, endpoint)
    return _status(manager, info)


@router.delete(
    "/llm-endpoint", response_model=EndpointStatus, dependencies=[Depends(require_admin)]
)
async def reset_llm_endpoint(
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> EndpointStatus:
    """Forget the dashboard value and fall back to LLM_REMOTE_ENDPOINT."""
    info = await manager.llm_endpoint_registry.clear(session)
    return _status(manager, info)


@router.post(
    "/llm-endpoint/test",
    response_model=EndpointTestResult,
    dependencies=[Depends(require_admin)],
)
async def test_llm_endpoint(
    body: EndpointTest,
    manager: ModelManager = Depends(get_model_manager),
) -> EndpointTestResult:
    """Send a tiny prompt to *endpoint* (default: the current one)."""
    remote = manager.get_remote_llm()
    target = (body.endpoint or "").strip() or manager.llm_endpoint_registry.current.endpoint
    if remote is None:
        return EndpointTestResult(
            ok=False, endpoint=target, message="LLM_PROVIDER is not 'remote' on this server."
        )
    if target and not is_valid_endpoint(target):
        return EndpointTestResult(ok=False, endpoint=target, message="Invalid endpoint format.")
    try:
        reply, latency = await remote.check(target)
    except LLMUnavailableError:
        return EndpointTestResult(
            ok=False, endpoint=target, message="N-ATLaS did not respond. See server logs."
        )
    return EndpointTestResult(ok=True, endpoint=target, latency_ms=latency, reply=reply)


# ── Overview ─────────────────────────────────────────────────────────────────


def _memory() -> dict[str, int | None]:
    """System and process memory in MB (Linux /proc; None elsewhere)."""
    info: dict[str, int | None] = {"total_mb": None, "available_mb": None, "process_mb": None}
    try:
        with open("/proc/meminfo") as f:
            meminfo = {line.split(":")[0]: int(line.split()[1]) for line in f}
        info["total_mb"] = meminfo["MemTotal"] // 1024
        info["available_mb"] = meminfo["MemAvailable"] // 1024
        with open("/proc/self/status") as f:
            rss = next(line for line in f if line.startswith("VmRSS:"))
        info["process_mb"] = int(rss.split()[1]) // 1024
    except (OSError, KeyError, StopIteration, ValueError):
        pass
    return info


def _iso(ts: float | None) -> str | None:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None


@router.get("/overview", dependencies=[Depends(require_admin)])
async def overview(
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    settings = get_settings()
    endpoint = await manager.llm_endpoint_registry.refresh(session, force=True)
    remote = manager.get_remote_llm()
    disk = shutil.disk_usage("/")
    now = datetime.now(timezone.utc)
    logs = InteractionLogRepository(session)
    started_at = request.app.state.started_at

    try:
        load = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        load = None

    return {
        "server": {
            "status": "ok",
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "started_at": _iso(started_at),
            "uptime_seconds": round(time.time() - started_at),
            "device": manager.get_device(),
            "cpu_count": os.cpu_count(),
            "load_average": load,
            "memory": _memory(),
            "disk": {"total_gb": round(disk.total / 1e9, 1), "free_gb": round(disk.free / 1e9, 1)},
            "python": platform.python_version(),
        },
        "models": {
            name: {
                "model_id": s.model_id,
                "loaded": s.is_loaded,
                "device": s.device,
                "mock": s.is_mock,
                "load_time_ms": s.load_time_ms,
            }
            for name, s in manager.get_all_status().items()
        },
        "llm": {
            **_status(manager, endpoint).model_dump(mode="json"),
            "last_success_at": _iso(remote.last_success_at) if remote else None,
            "last_failure_at": _iso(remote.last_failure_at) if remote else None,
            "last_error": remote.last_error if remote else None,
        },
        "stats_24h": await logs.stats(now - timedelta(hours=24)),
        "stats_7d": await logs.stats(now - timedelta(days=7)),
    }


# ── Activity ─────────────────────────────────────────────────────────────────


class InteractionOut(BaseModel):
    id: str
    created_at: datetime | None
    conversation_id: str | None
    input_type: str
    language: str
    language_source: str | None
    language_confidence: float | None
    prompt: str | None
    response: str | None
    success: bool
    llm_available: bool | None
    error_type: str | None
    error_message: str | None
    asr_model: str | None
    llm_model: str | None
    processing_time_ms: int | None
    detection_ms: int | None
    asr_ms: int | None
    llm_ms: int | None
    request_id: str | None


class InteractionPage(BaseModel):
    total: int
    items: list[InteractionOut]


@router.get("/interactions", response_model=InteractionPage, dependencies=[Depends(require_admin)])
async def list_interactions(
    session: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    status_filter: Literal["ok", "error"] | None = Query(None, alias="status"),
    language: str | None = None,
    input_type: Literal["voice", "text", "translation"] | None = None,
    search: str | None = Query(None, max_length=200),
) -> InteractionPage:
    rows, total = await InteractionLogRepository(session).list(
        limit=limit,
        offset=offset,
        status=status_filter,
        language=language,
        input_type=input_type,
        search=search,
    )
    items = [
        InteractionOut(
            id=str(r.id),
            # SQLite returns naive datetimes; they are stored in UTC
            created_at=(
                r.created_at.replace(tzinfo=timezone.utc)
                if r.created_at and r.created_at.tzinfo is None
                else r.created_at
            ),
            conversation_id=str(r.conversation_id) if r.conversation_id else None,
            **{
                field: getattr(r, field)
                for field in InteractionOut.model_fields
                if field not in ("id", "created_at", "conversation_id")
            },
        )
        for r in rows
    ]
    return InteractionPage(total=total, items=items)


# ── Settings ─────────────────────────────────────────────────────────────────


class SettingsUpdate(BaseModel):
    changes: dict[str, str | int | float]


@router.get("/settings", dependencies=[Depends(require_admin)])
async def get_runtime_settings(request: Request, session: AsyncSession = Depends(get_db)) -> dict:
    store = request.app.state.runtime_settings
    await store.refresh(session, force=True)
    return {"settings": store.describe()}


@router.put("/settings", dependencies=[Depends(require_admin)])
async def update_runtime_settings(
    body: SettingsUpdate, request: Request, session: AsyncSession = Depends(get_db)
) -> dict:
    store = request.app.state.runtime_settings
    try:
        await store.update(session, body.changes)
    except InvalidSettingError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return {"settings": store.describe()}


@router.delete("/settings/{key}", dependencies=[Depends(require_admin)])
async def reset_runtime_setting(
    key: str, request: Request, session: AsyncSession = Depends(get_db)
) -> dict:
    store = request.app.state.runtime_settings
    try:
        await store.reset(session, key)
    except InvalidSettingError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return {"settings": store.describe()}


# ── N-ATLaS on Kaggle: start / stop / schedule ───────────────────────────────


class StartRequest(BaseModel):
    hours: float | None = Field(default=None, gt=0, le=11)


@router.get("/natlas", dependencies=[Depends(require_admin)])
async def natlas_status(
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    return await request.app.state.natlas_runner.describe(session, manager.llm_endpoint_registry)


@router.post("/natlas/start", dependencies=[Depends(require_admin)])
async def natlas_start(
    body: StartRequest,
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    runner = request.app.state.natlas_runner
    try:
        await runner.start(session, hours=body.hours, trigger="manual")
    except RunnerError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await runner.describe(session, manager.llm_endpoint_registry)


@router.post("/natlas/stop", dependencies=[Depends(require_admin)])
async def natlas_stop(
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    runner = request.app.state.natlas_runner
    try:
        await runner.stop(session, trigger="manual")
    except RunnerError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return await runner.describe(session, manager.llm_endpoint_registry)


@router.put("/natlas/schedule", dependencies=[Depends(require_admin)])
async def natlas_schedule(
    body: dict,
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    runner = request.app.state.natlas_runner
    try:
        await runner.set_schedule(session, body)
    except RunnerError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    return await runner.describe(session, manager.llm_endpoint_registry)

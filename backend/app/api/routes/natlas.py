"""
Callbacks from the Kaggle notebook when it was started from the admin dashboard.

Authenticated by the one-time run key the backend put in that run's notebook,
not by the admin token, so a leaked key only affects one run.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_model_manager
from app.db.database import get_db
from app.services.model_manager import ModelManager
from app.services.natlas_runner import RunnerError

router = APIRouter(prefix="/natlas", tags=["natlas"], include_in_schema=False)


class Callback(BaseModel):
    run_key: str = Field(..., min_length=20, max_length=200)
    endpoint: str | None = Field(default=None, max_length=500)


def _forbidden(exc: RunnerError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))


@router.post("/register")
async def register(
    body: Callback,
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    try:
        await request.app.state.natlas_runner.on_register(
            session, manager.llm_endpoint_registry, body.run_key, body.endpoint or ""
        )
    except RunnerError as exc:
        raise _forbidden(exc) from exc
    return {"ok": True}


@router.post("/heartbeat")
async def heartbeat(
    body: Callback,
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    try:
        action = await request.app.state.natlas_runner.on_heartbeat(
            session, manager.llm_endpoint_registry, body.run_key, body.endpoint or ""
        )
    except RunnerError as exc:
        # An unknown key means this run is no longer wanted: tell it to stop
        return {"action": "stop", "reason": str(exc)}
    return {"action": action}


@router.post("/stopped")
async def stopped(
    body: Callback,
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> dict:
    try:
        await request.app.state.natlas_runner.on_stopped(
            session, manager.llm_endpoint_registry, body.run_key
        )
    except RunnerError:
        pass  # already finished
    return {"ok": True}

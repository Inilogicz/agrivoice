"""Health check routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.deps import get_model_manager
from app.services.model_manager import ModelManager

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", summary="Basic liveness check")
async def health() -> dict:
    """Returns 200 if the server is alive."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness check with model status")
async def readiness(
    manager: ModelManager = Depends(get_model_manager),
) -> dict:
    """
    Returns model load status.

    Returns 200 if all models are loaded.
    No secrets, paths, or internal details are exposed.
    """
    statuses = manager.get_all_status()
    all_ready = all(s.is_loaded for s in statuses.values())

    model_info = {
        name: {
            "loaded": s.is_loaded,
            "device": s.device,
            "mock": s.is_mock,
        }
        for name, s in statuses.items()
    }

    return {
        "status": "ready" if all_ready else "not_ready",
        "device": manager.get_device(),
        "models": model_info,
    }

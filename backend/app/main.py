"""
AgriVoice FastAPI application factory.

Responsibilities:
  - Configure CORS, middleware, and error handlers
  - Manage model lifecycle via lifespan context manager
  - Register all API routers
  - Expose OpenAPI docs only in non-production environments
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    admin,
    chat,
    conversations,
    feedback,
    health,
    languages,
    natlas,
    speech,
    translate,
    voice,
)
from app.core.config import get_settings
from app.core.exceptions import AgriVoiceError, to_http_exception
from app.core.logging import configure_logging, get_logger, set_request_id
from app.db.database import AsyncSessionLocal, create_sqlite_schema, is_sqlite
from app.services.natlas_runner import NatlasRunner, run_scheduler
from app.services.runtime_settings import RuntimeSettingsStore
from app.services.model_manager import ModelManager

logger = get_logger(__name__)


class DynamicCORSMiddleware(CORSMiddleware):
    """CORS whose allowed origins follow CORS_ORIGINS live (editable in /admin)."""

    def is_allowed_origin(self, origin: str) -> bool:
        allowed = get_settings().cors_origins_list
        return "*" in allowed or origin in allowed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan: initialize and tear down the model manager.

    Models are loaded once at startup and shared across all requests via
    app.state.model_manager.
    """
    configure_logging()
    settings = get_settings()

    logger.info(
        "AgriVoice starting",
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )

    if is_sqlite:
        await create_sqlite_schema()

    # Apply settings saved from the admin dashboard before serving requests
    async with AsyncSessionLocal() as session:
        await app.state.runtime_settings.refresh(session, force=True)

    manager = ModelManager()
    try:
        manager.initialize()
        logger.info("All models loaded successfully")
    except Exception as exc:
        logger.error("FATAL: Model initialization failed", error=str(exc))
        raise

    app.state.model_manager = manager

    # Starts/stops N-ATLaS on Kaggle on schedule and notices ended sessions
    scheduler = asyncio.create_task(run_scheduler(app.state, AsyncSessionLocal))

    yield  # Server is running

    scheduler.cancel()
    logger.info("AgriVoice shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Multilingual, voice-first agricultural assistant for Nigerian smallholder farmers. "
            "Supports Yoruba, Hausa, Igbo, and Nigerian-accented English via the N-ATLaS model family."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # Admin-editable settings (CORS origins, answer length, …); see /admin
    app.state.runtime_settings = RuntimeSettingsStore()
    app.state.natlas_runner = NatlasRunner()
    app.state.started_at = time.time()

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        DynamicCORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ──────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ── Domain exception handler ──────────────────────────────────────────────
    @app.exception_handler(AgriVoiceError)
    async def agrivoice_exception_handler(request: Request, exc: AgriVoiceError):
        # Same {"detail": {...}} shape as errors raised inside route handlers
        http_exc = to_http_exception(exc)
        return JSONResponse(
            status_code=http_exc.status_code,
            content={"detail": http_exc.detail},
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    api_prefix = settings.API_V1_PREFIX
    app.include_router(health.router, prefix=api_prefix)
    app.include_router(voice.router, prefix=api_prefix)
    app.include_router(languages.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(translate.router, prefix=api_prefix)
    app.include_router(speech.router, prefix=api_prefix)
    app.include_router(conversations.router, prefix=api_prefix)
    app.include_router(feedback.router, prefix=api_prefix)
    app.include_router(admin.router, prefix=api_prefix)
    app.include_router(natlas.router, prefix=api_prefix)
    app.include_router(admin.page_router)

    return app


app = create_app()

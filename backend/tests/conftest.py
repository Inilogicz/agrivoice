"""
conftest.py: Shared pytest fixtures for AgriVoice tests.

Uses an in-memory SQLite database so tests run without a real Postgres instance.
All model adapters are mocked — no GPU required.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.database import get_db
from app.main import create_app
from app.services.mock_adapters import (
    MockASRAdapter,
    MockLanguageDetector,
    MockLLMAdapter,
    MockSpeechSynthesizer,
)
from app.services.llm_endpoint import LLMEndpointRegistry
from app.services.model_manager import ModelManager


# ── Test settings override ─────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        ENVIRONMENT="development",
        DEV_USE_MOCK_MODELS=True,
        DATABASE_URL="sqlite+aiosqlite:///./test.db",
        HF_TOKEN=None,
    )


# ── In-memory async DB ─────────────────────────────────────────────────────────
@pytest_asyncio.fixture(scope="function")
async def test_session():
    """Create a fresh in-memory SQLite DB for each test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    TestSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with TestSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# ── Mock ModelManager ──────────────────────────────────────────────────────────
@pytest.fixture
def mock_model_manager() -> ModelManager:
    """Return a ModelManager pre-loaded with mock adapters."""
    manager = ModelManager.__new__(ModelManager)
    manager._settings = get_settings()
    manager._device = "cpu"
    manager._asr_adapters = {
        "yo": MockASRAdapter("yo"),
        "ha": MockASRAdapter("ha"),
        "ig": MockASRAdapter("ig"),
        "en-ng": MockASRAdapter("en-ng"),
    }
    manager._llm_adapter = MockLLMAdapter()
    manager._language_detector = MockLanguageDetector()
    manager._speech = MockSpeechSynthesizer()
    manager._status = {}
    manager.llm_endpoint_registry = LLMEndpointRegistry()
    return manager


# ── Test FastAPI app ───────────────────────────────────────────────────────────
@pytest_asyncio.fixture
async def test_app(mock_model_manager, test_session) -> FastAPI:
    """Build app with overridden DB and mock models."""
    app = create_app()

    # Override model manager
    app.state.model_manager = mock_model_manager

    # Override DB dependency
    async def override_get_db():
        yield test_session

    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest_asyncio.fixture
async def client(test_app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=test_app),
        base_url="http://test",
    ) as c:
        yield c

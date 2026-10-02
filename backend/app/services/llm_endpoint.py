"""
Live registry of where the remote N-ATLaS endpoint is.

The admin dashboard (or the Kaggle notebook, automatically) saves a new
endpoint to the database; every request re-reads it at most every
_REFRESH_SECONDS, so a change takes effect without a restart or redeploy.
Falls back to LLM_REMOTE_ENDPOINT when nothing has been saved.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.app_settings import AppSettingsRepository

logger = get_logger(__name__)

SETTING_KEY = "llm_remote_endpoint"
_REFRESH_SECONDS = 15

# "owner/space-name" or an http(s) URL (e.g. a *.gradio.live share link)
_SPACE_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
_URL = re.compile(r"^https?://\S+$")


def is_valid_endpoint(value: str) -> bool:
    return bool(_SPACE_ID.match(value) or _URL.match(value))


@dataclass(frozen=True)
class EndpointInfo:
    endpoint: str | None
    source: Literal["dashboard", "env", "none"]
    updated_at: datetime | None


class LLMEndpointRegistry:
    def __init__(self) -> None:
        self._info = self._from_env()
        self._last_refresh = 0.0

    @property
    def current(self) -> EndpointInfo:
        return self._info

    async def refresh(self, session: AsyncSession, *, force: bool = False) -> EndpointInfo:
        if not force and time.monotonic() - self._last_refresh < _REFRESH_SECONDS:
            return self._info
        try:
            setting = await AppSettingsRepository(session).get(SETTING_KEY)
        except Exception as exc:
            # Keep serving the last known endpoint if the DB hiccups
            logger.warning("Could not read N-ATLaS endpoint from DB", error=str(exc))
            return self._info
        self._info = (
            EndpointInfo(setting.value, "dashboard", setting.updated_at)
            if setting
            else self._from_env()
        )
        self._last_refresh = time.monotonic()
        return self._info

    async def save(self, session: AsyncSession, endpoint: str) -> EndpointInfo:
        setting = await AppSettingsRepository(session).set(SETTING_KEY, endpoint)
        await session.commit()
        self._info = EndpointInfo(setting.value, "dashboard", setting.updated_at)
        self._last_refresh = time.monotonic()
        logger.info("N-ATLaS endpoint updated", endpoint=endpoint)
        return self._info

    async def clear(self, session: AsyncSession) -> EndpointInfo:
        await AppSettingsRepository(session).delete(SETTING_KEY)
        await session.commit()
        self._info = self._from_env()
        self._last_refresh = time.monotonic()
        return self._info

    @staticmethod
    def _from_env() -> EndpointInfo:
        env = get_settings().LLM_REMOTE_ENDPOINT
        return EndpointInfo(env, "env", None) if env else EndpointInfo(None, "none", None)

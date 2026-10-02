"""
Settings an admin can change from the dashboard, without code or restarts.

Each editable setting is a field of app.core.config.Settings. Overrides are
stored in the app_settings table; the store applies them onto the cached
Settings object, so every get_settings() reader sees the new value. Requests
re-read the table at most every _REFRESH_SECONDS. Removing an override
restores the value from the environment (.env).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models.app_setting import AppSetting

logger = get_logger(__name__)

_PREFIX = "setting:"
_REFRESH_SECONDS = 15


@dataclass(frozen=True)
class SettingSpec:
    key: str
    label: str
    help: str
    kind: Literal["int", "float", "origins"]
    min: float | None = None
    max: float | None = None


EDITABLE_SETTINGS: dict[str, SettingSpec] = {
    spec.key: spec
    for spec in [
        SettingSpec(
            "CORS_ORIGINS",
            "Allowed frontend addresses",
            "Websites allowed to call the API from a browser, comma-separated "
            "(e.g. http://localhost:3000, https://agrivoice.vercel.app).",
            "origins",
        ),
        SettingSpec(
            "LLM_MAX_NEW_TOKENS",
            "Maximum answer length (tokens)",
            "Longer answers take longer: about 13 tokens per second on Kaggle.",
            "int", 32, 1000,
        ),
        SettingSpec(
            "REQUEST_TIMEOUT_SECONDS",
            "N-ATLaS timeout (seconds)",
            "How long to wait for an answer before showing 'temporarily unavailable'.",
            "int", 30, 600,
        ),
        SettingSpec(
            "LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD",
            "Language detection confidence (0–1)",
            "Below this, the user is asked to pick their language. Higher = asks more often.",
            "float", 0.3, 0.99,
        ),
        SettingSpec(
            "LANGUAGE_DETECTION_MAX_SECONDS",
            "Seconds of audio used for detection",
            "Shorter is faster. 8 s stayed accurate in testing; 5 s misread Hausa.",
            "float", 4, 30,
        ),
        SettingSpec(
            "NATLAS_DEFAULT_RUN_HOURS",
            "N-ATLaS run length (hours)",
            "How long N-ATLaS stays on after Start, unless stopped earlier.",
            "float", 0.5, 11,
        ),
        SettingSpec(
            "NATLAS_WEEKLY_HOURS_LIMIT",
            "Weekly GPU hours limit",
            "Start and the schedule refuse to run past this (Kaggle allows 30 h/week).",
            "float", 1, 30,
        ),
        SettingSpec(
            "MAX_AUDIO_SIZE_MB",
            "Maximum upload size (MB)",
            "Recordings larger than this are rejected.",
            "int", 1, 25,
        ),
    ]
}


class InvalidSettingError(ValueError):
    pass


def _parse(spec: SettingSpec, raw: Any) -> Any:
    """Validate and convert a value for *spec*; raises InvalidSettingError."""
    if spec.kind == "origins":
        origins = [o.strip().rstrip("/") for o in str(raw).split(",") if o.strip()]
        if not origins:
            raise InvalidSettingError("Enter at least one address.")
        for origin in origins:
            if origin == "*":
                continue
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.path:
                raise InvalidSettingError(
                    f"'{origin}' must look like https://example.com (no path)."
                )
        return ",".join(origins)

    try:
        value = int(raw) if spec.kind == "int" else float(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidSettingError(f"{spec.label} must be a number.") from exc
    if spec.min is not None and value < spec.min or spec.max is not None and value > spec.max:
        raise InvalidSettingError(f"{spec.label} must be between {spec.min:g} and {spec.max:g}.")
    return value


class RuntimeSettingsStore:
    def __init__(self) -> None:
        settings = get_settings()
        # Values from the environment, restored when an override is removed
        self._defaults = {key: getattr(settings, key) for key in EDITABLE_SETTINGS}
        self._overridden: set[str] = set()
        self._last_refresh = 0.0

    def describe(self) -> list[dict[str, Any]]:
        settings = get_settings()
        return [
            {
                "key": spec.key,
                "label": spec.label,
                "help": spec.help,
                "kind": spec.kind,
                "min": spec.min,
                "max": spec.max,
                "value": getattr(settings, spec.key),
                "default": self._defaults[spec.key],
                "overridden": spec.key in self._overridden,
            }
            for spec in EDITABLE_SETTINGS.values()
        ]

    async def refresh(self, session: AsyncSession, *, force: bool = False) -> None:
        if not force and time.monotonic() - self._last_refresh < _REFRESH_SECONDS:
            return
        try:
            rows = await session.scalars(
                select(AppSetting).where(AppSetting.key.startswith(_PREFIX))
            )
            stored = {row.key.removeprefix(_PREFIX): row.value for row in rows}
        except Exception as exc:
            logger.warning("Could not read runtime settings", error=str(exc))
            return
        self._apply(stored)
        self._last_refresh = time.monotonic()

    async def update(self, session: AsyncSession, changes: dict[str, Any]) -> None:
        """Validate all *changes* first, then save them together."""
        parsed: dict[str, Any] = {}
        for key, raw in changes.items():
            spec = EDITABLE_SETTINGS.get(key)
            if spec is None:
                raise InvalidSettingError(f"'{key}' can't be changed from the dashboard.")
            parsed[key] = _parse(spec, raw)

        for key, value in parsed.items():
            row = await session.get(AppSetting, _PREFIX + key)
            if row is None:
                session.add(AppSetting(key=_PREFIX + key, value=str(value)))
            else:
                row.value = str(value)
        await session.commit()
        logger.info("Runtime settings updated", keys=sorted(parsed))
        await self.refresh(session, force=True)

    async def reset(self, session: AsyncSession, key: str) -> None:
        if key not in EDITABLE_SETTINGS:
            raise InvalidSettingError(f"'{key}' can't be changed from the dashboard.")
        row = await session.get(AppSetting, _PREFIX + key)
        if row is not None:
            await session.delete(row)
            await session.commit()
        logger.info("Runtime setting reset to default", key=key)
        await self.refresh(session, force=True)

    def _apply(self, stored: dict[str, str]) -> None:
        settings = get_settings()
        self._overridden = set()
        for key, spec in EDITABLE_SETTINGS.items():
            value = self._defaults[key]
            if key in stored:
                try:
                    value = _parse(spec, stored[key])
                    self._overridden.add(key)
                except InvalidSettingError:
                    logger.warning("Ignoring invalid stored setting", key=key)
            setattr(settings, key, value)

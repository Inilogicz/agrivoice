"""
Start, stop and schedule N-ATLaS on Kaggle from the admin dashboard.

Kaggle's API can start a notebook (by pushing a new version) and report its
status, but it can't stop one. So each run gets a one-time key: the notebook
registers its gradio.live link with it, checks in every minute, and shuts
itself down when the heartbeat reply says "stop".

Runs started through the API can't read Kaggle Secrets, so the notebook gets
the Hugging Face token from a private Kaggle dataset (KAGGLE_SECRETS_DATASET).

State lives in the app_settings table, so it survives backend restarts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from zoneinfo import ZoneInfo

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.repositories.app_settings import AppSettingsRepository
from app.services.llm_endpoint import LLMEndpointRegistry, is_valid_endpoint

logger = get_logger(__name__)

TEMPLATE = Path(__file__).resolve().parent.parent / "resources" / "natlas_kaggle.ipynb"
RUN_CONFIG_LINE = "RUN_CONFIG = {}"

STATE_KEY = "natlas_runner_state"
HISTORY_KEY = "natlas_runner_history"
SCHEDULE_KEY = "natlas_runner_schedule"
_HISTORY_LIMIT = 200
_HEARTBEAT_STALE_SECONDS = 300
_MAX_RUN_HOURS = 11.0  # Kaggle ends GPU sessions after roughly 12 hours

Status = Literal["offline", "starting", "online", "stopping", "failed"]
ACTIVE: tuple[Status, ...] = ("starting", "online", "stopping")


class RunnerError(Exception):
    """A start/stop request that can't be carried out; message is user-facing."""


class KaggleNotebooks(Protocol):
    def push(self, folder: str) -> None: ...
    def status(self, notebook_id: str) -> str: ...


class KaggleApiNotebooks:
    """Real Kaggle API (authenticates from KAGGLE_API_TOKEN)."""

    def __init__(self) -> None:
        self._api: Any | None = None

    def _client(self) -> Any:
        if self._api is None:
            from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    def push(self, folder: str) -> None:
        response = self._client().kernels_push(folder)
        error = getattr(response, "error", None)
        if error:
            raise RunnerError(f"Kaggle refused the notebook: {error}")

    def status(self, notebook_id: str) -> str:
        result = self._client().kernels_status(notebook_id)
        return str(getattr(result, "status", result)).rsplit(".", 1)[-1].lower()


@dataclass
class RunState:
    status: Status = "offline"
    run_id: str | None = None
    key_hash: str | None = None
    trigger: Literal["manual", "schedule"] | None = None
    hours: float | None = None
    started_at: float | None = None
    registered_at: float | None = None
    last_heartbeat_at: float | None = None
    stop_requested_at: float | None = None
    ended_at: float | None = None
    last_manual_stop_at: float | None = None
    endpoint: str | None = None
    message: str | None = None
    kaggle_status: str | None = None


@dataclass
class Schedule:
    enabled: bool = False
    timezone: str = "Africa/Lagos"
    days: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # Mon=0
    start: str = "08:00"
    end: str = "17:00"
    # Optional date range (YYYY-MM-DD, inclusive, in the schedule's timezone)
    active_from: str | None = None
    active_until: str | None = None

    def window(self, at: datetime) -> tuple[datetime, datetime] | None:
        """Today's window (in UTC) if *at* falls inside it, else None."""
        local = at.astimezone(ZoneInfo(self.timezone))
        today = local.date().isoformat()
        if self.active_from and today < self.active_from:
            return None
        if self.active_until and today > self.active_until:
            return None
        if local.weekday() not in self.days:
            return None
        sh, sm = map(int, self.start.split(":"))
        eh, em = map(int, self.end.split(":"))
        start = local.replace(hour=sh, minute=sm, second=0, microsecond=0)
        end = local.replace(hour=eh, minute=em, second=0, microsecond=0)
        if start <= local < end:
            return start.astimezone(timezone.utc), end.astimezone(timezone.utc)
        return None


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def validate_schedule(data: dict[str, Any]) -> Schedule:
    try:
        schedule = Schedule(
            enabled=bool(data.get("enabled", False)),
            timezone=str(data.get("timezone", "Africa/Lagos")),
            days=sorted({int(d) for d in data.get("days", [])}),
            start=str(data.get("start", "08:00")),
            end=str(data.get("end", "17:00")),
            active_from=str(data["active_from"]) if data.get("active_from") else None,
            active_until=str(data["active_until"]) if data.get("active_until") else None,
        )
        for value in (schedule.active_from, schedule.active_until):
            if value:
                datetime.strptime(value, "%Y-%m-%d")
        ZoneInfo(schedule.timezone)
        sh, sm = map(int, schedule.start.split(":"))
        eh, em = map(int, schedule.end.split(":"))
    except Exception as exc:  # noqa: BLE001
        raise RunnerError(
            "Invalid schedule: use days 0–6, times like 08:00, dates like 2026-10-15, a valid timezone."
        ) from exc
    if not all(0 <= d <= 6 for d in schedule.days):
        raise RunnerError("Days must be 0 (Monday) to 6 (Sunday).")
    if not (0 <= sh < 24 and 0 <= eh < 24 and 0 <= sm < 60 and 0 <= em < 60):
        raise RunnerError("Times must be between 00:00 and 23:59.")
    if (eh, em) <= (sh, sm):
        raise RunnerError("The end time must be after the start time.")
    if schedule.enabled and not schedule.days:
        raise RunnerError("Pick at least one day.")
    if schedule.active_from and schedule.active_until and schedule.active_until < schedule.active_from:
        raise RunnerError("The 'until' date must be on or after the 'from' date.")
    return schedule


class NatlasRunner:
    def __init__(self, kaggle: KaggleNotebooks | None = None) -> None:
        self._kaggle = kaggle or KaggleApiNotebooks()
        self._lock = asyncio.Lock()
        self._last_kaggle_check = 0.0

    # ── persistence ──────────────────────────────────────────────────────

    async def _load(self, session: AsyncSession) -> RunState:
        row = await AppSettingsRepository(session).get(STATE_KEY)
        return RunState(**json.loads(row.value)) if row else RunState()

    async def _save(self, session: AsyncSession, state: RunState) -> None:
        await AppSettingsRepository(session).set(STATE_KEY, json.dumps(asdict(state)))
        await session.commit()

    async def _history(self, session: AsyncSession) -> list[dict[str, Any]]:
        row = await AppSettingsRepository(session).get(HISTORY_KEY)
        return json.loads(row.value) if row else []

    async def _record(self, session: AsyncSession, state: RunState) -> None:
        if not state.started_at:
            return
        history = await self._history(session)
        history.append(
            {
                "run_id": state.run_id,
                "trigger": state.trigger,
                "started_at": state.started_at,
                "ended_at": state.ended_at or time.time(),
                "status": state.status,
                "message": state.message,
            }
        )
        await AppSettingsRepository(session).set(
            HISTORY_KEY, json.dumps(history[-_HISTORY_LIMIT:])
        )

    async def get_schedule(self, session: AsyncSession) -> Schedule:
        row = await AppSettingsRepository(session).get(SCHEDULE_KEY)
        return Schedule(**json.loads(row.value)) if row else Schedule()

    async def set_schedule(self, session: AsyncSession, data: dict[str, Any]) -> Schedule:
        schedule = validate_schedule(data)
        await AppSettingsRepository(session).set(SCHEDULE_KEY, json.dumps(asdict(schedule)))
        await session.commit()
        logger.info("N-ATLaS schedule updated", **asdict(schedule))
        return schedule

    # ── reporting ────────────────────────────────────────────────────────

    def setup_checks(self) -> dict[str, bool]:
        s = get_settings()
        return {
            "provider_is_remote": s.LLM_PROVIDER == "remote" and not s.DEV_USE_MOCK_MODELS,
            "kaggle_api_token": bool(s.KAGGLE_API_TOKEN),
            "kaggle_notebook_id": bool(s.KAGGLE_NOTEBOOK_ID),
            "kaggle_secrets_dataset": bool(s.KAGGLE_SECRETS_DATASET),
            "public_base_url": bool(s.PUBLIC_BASE_URL),
        }

    async def hours_used(self, session: AsyncSession, state: RunState | None = None) -> float:
        """GPU hours used in the last 7 days, by runs this backend knows about."""
        since = time.time() - 7 * 86400
        total = 0.0
        for run in await self._history(session):
            start, end = max(run["started_at"], since), run["ended_at"]
            total += max(0.0, end - start)
        state = state or await self._load(session)
        if state.status in ACTIVE and state.started_at:
            total += max(0.0, time.time() - max(state.started_at, since))
        return round(total / 3600, 2)

    async def describe(
        self, session: AsyncSession, registry: LLMEndpointRegistry | None = None
    ) -> dict[str, Any]:
        await self.reconcile(session, registry)
        state = await self._load(session)
        schedule = await self.get_schedule(session)
        s = get_settings()
        now = time.time()
        heartbeat_stale = (
            state.status == "online"
            and state.last_heartbeat_at is not None
            and now - state.last_heartbeat_at > _HEARTBEAT_STALE_SECONDS
        )
        window = schedule.window(datetime.now(timezone.utc)) if schedule.enabled else None
        history = await self._history(session)
        return {
            "state": {
                **{k: v for k, v in asdict(state).items() if k != "key_hash"},
                "heartbeat_stale": heartbeat_stale,
                "ends_at": (
                    state.started_at + state.hours * 3600
                    if state.status in ACTIVE and state.started_at and state.hours
                    else None
                ),
            },
            "setup": self.setup_checks(),
            "usage": {
                "hours_last_7_days": await self.hours_used(session, state),
                "weekly_limit": s.NATLAS_WEEKLY_HOURS_LIMIT,
                "default_run_hours": s.NATLAS_DEFAULT_RUN_HOURS,
            },
            "schedule": {**asdict(schedule), "in_window_now": window is not None},
            "recent_runs": list(reversed(history[-10:])),
            "notebook_url": (
                f"https://www.kaggle.com/code/{s.KAGGLE_NOTEBOOK_ID}" if s.KAGGLE_NOTEBOOK_ID else None
            ),
        }

    # ── control ──────────────────────────────────────────────────────────

    async def start(
        self,
        session: AsyncSession,
        *,
        hours: float | None = None,
        trigger: Literal["manual", "schedule"] = "manual",
    ) -> RunState:
        async with self._lock:
            missing = [name for name, ok in self.setup_checks().items() if not ok]
            if missing:
                raise RunnerError("Setup incomplete: " + ", ".join(missing))
            state = await self._load(session)
            if state.status in ACTIVE:
                raise RunnerError(f"N-ATLaS is already {state.status}.")

            s = get_settings()
            remaining = s.NATLAS_WEEKLY_HOURS_LIMIT - await self.hours_used(session, state)
            if remaining < 0.25:
                raise RunnerError(
                    f"Weekly limit of {s.NATLAS_WEEKLY_HOURS_LIMIT:g} GPU hours reached."
                )
            run_hours = round(min(hours or s.NATLAS_DEFAULT_RUN_HOURS, remaining, _MAX_RUN_HOURS), 2)

            run_key = secrets.token_urlsafe(32)
            state = RunState(
                status="starting",
                run_id=secrets.token_hex(6),
                key_hash=_hash(run_key),
                trigger=trigger,
                hours=run_hours,
                started_at=time.time(),
                last_manual_stop_at=state.last_manual_stop_at,
                message="Pushing notebook to Kaggle…",
            )
            await self._save(session, state)

            try:
                await asyncio.to_thread(self._push, run_key, run_hours)
            except Exception as exc:  # noqa: BLE001
                state.status, state.ended_at = "failed", time.time()
                state.message = f"Could not start on Kaggle: {exc}"[:500]
                await self._record(session, state)
                await self._save(session, state)
                logger.error("N-ATLaS start failed", error=str(exc))
                raise RunnerError(state.message) from exc

            state.message = "Kaggle is loading N-ATLaS (about 3–6 minutes)."
            await self._save(session, state)
            logger.info("N-ATLaS start requested", run_id=state.run_id, hours=run_hours, trigger=trigger)
            return state

    def _push(self, run_key: str, run_hours: float) -> None:
        s = get_settings()
        notebook = TEMPLATE.read_text(encoding="utf-8")
        run_config = {
            "backend_url": (s.PUBLIC_BASE_URL or "").rstrip("/"),
            "run_key": run_key,
            "run_hours": run_hours,
        }
        line = f"RUN_CONFIG = {json.dumps(run_config)}"
        # The template stores source lines as JSON strings
        notebook = notebook.replace(json.dumps(RUN_CONFIG_LINE)[1:-1], json.dumps(line)[1:-1], 1)
        if run_key not in notebook:
            raise RunnerError("Notebook template is missing the RUN_CONFIG line.")

        slug = (s.KAGGLE_NOTEBOOK_ID or "").split("/", 1)[-1]
        metadata = {
            "id": s.KAGGLE_NOTEBOOK_ID,
            "title": slug,
            "code_file": "notebook.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [s.KAGGLE_SECRETS_DATASET],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "notebook.ipynb").write_text(notebook, encoding="utf-8")
            Path(folder, "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            self._kaggle.push(folder)

    async def stop(
        self, session: AsyncSession, *, trigger: Literal["manual", "schedule"] = "manual"
    ) -> RunState:
        async with self._lock:
            state = await self._load(session)
            if trigger == "manual":
                state.last_manual_stop_at = time.time()
            if state.status not in ACTIVE:
                await self._save(session, state)
                raise RunnerError("N-ATLaS isn't running.")
            state.status = "stopping"
            state.stop_requested_at = time.time()
            state.message = "Waiting for the notebook to shut down (up to about a minute)."
            await self._save(session, state)
            logger.info("N-ATLaS stop requested", run_id=state.run_id, trigger=trigger)
            return state

    # ── notebook callbacks ───────────────────────────────────────────────

    async def _verify(self, session: AsyncSession, run_key: str) -> RunState:
        state = await self._load(session)
        if not state.key_hash or not secrets.compare_digest(state.key_hash, _hash(run_key)):
            raise RunnerError("Unknown or expired run.")
        return state

    async def on_register(
        self, session: AsyncSession, registry: LLMEndpointRegistry, run_key: str, endpoint: str
    ) -> None:
        async with self._lock:
            state = await self._verify(session, run_key)
            if not is_valid_endpoint(endpoint):
                raise RunnerError("Invalid endpoint.")
            await registry.save(session, endpoint)
            now = time.time()
            state.endpoint = endpoint
            state.registered_at = state.registered_at or now
            state.last_heartbeat_at = now
            if state.status == "starting":
                state.status, state.message = "online", "N-ATLaS is answering questions."
            await self._save(session, state)

    async def on_heartbeat(
        self, session: AsyncSession, registry: LLMEndpointRegistry, run_key: str, endpoint: str
    ) -> Literal["continue", "stop"]:
        async with self._lock:
            state = await self._verify(session, run_key)
            state.last_heartbeat_at = time.time()
            if state.status == "starting":
                state.status, state.message = "online", "N-ATLaS is answering questions."
            if state.status != "stopping" and is_valid_endpoint(endpoint):
                current = await registry.refresh(session, force=True)
                if current.endpoint != endpoint:  # e.g. the backend's DB was reset
                    await registry.save(session, endpoint)
                state.endpoint = endpoint
            await self._save(session, state)
            return "stop" if state.status == "stopping" else "continue"

    async def on_stopped(
        self, session: AsyncSession, registry: LLMEndpointRegistry, run_key: str
    ) -> None:
        async with self._lock:
            state = await self._verify(session, run_key)
            await self._finish(session, registry, state, "offline", "N-ATLaS shut down.")

    async def _finish(
        self,
        session: AsyncSession,
        registry: LLMEndpointRegistry,
        state: RunState,
        status: Status,
        message: str,
    ) -> None:
        state.status, state.ended_at, state.message = status, time.time(), message
        state.key_hash = None
        await self._record(session, state)
        # Forget the dead link so requests fail fast instead of waiting for a timeout
        current = await registry.refresh(session, force=True)
        if state.endpoint and current.endpoint == state.endpoint:
            await registry.clear(session)
        await self._save(session, state)
        logger.info("N-ATLaS run ended", run_id=state.run_id, status=status, message=message)

    # ── background ───────────────────────────────────────────────────────

    async def reconcile(self, session: AsyncSession, registry: LLMEndpointRegistry | None = None) -> None:
        """Catch runs that ended without telling us (Kaggle limit, crash, cancel)."""
        state = await self._load(session)
        if state.status not in ACTIVE or not get_settings().KAGGLE_NOTEBOOK_ID:
            return
        if time.time() - self._last_kaggle_check < 30:
            return
        self._last_kaggle_check = time.time()
        try:
            kaggle_status = await asyncio.to_thread(
                self._kaggle.status, get_settings().KAGGLE_NOTEBOOK_ID or ""
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read Kaggle status", error=str(exc))
            return
        async with self._lock:
            state = await self._load(session)
            if state.status not in ACTIVE:
                return
            state.kaggle_status = kaggle_status
            # Ignore Kaggle's previous run's final status during the first minutes
            just_started = time.time() - (state.started_at or 0) < 180
            ended = kaggle_status in ("complete", "error", "cancelrequested", "cancelacknowledged")
            if ended and not just_started and registry is not None:
                failed = kaggle_status == "error" or state.status == "starting"
                await self._finish(
                    session,
                    registry,
                    state,
                    "failed" if failed else "offline",
                    f"Kaggle session ended ({kaggle_status}).",
                )
            else:
                await self._save(session, state)

    async def tick(self, session: AsyncSession, registry: LLMEndpointRegistry) -> None:
        """Run once a minute: reconcile with Kaggle and apply the schedule."""
        await self.reconcile(session, registry)
        schedule = await self.get_schedule(session)
        if not schedule.enabled:
            return
        now = datetime.now(timezone.utc)
        window = schedule.window(now)
        state = await self._load(session)

        if window and state.status not in ACTIVE:
            window_start, window_end = window
            stopped_by_hand = (
                state.last_manual_stop_at is not None
                and state.last_manual_stop_at >= window_start.timestamp()
            )
            failed_this_window = (
                state.status == "failed"
                and state.ended_at is not None
                and state.ended_at >= window_start.timestamp()
            )
            if not stopped_by_hand and not failed_this_window:
                hours = max(0.5, (window_end - now).total_seconds() / 3600 + 0.1)
                try:
                    await self.start(session, hours=hours, trigger="schedule")
                except RunnerError as exc:
                    logger.warning("Scheduled N-ATLaS start skipped", reason=str(exc))
        elif not window and state.status in ("starting", "online") and state.trigger == "schedule":
            await self.stop(session, trigger="schedule")


async def run_scheduler(app_state: Any, session_factory: Any, interval: float = 60) -> None:
    """Background task started in the app lifespan."""
    while True:
        try:
            async with session_factory() as session:
                await app_state.natlas_runner.tick(session, app_state.model_manager.llm_endpoint_registry)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("N-ATLaS scheduler tick failed", error=str(exc))
        await asyncio.sleep(interval)


__all__ = [
    "ACTIVE",
    "KaggleApiNotebooks",
    "NatlasRunner",
    "RunnerError",
    "RunState",
    "Schedule",
    "run_scheduler",
]

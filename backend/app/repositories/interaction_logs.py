"""InteractionLog repository for observability and evaluation."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.interaction_log import InteractionLog


class InteractionLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        conversation_id: uuid.UUID | None,
        language: str,
        input_type: str = "voice",
        asr_model: str | None = None,
        llm_model: str | None = None,
        processing_time_ms: int | None = None,
        success: bool = True,
        error_type: str | None = None,
        error_message: str | None = None,
        prompt: str | None = None,
        response: str | None = None,
        language_source: str | None = None,
        language_confidence: float | None = None,
        llm_available: bool | None = None,
        detection_ms: int | None = None,
        asr_ms: int | None = None,
        llm_ms: int | None = None,
        request_id: str | None = None,
    ) -> InteractionLog:
        log = InteractionLog(
            conversation_id=conversation_id,
            language=language,
            input_type=input_type,
            asr_model=asr_model,
            llm_model=llm_model,
            processing_time_ms=processing_time_ms,
            success=success,
            error_type=error_type,
            error_message=error_message,
            prompt=prompt,
            response=response,
            language_source=language_source,
            language_confidence=language_confidence,
            llm_available=llm_available,
            detection_ms=detection_ms,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
            request_id=request_id,
        )
        self.session.add(log)
        await self.session.flush()
        return log

    async def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,  # "ok" | "error"
        language: str | None = None,
        input_type: str | None = None,
        search: str | None = None,
    ) -> tuple[list[InteractionLog], int]:
        """Newest first, with the total count matching the filters."""
        filters: list[ColumnElement[bool]] = []
        if status == "ok":
            filters.append(InteractionLog.success.is_(True))
        elif status == "error":
            filters.append(InteractionLog.success.is_(False))
        if language:
            filters.append(InteractionLog.language == language)
        if input_type:
            filters.append(InteractionLog.input_type == input_type)
        if search:
            like = f"%{search}%"
            filters.append(InteractionLog.prompt.ilike(like) | InteractionLog.response.ilike(like))

        total = await self.session.scalar(
            select(func.count()).select_from(InteractionLog).where(*filters)
        )
        rows = await self.session.scalars(
            select(InteractionLog)
            .where(*filters)
            .order_by(InteractionLog.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows), total or 0

    async def stats(self, since: datetime) -> dict[str, Any]:
        """Request counts, error rate, latency and per-language totals since *since*."""
        window = InteractionLog.created_at >= since
        row = (
            await self.session.execute(
                select(
                    func.count(),
                    # Real failures only; "N-ATLaS offline" is counted separately below
                    func.sum(
                        case(
                            (
                                InteractionLog.success.is_(False)
                                & (InteractionLog.error_type.is_distinct_from("LLM_UNAVAILABLE")),
                                1,
                            ),
                            else_=0,
                        )
                    ),
                    func.sum(case((InteractionLog.llm_available.is_(False), 1), else_=0)),
                    func.avg(
                        case((InteractionLog.success.is_(True), InteractionLog.processing_time_ms))
                    ),
                    func.avg(InteractionLog.detection_ms),
                    func.avg(InteractionLog.asr_ms),
                    func.avg(InteractionLog.llm_ms),
                ).where(window)
            )
        ).one()
        total, errors, llm_offline, avg_ms, avg_det, avg_asr, avg_llm = row

        by_language = {
            lang: count
            for lang, count in (
                await self.session.execute(
                    select(InteractionLog.language, func.count())
                    .where(window)
                    .group_by(InteractionLog.language)
                )
            ).all()
        }
        by_error = {
            code: count
            for code, count in (
                await self.session.execute(
                    select(InteractionLog.error_type, func.count())
                    .where(window, InteractionLog.error_type.is_not(None))
                    .group_by(InteractionLog.error_type)
                )
            ).all()
        }

        def _round(value: Any) -> int | None:
            return round(value) if value is not None else None

        return {
            "requests": total or 0,
            "errors": int(errors or 0),
            "llm_offline": int(llm_offline or 0),
            "avg_processing_ms": _round(avg_ms),
            "avg_detection_ms": _round(avg_det),
            "avg_asr_ms": _round(avg_asr),
            "avg_llm_ms": _round(avg_llm),
            "by_language": by_language,
            "by_error": by_error,
        }

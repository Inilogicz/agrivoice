"""
Text chat route (non-voice path).

Accepts plain text in one of the supported languages and returns
an agricultural advisory response from N-ATLaS.

This endpoint is useful for:
  - Frontend fallback when microphone is unavailable
  - Testing LLM responses without ASR
  - Debugging language-specific response quality
"""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_conversation_repo,
    get_interaction_log_repo,
    get_llm_service,
    get_message_repo,
)
from app.core.exceptions import AgriVoiceError, to_http_exception
from app.core.logging import get_logger, get_request_id
from app.db.database import get_db
from app.repositories.conversations import ConversationRepository, MessageRepository
from app.repositories.interaction_logs import InteractionLogRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Text-based chat with N-ATLaS",
    description="Send a text message in one of the supported languages and receive an agricultural response.",
)
async def chat(
    request: ChatRequest,
    llm_service: LLMService = Depends(get_llm_service),
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
    msg_repo: MessageRepository = Depends(get_message_repo),
    log_repo: InteractionLogRepository = Depends(get_interaction_log_repo),
    session: AsyncSession = Depends(get_db),
) -> ChatResponse:
    t0 = time.time()

    async def log_failure(error_type: str, message: str) -> None:
        """Record a failed request for the admin activity log; never masks the error."""
        try:
            await session.rollback()
            await log_repo.create(
                conversation_id=None,
                language=request.language,
                input_type="text",
                processing_time_ms=round((time.time() - t0) * 1000),
                success=False,
                error_type=error_type,
                error_message=message[:2000],
                prompt=request.message,
                language_source="user_selected",
                llm_available=False if error_type == "LLM_UNAVAILABLE" else None,
                request_id=get_request_id(),
            )
            await session.commit()
        except Exception as log_exc:  # noqa: BLE001
            logger.warning("Could not record failed interaction", error=str(log_exc))

    try:
        # Get or create conversation
        conversation = await conv_repo.get_or_create(
            conversation_id=request.conversation_id,
            language=request.language,
        )

        # Build history from prior messages
        prior = await conv_repo.get_by_id(conversation.id)
        history: list[dict[str, str]] | None = None
        if prior and prior.messages:
            history = [
                {"role": msg.role, "content": msg.content}
                for msg in sorted(prior.messages, key=lambda m: m.created_at)
            ]

        # Generate
        result = await llm_service.generate_response(
            user_message=request.message,
            language=request.language,  # type: ignore[arg-type]
            conversation_history=history,
        )

        elapsed = round((time.time() - t0) * 1000)

        # Persist
        await msg_repo.create(
            conversation_id=conversation.id,
            role="user",
            content=request.message,
            language=request.language,
            input_type="text",
        )
        assistant_msg = await msg_repo.create(
            conversation_id=conversation.id,
            role="assistant",
            content=result.text,
            language=request.language,
            input_type="text",
            model_used=result.llm_model,
        )
        await log_repo.create(
            conversation_id=conversation.id,
            language=request.language,
            input_type="text",
            llm_model=result.llm_model,
            processing_time_ms=elapsed,
            success=True,
            prompt=request.message,
            response=result.text,
            language_source="user_selected",
            llm_available=True,
            llm_ms=result.processing_time_ms,
            request_id=get_request_id(),
        )
        await session.commit()

        return ChatResponse(
            conversation_id=conversation.id,
            message_id=assistant_msg.id,
            language=request.language,
            response=result.text,
            llm_model=result.llm_model,
            processing_time_ms=elapsed,
        )

    except AgriVoiceError as exc:
        await log_failure(exc.error_code, exc.message)
        raise to_http_exception(exc) from exc
    except HTTPException as exc:
        await log_failure("HTTP_" + str(exc.status_code), str(exc.detail))
        raise
    except Exception as exc:
        logger.error("Unexpected error in chat", error=str(exc), exc_info=True)
        await log_failure("GENERATION_ERROR", f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "GENERATION_ERROR", "message": "An internal error occurred."},
        ) from exc

"""Translate a message (usually an answer) into another supported language."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_interaction_log_repo, get_llm_service
from app.core.exceptions import (
    AgriVoiceError,
    AlreadyInLanguageError,
    MessageNotFoundError,
    UnsupportedLanguageError,
    to_http_exception,
)
from app.core.logging import get_logger, get_request_id
from app.db.database import get_db
from app.db.models.message import Message
from app.models.languages import SUPPORTED_LANGUAGES, parse_language
from app.repositories.interaction_logs import InteractionLogRepository
from app.repositories.translations import TranslationRepository
from app.schemas.translation import TranslateRequest, TranslateResponse
from app.services.llm_service import LLMService

router = APIRouter(prefix="/translate", tags=["translate"])
logger = get_logger(__name__)


@router.post(
    "",
    response_model=TranslateResponse,
    summary="Translate a message",
    description=(
        "Translate a saved message (e.g. an answer, via its `message_id`) into another "
        "supported language with N-ATLaS: Yoruba, Hausa, Igbo or Nigerian English. "
        "Translations are saved, so asking again returns instantly (`cached: true`)."
    ),
)
async def translate(
    request: TranslateRequest,
    llm_service: LLMService = Depends(get_llm_service),
    log_repo: InteractionLogRepository = Depends(get_interaction_log_repo),
    session: AsyncSession = Depends(get_db),
) -> TranslateResponse:
    t0 = time.time()
    source_text: str | None = None
    target_code = request.target_language

    async def log(*, success: bool, error_type: str | None = None, message: str | None = None,
                  response: str | None = None, conversation_id=None, llm_ms: int | None = None) -> None:
        try:
            if not success:
                await session.rollback()
            await log_repo.create(
                conversation_id=conversation_id,
                language=target_code,
                input_type="translation",
                llm_model=llm_service.model_id if success else None,
                processing_time_ms=round((time.time() - t0) * 1000),
                success=success,
                error_type=error_type,
                error_message=message[:2000] if message else None,
                prompt=source_text,
                response=response,
                language_source="user_selected",
                llm_available=False if error_type == "LLM_UNAVAILABLE" else (True if success else None),
                llm_ms=llm_ms,
                request_id=get_request_id(),
            )
            await session.commit()
        except Exception as log_exc:  # noqa: BLE001
            logger.warning("Could not record translation", error=str(log_exc))

    try:
        target = parse_language(request.target_language)
        if target is None:
            raise UnsupportedLanguageError(request.target_language)
        target_code = target

        message = await session.get(Message, request.message_id)
        if message is None:
            raise MessageNotFoundError(str(request.message_id))
        source_text = message.content
        source = parse_language(message.language) or "en-ng"
        if source == target:
            raise AlreadyInLanguageError(SUPPORTED_LANGUAGES[target].name)

        translations = TranslationRepository(session)
        cached = await translations.get(message.id, target)
        if cached is not None:
            return TranslateResponse(
                message_id=message.id,
                source_language=source,
                target_language=target,
                target_language_name=SUPPORTED_LANGUAGES[target].name,
                text=cached.content,
                llm_model=cached.model_used,
                cached=True,
                processing_time_ms=round((time.time() - t0) * 1000),
            )

        llm_ms = 0
        source_for_target, text_for_target = source, message.content
        if source != "en-ng" and target != "en-ng":
            # Between two Nigerian languages, go through English: N-ATLaS mixed
            # Hausa words into a direct Hausa→Yoruba translation, but translated
            # Hausa→English and English→Yoruba cleanly. The English step is saved too.
            english = await translations.get(message.id, "en-ng")
            if english is None:
                step = await llm_service.translate(message.content, source, "en-ng")
                llm_ms += step.processing_time_ms
                english = await translations.create(
                    message_id=message.id,
                    language="en-ng",
                    content=step.text.strip(),
                    model_used=step.llm_model,
                )
                await session.commit()
            source_for_target, text_for_target = "en-ng", english.content

        result = await llm_service.translate(text_for_target, source_for_target, target)
        llm_ms += result.processing_time_ms
        text = result.text.strip()
        await translations.create(
            message_id=message.id, language=target, content=text, model_used=result.llm_model
        )
        await session.commit()
        await log(
            success=True,
            response=text,
            conversation_id=message.conversation_id,
            llm_ms=llm_ms,
        )
        return TranslateResponse(
            message_id=message.id,
            source_language=source,
            target_language=target,
            target_language_name=SUPPORTED_LANGUAGES[target].name,
            text=text,
            llm_model=result.llm_model,
            cached=False,
            processing_time_ms=round((time.time() - t0) * 1000),
        )

    except AgriVoiceError as exc:
        await log(success=False, error_type=exc.error_code, message=exc.message)
        raise to_http_exception(exc) from exc
    except Exception as exc:
        logger.error("Unexpected error in translate", error=str(exc), exc_info=True)
        await log(success=False, error_type="TRANSLATION_ERROR", message=f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "TRANSLATION_ERROR", "message": "An internal error occurred."},
        ) from exc

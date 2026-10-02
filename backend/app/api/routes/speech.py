"""Spoken answers, and which optional features this server currently offers."""
from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_interaction_log_repo, get_model_manager, refresh_runtime_state
from app.core.config import get_settings
from app.core.exceptions import (
    AgriVoiceError,
    MessageNotFoundError,
    SpeechDisabledError,
    TranslationNotFoundError,
    UnsupportedLanguageError,
    to_http_exception,
)
from app.core.logging import get_logger, get_request_id
from app.db.database import get_db
from app.db.models.message import Message
from app.models.languages import SUPPORTED_LANGUAGES, parse_language
from app.repositories.interaction_logs import InteractionLogRepository
from app.repositories.translations import TranslationRepository
from app.services.model_manager import ModelManager
from app.services.speech_service import SpeechService

router = APIRouter(tags=["speech"])
logger = get_logger(__name__)


def _speech_on(manager: ModelManager) -> bool:
    return get_settings().TTS_ENABLED and manager.get_speech_synthesizer() is not None


@router.get(
    "/features",
    summary="Optional features available now",
    description="Use `speech` to decide whether to show a Listen button.",
    dependencies=[Depends(refresh_runtime_state)],
)
async def features(manager: ModelManager = Depends(get_model_manager)) -> dict:
    return {"speech": _speech_on(manager), "translation": True}


@router.get(
    "/messages/{message_id}/audio",
    summary="Listen to a message",
    description=(
        "MP3 of a message read aloud (YarnGPT2 Nigerian voices). Use it directly as "
        "`<audio src>`. Without `language` it reads the message as written; with a "
        "different language it reads the saved translation (translate it first). The "
        "first request generates the audio (~30–75 s for a full answer); later ones "
        "return it in about a second."
    ),
    response_class=FileResponse,
    responses={200: {"content": {"audio/mpeg": {}}}},
    dependencies=[Depends(refresh_runtime_state)],
)
async def message_audio(
    message_id: uuid.UUID,
    language: str | None = Query(default=None, description="yo, ha, ig or en-ng"),
    manager: ModelManager = Depends(get_model_manager),
    log_repo: InteractionLogRepository = Depends(get_interaction_log_repo),
    session: AsyncSession = Depends(get_db),
) -> FileResponse:
    t0 = time.time()
    text: str | None = None
    lang_code = language or "unknown"

    async def log(*, success: bool, error_type: str | None = None, message: str | None = None,
                  conversation_id: uuid.UUID | None = None, ms: int | None = None) -> None:
        try:
            if not success:
                await session.rollback()
            await log_repo.create(
                conversation_id=conversation_id,
                language=lang_code,
                input_type="speech",
                llm_model=synth.model_id if synth else None,
                processing_time_ms=round((time.time() - t0) * 1000),
                success=success,
                error_type=error_type,
                error_message=message[:2000] if message else None,
                prompt=text,
                llm_ms=ms,
                request_id=get_request_id(),
            )
            await session.commit()
        except Exception as log_exc:  # noqa: BLE001
            logger.warning("Could not record speech request", error=str(log_exc))

    synth = manager.get_speech_synthesizer()
    try:
        if not _speech_on(manager) or synth is None:
            raise SpeechDisabledError()

        message = await session.get(Message, message_id)
        if message is None:
            raise MessageNotFoundError(str(message_id))
        source = parse_language(message.language) or "en-ng"
        target = source
        if language:
            parsed = parse_language(language)
            if parsed is None:
                raise UnsupportedLanguageError(language)
            target = parsed
        lang_code = target

        if target == source:
            text = message.content
        else:
            translation = await TranslationRepository(session).get(message.id, target)
            if translation is None:
                raise TranslationNotFoundError(SUPPORTED_LANGUAGES[target].name)
            text = translation.content

        audio = await SpeechService(synth).audio_for(message.id, target, text)
        if not audio.cached:
            await log(success=True, conversation_id=message.conversation_id, ms=audio.generation_ms)
        return FileResponse(
            audio.path,
            media_type="audio/mpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except AgriVoiceError as exc:
        if exc.error_code != "SPEECH_DISABLED":
            await log(success=False, error_type=exc.error_code, message=exc.message)
        raise to_http_exception(exc) from exc
    except Exception as exc:
        logger.error("Unexpected error in message_audio", error=str(exc), exc_info=True)
        await log(success=False, error_type="SPEECH_ERROR", message=f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "SPEECH_ERROR", "message": "An internal error occurred."},
        ) from exc

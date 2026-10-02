"""
Voice pipeline API route.

Accepts an audio file upload, runs the full voice pipeline,
persists conversation/message/interaction log records,
and returns transcript + response.
"""
from __future__ import annotations

import dataclasses
import time
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import (
    get_conversation_repo,
    get_interaction_log_repo,
    get_message_repo,
    get_voice_pipeline,
)
from app.core.exceptions import AgriVoiceError, UnsupportedLanguageError, to_http_exception
from app.core.logging import get_logger, get_request_id
from app.db.database import get_db
from app.models.base import SupportedLanguage
from app.models.languages import parse_language
from app.repositories.conversations import ConversationRepository, MessageRepository
from app.repositories.interaction_logs import InteractionLogRepository
from app.schemas.voice import VoiceAskResponse
from app.services.voice_pipeline import VoicePipeline
from app.utils.audio import create_temp_wav, validate_audio_file
from app.utils.cleanup import safe_delete

router = APIRouter(prefix="/voice", tags=["voice"])
logger = get_logger(__name__)


@router.post(
    "/ask",
    response_model=VoiceAskResponse,
    summary="Submit a voice query",
    description=(
        "Upload an audio file. If `language` is omitted, the system auto-detects it "
        "(Yoruba, Hausa, Igbo, Nigerian English). If detection isn't confident, it "
        "returns 422 `LANGUAGE_NOT_DETECTED` with `available_languages`; resend the "
        "audio with `language` set to the user's choice. The speech is then "
        "transcribed and answered in that language."
    ),
    status_code=status.HTTP_200_OK,
)
async def voice_ask(
    audio: UploadFile = File(..., description="Audio file (WAV/WebM/MP3/M4A/OGG, max 25 MB)"),
    conversation_id: str | None = Form(default=None, description="Optional existing conversation ID"),
    language: str | None = Form(
        default=None,
        description="Language the user chose (yo/ha/ig/en-ng). Omit to auto-detect.",
    ),
    pipeline: VoicePipeline = Depends(get_voice_pipeline),
    conv_repo: ConversationRepository = Depends(get_conversation_repo),
    msg_repo: MessageRepository = Depends(get_message_repo),
    log_repo: InteractionLogRepository = Depends(get_interaction_log_repo),
    session: AsyncSession = Depends(get_db),
) -> VoiceAskResponse:
    """Full voice pipeline: detect → transcribe → generate → persist."""
    started = time.time()
    wav_path: str | None = None

    async def log_failure(error_type: str, message: str) -> None:
        """Record a failed request for the admin activity log; never masks the error."""
        try:
            await session.rollback()
            await log_repo.create(
                conversation_id=None,
                language=language or "unknown",
                input_type="voice",
                processing_time_ms=round((time.time() - started) * 1000),
                success=False,
                error_type=error_type,
                error_message=message[:2000],
                language_source="user_selected" if language else None,
                request_id=get_request_id(),
            )
            await session.commit()
        except Exception as log_exc:  # noqa: BLE001
            logger.warning("Could not record failed interaction", error=str(log_exc))

    try:
        selected_language: SupportedLanguage | None = None
        if language:
            selected_language = parse_language(language)
            if selected_language is None:
                raise UnsupportedLanguageError(language)

        # Validate upload
        content = await audio.read()
        validate_audio_file(
            file_path=audio.filename or "",
            content_type=audio.content_type or "",
            file_size=len(content),
        )

        # Get file extension for temp file naming
        ext = "." + (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()

        # Convert to 16 kHz WAV
        wav_path = create_temp_wav(content, suffix=ext)

        # Parse conversation ID
        conv_uuid: uuid.UUID | None = None
        if conversation_id:
            try:
                conv_uuid = uuid.UUID(conversation_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Invalid conversation_id format. Must be a valid UUID.",
                )

        # Fetch prior conversation history for multi-turn context
        history: list[dict[str, str]] | None = None
        if conv_uuid:
            existing_conv = await conv_repo.get_by_id(conv_uuid)
            if existing_conv and existing_conv.messages:
                history = [
                    {"role": msg.role, "content": msg.content}
                    for msg in sorted(existing_conv.messages, key=lambda m: m.created_at)
                ]

        # Run voice pipeline
        result = await pipeline.process(
            wav_path,
            language=selected_language,
            conversation_history=history,
        )

        # Persist conversation + messages
        conversation = await conv_repo.get_or_create(
            conversation_id=conv_uuid,
            language=result.language,
        )
        await msg_repo.create(
            conversation_id=conversation.id,
            role="user",
            content=result.transcript,
            language=result.language,
            input_type="voice",
            model_used=result.asr_model,
        )
        assistant_msg = None
        if result.llm_available:
            assistant_msg = await msg_repo.create(
                conversation_id=conversation.id,
                role="assistant",
                content=result.response,
                language=result.language,
                input_type="voice",
                model_used=result.llm_model,
            )
        await log_repo.create(
            conversation_id=conversation.id,
            language=result.language,
            input_type="voice",
            asr_model=result.asr_model,
            llm_model=result.llm_model,
            processing_time_ms=result.processing_time_ms,
            success=result.llm_available,
            error_type=None if result.llm_available else "LLM_UNAVAILABLE",
            prompt=result.transcript,
            response=result.response if result.llm_available else None,
            language_source=result.language_source,
            language_confidence=result.language_confidence,
            llm_available=result.llm_available,
            detection_ms=result.detection_ms,
            asr_ms=result.asr_ms,
            llm_ms=result.llm_ms,
            request_id=get_request_id(),
        )
        await session.commit()

        response_fields = dataclasses.asdict(result)
        for internal in ("detection_ms", "asr_ms", "llm_ms"):
            response_fields.pop(internal)
        return VoiceAskResponse(
            conversation_id=conversation.id,
            message_id=assistant_msg.id if assistant_msg else None,
            **response_fields,
        )

    except AgriVoiceError as exc:
        await log_failure(exc.error_code, exc.message)
        raise to_http_exception(exc) from exc
    except HTTPException as exc:
        await log_failure("HTTP_" + str(exc.status_code), str(exc.detail))
        raise
    except Exception as exc:
        logger.error("Unexpected error in voice_ask", error=str(exc), exc_info=True)
        await log_failure("PIPELINE_ERROR", f"{type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "PIPELINE_ERROR", "message": "An internal error occurred."},
        ) from exc
    finally:
        # Always clean up temp audio
        if wav_path:
            safe_delete(wav_path)

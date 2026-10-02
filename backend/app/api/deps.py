"""
Dependency injection factories for FastAPI routes.

All service objects are built here, sourcing the ModelManager from
app.state. Routes import from this module, never from services directly.
"""
from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.repositories.conversations import ConversationRepository, MessageRepository
from app.repositories.feedback import FeedbackRepository
from app.repositories.interaction_logs import InteractionLogRepository
from app.services.language_detection import LanguageDetectionService
from app.services.language_router import LanguageRouter
from app.services.llm_service import LLMService
from app.services.model_manager import ModelManager
from app.services.prompt_service import PromptService
from app.services.voice_pipeline import VoicePipeline


def get_model_manager(request: Request) -> ModelManager:
    return request.app.state.model_manager


async def refresh_runtime_state(
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    session: AsyncSession = Depends(get_db),
) -> None:
    """Pick up settings and N-ATLaS endpoint changes made from the admin dashboard."""
    await request.app.state.runtime_settings.refresh(session)
    if manager.get_remote_llm() is not None:
        await manager.llm_endpoint_registry.refresh(session)


def get_voice_pipeline(
    request: Request,
    manager: ModelManager = Depends(get_model_manager),
    _: None = Depends(refresh_runtime_state),
) -> VoicePipeline:
    detector = manager.get_language_detector()
    llm_adapter = manager.get_llm_adapter()
    return VoicePipeline(
        language_detection=LanguageDetectionService(detector),
        language_router=LanguageRouter(manager),
        llm_service=LLMService(llm_adapter, PromptService()),
    )


def get_llm_service(
    manager: ModelManager = Depends(get_model_manager),
    _: None = Depends(refresh_runtime_state),
) -> LLMService:
    return LLMService(manager.get_llm_adapter(), PromptService())


def get_conversation_repo(session: AsyncSession = Depends(get_db)) -> ConversationRepository:
    return ConversationRepository(session)


def get_message_repo(session: AsyncSession = Depends(get_db)) -> MessageRepository:
    return MessageRepository(session)


def get_feedback_repo(session: AsyncSession = Depends(get_db)) -> FeedbackRepository:
    return FeedbackRepository(session)


def get_interaction_log_repo(
    session: AsyncSession = Depends(get_db),
) -> InteractionLogRepository:
    return InteractionLogRepository(session)

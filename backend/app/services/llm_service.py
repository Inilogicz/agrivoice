"""
LLM service: wraps the LLM adapter with prompt construction and logging.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.base import BaseLLMAdapter, SupportedLanguage
from app.schemas.voice import GenerationResult
from app.services.prompt_service import PromptService

logger = get_logger(__name__)


class LLMService:
    """Orchestrates language model generation with agricultural context."""

    def __init__(self, llm_adapter: BaseLLMAdapter, prompt_service: PromptService) -> None:
        self._adapter = llm_adapter
        self._prompt_service = prompt_service

    @property
    def model_id(self) -> str:
        return self._adapter.model_id

    async def generate_response(
        self,
        user_message: str,
        language: SupportedLanguage,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> GenerationResult:
        """
        Generate an agricultural advisory response from N-ATLaS.

        Args:
            user_message: User's input text (in their language).
            language: Confirmed language code.
            conversation_history: Prior turns for multi-turn context.

        Returns:
            GenerationResult with the response text.
        """
        messages = self._prompt_service.build_chat_messages(
            user_message=user_message,
            language=language,
            conversation_history=conversation_history,
        )

        logger.info(
            "Generating response from N-ATLaS",
            model=self._adapter.model_id,
            language=language,
            turn_count=len(messages),
        )

        result = await self._adapter.generate(messages=messages, language=language)

        logger.info(
            "LLM generation complete",
            model=self._adapter.model_id,
            duration_ms=result.processing_time_ms,
        )
        return result

    async def translate(
        self,
        text: str,
        source: SupportedLanguage,
        target: SupportedLanguage,
    ) -> GenerationResult:
        """Translate *text* between two supported languages with N-ATLaS."""
        messages = self._prompt_service.build_translation_messages(text, source, target)
        # Translations can run longer than the original (e.g. Yoruba tone marks
        # cost extra tokens); the N-ATLaS app caps generation at 1000 tokens.
        limit = min(1000, max(600, get_settings().LLM_MAX_NEW_TOKENS * 2))
        logger.info("Translating with N-ATLaS", source=source, target=target, chars=len(text))
        return await self._adapter.generate(messages=messages, language=target, max_new_tokens=limit)

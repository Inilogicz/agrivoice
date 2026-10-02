"""
Language router: maps a detected language code to the correct ASR adapter.

This is a pure routing table — no ML, no inference.
"""
from __future__ import annotations

from app.core.exceptions import UnsupportedLanguageError
from app.models.base import BaseASRAdapter, SupportedLanguage
from app.services.model_manager import ModelManager

SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset({"yo", "ha", "ig", "en-ng"})


class LanguageRouter:
    """Routes a detected language to the appropriate ASR model."""

    def __init__(self, model_manager: ModelManager) -> None:
        self._manager = model_manager

    def route(self, language: str) -> BaseASRAdapter:
        """
        Return the ASR adapter for the given language code.

        Args:
            language: One of "yo", "ha", "ig", "en-ng".

        Raises:
            UnsupportedLanguageError: If the language is not in the supported set.
        """
        if language not in SUPPORTED_LANGUAGE_CODES:
            raise UnsupportedLanguageError(language)

        lang: SupportedLanguage = language  # type: ignore[assignment]
        return self._manager.get_asr_adapter(lang)

"""
Language resolution service.

Decides which NCAIR language a voice request is in: the user's explicit
choice wins; otherwise the audio is auto-detected. If detection isn't
confident, raises LanguageNotDetectedError so the client can ask the user
to pick from the supported languages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from app.core.exceptions import LanguageNotDetectedError
from app.core.logging import get_logger
from app.models.base import BaseLanguageDetector, SupportedLanguage
from app.models.languages import SUPPORTED_LANGUAGES

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedLanguage:
    language: SupportedLanguage
    confidence: float
    source: Literal["detected", "user_selected"]


class LanguageDetectionService:
    """Resolves the request language from user choice or auto-detection."""

    def __init__(self, detector: BaseLanguageDetector) -> None:
        self._detector = detector

    async def resolve(
        self,
        audio_path: str,
        language: SupportedLanguage | None = None,
    ) -> ResolvedLanguage:
        """
        Args:
            audio_path: Path to preprocessed 16 kHz WAV.
            language: Language the user explicitly chose, if any.

        Raises:
            LanguageNotDetectedError: No user choice and detection wasn't confident.
        """
        if language is not None:
            logger.info("Using user-selected language", language=language)
            return ResolvedLanguage(language=language, confidence=1.0, source="user_selected")

        result = await self._detector.detect(audio_path)
        logger.info(
            "Language detection result",
            language=result.language,
            detected_label=result.detected_label,
            confidence=result.confidence,
        )

        if result.language is None:
            raise LanguageNotDetectedError(
                available_languages=[asdict(info) for info in SUPPORTED_LANGUAGES.values()]
            )

        return ResolvedLanguage(
            language=result.language, confidence=result.confidence, source="detected"
        )

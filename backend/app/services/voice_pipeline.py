"""
Voice pipeline service: orchestrates the full audio → response flow.

Pipeline:
  1. Validate and preprocess audio
  2. Detect language (or apply override)
  3. Route to correct ASR model
  4. Transcribe to text
  5. Generate response via N-ATLaS LLM (with agricultural system prompt)
  6. Return structured result

This service is the single point of orchestration. Route handlers call
only this service — they never touch individual model adapters directly.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.core.exceptions import LLMUnavailableError
from app.core.logging import get_logger
from app.models.base import SupportedLanguage
from app.models.languages import SUPPORTED_LANGUAGES
from app.services.asr_service import ASRService
from app.services.language_detection import LanguageDetectionService
from app.services.language_router import LanguageRouter
from app.services.llm_service import LLMService

logger = get_logger(__name__)


@dataclass(frozen=True)
class VoicePipelineResult:
    """Pipeline output. The caller attaches a conversation ID after persisting."""

    language: SupportedLanguage
    language_name: str
    language_source: Literal["detected", "user_selected"]
    language_confidence: float
    transcript: str
    response: str
    asr_model: str
    llm_model: str
    processing_time_ms: int
    llm_available: bool = True
    # Per-step timings, for the admin activity log (not part of the API response)
    detection_ms: int | None = None
    asr_ms: int | None = None
    llm_ms: int | None = None


class VoicePipeline:
    """End-to-end voice processing pipeline."""

    def __init__(
        self,
        language_detection: LanguageDetectionService,
        language_router: LanguageRouter,
        llm_service: LLMService,
    ) -> None:
        self._detection = language_detection
        self._router = language_router
        self._llm = llm_service

    async def process(
        self,
        audio_path: str,
        *,
        language: SupportedLanguage | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> VoicePipelineResult:
        """
        Run the full voice pipeline on a preprocessed audio file.

        Args:
            audio_path: Path to 16 kHz WAV file.
            language: Language the user chose; None means auto-detect.
            conversation_history: Prior conversation turns for context.

        Returns:
            VoicePipelineResult with transcript, response, and metadata.
        """
        pipeline_start = time.time()

        # Step 1: Resolve language (user choice, else auto-detect)
        resolved = await self._detection.resolve(audio_path, language=language)
        language = resolved.language
        detection_ms = round((time.time() - pipeline_start) * 1000)

        # Step 2: Route to appropriate ASR model
        asr_adapter = self._router.route(language)
        asr_service = ASRService(asr_adapter=asr_adapter)

        # Step 3: Transcription
        asr_start = time.time()
        transcription = await asr_service.transcribe(audio_path)
        asr_ms = round((time.time() - asr_start) * 1000)

        # Step 4: LLM generation (same language, agricultural context).
        # If N-ATLaS is offline, still return the transcript with a notice.
        llm_available = True
        llm_start = time.time()
        try:
            generation = await self._llm.generate_response(
                user_message=transcription.text,
                language=language,
                conversation_history=conversation_history,
            )
            response_text, llm_model = generation.text, generation.llm_model
        except LLMUnavailableError as exc:
            llm_available = False
            response_text, llm_model = exc.message, self._llm.model_id
        llm_ms = round((time.time() - llm_start) * 1000)

        total_ms = round((time.time() - pipeline_start) * 1000)

        logger.info(
            "Voice pipeline complete",
            language=language,
            asr_model=asr_adapter.model_id,
            llm_model=llm_model,
            llm_available=llm_available,
            total_ms=total_ms,
        )

        return VoicePipelineResult(
            language=language,
            language_name=SUPPORTED_LANGUAGES[language].name,
            language_source=resolved.source,
            language_confidence=resolved.confidence,
            transcript=transcription.text,
            response=response_text,
            asr_model=asr_adapter.model_id,
            llm_model=llm_model,
            processing_time_ms=total_ms,
            llm_available=llm_available,
            detection_ms=detection_ms,
            asr_ms=asr_ms,
            llm_ms=llm_ms,
        )

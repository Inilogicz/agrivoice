"""
ASR service: wraps ASR adapter with logging and timing.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.base import BaseASRAdapter
from app.schemas.voice import TranscriptionResult

logger = get_logger(__name__)


class ASRService:
    """Orchestrates speech-to-text transcription."""

    def __init__(self, asr_adapter: BaseASRAdapter) -> None:
        self._adapter = asr_adapter
        self._settings = get_settings()

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio at *audio_path* using the configured ASR adapter.

        Args:
            audio_path: Path to a 16 kHz WAV file.

        Returns:
            TranscriptionResult with transcript text.
        """
        logger.info(
            "Starting ASR transcription",
            model=self._adapter.model_id,
            language=self._adapter.language,
        )

        result = await self._adapter.transcribe(audio_path)

        if self._settings.LOG_TRANSCRIPTS:
            # Only log transcript in development mode (PII risk)
            logger.debug("Transcription result", transcript=result.text)

        logger.info(
            "ASR transcription complete",
            model=self._adapter.model_id,
            duration_ms=result.processing_time_ms,
        )
        return result

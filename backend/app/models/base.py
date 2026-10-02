"""
Abstract base classes for ASR and LLM model adapters.

The HTTP layer depends only on these interfaces, never on Transformers directly.
Real and mock implementations both satisfy these contracts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from app.schemas.voice import GenerationResult, LanguageDetectionResult, TranscriptionResult

SupportedLanguage = Literal["yo", "ha", "ig", "en-ng"]


class BaseASRAdapter(ABC):
    """Contract for speech-to-text adapters."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the official HuggingFace model identifier."""

    @property
    @abstractmethod
    def language(self) -> SupportedLanguage:
        """Return the language code this adapter handles."""

    @abstractmethod
    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe audio at *audio_path* to text.

        Args:
            audio_path: Absolute path to a 16 kHz WAV file.

        Returns:
            TranscriptionResult with transcript text and timing info.
        """


class BaseLLMAdapter(ABC):
    """Contract for language model generation adapters."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the official HuggingFace model identifier."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        language: SupportedLanguage,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """
        Generate a response given a list of chat messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            language: The target language code for the response.
            max_new_tokens: Length limit; None uses LLM_MAX_NEW_TOKENS.

        Returns:
            GenerationResult with generated text and timing info.
        """


class BaseLanguageDetector(ABC):
    """Contract for automatic language detection adapters."""

    @abstractmethod
    async def detect(self, audio_path: str) -> LanguageDetectionResult:
        """
        Detect the spoken language in the given audio file.

        Args:
            audio_path: Absolute path to a 16 kHz WAV file.

        Returns:
            LanguageDetectionResult with language code and confidence score.
        """

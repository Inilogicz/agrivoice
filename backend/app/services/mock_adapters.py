"""
Mock model adapters for local development.

These adapters satisfy the same interfaces as the real N-ATLaS adapters but
produce deterministic placeholder outputs without loading any models.

Mock adapters are only loaded with DEV_USE_MOCK_MODELS=true or
LLM_PROVIDER=mock, ModelManager refuses them when ENVIRONMENT=production,
and every mock output is labelled "[MOCK]".
"""
from __future__ import annotations

import asyncio
import time

from app.models.base import BaseASRAdapter, BaseLLMAdapter, BaseLanguageDetector, SupportedLanguage
from app.schemas.voice import GenerationResult, LanguageDetectionResult, TranscriptionResult

_MOCK_TRANSCRIPTS: dict[SupportedLanguage, str] = {
    "yo": "[MOCK] Ẹ jẹ kí n sọ nipa àjàrí àgbàdo",
    "ha": "[MOCK] Ina son sanin yadda ake noman masara",
    "ig": "[MOCK] Achọrọ m ịmụta ọrụ ugbo mmiri",
    "en-ng": "[MOCK] I want to know about maize farming",
}

_MOCK_RESPONSES: dict[SupportedLanguage, str] = {
    "yo": (
        "[MOCK RESPONSE — not from N-ATLaS] "
        "Àgbàdo jẹ irúgbìn tó ṣe pàtàkì fún àwọn àgbẹ kékeré ní Nàìjíríà."
    ),
    "ha": (
        "[MOCK RESPONSE — not from N-ATLaS] "
        "Masara kayan gonaki ne masu muhimmanci ga manoman karamin gonaki a Najeriya."
    ),
    "ig": (
        "[MOCK RESPONSE — not from N-ATLaS] "
        "Ọka bụ ọchịchọ mkpụrụ ihe ọ bụla dị mkpa maka ndị ọrụ ugbo obere na Naịjirịa."
    ),
    "en-ng": (
        "[MOCK RESPONSE — not from N-ATLaS] "
        "Maize is one of the most important staple crops for smallholder farmers in Nigeria."
    ),
}


class MockASRAdapter(BaseASRAdapter):
    """Development-only ASR adapter."""

    def __init__(self, language: SupportedLanguage) -> None:
        self._language = language

    @property
    def model_id(self) -> str:
        return f"MOCK/ASR-{self._language}"

    @property
    def language(self) -> SupportedLanguage:
        return self._language

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        await asyncio.sleep(0.05)  # Simulate latency
        return TranscriptionResult(
            text=_MOCK_TRANSCRIPTS[self._language],
            language=self._language,
            asr_model=self.model_id,
            processing_time_ms=50,
        )


class MockLLMAdapter(BaseLLMAdapter):
    """Development-only LLM adapter."""

    @property
    def model_id(self) -> str:
        return "MOCK/N-ATLaS"

    async def generate(
        self,
        messages: list[dict[str, str]],
        language: SupportedLanguage,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        await asyncio.sleep(0.1)  # Simulate latency
        return GenerationResult(
            text=_MOCK_RESPONSES.get(language, _MOCK_RESPONSES["en-ng"]),
            language=language,
            llm_model=self.model_id,
            processing_time_ms=100,
        )


class MockLanguageDetector(BaseLanguageDetector):
    """Development-only language detector that always returns 'en-ng'."""

    async def detect(self, audio_path: str) -> LanguageDetectionResult:
        await asyncio.sleep(0.02)
        return LanguageDetectionResult(
            language="en-ng",
            confidence=0.99,
            is_fallback=False,
        )

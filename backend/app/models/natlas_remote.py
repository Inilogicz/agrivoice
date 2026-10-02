"""
N-ATLaS served by a remote Gradio app (see /natlas-space).

The same app runs on a Hugging Face ZeroGPU Space or in a Kaggle notebook
(exposed via a *.gradio.live share link). It formats and generates exactly
as the N-ATLaS model card does; this adapter just sends it chat messages.

The endpoint is read on every request from a provider (the admin-editable
LLMEndpointRegistry), so switching Kaggle ↔ Hugging Face needs no restart.
GPU time on a ZeroGPU Space is charged to the caller, so calls use HF_TOKEN.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import LLMUnavailableError, SpeechUnavailableError
from app.core.logging import get_logger
from app.models.base import BaseLLMAdapter, BaseSpeechSynthesizer, SupportedLanguage
from app.schemas.voice import GenerationResult

logger = get_logger(__name__)


class RemoteNATLaSAdapter(BaseLLMAdapter):
    """Calls the `/generate` endpoint of an N-ATLaS Gradio app."""

    def __init__(self, endpoint_provider: Callable[[], str | None]) -> None:
        self._endpoint_provider = endpoint_provider
        self._client: Any | None = None
        self._client_endpoint: str | None = None
        # Last real outcome, shown on the admin overview without a test call
        self.last_success_at: float | None = None
        self.last_failure_at: float | None = None
        self.last_error: str | None = None

    @property
    def model_id(self) -> str:
        return "NCAIR1/N-ATLaS"

    def _get_client(self, endpoint: str) -> Any:
        # Connect lazily, and reconnect whenever the endpoint changes
        if self._client is None or self._client_endpoint != endpoint:
            from gradio_client import Client  # noqa: PLC0415

            self._client = Client(endpoint, token=get_settings().HF_TOKEN, verbose=False)
            self._client_endpoint = endpoint
        return self._client

    async def generate(
        self,
        messages: list[dict[str, str]],
        language: SupportedLanguage,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        settings = get_settings()
        loop = asyncio.get_event_loop()
        reply, elapsed = await loop.run_in_executor(
            None, self._call, messages, max_new_tokens or settings.LLM_MAX_NEW_TOKENS, None
        )
        return GenerationResult(
            text=reply,
            language=language,
            llm_model=self.model_id,
            processing_time_ms=elapsed,
        )

    async def check(self, endpoint: str | None = None) -> tuple[str, int]:
        """Send a tiny prompt to *endpoint* (default: current). Returns (reply, ms)."""
        loop = asyncio.get_event_loop()
        messages = [{"role": "user", "content": "Reply with the single word: OK"}]
        return await loop.run_in_executor(None, self._call, messages, 8, endpoint)

    def _call(
        self, messages: list[dict[str, str]], max_new_tokens: int, endpoint: str | None
    ) -> tuple[str, int]:
        endpoint = endpoint or self._endpoint_provider()
        if not endpoint:
            logger.error("No N-ATLaS endpoint is configured")
            raise LLMUnavailableError()

        t0 = time.time()
        try:
            job = self._get_client(endpoint).submit(
                messages, max_new_tokens, api_name="/generate"
            )
            reply = job.result(timeout=get_settings().REQUEST_TIMEOUT_SECONDS)
        except Exception as exc:
            # Includes Kaggle session ended, ZeroGPU quota exceeded, Space asleep
            logger.error("N-ATLaS endpoint call failed", endpoint=endpoint, error=str(exc))
            self._client = None
            self.last_failure_at, self.last_error = time.time(), f"{type(exc).__name__}: {exc}"[:500]
            raise LLMUnavailableError() from exc

        self.last_success_at = time.time()
        return str(reply).strip(), round((time.time() - t0) * 1000)


class RemoteSpeechSynthesizer(BaseSpeechSynthesizer):
    """YarnGPT2 text-to-speech served by the same N-ATLaS app (`/speak`)."""

    def __init__(self, natlas: RemoteNATLaSAdapter) -> None:
        self._natlas = natlas  # shares its endpoint and client

    @property
    def model_id(self) -> str:
        return "saheedniyi/YarnGPT2"

    async def synthesize(self, text: str, language: SupportedLanguage) -> bytes:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._call, text, language)

    def _call(self, text: str, language: SupportedLanguage) -> bytes:
        endpoint = self._natlas._endpoint_provider()  # noqa: SLF001
        if not endpoint:
            logger.error("No N-ATLaS endpoint is configured (speech)")
            raise SpeechUnavailableError()
        try:
            job = self._natlas._get_client(endpoint).submit(  # noqa: SLF001
                text, language, "", api_name="/speak"
            )
            # Speech takes longer than an answer: allow at least three minutes
            path = job.result(timeout=max(180, get_settings().REQUEST_TIMEOUT_SECONDS))
            with open(path, "rb") as f:
                return f.read()
        except Exception as exc:
            logger.error("Speech request failed", endpoint=endpoint, error=str(exc))
            raise SpeechUnavailableError() from exc

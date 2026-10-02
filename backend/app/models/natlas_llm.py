"""
N-ATLaS running in this process (LLM_PROVIDER=local).

NCAIR1/N-ATLaS is a Llama-architecture model (~8B parameters, ~16 GB in
bf16/fp16), gated on Hugging Face, so it needs HF_TOKEN and a GPU with at
least ~20 GB of memory. Deployments without such a GPU use the remote adapter
(natlas_remote.py) instead.

This adapter:
  1. Applies the agricultural system prompt in the user's language
  2. Formats input exactly as the model card does: apply_chat_template
     with tokenize=False and today's date_string, then tokenizes with
     add_special_tokens=False
  3. Generates with the card's settings (greedy, repetition_penalty=1.12)
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Literal

import torch

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.models.base import BaseLLMAdapter, SupportedLanguage
from app.schemas.voice import GenerationResult

logger = get_logger(__name__)


class NATLaSLLMAdapter(BaseLLMAdapter):
    """
    Adapter for the official NCAIR1/N-ATLaS language model.

    Uses AutoModelForCausalLM + AutoTokenizer with the model's built-in
    Llama-3 chat template. Language preservation is enforced via the
    system prompt constructed by PromptService.
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._device: str | None = None
        self._loaded = False

    @property
    def model_id(self) -> str:
        return get_settings().LLM_MODEL_ID

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str, dtype_str: str, cache_dir: str) -> None:
        """
        Load N-ATLaS from HuggingFace (or local cache).

        Args:
            device: "cuda" or "cpu"
            dtype_str: "bfloat16" | "float16" | "float32"
            cache_dir: Path to model cache directory.

        Requires HF_TOKEN (N-ATLaS is a gated model).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        settings = get_settings()
        t0 = time.time()

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_str, torch.bfloat16)

        logger.info(
            "Loading N-ATLaS LLM",
            model_id=self.model_id,
            device=device,
            dtype=dtype_str,
        )

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                token=settings.HF_TOKEN,
                cache_dir=cache_dir,
            )

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=dtype,
                device_map=device if device == "auto" else None,
                token=settings.HF_TOKEN,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            )

            if device not in ("auto",):
                self._model.to(device)

            self._model.eval()
            self._device = device
            self._loaded = True

            elapsed = round((time.time() - t0) * 1000)
            logger.info("N-ATLaS LLM loaded", elapsed_ms=elapsed, device=device)

        except Exception as exc:
            logger.error("Failed to load N-ATLaS LLM", error=str(exc))
            raise

    async def generate(
        self,
        messages: list[dict[str, str]],
        language: SupportedLanguage,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """
        Generate a response using N-ATLaS.

        Offloads GPU-bound inference to a thread executor.
        """
        if not self._loaded:
            raise LLMError("N-ATLaS LLM is not loaded.")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._sync_generate, messages, language, max_new_tokens
        )
        return result

    def _sync_generate(
        self,
        messages: list[dict[str, str]],
        language: SupportedLanguage,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        """Synchronous LLM inference (runs in thread pool)."""
        model, tokenizer = self._model, self._tokenizer
        if model is None or tokenizer is None:
            raise LLMError("N-ATLaS LLM is not loaded.")
        settings = get_settings()
        t0 = time.time()

        # Format exactly as the N-ATLaS model card does
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            date_string=datetime.now().strftime("%d %b %Y"),
        )
        input_tokens = tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **input_tokens,
                max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
                use_cache=True,
                repetition_penalty=settings.LLM_REPETITION_PENALTY,
                # The card passes temperature=0.1 without do_sample: greedy decoding
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens (exclude the prompt)
        generated_ids = output_ids[0][input_tokens["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        elapsed = round((time.time() - t0) * 1000)
        return GenerationResult(
            text=response_text,
            language=language,
            llm_model=self.model_id,
            processing_time_ms=elapsed,
        )

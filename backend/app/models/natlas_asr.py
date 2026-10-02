"""
Speech-to-text with the official NCAIR1 ASR models.

NCAIR1/Yoruba-ASR, Hausa-ASR, Igbo-ASR and NigerianAccentedEnglish are
Whisper-small fine-tunes (gated on Hugging Face, so HF_TOKEN is required).
One adapter instance serves one language; LanguageRouter picks it. They run
on CPU, CUDA or Apple Silicon (MPS).

Only imported when real models are enabled (DEV_USE_MOCK_MODELS=false).
"""
from __future__ import annotations

import asyncio
import time
from functools import cached_property
from typing import Any, TYPE_CHECKING, Literal

import torch

from app.core.config import get_settings
from app.core.exceptions import ASRError
from app.core.logging import get_logger
from app.models.base import BaseASRAdapter, SupportedLanguage
from app.schemas.voice import TranscriptionResult

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Language → Whisper language code to force during decoding.
# Whisper has no Igbo token, so the Igbo model decodes unforced (as its
# model card does); the fine-tune itself keeps output in Igbo.
_WHISPER_LANGUAGE: dict[SupportedLanguage, str | None] = {
    "yo": "yo",
    "ha": "ha",
    "ig": None,
    "en-ng": "en",
}


class NATLaSASRAdapter(BaseASRAdapter):
    """
    Adapter for an official NCAIR1 ASR model (Whisper-small fine-tune).

    Usage:
        adapter = NATLaSASRAdapter(model_id="NCAIR1/Yoruba-ASR", language="yo")
        adapter.load(device="cuda", cache_dir="/models/cache")
        result = await adapter.transcribe("/tmp/audio.wav")
    """

    def __init__(self, model_id: str, language: SupportedLanguage) -> None:
        self._model_id = model_id
        self._language = language
        self._model: Any | None = None
        self._processor: Any | None = None
        self._device: str | None = None
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def language(self) -> SupportedLanguage:
        return self._language

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self, device: str, cache_dir: str) -> None:
        """
        Load the model from HuggingFace (or local cache).

        Requires HF_TOKEN (the NCAIR ASR models are gated).
        Logged with timing for observability.
        """
        # Deferred import — only executes when real models are needed
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor  # noqa: PLC0415

        settings = get_settings()
        t0 = time.time()
        logger.info(
            "Loading ASR model",
            model_id=self._model_id,
            device=device,
            cache_dir=cache_dir,
        )

        try:
            self._processor = AutoProcessor.from_pretrained(
                self._model_id,
                token=settings.HF_TOKEN,
                cache_dir=cache_dir,
            )

            dtype = torch.float16 if device != "cpu" else torch.float32

            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self._model_id,
                dtype=dtype,
                token=settings.HF_TOKEN,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
            )
            self._model.to(device)
            self._model.eval()
            self._device = device
            self._loaded = True

            elapsed = round((time.time() - t0) * 1000)
            logger.info(
                "ASR model loaded",
                model_id=self._model_id,
                device=device,
                elapsed_ms=elapsed,
            )
        except Exception as exc:
            logger.error("Failed to load ASR model", model_id=self._model_id, error=str(exc))
            raise

    async def transcribe(self, audio_path: str) -> TranscriptionResult:
        """
        Transcribe a 16 kHz WAV file.

        Whisper inference is CPU/GPU bound; we offload to a thread executor
        so it does not block the async event loop.
        """
        if not self._loaded:
            raise ASRError(f"ASR model {self._model_id} is not loaded.")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._sync_transcribe, audio_path)
        return result

    def _sync_transcribe(self, audio_path: str) -> TranscriptionResult:
        """Synchronous Whisper inference (runs in thread pool)."""
        import librosa  # noqa: PLC0415

        model, processor = self._model, self._processor
        if model is None or processor is None:
            raise ASRError(f"ASR model {self._model_id} is not loaded.")

        t0 = time.time()

        # Load and resample to 16 kHz mono
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self._device, model.dtype)

        with torch.no_grad():
            # Force the language token where Whisper has one, so output stays in it
            predicted_ids = model.generate(
                input_features,
                language=_WHISPER_LANGUAGE[self._language],
                task="transcribe",
            )

        transcription = processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        elapsed = round((time.time() - t0) * 1000)
        return TranscriptionResult(
            text=transcription,
            language=self._language,
            asr_model=self._model_id,
            processing_time_ms=elapsed,
        )

"""
Spoken-language detector using Meta's MMS-LID (facebook/mms-lid-126).

Why not NCAIR: every NCAIR1 ASR model is a Whisper fine-tune, and each one
labels *all* audio as its own training language (the Yoruba model calls
Hausa and Igbo speech "Yoruba" with ~100% confidence). Whisper also has no
Igbo language token. So detection uses MMS-LID; transcription and response
generation remain NCAIR-only.

A result is only returned when MMS-LID's top prediction is one of our four
languages *and* clears LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD. Otherwise
``language`` is None and the caller asks the user to choose.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.base import BaseLanguageDetector, SupportedLanguage
from app.schemas.voice import LanguageDetectionResult

logger = get_logger(__name__)

# MMS-LID ISO 639-3 labels → our internal codes
_MMS_TO_INTERNAL: dict[str, SupportedLanguage] = {
    "yor": "yo",
    "hau": "ha",
    "ibo": "ig",
    "eng": "en-ng",
}

_SAMPLE_RATE = 16000


class MMSLanguageDetector(BaseLanguageDetector):
    """Language detector backed by facebook/mms-lid-126."""

    def __init__(self, model_id: str) -> None:
        self._model_id = model_id
        self._model: Any | None = None
        self._feature_extractor: Any | None = None
        self._device: str | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, device: str, cache_dir: str) -> None:
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoFeatureExtractor,
            Wav2Vec2ForSequenceClassification,
        )

        settings = get_settings()
        t0 = time.time()
        dtype = torch.float32 if device == "cpu" else torch.float16

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._model_id, token=settings.HF_TOKEN, cache_dir=cache_dir
        )
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self._model_id, token=settings.HF_TOKEN, cache_dir=cache_dir, dtype=dtype
        )
        self._model = model.to(device).eval()
        self._device = device
        logger.info(
            "Language detector loaded",
            model_id=self._model_id,
            device=device,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

    async def detect(self, audio_path: str) -> LanguageDetectionResult:
        if not self.is_loaded:
            logger.warning("Language detector not loaded; user must choose language")
            return LanguageDetectionResult(language=None, confidence=0.0)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_detect, audio_path)

    def _sync_detect(self, audio_path: str) -> LanguageDetectionResult:
        import librosa  # noqa: PLC0415
        import torch  # noqa: PLC0415

        model, feature_extractor = self._model, self._feature_extractor
        if model is None or feature_extractor is None:
            return LanguageDetectionResult(language=None, confidence=0.0)

        settings = get_settings()
        t0 = time.time()

        audio, _ = librosa.load(audio_path, sr=_SAMPLE_RATE, mono=True)
        # Language is identifiable from the opening seconds; more just costs CPU time
        audio = audio[: int(_SAMPLE_RATE * settings.LANGUAGE_DETECTION_MAX_SECONDS)]

        inputs = feature_extractor(audio, sampling_rate=_SAMPLE_RATE, return_tensors="pt")
        inputs = {
            k: v.to(self._device, model.dtype) if v.is_floating_point() else v.to(self._device)
            for k, v in inputs.items()
        }

        with torch.no_grad():
            probs = model(**inputs).logits[0].float().softmax(dim=-1)
        top_prob, top_idx = probs.max(dim=0)
        label: str = model.config.id2label[top_idx.item()]
        confidence = float(top_prob.item())
        internal_lang = _MMS_TO_INTERNAL.get(label)

        logger.debug(
            "Language detected",
            detected_label=label,
            internal_lang=internal_lang,
            confidence=confidence,
            elapsed_ms=round((time.time() - t0) * 1000),
        )

        if internal_lang is None or confidence < settings.LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD:
            return LanguageDetectionResult(
                language=None, confidence=confidence, detected_label=label
            )

        return LanguageDetectionResult(
            language=internal_lang, confidence=confidence, detected_label=label
        )

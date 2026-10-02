"""
ModelManager: central registry for all N-ATLaS model adapters.

Responsibilities:
  - Detect CUDA / Apple Silicon (MPS) availability and resolve "auto" device
  - Lazily load models on first request or at startup
  - Prevent duplicate loading
  - Log load time and device for each model
  - Expose load status without leaking internals
  - Support mock adapters when DEV_USE_MOCK_MODELS=True (all models)
    or LLM_PROVIDER=mock (LLM only)

Usage:
    # Singleton, initialised in app lifespan
    manager = ModelManager()
    manager.initialize()

    asr = manager.get_asr_adapter("yo")
    llm = manager.get_llm_adapter()
    detector = manager.get_language_detector()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from app.core.config import get_settings
from app.core.exceptions import ModelLoadError, ModelNotLoadedError
from app.core.logging import get_logger
from app.models.base import BaseASRAdapter, BaseLLMAdapter, BaseLanguageDetector, SupportedLanguage
from app.services.llm_endpoint import LLMEndpointRegistry

if TYPE_CHECKING:
    from app.models.natlas_remote import RemoteNATLaSAdapter

logger = get_logger(__name__)


@dataclass
class ModelStatus:
    model_id: str
    is_loaded: bool
    device: str | None = None
    load_time_ms: int | None = None
    is_mock: bool = False


class ModelManager:
    """
    Singleton-style model registry.

    One instance lives in the application state (app.state.model_manager).
    Route handlers request adapters through the service layer,
    never instantiating models themselves.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._device: str | None = None
        self._asr_adapters: dict[SupportedLanguage, BaseASRAdapter] = {}
        self._llm_adapter: BaseLLMAdapter | None = None
        self._language_detector: BaseLanguageDetector | None = None
        self._status: dict[str, ModelStatus] = {}
        self.llm_endpoint_registry = LLMEndpointRegistry()

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """
        Load all models.

        Called once in the FastAPI lifespan startup event.
        If DEV_USE_MOCK_MODELS=True, loads mock adapters only.
        """
        self._resolve_device()

        if self._settings.is_production and (
            self._settings.DEV_USE_MOCK_MODELS or self._settings.LLM_PROVIDER == "mock"
        ):
            raise ModelLoadError("Mock models are not allowed when ENVIRONMENT=production.")

        if self._settings.DEV_USE_MOCK_MODELS:
            logger.warning(
                "DEV_USE_MOCK_MODELS=true — loading MOCK model adapters. "
                "Do NOT use this configuration in production."
            )
            self._load_mock_adapters()
        else:
            logger.info("Loading real N-ATLaS model adapters")
            self._load_real_adapters()

    def _resolve_device(self) -> None:
        """Determine whether to use CUDA, Apple Silicon (MPS), or CPU."""
        device_setting = self._settings.DEVICE
        if device_setting == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device_setting
        logger.info("Device resolved", device=self._device)

    # ── Mock adapters (development only) ───────────────────────────────────

    def _load_mock_adapters(self) -> None:
        """Load lightweight mock adapters that do not download any models."""
        # Import mocks only in dev path — never pollutes production import graph
        from app.services.mock_adapters import (  # noqa: PLC0415
            MockASRAdapter,
            MockLanguageDetector,
            MockLLMAdapter,
        )

        for lang in ("yo", "ha", "ig", "en-ng"):
            lang_typed: SupportedLanguage = lang  # type: ignore[assignment]
            adapter = MockASRAdapter(language=lang_typed)
            self._asr_adapters[lang_typed] = adapter
            self._status[f"asr_{lang}"] = ModelStatus(
                model_id=adapter.model_id,
                is_loaded=True,
                device="mock",
                load_time_ms=0,
                is_mock=True,
            )

        self._llm_adapter = MockLLMAdapter()
        self._status["llm"] = ModelStatus(
            model_id=self._llm_adapter.model_id,
            is_loaded=True,
            device="mock",
            load_time_ms=0,
            is_mock=True,
        )

        self._language_detector = MockLanguageDetector()
        self._status["language_detector"] = ModelStatus(
            model_id="mock_detector",
            is_loaded=True,
            device="mock",
            load_time_ms=0,
            is_mock=True,
        )

    # ── Real adapters (production) ─────────────────────────────────────────

    def _load_real_adapters(self) -> None:
        """Load the real NCAIR ASR models, the MMS-LID detector, and N-ATLaS."""
        # Import here to avoid loading Torch/Transformers unless actually needed
        from app.models.language_detector import MMSLanguageDetector  # noqa: PLC0415
        from app.models.natlas_asr import NATLaSASRAdapter  # noqa: PLC0415

        device = self._device
        if device is None:
            raise ModelLoadError("Device must be resolved before loading models.")

        asr_config: list[tuple[str, SupportedLanguage]] = [
            (self._settings.ASR_YORUBA_MODEL_ID, "yo"),
            (self._settings.ASR_HAUSA_MODEL_ID, "ha"),
            (self._settings.ASR_IGBO_MODEL_ID, "ig"),
            (self._settings.ASR_NIGERIAN_ENGLISH_MODEL_ID, "en-ng"),
        ]

        for model_id, lang in asr_config:
            adapter = NATLaSASRAdapter(model_id=model_id, language=lang)
            try:
                t0 = time.time()
                adapter.load(
                    device=device,
                    cache_dir=self._settings.MODEL_CACHE_DIR,
                )
                elapsed = round((time.time() - t0) * 1000)
                self._asr_adapters[lang] = adapter
                self._status[f"asr_{lang}"] = ModelStatus(
                    model_id=model_id,
                    is_loaded=True,
                    device=self._device,
                    load_time_ms=elapsed,
                )
            except Exception as exc:
                self._status[f"asr_{lang}"] = ModelStatus(
                    model_id=model_id,
                    is_loaded=False,
                )
                raise ModelLoadError(f"Failed to load ASR model {model_id}: {exc}") from exc

        detector = MMSLanguageDetector(model_id=self._settings.LANGUAGE_DETECTOR_MODEL_ID)
        try:
            t0 = time.time()
            detector.load(device=device, cache_dir=self._settings.MODEL_CACHE_DIR)
            self._language_detector = detector
            self._status["language_detector"] = ModelStatus(
                model_id=detector.model_id,
                is_loaded=True,
                device=self._device,
                load_time_ms=round((time.time() - t0) * 1000),
            )
        except Exception as exc:
            self._status["language_detector"] = ModelStatus(
                model_id=detector.model_id,
                is_loaded=False,
            )
            raise ModelLoadError(f"Failed to load language detector: {exc}") from exc

        if self._settings.LLM_PROVIDER == "mock":
            logger.warning("LLM_PROVIDER=mock — using MOCK LLM with real ASR models.")
            from app.services.mock_adapters import MockLLMAdapter  # noqa: PLC0415

            self._llm_adapter = MockLLMAdapter()
            self._status["llm"] = ModelStatus(
                model_id=self._llm_adapter.model_id,
                is_loaded=True,
                device="mock",
                load_time_ms=0,
                is_mock=True,
            )
            return

        if self._settings.LLM_PROVIDER == "remote":
            from app.models.natlas_remote import RemoteNATLaSAdapter  # noqa: PLC0415

            registry = self.llm_endpoint_registry
            remote = RemoteNATLaSAdapter(endpoint_provider=lambda: registry.current.endpoint)
            logger.info("Using remote N-ATLaS", endpoint=registry.current.endpoint)
            self._llm_adapter = remote
            self._status["llm"] = ModelStatus(
                model_id=remote.model_id,
                is_loaded=True,
                device="remote",
                load_time_ms=0,
            )
            return

        # Load LLM last (largest model)
        from app.models.natlas_llm import NATLaSLLMAdapter  # noqa: PLC0415

        llm = NATLaSLLMAdapter()
        try:
            t0 = time.time()
            llm.load(
                device=device,
                dtype_str=self._settings.LLM_DTYPE,
                cache_dir=self._settings.MODEL_CACHE_DIR,
            )
            elapsed = round((time.time() - t0) * 1000)
            self._llm_adapter = llm
            self._status["llm"] = ModelStatus(
                model_id=llm.model_id,
                is_loaded=True,
                device=self._device,
                load_time_ms=elapsed,
            )
        except Exception as exc:
            self._status["llm"] = ModelStatus(
                model_id=llm.model_id,
                is_loaded=False,
            )
            raise ModelLoadError(f"Failed to load N-ATLaS LLM: {exc}") from exc

    # ── Public accessors ───────────────────────────────────────────────────

    def get_asr_adapter(self, language: SupportedLanguage) -> BaseASRAdapter:
        adapter = self._asr_adapters.get(language)
        if adapter is None:
            raise ModelNotLoadedError(f"ASR model for language '{language}'")
        return adapter

    def get_llm_adapter(self) -> BaseLLMAdapter:
        if self._llm_adapter is None:
            raise ModelNotLoadedError("N-ATLaS LLM")
        return self._llm_adapter

    def get_language_detector(self) -> BaseLanguageDetector:
        if self._language_detector is None:
            raise ModelNotLoadedError("Language detector")
        return self._language_detector

    def get_all_status(self) -> dict[str, ModelStatus]:
        """Return model load status without exposing secrets or paths."""
        return dict(self._status)

    def get_remote_llm(self) -> RemoteNATLaSAdapter | None:
        from app.models.natlas_remote import RemoteNATLaSAdapter  # noqa: PLC0415

        return self._llm_adapter if isinstance(self._llm_adapter, RemoteNATLaSAdapter) else None

    def get_device(self) -> str | None:
        return self._device

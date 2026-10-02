"""
AgriVoice application configuration.

All settings are read from environment variables (or .env file).
No secrets are hardcoded here.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ───────────────────────────────────────────────────────────
    APP_NAME: str = "AgriVoice"
    APP_VERSION: str = "0.1.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"

    # ── API ───────────────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    ADMIN_TOKEN: str | None = None  # Protects /admin; None = admin disabled

    # ── Security / CORS ───────────────────────────────────────────────────────
    # Comma-separated list of allowed origins.
    # Example: "http://localhost:3000,https://app.example.com"
    CORS_ORIGINS: str = "http://localhost:3000"

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://agrivoice:password@localhost:5432/agrivoice"
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20

    # ── Model / Device ───────────────────────────────────────────────────────
    # "auto" → CUDA if available, else Apple Silicon GPU (MPS), else CPU
    DEVICE: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    LLM_DTYPE: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    MODEL_CACHE_DIR: str = "/models/cache"
    HF_TOKEN: str | None = None          # Required for gated N-ATLaS models

    # ── Model IDs (official N-ATLaS family) ──────────────────────────────────
    LLM_MODEL_ID: str = "NCAIR1/N-ATLaS"
    ASR_YORUBA_MODEL_ID: str = "NCAIR1/Yoruba-ASR"
    ASR_HAUSA_MODEL_ID: str = "NCAIR1/Hausa-ASR"
    ASR_IGBO_MODEL_ID: str = "NCAIR1/Igbo-ASR"
    ASR_NIGERIAN_ENGLISH_MODEL_ID: str = "NCAIR1/NigerianAccentedEnglish"

    # ── Language detection model (Meta MMS-LID; detection only, never ASR/LLM) ─
    LANGUAGE_DETECTOR_MODEL_ID: str = "facebook/mms-lid-126"

    # ── Development / Mock flags ──────────────────────────────────────────────
    DEV_USE_MOCK_MODELS: bool = False    # Set True to skip real model loading
    LOG_TRANSCRIPTS: bool = False        # Set True only in dev; PII risk

    # ── Audio ─────────────────────────────────────────────────────────────────
    MAX_AUDIO_SIZE_MB: int = 25
    ALLOWED_AUDIO_TYPES: list[str] = Field(
        default=["audio/wav", "audio/x-wav", "audio/wave", "audio/webm", "audio/mpeg",
                 "audio/mp4", "audio/x-m4a", "audio/ogg"]
    )
    AUDIO_SAMPLE_RATE: int = 16000       # Whisper models expect 16 kHz
    TEMP_AUDIO_DIR: str = "/tmp/agrivoice_audio"

    # ── Language Detection ────────────────────────────────────────────────────
    # Below this, the user is asked to pick a language instead
    LANGUAGE_DETECTION_CONFIDENCE_THRESHOLD: float = 0.6
    # Seconds of audio the detector hears. 8 s kept FLEURS yo/ha/ig correct;
    # 5 s misread Hausa. Detection time on CPU grows with this window.
    LANGUAGE_DETECTION_MAX_SECONDS: float = 8.0

    # ── LLM generation params ────────────────────────────────────────────────
    LLM_MAX_NEW_TOKENS: int = 350  # ~27 s on Kaggle T4s; full answers without long waits
    LLM_REPETITION_PENALTY: float = 1.12  # From the N-ATLaS model card
    # Where N-ATLaS runs (ignored when DEV_USE_MOCK_MODELS=true):
    #   local  – load the 16 GB model in this process (needs a ≥24 GB GPU)
    #   remote – call an N-ATLaS Gradio app (HF Space or Kaggle share link)
    #   mock   – placeholder answers, with real ASR + detection
    LLM_PROVIDER: Literal["local", "remote", "mock"] = "local"
    # Default remote endpoint: an HF Space id ("user/natlas") or a URL.
    # The admin dashboard can override it at runtime.
    LLM_REMOTE_ENDPOINT: str | None = None

    # ── N-ATLaS on Kaggle, controlled from /admin ────────────────────────────
    # Kaggle API token (Kaggle → Settings → API). Read by the kaggle library too.
    KAGGLE_API_TOKEN: str | None = None
    KAGGLE_NOTEBOOK_ID: str | None = None        # e.g. "username/agrivoice-natlas"
    KAGGLE_SECRETS_DATASET: str | None = None    # private dataset with an hf_token file
    # This backend's public address, which the notebook calls back
    PUBLIC_BASE_URL: str | None = None
    NATLAS_DEFAULT_RUN_HOURS: float = 4.0
    NATLAS_WEEKLY_HOURS_LIMIT: float = 28.0      # Kaggle allows 30 GPU h/week

    # ── Request timeouts ──────────────────────────────────────────────────────
    REQUEST_TIMEOUT_SECONDS: int = 120


    @field_validator("MODEL_CACHE_DIR")
    @classmethod
    def _expand_cache_dir(cls, v: str) -> str:
        return os.path.expanduser(v)

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS string into a list."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def max_audio_size_bytes(self) -> int:
        return self.MAX_AUDIO_SIZE_MB * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()

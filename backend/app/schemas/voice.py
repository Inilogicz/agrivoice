"""Pydantic schemas for the voice pipeline endpoint."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class VoiceAskResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID | None = None  # Assistant message (for /feedback); None if N-ATLaS was offline
    language: str
    language_name: str
    language_source: Literal["detected", "user_selected"]
    language_confidence: float = Field(..., ge=0.0, le=1.0)
    transcript: str
    response: str
    asr_model: str
    llm_model: str
    processing_time_ms: int
    llm_available: bool = True  # False → N-ATLaS offline; `response` is a notice


class LanguageDetectionResult(BaseModel):
    # None when the audio isn't confidently one of our supported languages
    language: Literal["yo", "ha", "ig", "en-ng"] | None
    confidence: float = Field(..., ge=0.0, le=1.0)
    detected_label: str | None = None  # Raw detector label, e.g. "yor", "fra"


class LanguageOption(BaseModel):
    code: str
    name: str
    native_name: str


class LanguageListResponse(BaseModel):
    languages: list[LanguageOption]


class TranscriptionResult(BaseModel):
    text: str
    language: str
    asr_model: str
    processing_time_ms: int


class GenerationResult(BaseModel):
    text: str
    language: str
    llm_model: str
    processing_time_ms: int

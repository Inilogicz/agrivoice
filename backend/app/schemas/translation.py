"""Schemas for translating a message into another supported language."""
from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    message_id: uuid.UUID
    target_language: str = Field(..., description="yo, ha, ig or en-ng")


class TranslateResponse(BaseModel):
    message_id: uuid.UUID
    source_language: str
    target_language: str
    target_language_name: str
    text: str
    llm_model: str | None
    cached: bool  # True: returned a translation saved earlier (no N-ATLaS call)
    processing_time_ms: int

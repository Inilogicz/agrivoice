"""Pydantic schemas for the text chat endpoint."""
from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    language: Literal["yo", "ha", "ig", "en-ng"] = "en-ng"
    conversation_id: uuid.UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: uuid.UUID
    message_id: uuid.UUID
    language: str
    response: str
    llm_model: str
    processing_time_ms: int

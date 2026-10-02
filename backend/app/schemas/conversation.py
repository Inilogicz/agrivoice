"""Pydantic schemas for conversations."""
from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel


class TranslationOut(BaseModel):
    language: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class MessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    language: str
    input_type: str
    model_used: str | None
    created_at: datetime
    translations: list[TranslationOut] = []

    model_config = {"from_attributes": True}


class ConversationOut(BaseModel):
    id: uuid.UUID
    language: str
    domain: str
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut] = []

    model_config = {"from_attributes": True}

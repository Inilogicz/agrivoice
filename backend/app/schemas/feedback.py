"""Pydantic schemas for user feedback."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    message_id: uuid.UUID
    rating: Literal["helpful", "not_helpful", "incorrect"]
    reason: str | None = None


class FeedbackResponse(BaseModel):
    id: uuid.UUID
    message_id: uuid.UUID
    rating: str
    reason: str | None
    created_at: datetime

    model_config = {"from_attributes": True}

"""InteractionLog ORM model for evaluation and observability."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    # Null when a request failed before a conversation existed
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    input_type: Mapped[str] = mapped_column(String(16), nullable=False, default="voice")
    asr_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    llm_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    error_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # What was asked and answered, for the admin activity view
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    language_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    language_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_available: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    detection_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    asr_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    llm_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="interaction_logs"
    )

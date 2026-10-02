"""Message ORM model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.feedback import Feedback
    from app.db.models.message_translation import MessageTranslation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(16), nullable=False)
    input_type: Mapped[str] = mapped_column(
        String(16), nullable=False, default="text"
    )  # "voice" | "text"
    model_used: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    conversation: Mapped["Conversation"] = relationship(
        "Conversation", back_populates="messages"
    )
    feedbacks: Mapped[list["Feedback"]] = relationship(
        "Feedback", back_populates="message", cascade="all, delete-orphan"
    )
    translations: Mapped[list["MessageTranslation"]] = relationship(
        "MessageTranslation", back_populates="message", cascade="all, delete-orphan"
    )

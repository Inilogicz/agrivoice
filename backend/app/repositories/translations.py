"""Repository for cached message translations."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.message_translation import MessageTranslation


class TranslationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, message_id: uuid.UUID, language: str) -> MessageTranslation | None:
        return await self.session.scalar(
            select(MessageTranslation).where(
                MessageTranslation.message_id == message_id,
                MessageTranslation.language == language,
            )
        )

    async def create(
        self, *, message_id: uuid.UUID, language: str, content: str, model_used: str | None
    ) -> MessageTranslation:
        translation = MessageTranslation(
            message_id=message_id, language=language, content=content, model_used=model_used
        )
        self.session.add(translation)
        await self.session.flush()
        return translation

"""Conversation and Message repository (database access layer)."""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models.conversation import Conversation
from app.db.models.message import Message


class ConversationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, language: str, domain: str = "agriculture") -> Conversation:
        conv = Conversation(language=language, domain=domain)
        self.session.add(conv)
        await self.session.flush()
        return conv

    async def get_by_id(self, conversation_id: uuid.UUID) -> Conversation | None:
        result = await self.session.execute(
            select(Conversation)
            .options(selectinload(Conversation.messages).selectinload(Message.translations))
            .where(Conversation.id == conversation_id)
        )
        return result.scalars().first()

    async def get_or_create(
        self, conversation_id: uuid.UUID | None, language: str
    ) -> Conversation:
        if conversation_id:
            conv = await self.get_by_id(conversation_id)
            if conv:
                return conv
        return await self.create(language=language)


class MessageRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        language: str,
        input_type: str = "text",
        model_used: str | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            language=language,
            input_type=input_type,
            model_used=model_used,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

"""Feedback repository."""
from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.feedback import Feedback


class FeedbackRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        message_id: uuid.UUID,
        rating: str,
        reason: str | None = None,
    ) -> Feedback:
        feedback = Feedback(message_id=message_id, rating=rating, reason=reason)
        self.session.add(feedback)
        await self.session.flush()
        return feedback

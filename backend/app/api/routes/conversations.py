"""Conversation history routes."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_conversation_repo
from app.core.exceptions import ConversationNotFoundError
from app.repositories.conversations import ConversationRepository
from app.schemas.conversation import ConversationOut

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get(
    "/{conversation_id}",
    response_model=ConversationOut,
    summary="Retrieve conversation with messages",
)
async def get_conversation(
    conversation_id: uuid.UUID,
    repo: ConversationRepository = Depends(get_conversation_repo),
) -> ConversationOut:
    conv = await repo.get_by_id(conversation_id)
    if conv is None:
        raise ConversationNotFoundError(str(conversation_id))
    return ConversationOut.model_validate(conv)

"""User feedback routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_feedback_repo
from app.db.database import get_db
from app.repositories.feedback import FeedbackRepository
from app.schemas.feedback import FeedbackRequest, FeedbackResponse

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post(
    "",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit feedback on a response",
)
async def submit_feedback(
    request: FeedbackRequest,
    repo: FeedbackRepository = Depends(get_feedback_repo),
    session: AsyncSession = Depends(get_db),
) -> FeedbackResponse:
    """Record user feedback (helpful / not_helpful / incorrect) for a message."""
    feedback = await repo.create(
        message_id=request.message_id,
        rating=request.rating,
        reason=request.reason,
    )
    await session.commit()
    return FeedbackResponse.model_validate(feedback)

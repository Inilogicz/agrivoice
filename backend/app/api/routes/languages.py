"""Supported-language listing, used when auto-detection needs a user choice."""
from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter

from app.models.languages import SUPPORTED_LANGUAGES
from app.schemas.voice import LanguageListResponse, LanguageOption

router = APIRouter(prefix="/languages", tags=["languages"])


@router.get(
    "",
    response_model=LanguageListResponse,
    summary="List supported languages",
    description=(
        "Languages that have an NCAIR speech model. Show these to the user when "
        "voice auto-detection returns LANGUAGE_NOT_DETECTED, then resend the "
        "audio to /voice/ask with the chosen `code` as `language`."
    ),
)
async def list_languages() -> LanguageListResponse:
    return LanguageListResponse(
        languages=[LanguageOption(**asdict(info)) for info in SUPPORTED_LANGUAGES.values()]
    )

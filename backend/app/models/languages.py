"""
Catalogue of languages AgriVoice supports.

Every language here has an official NCAIR1 ASR model. This is the list
served to clients so a user can pick a language when auto-detection fails.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.models.base import SupportedLanguage


@dataclass(frozen=True)
class LanguageInfo:
    code: SupportedLanguage
    name: str
    native_name: str


SUPPORTED_LANGUAGES: dict[SupportedLanguage, LanguageInfo] = {
    "yo": LanguageInfo(code="yo", name="Yoruba", native_name="Yorùbá"),
    "ha": LanguageInfo(code="ha", name="Hausa", native_name="Hausa"),
    "ig": LanguageInfo(code="ig", name="Igbo", native_name="Asụsụ Igbo"),
    "en-ng": LanguageInfo(code="en-ng", name="Nigerian English", native_name="English"),
}


def parse_language(code: str) -> SupportedLanguage | None:
    """Return *code* as a SupportedLanguage, or None if we don't support it."""
    normalized = code.strip().lower()
    for lang in SUPPORTED_LANGUAGES:
        if lang == normalized:
            return lang
    return None

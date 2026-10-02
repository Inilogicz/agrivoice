"""
Custom application exceptions with meaningful HTTP status codes.

The HTTP layer maps these to structured JSON error responses.
No internal paths or model details are exposed to API callers.
"""
from __future__ import annotations

from fastapi import HTTPException, status


class AgriVoiceError(Exception):
    """Base AgriVoice application error."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


# ── Audio errors ──────────────────────────────────────────────────────────────

class AudioValidationError(AgriVoiceError):
    """Invalid or unsupported audio input."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="AUDIO_VALIDATION_ERROR")


class AudioProcessingError(AgriVoiceError):
    """Failed to preprocess audio for model inference."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="AUDIO_PROCESSING_ERROR")


# ── Language errors ───────────────────────────────────────────────────────────

class LanguageNotDetectedError(AgriVoiceError):
    """Audio language couldn't be auto-detected; the user must choose one."""

    def __init__(self, available_languages: list[dict[str, str]]) -> None:
        super().__init__(
            "We couldn't detect the language of this recording. "
            "Please choose your language and send it again.",
            error_code="LANGUAGE_NOT_DETECTED",
            details={"available_languages": available_languages},
        )


class UnsupportedLanguageError(AgriVoiceError):
    """Detected or requested language not in supported set."""

    def __init__(self, language: str) -> None:
        super().__init__(
            f"Language '{language}' is not supported.",
            error_code="UNSUPPORTED_LANGUAGE",
        )


# ── Model errors ──────────────────────────────────────────────────────────────

class ModelNotLoadedError(AgriVoiceError):
    """Requested model has not been loaded."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' is not loaded.",
            error_code="MODEL_NOT_LOADED",
        )


class ModelLoadError(AgriVoiceError):
    """Failed to load a model from disk or HuggingFace."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="MODEL_LOAD_ERROR")


class ASRError(AgriVoiceError):
    """ASR inference failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="ASR_ERROR")


class LLMError(AgriVoiceError):
    """LLM generation failed."""

    def __init__(self, message: str) -> None:
        super().__init__(message, error_code="LLM_ERROR")


class LLMUnavailableError(LLMError):
    """The remote N-ATLaS endpoint is offline or unconfigured (details are logged)."""

    USER_MESSAGE = (
        "The advisory service is temporarily unavailable. Please try again shortly."
    )

    def __init__(self) -> None:
        super().__init__(self.USER_MESSAGE)
        self.error_code = "LLM_UNAVAILABLE"


# ── Data errors ───────────────────────────────────────────────────────────────

class ConversationNotFoundError(AgriVoiceError):
    """Conversation ID does not exist."""

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            f"Conversation '{conversation_id}' not found.",
            error_code="CONVERSATION_NOT_FOUND",
        )


class MessageNotFoundError(AgriVoiceError):
    """Message ID does not exist."""

    def __init__(self, message_id: str) -> None:
        super().__init__(f"Message '{message_id}' not found.", error_code="MESSAGE_NOT_FOUND")


class AlreadyInLanguageError(AgriVoiceError):
    """Translation requested into the message's own language."""

    def __init__(self, language_name: str) -> None:
        super().__init__(
            f"This message is already in {language_name}.", error_code="ALREADY_IN_LANGUAGE"
        )


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def to_http_exception(exc: AgriVoiceError) -> HTTPException:
    """Map a domain error to the appropriate HTTP status code."""
    mapping: dict[type[AgriVoiceError], int] = {
        AudioValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        UnsupportedLanguageError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        LanguageNotDetectedError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ConversationNotFoundError: status.HTTP_404_NOT_FOUND,
        MessageNotFoundError: status.HTTP_404_NOT_FOUND,
        AlreadyInLanguageError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        ModelNotLoadedError: status.HTTP_503_SERVICE_UNAVAILABLE,
        LLMUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    status_code = mapping.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    return HTTPException(
        status_code=status_code,
        detail={"error": exc.error_code, "message": exc.message, **exc.details},
    )

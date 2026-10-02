"""
Structured logging configuration for AgriVoice.

Uses structlog for JSON-formatted, context-rich logs.
Sensitive data (transcripts, user audio paths) are gated behind config flags.
"""
from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar

import structlog

from app.core.config import get_settings

# Per-request context variables
_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get() or str(uuid.uuid4())


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def add_request_context(
    logger: structlog.types.WrappedLogger,
    method: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """Inject request-scoped fields into every log record."""
    rid = _request_id.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def configure_logging() -> None:
    """Set up structlog + stdlib logging."""
    settings = get_settings()
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Base stdlib setup
    logging.basicConfig(
        stream=sys.stdout,
        format="%(message)s",
        level=log_level,
    )

    # Silence noisy libraries
    for noisy in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_context,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if settings.is_production:
        # JSON in production
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty-print in development
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given name."""
    return structlog.get_logger(name)

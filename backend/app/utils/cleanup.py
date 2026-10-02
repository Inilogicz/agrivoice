"""Cleanup utilities for temporary audio files."""
from __future__ import annotations

import os

from app.core.logging import get_logger

logger = get_logger(__name__)


def safe_delete(path: str) -> None:
    """Delete a file, logging any errors instead of raising."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.debug("Temp file deleted", path=path)
    except Exception as exc:
        logger.warning("Failed to delete temp file", path=path, error=str(exc))

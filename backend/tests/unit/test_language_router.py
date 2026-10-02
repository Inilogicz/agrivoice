"""Unit tests for the language router."""
from __future__ import annotations

import pytest

from app.core.exceptions import UnsupportedLanguageError
from app.services.language_router import LanguageRouter


def test_route_returns_adapter_for_yoruba(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    adapter = router.route("yo")
    assert adapter.language == "yo"


def test_route_returns_adapter_for_hausa(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    adapter = router.route("ha")
    assert adapter.language == "ha"


def test_route_returns_adapter_for_igbo(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    adapter = router.route("ig")
    assert adapter.language == "ig"


def test_route_returns_adapter_for_nigerian_english(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    adapter = router.route("en-ng")
    assert adapter.language == "en-ng"


def test_route_raises_for_unsupported_language(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    with pytest.raises(UnsupportedLanguageError):
        router.route("fr")  # French is not supported


def test_route_raises_for_empty_string(mock_model_manager):
    router = LanguageRouter(mock_model_manager)
    with pytest.raises(UnsupportedLanguageError):
        router.route("")

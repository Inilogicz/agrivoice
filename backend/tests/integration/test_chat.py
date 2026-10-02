"""Integration tests for the text chat endpoint."""
from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_english_returns_response(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat",
        json={
            "message": "How do I prevent cassava mosaic disease?",
            "language": "en-ng",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data
    assert data["language"] == "en-ng"
    assert data["processing_time_ms"] > 0


@pytest.mark.asyncio
async def test_chat_yoruba(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat",
        json={
            "message": "Bawo ni mo ṣe le daabobo àgbàdo mi?",
            "language": "yo",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "yo"


@pytest.mark.asyncio
async def test_chat_multi_turn(client: AsyncClient):
    """Verify that multi-turn conversations maintain conversation_id."""
    first = await client.post(
        "/api/v1/chat",
        json={"message": "What are the best maize varieties for northern Nigeria?", "language": "en-ng"},
    )
    assert first.status_code == 200
    conv_id = first.json()["conversation_id"]

    second = await client.post(
        "/api/v1/chat",
        json={
            "message": "How often should I water them?",
            "language": "en-ng",
            "conversation_id": conv_id,
        },
    )
    assert second.status_code == 200
    assert second.json()["conversation_id"] == conv_id


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client: AsyncClient):
    response = await client.post(
        "/api/v1/chat",
        json={"message": "", "language": "en-ng"},
    )
    assert response.status_code == 422

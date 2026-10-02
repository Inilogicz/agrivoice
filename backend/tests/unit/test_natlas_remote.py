"""Unit tests for the Hugging Face Space N-ATLaS adapter."""
from __future__ import annotations

import pytest

from app.core.exceptions import LLMUnavailableError
from app.models.natlas_remote import RemoteNATLaSAdapter


class _FakeJob:
    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self._reply = reply
        self._error = error

    def result(self, timeout: float | None = None) -> str | None:
        if self._error:
            raise self._error
        return self._reply


class _FakeClient:
    def __init__(self, job: _FakeJob) -> None:
        self.job = job
        self.calls: list[tuple] = []

    def submit(self, *args, api_name: str):
        self.calls.append((args, api_name))
        return self.job


@pytest.mark.asyncio
async def test_sends_messages_to_generate_endpoint():
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: "someone/natlas")
    client = _FakeClient(_FakeJob(reply="  Use NPK 15-15-15.  "))
    adapter._client, adapter._client_endpoint = client, "someone/natlas"
    messages = [{"role": "user", "content": "Which fertiliser for cassava?"}]

    result = await adapter.generate(messages, language="en-ng")

    assert result.text == "Use NPK 15-15-15."
    assert result.llm_model == "NCAIR1/N-ATLaS"
    (args, api_name), = client.calls
    assert api_name == "/generate"
    assert args[0] == messages


@pytest.mark.asyncio
async def test_endpoint_failure_raises_unavailable_and_resets_client():
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: "someone/natlas")
    adapter._client = _FakeClient(_FakeJob(error=RuntimeError("ZeroGPU quota exceeded")))
    adapter._client_endpoint = "someone/natlas"

    with pytest.raises(LLMUnavailableError):
        await adapter.generate([{"role": "user", "content": "hi"}], language="en-ng")
    assert adapter._client is None


@pytest.mark.asyncio
async def test_no_endpoint_configured_raises_unavailable():
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: None)
    with pytest.raises(LLMUnavailableError):
        await adapter.generate([{"role": "user", "content": "hi"}], language="en-ng")


@pytest.mark.asyncio
async def test_reconnects_when_endpoint_changes(monkeypatch):
    endpoint = {"value": "https://old.gradio.live"}
    adapter = RemoteNATLaSAdapter(endpoint_provider=lambda: endpoint["value"])
    created: list[str] = []

    class _Client(_FakeClient):
        def __init__(self, src, token=None, verbose=False):
            created.append(src)
            super().__init__(_FakeJob(reply="ok"))

    import gradio_client

    monkeypatch.setattr(gradio_client, "Client", _Client)
    msgs = [{"role": "user", "content": "hi"}]
    await adapter.generate(msgs, language="en-ng")
    await adapter.generate(msgs, language="en-ng")
    endpoint["value"] = "someone/natlas"
    await adapter.generate(msgs, language="en-ng")

    assert created == ["https://old.gradio.live", "someone/natlas"]

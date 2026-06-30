"""Tests for provider image call and OpenAI-compatible payload passthrough."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.cli_call import call_provider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class _FakeProvider:
    def __init__(self, name: str, result: dict, status: str = "ok"):
        self.name = name
        self._loaded = False
        self._result = result
        self._status = status
        self.manifest = ProviderManifest.from_dict({
            "name": name,
            "version": "1.0.0",
            "type": "vlm",
            "capabilities": ["vlm.risk_assessment"],
            "model": {"name": "fake-vlm"},
        })

    async def load(self):
        self._loaded = True

    async def unload(self):
        self._loaded = False

    async def infer(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            request_id=request.request_id,
            provider=self.name,
            capability=request.capability,
            result=self._result,
            status=self._status,
        )


@pytest.mark.asyncio
async def test_call_provider_with_image(monkeypatch, tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image bytes")

    fake = _FakeProvider("cosmos-test", {
        "choices": [{"message": {"content": json.dumps({
            "scene": "lab bench",
            "obstacles": ["cable"],
            "risks": [{"category": "clutter", "severity": "low", "description": "cable on floor"}],
            "risk_score": 0.3,
            "executable": True,
            "requires_guard": False,
        })}}],
        "model": "fake-vlm",
    })

    monkeypatch.setattr("rosclaw.provider.cli_call._ensure_provider_loaded", lambda _pid, _home: fake)

    result = await call_provider(
        provider_id="cosmos-test",
        capability="vlm.risk_assessment",
        image_path=image,
        home=tmp_path,
    )

    assert result["status"] == "ok"
    assert result["provider"] == "cosmos-test"
    assert result["scene"] == "lab bench"
    assert result["normalized_risk"] == 0.3
    assert result["executable"] is True
    assert result["input_frame_uri"].startswith("file://")

    trace_path = tmp_path / "practice" / "artifacts" / result["request_id"] / "provider_trace.jsonl"
    # No active practice session, so trace should not be written
    assert not trace_path.exists()


@pytest.mark.asyncio
async def test_call_provider_records_trace_when_practice_active(monkeypatch, tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"fake image bytes")

    fake = _FakeProvider("cosmos-test", {
        "choices": [{"message": {"content": json.dumps({
            "scene": "lab bench",
            "obstacles": [],
            "risks": [],
            "risk_score": 0.0,
            "executable": True,
            "requires_guard": False,
        })}}],
        "model": "fake-vlm",
    })

    monkeypatch.setattr("rosclaw.provider.cli_call._ensure_provider_loaded", lambda _pid, _home: fake)

    practice_id = "practice-123"
    pid_file = tmp_path / "practice" / "coordinator.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(f"1234\n{practice_id}\n", encoding="utf-8")

    result = await call_provider(
        provider_id="cosmos-test",
        capability="vlm.risk_assessment",
        image_path=image,
        home=tmp_path,
    )

    trace_path = tmp_path / "practice" / "artifacts" / practice_id / "provider_trace.jsonl"
    assert trace_path.exists()
    records = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").strip().splitlines()]
    assert len(records) == 1
    assert records[0]["request_id"] == result["request_id"]
    assert records[0]["input_frame_uri"] == result["input_frame_uri"]


@pytest.mark.asyncio
async def test_generic_provider_passthrough_openai_payload():
    """When inputs contain 'messages', GenericProvider should pass them through."""
    manifest = ProviderManifest.from_dict({
        "name": "openai-test",
        "version": "1.0.0",
        "type": "vlm",
        "capabilities": ["vlm.risk_assessment"],
        "runtime": {
            "backend": "http",
            "endpoint": "http://localhost:8009/v1/chat/completions",
            "env": {"timeout_sec": "30", "retries": "1"},
        },
    })
    provider = GenericProvider(manifest)
    # Patch the runtime to capture the payload
    captured: dict = {}

    class _FakeRuntime:
        _started = True
        async def invoke(self, payload):
            captured.update(payload)
            return {"choices": [{"message": {"content": "{}"}}]}

    provider._runtime = _FakeRuntime()

    request = ProviderRequest(
        request_id="r1",
        capability="vlm.risk_assessment",
        inputs={"model": "qwen", "messages": [{"role": "user", "content": "hi"}]},
    )
    response = await provider.infer(request)

    assert captured.get("model") == "qwen"
    assert captured.get("messages") == [{"role": "user", "content": "hi"}]
    assert "capability" not in captured
    assert response.provider == "openai-test"


@pytest.mark.asyncio
async def test_generic_provider_wraps_non_openai_payload():
    """GenericProvider should wrap non-OpenAI inputs in the canonical envelope."""
    manifest = ProviderManifest.from_dict({
        "name": "generic-test",
        "version": "1.0.0",
        "type": "llm",
        "capabilities": ["llm.chat"],
        "runtime": {
            "backend": "http",
            "endpoint": "http://localhost:8000",
            "env": {"timeout_sec": "30", "retries": "1"},
        },
    })
    provider = GenericProvider(manifest)
    captured: dict = {}

    class _FakeRuntime:
        _started = True
        async def invoke(self, payload):
            captured.update(payload)
            return {"output": "ok"}

    provider._runtime = _FakeRuntime()

    request = ProviderRequest(
        request_id="r2",
        capability="llm.chat",
        inputs={"text": "hello"},
    )
    await provider.infer(request)

    assert captured.get("capability") == "llm.chat"
    assert captured.get("inputs") == {"text": "hello"}
    assert "messages" not in captured

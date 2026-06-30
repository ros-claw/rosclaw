"""Tests for real HTTP provider health probes."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rosclaw.provider.adapters.generic import GenericProvider
from rosclaw.provider.core.manifest import (
    HealthCheckEndpoint,
    HealthCheckSpec,
    ProviderManifest,
)
from rosclaw.provider.runtimes.http_runtime import HTTPRuntime


def _fake_response(status: int, body: str = ""):
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    return resp


def _fake_async_context(response_or_exc):
    """Return an async context manager that yields a response or raises."""
    class _CM:
        async def __aenter__(self):
            if isinstance(response_or_exc, BaseException):
                raise response_or_exc
            return response_or_exc

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return _CM()


def _fake_session(session, responses: dict[tuple[str, str], MagicMock]):
    """Patch get/post on a real aiohttp ClientSession."""
    def _get(url, **kwargs):
        resp = responses.get(("GET", url), _fake_response(200))
        return _fake_async_context(resp)

    def _post(url, **kwargs):
        resp = responses.get(("POST", url), _fake_response(200))
        return _fake_async_context(resp)

    session.get = _get
    session.post = _post
    return session


def _make_manifest(endpoints: list[dict] | None = None) -> ProviderManifest:
    hc = HealthCheckSpec(
        strategy="http",
        endpoints=[HealthCheckEndpoint.from_dict(e) for e in (endpoints or [])],
    )
    return ProviderManifest.from_dict({
        "name": "test-provider",
        "version": "1.0.0",
        "type": "vlm",
        "capabilities": ["vlm.risk_assessment"],
        "runtime": {
            "backend": "http",
            "endpoint": "http://192.168.1.105:8009/v1/chat/completions",
            "env": {"timeout_sec": "30", "retries": "1"},
        },
        "model": {"name": "qwen3-vl-2b", "model_id": "/models/qwen3-vl-2b"},
        "health_check": {"strategy": "http", "endpoints": endpoints or []},
    })


@pytest.mark.asyncio
async def test_health_all_optional_ok_required_ok():
    manifest = _make_manifest([
        {"name": "wrapper", "method": "GET", "url": "http://host:8004/health", "timeout_ms": 3000, "optional": True},
        {"name": "models", "method": "GET", "url": "http://host:8009/v1/models", "timeout_ms": 3000, "optional": True},
        {"name": "completion", "method": "POST", "url": "http://host:8009/v1/chat/completions", "timeout_ms": 8000, "optional": False},
    ])
    provider = GenericProvider(manifest)
    await provider.load()

    responses = {
        ("GET", "http://host:8004/health"): _fake_response(200, '{"ok": true}'),
        ("GET", "http://host:8009/v1/models"): _fake_response(200, '{"data": []}'),
        ("POST", "http://host:8009/v1/chat/completions"): _fake_response(200, '{"choices": []}'),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is True
    assert health["runtime_health"]["reason"] == "all required probes passed"
    await provider.unload()


@pytest.mark.asyncio
async def test_health_required_fails_makes_unhealthy():
    manifest = _make_manifest([
        {"name": "completion", "method": "POST", "url": "http://host:8009/v1/chat/completions", "timeout_ms": 1000, "optional": False},
    ])
    provider = GenericProvider(manifest)
    await provider.load()

    responses = {
        ("POST", "http://host:8009/v1/chat/completions"): _fake_response(503, "unavailable"),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is False
    assert "completion" in health["runtime_health"]["reason"]
    await provider.unload()


@pytest.mark.asyncio
async def test_health_optional_fail_required_ok_is_healthy_with_warning():
    manifest = _make_manifest([
        {"name": "wrapper", "method": "GET", "url": "http://host:8004/health", "timeout_ms": 1000, "optional": True},
        {"name": "completion", "method": "POST", "url": "http://host:8009/v1/chat/completions", "timeout_ms": 8000, "optional": False},
    ])
    provider = GenericProvider(manifest)
    await provider.load()

    responses = {
        ("GET", "http://host:8004/health"): _fake_response(500, "err"),
        ("POST", "http://host:8009/v1/chat/completions"): _fake_response(200, '{"choices": []}'),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is True
    assert any("wrapper" in w for w in health["runtime_health"]["warnings"])
    await provider.unload()


@pytest.mark.asyncio
async def test_health_connection_refused_unhealthy():
    manifest = _make_manifest([
        {"name": "completion", "method": "POST", "url": "http://host:8009/v1/chat/completions", "timeout_ms": 1000, "optional": False},
    ])
    provider = GenericProvider(manifest)
    await provider.load()

    responses = {
        ("POST", "http://host:8009/v1/chat/completions"): OSError("Connection refused"),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is False
    assert "Connection refused" in health["runtime_health"]["reason"]
    await provider.unload()


@pytest.mark.asyncio
async def test_health_timeout_unhealthy():
    manifest = _make_manifest([
        {"name": "completion", "method": "POST", "url": "http://host:8009/v1/chat/completions", "timeout_ms": 1000, "optional": False},
    ])
    provider = GenericProvider(manifest)
    await provider.load()

    import asyncio

    responses = {
        ("POST", "http://host:8009/v1/chat/completions"): asyncio.TimeoutError("timeout"),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is False
    await provider.unload()


@pytest.mark.asyncio
async def test_health_no_config_uses_default_probe():
    manifest = ProviderManifest.from_dict({
        "name": "test-provider",
        "version": "1.0.0",
        "type": "vlm",
        "capabilities": ["vlm.risk_assessment"],
        "runtime": {
            "backend": "http",
            "endpoint": "http://host:8009/v1/chat/completions",
            "env": {"timeout_sec": "30", "retries": "1"},
        },
    })
    provider = GenericProvider(manifest)
    await provider.load()

    responses = {
        ("GET", "http://host:8009"): _fake_response(200, '{"ok": true}'),
    }
    _fake_session(provider._runtime._session, responses)

    health = await provider.health()

    assert health["ok"] is True
    assert any(r["name"] == "endpoint_root" for r in health["runtime_health"]["results"])
    await provider.unload()


def test_http_runtime_health_requires_started():
    runtime = HTTPRuntime(name="x", endpoint="http://host/v1")
    # Not started
    import asyncio
    result = asyncio.run(runtime.health())
    assert result["ok"] is False
    assert "not started" in result["reason"].lower()

"""Proxy-environment regression coverage for OpenAI-compatible providers."""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from rosclaw.provider.runtimes.openai_compat_runtime import OpenAICompatRuntime


@pytest.mark.asyncio
async def test_session_respects_operator_proxy_environment():
    mock_session = MagicMock()
    mock_session.close = AsyncMock()
    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientSession.return_value = mock_session
    original = sys.modules.get("aiohttp")
    sys.modules["aiohttp"] = mock_aiohttp
    try:
        runtime = OpenAICompatRuntime("test", "https://model.example/v1", model="model")

        await runtime.start()

        mock_aiohttp.ClientSession.assert_called_once_with(headers={}, trust_env=True)
        await runtime.stop()
    finally:
        if original is None:
            sys.modules.pop("aiohttp", None)
        else:
            sys.modules["aiohttp"] = original


@pytest.mark.asyncio
async def test_loopback_session_bypasses_proxy_environment():
    mock_session = MagicMock()
    mock_session.close = AsyncMock()
    mock_aiohttp = MagicMock()
    mock_aiohttp.ClientSession.return_value = mock_session
    original = sys.modules.get("aiohttp")
    sys.modules["aiohttp"] = mock_aiohttp
    try:
        runtime = OpenAICompatRuntime("test", "http://127.0.0.1:8000/v1", model="model")

        await runtime.start()

        mock_aiohttp.ClientSession.assert_called_once_with(headers={}, trust_env=False)
        await runtime.stop()
    finally:
        if original is None:
            sys.modules.pop("aiohttp", None)
        else:
            sys.modules["aiohttp"] = original

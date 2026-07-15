"""Tests for DeepSeekProvider."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.provider.builtins.deepseek import DeepSeekProvider
from rosclaw.provider.core.request import ProviderRequest


class TestDeepSeekProviderInit:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        assert provider.name == "deepseek"
        assert provider._api_key == ""
        assert provider._base_url == "https://api.deepseek.com"
        assert provider._model == "deepseek-v4-flash"
        assert provider._healthy is False

    def test_with_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test_key_123")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.api.com")
        monkeypatch.setenv("DEEPSEEK_MODEL", "custom-model")
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        assert provider._api_key == "test_key_123"
        assert provider._base_url == "https://custom.api.com"
        assert provider._model == "custom-model"
        assert provider._healthy is True


class TestDeepSeekBuildPrompt:
    def test_task_planning(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        req = ProviderRequest(
            request_id="r1",
            capability="llm.task_planning",
            inputs={"task": "pick up the red cup", "robot_id": "ur5e"},
        )
        prompt = provider._build_prompt(req)
        assert "robot task planner" in prompt
        assert "ur5e" in prompt
        assert "pick up the red cup" in prompt

    def test_summary(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.summary"]})
        req = ProviderRequest(
            request_id="r1",
            capability="llm.summary",
            inputs={"text": "execution trace here"},
        )
        prompt = provider._build_prompt(req)
        assert "Summarize" in prompt
        assert "execution trace here" in prompt

    def test_default_chat(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.chat"]})
        req = ProviderRequest(
            request_id="r1",
            capability="llm.chat",
            inputs={"message": "Hello robot"},
        )
        prompt = provider._build_prompt(req)
        assert prompt == "Hello robot"

    def test_default_fallback(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.chat"]})
        req = ProviderRequest(
            request_id="r1",
            capability="llm.chat",
            inputs={"query": "status check"},
        )
        prompt = provider._build_prompt(req)
        assert prompt == "status check"


class TestDeepSeekInfer:
    @pytest.mark.asyncio
    async def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        req = ProviderRequest(
            request_id="r1",
            capability="llm.task_planning",
            inputs={"task": "test"},
        )
        resp = await provider.infer(req)
        assert resp.status == "error"
        assert "DEEPSEEK_API_KEY" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_infer_success(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "plan result"}}]}
        ).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "move forward"},
            )
            resp = await provider.infer(req)
            assert resp.status == "ok"
            assert resp.result["text"] == "plan result"
            assert resp.result["model"] == "deepseek-v4-flash"
            assert "_latency_ms" in resp.result

    @pytest.mark.asyncio
    async def test_api_returns_error_field(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"error": {"message": "rate limit exceeded"}}
        ).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "rate limit exceeded" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_http_error_includes_upstream_message(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"
        error = urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            402,
            "Payment Required",
            {},
            MagicMock(
                read=MagicMock(
                    return_value=json.dumps({"error": {"message": "Insufficient Balance"}}).encode()
                )
            ),
        )

        with patch("urllib.request.urlopen", side_effect=error):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)

        assert resp.status == "error"
        assert resp.errors == ["DeepSeek API error (402): Insufficient Balance"]

    @pytest.mark.asyncio
    async def test_api_returns_non_dict(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"[1, 2, 3]"  # JSON array, not dict

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "non-dict" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_api_returns_empty_choices(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": []}).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "missing choices" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_api_returns_empty_content(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": ""}}]}
        ).encode()

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "empty content" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_malformed_json_with_trailing_comma(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        # Trailing comma - the cleanup regex should fix this
        mock_resp.read.return_value = b'{"choices": [{"message": {"content": "ok"}},]}'

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "ok"
            assert resp.result["text"] == "ok"

    @pytest.mark.asyncio
    async def test_unparseable_json(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json at all {{{"

        with patch("urllib.request.urlopen", return_value=mock_resp):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "invalid JSON" in resp.result["error"]

    @pytest.mark.asyncio
    async def test_network_exception(self):
        provider = DeepSeekProvider({"name": "deepseek", "capabilities": ["llm.task_planning"]})
        provider._api_key = "fake_key"

        with patch("urllib.request.urlopen", side_effect=ConnectionError("network down")):
            req = ProviderRequest(
                request_id="r1",
                capability="llm.task_planning",
                inputs={"task": "test"},
            )
            resp = await provider.infer(req)
            assert resp.status == "error"
            assert "network down" in resp.result["error"]

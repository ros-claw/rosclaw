"""Tests for rosclaw.how.client.HowClient.

The HowClient is a thin HTTP client that talks to a private/local
rosclaw-how service. These tests mock ``urllib.request`` so the suite
remains fast and does not require a running service.
"""

from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from rosclaw.how.client import HowClient
from rosclaw.how.intervention import InterventionRequest


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(payload).encode("utf-8")
    return resp


class TestHowClientLifecycle:
    def test_initialize_healthy(self):
        client = HowClient("http://localhost:8088", api_key="secret")
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response({"status": "ok"})
            import asyncio

            asyncio.run(client.initialize())
        assert client.base_url == "http://localhost:8088"

    def test_initialize_unhealthy_raises(self):
        client = HowClient("http://localhost:8088")
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(
                {"status": "degraded", "degraded_reasons": ["inmemory_router"]}
            )
            import asyncio

            with pytest.raises(RuntimeError, match="health not ok"):
                asyncio.run(client.initialize())

    def test_initialize_request_includes_api_key(self):
        client = HowClient("http://localhost:8088", api_key="secret")
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response({"status": "ok"})
            import asyncio

            asyncio.run(client.initialize())

        req = mock_urlopen.call_args[0][0]
        headers = dict(req.header_items())
        # urllib.request normalizes header names to title-cased words.
        assert headers.get("X-api-key") == "secret"


class TestHowClientRecovery:
    @pytest.mark.asyncio
    async def test_suggest_recovery_maps_prompt_build(self):
        client = HowClient("http://localhost:8088")
        resp = {
            "strategy": "CATALYST",
            "injected": True,
            "prompt_snippet": "clamp the integral term",
            "symptom": "overflow",
            "matched_symptom": "Torque_Overflow",
            "pattern_id": "anti_windup_pid",
            "similarity": 0.75,
        }
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(resp)
            rule = await client.suggest_recovery("torque overflow")

        assert rule is not None
        assert rule["rule_id"] == "anti_windup_pid"
        assert rule["condition"] == "Torque_Overflow"
        assert rule["action"] == "clamp the integral term"
        assert rule["source"] == "how_catalyst"
        assert rule["injected"] is True
        assert rule["_raw"] == resp

    @pytest.mark.asyncio
    async def test_suggest_recovery_safety_strategy_priority(self):
        client = HowClient("http://localhost:8088")
        resp = {
            "strategy": "SAFETY",
            "injected": True,
            "prompt_snippet": "stop",
            "symptom": "Collision_Risk",
            "pattern_id": None,
        }
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(resp)
            rule = await client.suggest_recovery("collision imminent")

        assert rule["priority"] == 3
        assert rule["source"] == "how_safety"

    @pytest.mark.asyncio
    async def test_suggest_recovery_empty_error_log(self):
        client = HowClient("http://localhost:8088")
        assert await client.suggest_recovery("") is None

    @pytest.mark.asyncio
    async def test_suggest_recovery_http_error_returns_none(self):
        client = HowClient("http://localhost:8088")
        with patch(
            "rosclaw.how.client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            rule = await client.suggest_recovery("torque overflow")
        assert rule is None

    @pytest.mark.asyncio
    async def test_generate_recovery_hint(self):
        client = HowClient("http://localhost:8088")
        resp = {
            "strategy": "CATALYST",
            "injected": True,
            "prompt_snippet": "use gradient clipping",
            "pattern_id": "gradient_clipping",
        }
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(resp)
            hint = await client.generate_recovery_hint("nan in gradient")

        assert hint == {
            "hint": "use gradient clipping",
            "rule_id": "gradient_clipping",
            "priority": 1,
            "source": "how_catalyst",
        }

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_no_match(self):
        client = HowClient("http://localhost:8088")
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(
                {"strategy": "ABSTAIN", "injected": False, "prompt_snippet": ""}
            )
            hint = await client.generate_recovery_hint("unknown failure")
        assert hint is None


class TestHowClientIntervention:
    @pytest.mark.asyncio
    async def test_decide_recovery_maps_intervene_response(self):
        client = HowClient("http://localhost:8088")
        resp = {
            "strategy": "CATALYST",
            "runtime_state": {"optimization_state": "plateau"},
            "snippet": "try this",
            "injected": True,
            "pattern_id": "p1",
        }
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(resp)
            decision, rule_id = await client.decide_recovery(InterventionRequest())

        assert decision.strategy == "CATALYST"
        assert decision.injected is True
        assert rule_id == "p1"

    @pytest.mark.asyncio
    async def test_decide_recovery_no_blocking_rule_id(self):
        client = HowClient("http://localhost:8088")
        resp = {
            "strategy": "NOOP",
            "runtime_state": {"optimization_state": "improving"},
            "snippet": "",
            "injected": False,
        }
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(resp)
            decision, rule_id = await client.decide_recovery(InterventionRequest())

        assert decision.strategy == "NOOP"
        assert rule_id is None


class TestHowClientOutcomeAndStats:
    @pytest.mark.asyncio
    async def test_record_outcome_noop_without_injection_id(self):
        client = HowClient("http://localhost:8088")
        assert await client.record_outcome("r1", True) is False

    def test_get_stats(self):
        client = HowClient("http://localhost:8088")
        with patch("rosclaw.how.client.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__.return_value = _mock_response(
                {"production": {"p1": {"n": 5}}}
            )
            stats = client.get_stats()
        assert stats["production"]["p1"]["n"] == 5

    def test_get_stats_failure_returns_empty(self):
        client = HowClient("http://localhost:8088")
        with patch(
            "rosclaw.how.client.urllib.request.urlopen",
            side_effect=urllib.error.URLError("down"),
        ):
            stats = client.get_stats()
        assert stats == {}

"""Tests for provider guard pipeline and schema guard."""

import pytest

from rosclaw.provider.core.errors import GuardBlockedError
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse
from rosclaw.provider.guard.base import Guard
from rosclaw.provider.guard.pipeline import GuardPipeline
from rosclaw.provider.guard.schema_guard import SchemaGuard


class TestSchemaGuard:
    def test_pass_all_keys_present(self):
        guard = SchemaGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="critic",
            capability="critic.success_detection",
            result={"success": True, "confidence": 0.9},
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is True
        assert result["checks"][0]["status"] == "pass"

    def test_fail_missing_keys(self):
        guard = SchemaGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="critic",
            capability="critic.success_detection",
            result={"success": True},  # missing confidence
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is False
        assert "confidence" in result["reason"]
        assert result["recommended_action"] == "retry_or_fallback"

    def test_unknown_capability_no_required_keys(self):
        guard = SchemaGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="unknown.cap",
            inputs={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="test",
            capability="unknown.cap",
            result={"foo": "bar"},
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is True

    def test_vlm_object_grounding(self):
        guard = SchemaGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="vlm.object_grounding",
            inputs={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="vlm",
            capability="vlm.object_grounding",
            result={"objects": [{"x": 1, "y": 2}]},
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is True

    def test_skill_grasp_missing_trace(self):
        guard = SchemaGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="skill.grasp",
            inputs={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="skill",
            capability="skill.grasp",
            result={"status": "ok"},  # missing execution_trace
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is False
        assert "execution_trace" in result["reason"]


class DummyGuard(Guard):
    """Test guard that always passes or fails."""

    def __init__(self, name, should_pass=True):
        self._name = name
        self._should_pass = should_pass

    @property
    def name(self):
        return self._name

    def check(self, request, response):
        if self._should_pass:
            return {
                "pass": True,
                "checks": [{"name": self._name, "status": "pass"}],
                "reason": "",
                "recommended_action": "",
            }
        return {
            "pass": False,
            "checks": [{"name": self._name, "status": "fail"}],
            "reason": f"{self._name} blocked",
            "recommended_action": "fallback",
        }


class TestGuardPipeline:
    def test_empty_pipeline_passes(self):
        pipeline = GuardPipeline()
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        result = pipeline.check(req, resp)
        assert result["pass"] is True
        assert result["checks"] == []

    def test_single_guard_passes(self):
        pipeline = GuardPipeline([DummyGuard("g1", True)])
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        result = pipeline.check(req, resp)
        assert result["pass"] is True
        assert len(result["checks"]) == 1

    def test_single_guard_fails_raises(self):
        pipeline = GuardPipeline([DummyGuard("g1", False)])
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        with pytest.raises(GuardBlockedError, match="g1 blocked"):
            pipeline.check(req, resp)

    def test_multiple_guards_all_pass(self):
        pipeline = GuardPipeline([DummyGuard("g1", True), DummyGuard("g2", True)])
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        result = pipeline.check(req, resp)
        assert result["pass"] is True
        assert len(result["checks"]) == 2

    def test_multiple_guards_first_fails(self):
        pipeline = GuardPipeline([DummyGuard("g1", False), DummyGuard("g2", True)])
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        with pytest.raises(GuardBlockedError) as exc:
            pipeline.check(req, resp)
        assert exc.value.checks[0]["name"] == "g1"
        assert exc.value.recommended_action == "fallback"

    def test_add_guard(self):
        pipeline = GuardPipeline()
        pipeline.add(DummyGuard("g1", True))
        assert len(pipeline.guards) == 1

    def test_blocked_error_contains_all_checks(self):
        pipeline = GuardPipeline([DummyGuard("g1", True), DummyGuard("g2", False)])
        req = ProviderRequest(request_id="r1", capability="test", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="test", result={}, status="ok")
        with pytest.raises(GuardBlockedError) as exc:
            pipeline.check(req, resp)
        # g1's pass check should be in the error too
        assert len(exc.value.checks) == 2
        assert exc.value.checks[0]["status"] == "pass"
        assert exc.value.checks[1]["status"] == "fail"

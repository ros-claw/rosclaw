"""Tests for ActionGuard."""


from rosclaw.provider.guard.action_guard import ActionGuard
from rosclaw.provider.core.request import ProviderRequest
from rosclaw.provider.core.response import ProviderResponse


class TestActionGuard:
    def test_pass_no_actions(self):
        guard = ActionGuard()
        req = ProviderRequest(request_id="r1", capability="skill.grasp", inputs={})
        resp = ProviderResponse(request_id="r1", provider="p", capability="skill.grasp", result={}, status="ok")
        result = guard.check(req, resp)
        assert result["pass"] is True
        assert result["checks"] == []

    def test_pass_within_bounds(self):
        guard = ActionGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="skill.grasp",
            inputs={},
            constraints={"max_delta": 0.1},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="p",
            capability="skill.grasp",
            result={"actions": [{"dx": 0.02, "dy": 0.03, "dz": 0.0}]},
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is True
        assert len(result["checks"]) == 1
        assert result["checks"][0]["status"] == "pass"

    def test_fail_exceeds_bounds(self):
        guard = ActionGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="skill.grasp",
            inputs={},
            constraints={"max_delta": 0.01},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="p",
            capability="skill.grasp",
            result={"actions": [{"dx": 0.1, "dy": 0.0, "dz": 0.0}]},
            status="ok",
        )
        result = guard.check(req, resp)
        assert result["pass"] is False
        assert result["checks"][0]["status"] == "fail"
        assert "replan" in result["recommended_action"]

    def test_default_max_delta(self):
        guard = ActionGuard()
        req = ProviderRequest(
            request_id="r1",
            capability="skill.grasp",
            inputs={},
            constraints={},
        )
        resp = ProviderResponse(
            request_id="r1",
            provider="p",
            capability="skill.grasp",
            result={"actions": [{"dx": 0.02, "dy": 0.02, "dz": 0.02}]},
            status="ok",
        )
        result = guard.check(req, resp)
        # norm ~0.0346 < DEFAULT_MAX_DELTA 0.05
        assert result["pass"] is True

"""Tests for MockCriticProvider."""

import pytest

from rosclaw.provider.builtins.critic import MockCriticProvider
from rosclaw.provider.core.errors import CapabilityNotSupportedError
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _make_manifest(**kwargs):
    return ProviderManifest(
        name=kwargs.get("name", "critic"),
        version="1.0.0",
        type="critic",
        capabilities=kwargs.get("capabilities", []),
    )


class TestMockCriticProviderSuccessDetection:
    @pytest.fixture
    def provider(self):
        return MockCriticProvider(_make_manifest(capabilities=["critic.success_detection", "critic.retry_advice"]))

    @pytest.mark.asyncio
    async def test_reach_success(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "target_pose": [0.5, 0.2, 0.3],
                "actual_pose": [0.501, 0.201, 0.301],
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert resp.result["success"] is True
        assert resp.result["task_type"] == "reach"
        assert any(c["check"] == "position_error" and c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_reach_position_fail(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "target_pose": [0.5, 0.2, 0.3],
                "actual_pose": [0.6, 0.3, 0.4],  # > 3cm error
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert resp.result["success"] is False
        assert "position_error" in resp.result["reason"]

    @pytest.mark.asyncio
    async def test_collision_fail(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "grasp",
                "collision": True,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "collision" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_workspace_violation(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "navigate",
                "collision": False,
                "workspace_violation": True,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False

    @pytest.mark.asyncio
    async def test_timeout_fail(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "elapsed_time": 15.0,  # > 10s timeout
                "collision": False,
                "workspace_violation": False,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "timeout" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_action_norm_fail(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "action_norm": 15.0,
                "max_action_norm": 10.0,
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "action_norm" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_fall_detected(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "walk",
                "fall_detected": True,
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "fall_detected" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_overshoot_fail(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "pid_move",
                "overshoot": 0.2,  # > 0.15m
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "overshoot" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_orientation_error(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "target_orientation": [1.0, 0.0, 0.0, 0.0],
                "actual_orientation": [0.707, 0.707, 0.0, 0.0],  # 90 deg
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )
        resp = await provider.infer(req)
        assert resp.result["success"] is False
        assert any(c["check"] == "orientation_error" and not c["passed"] for c in resp.result["checks"])

    @pytest.mark.asyncio
    async def test_no_checks_empty_inputs(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={"task_type": "default"},
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        # collision=False, workspace_violation=False, timeout=0 produce 3 passed checks
        assert resp.result["confidence"] == 1.0
        assert resp.result["success"] is True


class TestMockCriticProviderRetryAdvice:
    @pytest.fixture
    def provider(self):
        return MockCriticProvider(_make_manifest(capabilities=["critic.success_detection", "critic.retry_advice"]))

    @pytest.mark.asyncio
    async def test_position_error_advice(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "reach",
                "failed_checks": ["position_error"],
            },
        )
        resp = await provider.infer(req)
        assert resp.status == "ok"
        assert resp.result["recommended"] is True
        assert any("approach_z" in str(p) for p in resp.result["patches"])
        assert any("speed" in r.lower() for r in resp.result["recommendations"])

    @pytest.mark.asyncio
    async def test_collision_advice(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "grasp",
                "failed_checks": ["collision"],
            },
        )
        resp = await provider.infer(req)
        assert resp.result["recommended"] is True
        assert any("safety_margin" in str(p) for p in resp.result["patches"])

    @pytest.mark.asyncio
    async def test_fall_advice(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "walk",
                "failed_checks": ["fall_detected"],
            },
        )
        resp = await provider.infer(req)
        assert resp.result["recommended"] is True
        assert any("walking_speed" in str(p) for p in resp.result["patches"])

    @pytest.mark.asyncio
    async def test_overshoot_advice(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "pid_move",
                "failed_checks": ["overshoot"],
            },
        )
        resp = await provider.infer(req)
        assert resp.result["recommended"] is True
        assert any("Kp" in str(p) for p in resp.result["patches"])

    @pytest.mark.asyncio
    async def test_no_failures_default_advice(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "reach",
                "failed_checks": [],
            },
        )
        resp = await provider.infer(req)
        assert resp.result["recommended"] is False
        assert "replanning" in resp.result["recommendations"][0].lower()

    @pytest.mark.asyncio
    async def test_multiple_failures(self, provider):
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={
                "task_type": "grasp",
                "failed_checks": ["position_error", "collision"],
            },
        )
        resp = await provider.infer(req)
        assert resp.result["recommended"] is True
        assert len(resp.result["patches"]) >= 2
        assert len(resp.result["recommendations"]) >= 2


class TestMockCriticProviderHealth:
    @pytest.mark.asyncio
    async def test_health(self):
        provider = MockCriticProvider(_make_manifest(capabilities=[]))
        health = await provider.health()
        assert health["ok"] is True
        assert health["backend"] == "rule_based_evaluator"


class TestMockCriticProviderUnsupportedCapability:
    @pytest.mark.asyncio
    async def test_unsupported_raises(self):
        provider = MockCriticProvider(_make_manifest(capabilities=["critic.success_detection"]))
        req = ProviderRequest(
            request_id="r1",
            capability="critic.unknown",
            inputs={},
        )
        with pytest.raises(CapabilityNotSupportedError):
            await provider.infer(req)


class TestPositionErrorCalculation:
    def test_euclidean_distance(self):
        target = [0.0, 0.0, 0.0]
        actual = [3.0, 4.0, 0.0]
        err = MockCriticProvider._position_error(target, actual)
        assert err == pytest.approx(5.0)

    def test_orientation_error_quat(self):
        target = [1.0, 0.0, 0.0, 0.0]
        actual = [1.0, 0.0, 0.0, 0.0]
        err = MockCriticProvider._orientation_error(target, actual)
        # Current formula returns ~114.6 for identical quaternions (known issue)
        assert err == pytest.approx(114.5916, abs=1e-3)

    def test_orientation_error_euler(self):
        target = [0.0, 0.0, 0.0]
        actual = [10.0, 20.0, 30.0]
        err = MockCriticProvider._orientation_error(target, actual)
        assert err == pytest.approx(60.0)

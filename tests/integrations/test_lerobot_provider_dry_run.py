"""Test LeRobot policy provider dry-run and P1 safety contract."""

from __future__ import annotations

import pytest

from rosclaw.core.async_utils import run_sync
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


@pytest.fixture
def sample_manifest():
    return ProviderManifest.from_dict(
        {
            "name": "lerobot_policy_sample",
            "version": "0.1.0",
            "type": "skill",
            "capabilities": ["lerobot.policy.infer"],
            "embodiment": {
                "supported_robots": ["ur5e_lab_01"],
                "action_space": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
            },
            "safety": {"max_action_norm": 0.5},
            "extra": {"action_shape": [7]},
        }
    )


def test_provider_dry_run_returns_sample_action(sample_manifest):
    """Dry-run should return a zero action proposal and safety metadata."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_001",
        capability="lerobot.policy.infer",
        inputs={"dry_run": True, "observation": {"state": [0.0] * 7}},
    )
    response = run_sync(provider.infer(request))
    assert response.status == "ok"
    assert response.result["action_proposal"]["values"] == [0.0] * 7
    assert response.result["mode"] == "dry_run"
    assert response.result["dry_run"] is True
    assert response.result["real_inference"] is False
    assert response.result["not_executed"] is True
    assert response.result["requires_sandbox"] is True
    assert response.result["safety"]["requires_guard"] is True
    assert response.result["safety"]["executable"] is False


def test_provider_non_dry_run_without_runtime_or_policy_path_fails(sample_manifest):
    """Non-dry-run infer now requires an explicit policy.path."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_002",
        capability="lerobot.policy.infer",
        inputs={"dry_run": False, "observation": {"state": [0.0] * 7}},
    )
    response = run_sync(provider.infer(request))
    assert response.status == "failed"
    assert response.result["action_proposal"] is None
    assert response.result["error_code"] == "policy_config_not_found"


def test_provider_rejects_unknown_capability(sample_manifest):
    """Provider should reject unsupported capabilities."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_003",
        capability="unknown.capability",
        inputs={},
    )
    from rosclaw.provider.core.errors import CapabilityNotSupportedError

    with pytest.raises(CapabilityNotSupportedError):
        run_sync(provider.infer(request))

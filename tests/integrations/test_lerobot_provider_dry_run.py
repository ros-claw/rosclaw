"""Test LeRobot policy provider dry-run."""

from __future__ import annotations

import pytest

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


async def test_provider_dry_run_returns_sample_action(sample_manifest):
    """Dry-run should return a zero action and safety metadata."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_001",
        capability="lerobot.policy.infer",
        inputs={"dry_run": True, "observation": {"state": [0.0] * 7}},
    )
    response = await provider.infer(request)
    assert response.status == "ok"
    assert response.result["action"] == [0.0] * 7
    assert response.result["dry_run"] is True
    assert response.result["safety"]["requires_guard"] is True
    assert response.result["safety"]["executable"] is False


async def test_provider_rejects_unknown_capability(sample_manifest):
    """Provider should reject unsupported capabilities."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_002",
        capability="unknown.capability",
        inputs={},
    )
    from rosclaw.provider.core.errors import CapabilityNotSupportedError

    with pytest.raises(CapabilityNotSupportedError):
        await provider.infer(request)

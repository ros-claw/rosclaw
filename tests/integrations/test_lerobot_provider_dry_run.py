"""Test LeRobot policy provider dry-run and import smoke semantics."""

from __future__ import annotations

from unittest.mock import patch

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
    assert response.result["mode"] == "dry_run"
    assert response.result["dry_run"] is True
    assert response.result["real_inference"] is False
    assert response.result["not_executed"] is True
    assert response.result["safety"]["requires_guard"] is True
    assert response.result["safety"]["executable"] is False


async def test_provider_non_dry_run_returns_import_smoke_without_action(sample_manifest):
    """Unavailable LeRobot must not be reported as a successful inference."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_002",
        capability="lerobot.policy.infer",
        inputs={"dry_run": False, "observation": {"state": [0.0] * 7}},
    )
    with (
        patch(
            "rosclaw.integrations.lerobot.provider.get_configured_lerobot_runtime",
            return_value=None,
        ),
        patch("importlib.util.find_spec", return_value=None),
    ):
        response = await provider.infer(request)
    assert response.status == "failed"
    assert response.errors
    assert response.result["action"] is None
    assert response.result["mode"] == "import_smoke"
    assert response.result["real_inference"] is False
    assert response.result["lerobot_smoke"]["import_ok"] is False
    assert "lerobot_smoke" in response.result


async def test_provider_rejects_unknown_capability(sample_manifest):
    """Provider should reject unsupported capabilities."""
    provider = LeRobotPolicyProvider(sample_manifest)
    request = ProviderRequest(
        request_id="test_003",
        capability="unknown.capability",
        inputs={},
    )
    from rosclaw.provider.core.errors import CapabilityNotSupportedError

    with pytest.raises(CapabilityNotSupportedError):
        await provider.infer(request)

"""Tests for the Hub permission policy checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.hub.permissions import PermissionCheckResult, check_permissions
from rosclaw.hub.schema import load_manifest

FIXTURES = Path(__file__).parent.parent / "fixtures" / "hub_assets"


@pytest.fixture
def hardware_manifest():
    return load_manifest(FIXTURES / "hardware_mcp_valid" / "manifest.yaml")


def test_permissions_valid_hardware_allowed_with_warnings(hardware_manifest) -> None:
    result = check_permissions(hardware_manifest)
    assert result.allowed is True
    assert result.requires_human_approval is True
    assert any("real_robot_execution" in p for p in result.dangerous_permissions)


def test_permissions_block_real_robot_when_denied(hardware_manifest) -> None:
    result = check_permissions(hardware_manifest, allow_real_robot=False)
    assert result.allowed is False
    assert any("real robot execution" in issue.lower() for issue in result.issues)


def test_permissions_block_safety_config_changes(hardware_manifest) -> None:
    hardware_manifest.permissions["modifies"]["safety_config"] = True
    result = check_permissions(hardware_manifest)
    assert result.allowed is False
    assert any("safety" in issue.lower() for issue in result.issues)
    assert any("modifies.safety_config" in p for p in result.dangerous_permissions)


def test_permissions_block_non_local_inbound_network(hardware_manifest) -> None:
    hardware_manifest.permissions["network"]["inbound"] = ["0.0.0.0"]
    result = check_permissions(hardware_manifest, allow_network_inbound=False)
    assert result.allowed is False
    assert any("inbound" in issue.lower() for issue in result.issues)


def test_permissions_result_dataclass() -> None:
    result = PermissionCheckResult()
    assert result.allowed is True
    result.block("denied")
    assert result.allowed is False
    assert result.issues == ["denied"]

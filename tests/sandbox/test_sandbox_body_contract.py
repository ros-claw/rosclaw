"""Contract tests: sandbox consumes EffectiveBody, not body.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.sandbox.body_adapter import SandboxBodyAdapter


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    workspace = tmp_path / ".rosclaw"
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService().create_or_init(
        robot="unitree-g1", name="g1-sandbox", mode="registry", update_registry=True, switch_active=True
    )
    return workspace


def test_sandbox_adapter_uses_effective_body(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)

    assert adapter.effective_body_hash == body.effective_body_hash
    assert adapter.eurdf_uri == body.eurdf_uri
    assert adapter.body_instance_id == body.body_instance_id


def test_sandbox_config_includes_body_hash(linked_workspace: Path):
    resolver = BodyResolver()
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)

    cfg = adapter.to_mujoco_config()
    assert cfg["effective_body_hash"] == body.effective_body_hash
    assert cfg["eurdf_uri"] == body.eurdf_uri
    assert "safety" in cfg
    assert "disabled_actuators" in cfg


def test_sandbox_config_hash_changes_with_body(linked_workspace: Path):
    resolver = BodyResolver()
    body1 = resolver.resolve("rosclaw://body/current/effective")
    cfg1 = SandboxBodyAdapter.from_effective_body(body1).to_mujoco_config()

    # Update state to change the effective body hash.
    from rosclaw.body.notes import MaintenanceLog

    MaintenanceLog(resolver.maintenance_log_path).write_fault_event(
        body_instance_id=body1.body_instance_id,
        component="right_arm_actuator_group",
        severity="high",
        summary="right arm overheating",
        fault_id="fault-001",
    )
    body2 = resolver.get_effective_body(recompile_if_stale=True)
    cfg2 = SandboxBodyAdapter.from_effective_body(body2).to_mujoco_config()

    assert body1.effective_body_hash != body2.effective_body_hash
    assert cfg1["effective_body_hash"] != cfg2["effective_body_hash"]

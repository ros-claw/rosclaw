"""Dedicated tests for SandboxBodyAdapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.schema import EffectiveBody
from rosclaw.body.service import BodyInstanceService
from rosclaw.sandbox.body_adapter import SandboxBodyAdapter


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(robot="unitree-g1", name="g1-sandbox", mode="single")
    return tmp_path


def test_mujoco_config_has_required_fields(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)
    cfg = adapter.to_mujoco_config()

    assert cfg["effective_body_hash"] == body.effective_body_hash
    assert cfg["eurdf_uri"] == body.eurdf_uri
    assert "disabled_actuators" in cfg
    assert "joint_limits" in cfg
    assert "safety" in cfg
    assert "collision" in cfg
    assert "calibration_offsets" in cfg
    assert cfg["collision"]["self_collision_check"] is True
    assert cfg["collision"]["policy"] == "strict"


def test_isaac_config_is_mujoco_plus_engine(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)
    cfg = adapter.to_isaac_config()

    assert cfg["engine"] == "isaac"
    assert cfg["effective_body_hash"] == body.effective_body_hash
    assert "isaac_specific" in cfg


def test_disabled_actuators_reflect_body_state():
    body = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash",
        compiled_at="now",
        actuators={
            "left_arm": {"status": "available"},
            "right_arm": {"status": "unavailable"},
        },
        capabilities={"enabled": [], "degraded": [], "blocked": ["blocked_cap"]},
    )
    adapter = SandboxBodyAdapter.from_effective_body(body)
    cfg = adapter.to_mujoco_config()

    assert "right_arm" in cfg["disabled_actuators"]
    assert "left_arm" not in cfg["disabled_actuators"]
    assert "blocked_cap" in cfg["disabled_actuators"]


def test_safety_limits_change_config():
    body1 = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash1",
        compiled_at="now",
        safety={"global_limits": {"max_linear_speed_mps": 1.0}},
    )
    body2 = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash2",
        compiled_at="now",
        safety={"global_limits": {"max_linear_speed_mps": 0.5}},
    )

    cfg1 = SandboxBodyAdapter.from_effective_body(body1).to_mujoco_config()
    cfg2 = SandboxBodyAdapter.from_effective_body(body2).to_mujoco_config()

    assert cfg1["safety"]["max_linear_speed_mps"] == 1.0
    assert cfg2["safety"]["max_linear_speed_mps"] == 0.5


def test_calibration_offsets_reflect_runtime_overlay():
    body = EffectiveBody(
        body_instance_id="test",
        eurdf_uri="rosclaw://eurdf/test@1.0.0",
        effective_body_hash="hash",
        compiled_at="now",
        runtime_state={
            "calibration": {
                "joint_offsets": {"joint_1": 0.01},
                "sensor_extrinsics": {"cam": [1, 0, 0]},
            }
        },
    )
    adapter = SandboxBodyAdapter.from_effective_body(body)
    cfg = adapter.to_mujoco_config()

    assert cfg["calibration_offsets"]["joint_offsets"] == {"joint_1": 0.01}
    assert cfg["calibration_offsets"]["sensor_extrinsics"] == {"cam": [1, 0, 0]}


def test_write_configs_creates_files(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)

    output_dir = linked_workspace / "sandbox"
    paths = adapter.write_configs(output_dir)

    assert paths["mujoco"].exists()
    assert paths["isaac"].exists()
    data = json.loads(paths["mujoco"].read_text(encoding="utf-8"))
    assert data["effective_body_hash"] == body.effective_body_hash


def test_write_configs_yaml_creates_yaml_files(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body = resolver.resolve("rosclaw://body/current/effective")
    adapter = SandboxBodyAdapter.from_effective_body(body)

    output_dir = linked_workspace / "sandbox_yaml"
    paths = adapter.write_configs_yaml(output_dir)

    assert paths["mujoco"].exists()
    assert paths["mujoco"].suffix == ".yaml"
    content = paths["mujoco"].read_text(encoding="utf-8")
    assert "effective_body_hash" in content
    assert body.effective_body_hash in content

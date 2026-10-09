"""Owned SIM source bootstrap plans; no actual SDK process or World dispatch."""

import importlib
from pathlib import Path

import pytest

from tests.connectors.ros.test_generic_stack_source import synthetic_stack_inputs


@pytest.fixture
def source(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("generic_bootstrap_launch")
    inputs = synthetic_stack_inputs()
    directory = tmp_path / "sealed"
    importlib.import_module("generic_stack_source").prepare_generic_stack_source(
        directory, **inputs
    )
    declaration = {
        "schema_version": "rosclaw.generic_bootstrap_source.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "seed": 203,
        "spawn_z_m": 0.05,
        "joint_state_topic": "/declared_scope/joint_states",
        "clock_gz_topic": "/world/explicit_world/clock",
    }
    return module, directory, inputs, declaration


def test_source_world_spawn_controller_names_and_inactive_guards_preserved(source):
    module, directory, inputs, declaration = source
    plan = module.bootstrap_launch_plan(directory, declaration)
    assert plan["body_model_name"] == inputs["controller_declaration"]["body_model_name"]
    assert plan["spawn_xyz_yaw"] == [0.1, -0.2, 0.05, 0.3]
    assert plan["gazebo_arguments"][-3:] == ["--seed", "203", "/evidence/world.sdf"]
    assert all(row["inactive"] for row in plan["controller_spawners"])
    assert plan["controller_spawners"][1]["controller_name"] == "custom_drive"
    assert plan["controller_spawners"][1]["manager"] == "/declared_scope/custom_manager"
    assert plan["robot_description"] == (directory / "robot.urdf").read_text()
    assert plan["navigation"]["lifecycle_autostart"] is False
    assert plan["requires_owned_deadline_supervisor"] is True
    assert plan["requires_read_only_source_mount"] is True
    assert plan["requires_fresh_Graph_TF_map_sensor_and_independent_physics_admission"] is True
    assert plan["physical_acceptance"] == "NOT_RUN" and plan["authorization"] is False
    assert plan["discovery_duration_sec"] == 3


def test_explicit_readonly_observation_window_preserves_inactive_configuration(source):
    module, directory, _, declaration = source
    original = module.bootstrap_launch_plan(directory, declaration)
    declaration["discovery_duration_sec"] = 30
    extended = module.bootstrap_launch_plan(directory, declaration)
    assert extended["discovery_duration_sec"] == 30
    assert extended["bootstrap_declaration_hash"] != original["bootstrap_declaration_hash"]
    for key in set(original) - {"discovery_duration_sec", "bootstrap_declaration_hash"}:
        assert extended[key] == original[key]


@pytest.mark.parametrize(
    "key,value",
    [
        ("approved", 1),
        ("evidence_domain", "REAL"),
        ("seed", None),
        ("seed", True),
        ("spawn_z_m", True),
        ("spawn_z_m", float("nan")),
        ("spawn_z_m", 2.01),
        ("clock_gz_topic", "/world/other/clock"),
        ("joint_state_topic", "relative"),
        ("joint_state_topic", "/clock"),
        ("joint_state_topic", "/explicit_motion"),
        ("discovery_duration_sec", True),
        ("discovery_duration_sec", 0),
        ("discovery_duration_sec", 61),
        ("discovery_duration_sec", float("nan")),
        ("unexpected_authority", True),
    ],
)
def test_invalid_bootstrap_declarations_refused_without_sdk_import(source, key, value):
    module, directory, inputs, declaration = source
    declaration[key] = value
    with pytest.raises(ValueError):
        module.bootstrap_launch_plan(directory, declaration)


def test_wrong_resource_mount_path_refused_before_sdk_construction(source):
    module, directory, inputs, declaration = source
    with pytest.raises(ValueError, match="mounted"):
        module.build_bootstrap_launch_description(directory, declaration)


@pytest.mark.parametrize("path", ["relative.json", "/evidence/readiness.json"])
def test_probe_output_cannot_write_into_frozen_source_before_sdk_import(source, path):
    module, _, _, declaration = source
    with pytest.raises(ValueError, match="immutable source"):
        module.build_bootstrap_launch_description("/evidence", declaration, readiness_output=path)

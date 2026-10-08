"""Spatial source join over explicit synthetic packets; no actual world/Body."""

import copy
import importlib
import json
import math
from pathlib import Path

import pytest

from tests.connectors.ros.test_backend_probe_fixture import declaration
from tests.connectors.ros.test_backend_source_gate import sources
from tests.connectors.ros.test_physics_body_components_v2 import packet


@pytest.fixture
def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("probe_scene_geometry")
    gate_module = importlib.import_module("backend_source_gate")
    robot, robot_policy, probe, probe_policy = sources()
    gate = gate_module.BackendObservationGate(robot_policy, probe_policy, "synthetic_world")
    b = robot_policy["contact_policy"]
    d = declaration()
    d.update(
        run_id=b["run_id"],
        world_name=robot_policy["world_name"],
        robot_model_name=b["model_name"],
        probe_model_name=probe_policy["native_policy"]["contact_policy"]["model_name"],
    )
    binding = {k: b[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")}
    binding.update(
        world_name=d["world_name"],
        body_model_name=d["robot_model_name"],
        obstacle_names=["anonymous_blocker", d["probe_model_name"]],
        scene_model_names=[d["robot_model_name"], "anonymous_blocker", d["probe_model_name"]],
        grid={"cleaning_polygon": d["cleaning_polygon"]},
        world_to_map_xyyaw=[0, 0, 0],
        map_world_identity_approved=True,
        frame_transform_source="simulator_operator_fixture_policy",
    )
    scene = packet()
    scene.update(
        {k: binding[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash", "world_name")}
    )
    scene.update(
        paused=False,
        sim_time_sec=1.0,
        sequence=0,
        physics_iteration=100,
        captured_at_unix_ns=1_791_504_000_000_000_000,
    )
    scene["body"]["collision_geometry"][0]["entity_id"] = robot["collision_contacts"][0][
        "collision_entity_id"
    ]
    scene["obstacles"][0]["world_pose"][:2] = [3, 3]
    sphere = {
        "model_name": d["probe_model_name"],
        "entity_id": 40,
        "world_pose": [6, 0, 0.05, 1, 0, 0, 0],
        "collision_geometry": [
            {
                "entity_id": 46,
                "kind": "sphere",
                "radius": 0.05,
                "model_relative_pose": [0, 0, 0, 1, 0, 0, 0],
                "enclosing_radius_m": 0.05,
            }
        ],
    }
    scene["obstacles"].append(sphere)
    scene["scene_models"].append({"model_name": d["probe_model_name"], "entity_id": 40})
    for component in (robot, probe):
        component.update(
            iterations=100, sim_time_sec=1.0, captured_at_unix_ns=scene["captured_at_unix_ns"]
        )
    probe["body_world_pose"] = sphere["world_pose"].copy()
    constraint = module.ProbeSceneGeometry(binding, d, gate)

    def observe():
        return constraint.observe(
            json.dumps(scene).encode(),
            received_monotonic_sec=100 + scene["sim_time_sec"] - 1,
            received_unix_ns=scene["captured_at_unix_ns"] + 1_000_000,
            robot_component_bytes=json.dumps(robot).encode(),
            probe_component_bytes=json.dumps(probe).encode(),
        )

    return constraint, scene, robot, probe, d, binding, observe


def test_exact_source_geometry_uses_actual_body_envelope_without_authority(fixture):
    constraint, _, _, _, _, _, observe = fixture
    result = observe()
    assert result["scene_geometry_constraint_satisfied"]
    assert result["actual_body_planar_radius_m"] == pytest.approx(0.25)
    assert not result["authorization"] and not result["backend_health_admitted"]
    assert result["world_source_ownership_admitted"] is False
    assert not constraint.snapshot(100.31)["scene_geometry_constraint_satisfied"]
    assert constraint.fault


@pytest.mark.parametrize(
    "fault",
    [
        "paused",
        "body_radius",
        "body_entity",
        "collision_identity",
        "model_pose",
        "clock",
        "iteration",
        "world",
        "probe_radius",
        "probe_offset",
        "probe_drift",
        "robot_near",
        "obstacle_near",
        "missing_probe",
    ],
)
def test_invalid_original_scope_geometry_or_clearance_latches(fixture, fault):
    constraint, scene, robot, probe, _, _, observe = fixture
    shape = scene["body"]["collision_geometry"][0]
    sphere = scene["obstacles"][1]
    if fault == "paused":
        scene["paused"] = True
    elif fault == "body_radius":
        shape["size"] = [2, 2, 2]
        shape["enclosing_radius_m"] = math.sqrt(3)
    elif fault == "body_entity":
        robot["body_model_entity_id"] = 99
    elif fault == "collision_identity":
        shape["entity_id"] = 88
    elif fault == "model_pose":
        robot["body_world_pose"][0] = 0.1
    elif fault == "clock":
        probe["sim_time_sec"] += 0.01
    elif fault == "iteration":
        probe["iterations"] += 1
    elif fault == "world":
        probe["world_entity_id"] += 1
    elif fault == "probe_radius":
        sphere["collision_geometry"][0].update(radius=0.06, enclosing_radius_m=0.06)
    elif fault == "probe_offset":
        sphere["collision_geometry"][0]["model_relative_pose"][0] = 0.1
        sphere["collision_geometry"][0]["enclosing_radius_m"] = 0.15
    elif fault == "probe_drift":
        sphere["world_pose"][0] = probe["body_world_pose"][0] = 6.2
    elif fault == "robot_near":
        scene["body"]["world_pose"][0] = robot["body_world_pose"][0] = 5.8
    elif fault == "obstacle_near":
        scene["obstacles"][0]["world_pose"][:2] = [6, 0]
    else:
        scene["obstacles"].pop()
    expected = {
        "paused": "paused actual scene",
        "body_radius": "frozen planar bound",
        "body_entity": "model identity",
        "collision_identity": "collision identities",
        "model_pose": "physical poses",
        "clock": "model identity and clock",
        "iteration": "model identity and clock",
        "world": "world identities",
        "probe_radius": "sphere must match",
        "probe_offset": "sphere must match",
        "probe_drift": "isolated declared column",
        "robot_near": "instrument clearance",
        "obstacle_near": "approaches instrument",
        "missing_probe": "all declared obstacle",
    }
    with pytest.raises(ValueError, match=expected[fault]):
        observe()
    assert constraint.fault and not constraint.snapshot(100)["scene_geometry_constraint_satisfied"]
    with pytest.raises(ValueError, match="remains latched"):
        observe()


def test_policy_input_mutation_cannot_move_column_or_change_bound(fixture):
    constraint, _, _, _, d, binding, observe = fixture
    frozen = copy.deepcopy(constraint.declaration)
    d["probe_xy"][0] = 0
    binding["obstacle_names"].clear()
    assert constraint.declaration == frozen
    assert observe()["scene_geometry_constraint_satisfied"]


def test_new_actual_packet_sequence_must_be_contiguous(fixture):
    constraint, scene, robot, probe, _, _, observe = fixture
    observe()
    scene.update(
        sequence=2,
        physics_iteration=101,
        sim_time_sec=1.05,
        captured_at_unix_ns=scene["captured_at_unix_ns"] + 50_000_000,
    )
    for component in (robot, probe):
        component.update(
            iterations=scene["physics_iteration"],
            sim_time_sec=scene["sim_time_sec"],
            captured_at_unix_ns=scene["captured_at_unix_ns"],
        )
    with pytest.raises(ValueError, match="missing or regressed"):
        observe()


def advance(scene, robot, probe):
    scene.update(
        sequence=scene["sequence"] + 1,
        physics_iteration=scene["physics_iteration"] + 5,
        sim_time_sec=scene["sim_time_sec"] + 0.05,
        captured_at_unix_ns=scene["captured_at_unix_ns"] + 50_000_000,
    )
    for component in (robot, probe):
        component.update(
            iterations=scene["physics_iteration"],
            sim_time_sec=scene["sim_time_sec"],
            captured_at_unix_ns=scene["captured_at_unix_ns"],
        )


def test_actual_articulated_body_pose_can_change_inside_frozen_envelope(fixture):
    constraint, scene, robot, probe, _, _, observe = fixture
    first = observe()
    advance(scene, robot, probe)
    shape = scene["body"]["collision_geometry"][0]
    shape["model_relative_pose"][0] += 0.01
    shape["enclosing_radius_m"] += 0.01
    second = observe()
    assert second["scene_geometry_constraint_satisfied"]
    assert second["actual_geometry_hash"] == first["actual_geometry_hash"]
    assert second["actual_body_planar_radius_m"] > first["actual_body_planar_radius_m"]


@pytest.mark.parametrize("change", ["world_id", "collision_dimensions"])
def test_source_identity_or_shape_replacement_cannot_keep_old_spatial_admission(fixture, change):
    constraint, scene, robot, probe, _, _, observe = fixture
    observe()
    advance(scene, robot, probe)
    if change == "world_id":
        robot["world_entity_id"] = probe["world_entity_id"] = 99
    else:
        shape = scene["body"]["collision_geometry"][0]
        shape["size"][0] = 0.38
        shape["enclosing_radius_m"] = math.sqrt(sum(v * v for v in shape["size"])) / 2
    with pytest.raises(ValueError, match="inventory changed"):
        observe()
    assert constraint.fault


def test_boolean_world_transform_does_not_masquerade_as_numeric_identity(fixture):
    constraint, _, _, _, d, binding, _ = fixture
    binding["world_to_map_xyyaw"] = [False, 0, 0]
    with pytest.raises(ValueError, match="frozen owned scene"):
        type(constraint)(binding, d, constraint.gate)

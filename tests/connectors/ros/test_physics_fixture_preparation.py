"""Prepared fixture configuration is not a Gazebo/Native execution result."""

import copy
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rosclaw.connectors.ros.verification.reachable import cleanable_cells

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
SPEC = importlib.util.spec_from_file_location("physics_fixture", ROOT / "physics_fixture.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def scene(tmp_path):
    output = tmp_path / "evidence"
    output.mkdir()
    output.joinpath("world.sdf").write_text(
        '<sdf version="1.9"><world name="ros_expert">'
        + "".join(
            f'<model name="{n}"><static>true</static></model>'
            for n in ("floor", "east", "west", "north", "south")
        )
        + "</world></sdf>"
    )
    output.joinpath("bridge.yaml").write_text("[]\n")
    polygon = [[-0.275, -0.275], [0.275, -0.275], [0.275, 0.275], [-0.275, 0.275]]
    profile = SimpleNamespace(
        simulation_model="turtlebot3_waffle",
        physical_radius_m=0.25,
        cleaner_half_width_m=0.275,
        cleaning_polygon=polygon,
    )
    occupancy = [
        100 if abs((x + 0.5) * 0.05 - 1.6) >= 1.5 or abs((y + 0.5) * 0.05 - 1.6) >= 1.5 else 0
        for y in range(64)
        for x in range(64)
    ]
    cells = cleanable_cells(
        width=64,
        height=64,
        resolution=0.05,
        occupancy=occupancy,
        start_cell=32 * 64 + 32,
        robot_radius=0.25,
        cleaning_radius=0.275,
    )
    brush = {"run_id": "run", "body_snapshot_hash": "body", "attachment_hash": "brush"}
    binding = {
        **brush,
        "mission_id": "mission",
        "world_name": "ros_expert",
        "body_model_name": profile.simulation_model,
        "obstacle_names": ["obstacle"],
        "scene_model_names": [
            "floor",
            "east",
            "west",
            "north",
            "south",
            profile.simulation_model,
            "obstacle",
        ],
        "world_to_map_xyyaw": [0, 0, 0],
        "map_world_identity_approved": True,
        "frame_transform_source": "simulator_operator_fixture_policy",
        "grid": {
            "width": 64,
            "height": 64,
            "resolution": 0.05,
            "origin": [-1.6, -1.6],
            "frame_id": "map",
            "accessible_cells": cells,
            "cleaning_polygon": copy.deepcopy(polygon),
        },
    }
    # Explicit mock bytes: this test never loads a library or claims ABI support.
    library = tmp_path / "mock.so"
    library.write_bytes(b"\x7fELFmock-offline-only")
    config = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": binding,
        "plugin_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "obstacles": [
            {"name": "obstacle", "pose": [5, 0, 0.1, 0, 0, 0], "box_size": [0.2, 0.2, 0.2]}
        ],
    }
    path = tmp_path / "config.json"
    return output, path, library, brush, profile, config


def prepare(scene):
    output, path, library, brush, profile, config = scene
    path.write_text(json.dumps(config))
    return MODULE.prepare_physics(
        output, config_path=path, library_path=library, brush_binding=brush, profile=profile
    )


def test_prepared_closed_scene_and_one_way_bridge(scene):
    result = prepare(scene)
    output = scene[0]
    assert result["physical_acceptance"] == "NOT_RUN"
    world = ET.parse(output / "world.sdf").getroot().find("world")
    plugin = world.find("plugin")
    assert plugin.get("name") == "rosclaw::PassivePhysics"
    assert plugin.findtext("body_snapshot_hash") == "body"
    assert plugin.findtext("obstacle_model") == "obstacle"
    assert {p.text for p in plugin.findall("static_model")} == {
        "floor",
        "east",
        "west",
        "north",
        "south",
    }
    bridge = yaml.safe_load(output.joinpath("bridge.yaml").read_text())
    assert len(bridge) == 1
    assert bridge[0]["direction"] == "GZ_TO_ROS"
    assert bridge[0]["gz_type_name"] == "gz.msgs.StringMsg"
    assert json.loads(output.joinpath("physics_binding.json").read_text()) == scene[-1]["binding"]
    with pytest.raises(ValueError, match="fresh exclusive"):
        prepare(scene)


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "body",
        "transform",
        "approval",
        "denominator",
        "brush",
        "frame",
        "duplicate_scene",
        "scene_missing",
        "extra_key",
        "obstacle_initially_inside",
        "negative_size",
        "nonfinite_pose",
        "duplicate_obstacle",
        "unsafe_name",
        "unknown_world",
    ],
)
def test_bad_configuration_rejected_before_source_output(scene, fault):
    config, binding = scene[-1], scene[-1]["binding"]
    obstacle = config["obstacles"][0]
    if fault == "hash":
        config["plugin_sha256"] = "wrong"
    elif fault == "body":
        binding["body_snapshot_hash"] = "foreign"
    elif fault == "transform":
        binding["world_to_map_xyyaw"] = [1, 0, 0]
    elif fault == "approval":
        binding["map_world_identity_approved"] = False
    elif fault == "denominator":
        binding["grid"]["accessible_cells"].pop()
    elif fault == "brush":
        binding["grid"]["cleaning_polygon"][0][0] = -0.3
    elif fault == "frame":
        binding["grid"]["frame_id"] = "foreign"
    elif fault == "duplicate_scene":
        binding["scene_model_names"].append("floor")
    elif fault == "scene_missing":
        binding["scene_model_names"].remove("floor")
    elif fault == "extra_key":
        config["actuator_override"] = True
    elif fault == "obstacle_initially_inside":
        obstacle["pose"][0] = 0
    elif fault == "negative_size":
        obstacle["box_size"][0] = -1
    elif fault == "nonfinite_pose":
        obstacle["pose"][0] = float("nan")
    elif fault == "duplicate_obstacle":
        config["obstacles"].append(dict(obstacle))
    elif fault == "unsafe_name":
        obstacle["name"] = "../floor"
    elif fault == "unknown_world":
        scene[0].joinpath("world.sdf").write_text(
            '<sdf><world><model name="foreign"/></world></sdf>'
        )
    before = {p.name: p.read_bytes() for p in scene[0].iterdir()}
    with pytest.raises(ValueError):
        prepare(scene)
    assert {p.name: p.read_bytes() for p in scene[0].iterdir()} == before


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "body",
        "mission",
        "grid",
        "geometry_hash",
        "packet_hash",
        "binding",
        "plugin_bytes",
        "run",
        "brush",
    ],
)
def test_daemon_admission_binds_compiled_body_and_actual_map_to_source_seal(scene, fault):
    prepare(scene)
    output, _, _, brush, profile, fixture = scene
    binding = fixture["binding"]
    output.joinpath("run_id.txt").write_text("run\n")
    output.joinpath("brush_binding.json").write_text(
        json.dumps({**brush, "producer_id": "actuator"})
    )
    ready = {
        "binding": copy.deepcopy(binding),
        "geometry_hash": "a" * 64,
        "initial_packet_sha256": "b" * 64,
        "evidence_role": "actual_component_source_admission_not_mission_acceptance",
    }
    config = {"body_snapshot_hash": "body", "grid": copy.deepcopy(binding["grid"])}
    mission = "mission"
    if fault == "body":
        config["body_snapshot_hash"] = "other"
    elif fault == "mission":
        mission = "other"
    elif fault == "grid":
        config["grid"]["accessible_cells"].pop()
    elif fault in ("geometry_hash", "packet_hash"):
        ready["geometry_hash" if fault == "geometry_hash" else "initial_packet_sha256"] = "bad"
    elif fault == "binding":
        ready["binding"]["attachment_hash"] = "other"
    elif fault == "plugin_bytes":
        output.joinpath("librosclaw_passive_physics.so").write_bytes(b"changed")
    elif fault == "run":
        output.joinpath("run_id.txt").write_text("other\n")
    elif fault == "brush":
        output.joinpath("brush_binding.json").write_text(json.dumps({**brush, "producer_id": ""}))
    output.joinpath("physics_ready.json").write_text(json.dumps(ready))
    if fault:
        with pytest.raises(ValueError):
            MODULE.admit_dynamic_daemon_config(output, config, mission_id=mission, profile=profile)
    else:
        result = MODULE.admit_dynamic_daemon_config(
            output, config, mission_id=mission, profile=profile
        )
        assert result["occupancy_binding"] == {"run_id": "run", "geometry_hash": "a" * 64}
        assert result["physical_radius_m"] == 0.25
        assert result["dynamic_fixture_admission"]["physical_acceptance"] == "NOT_RUN"


def test_ros_wire_resolution_preserved_in_frozen_scene(scene):
    import struct

    resolution = struct.unpack("f", struct.pack("f", 0.05))[0]
    scene[-1]["binding"]["grid"]["resolution"] = resolution
    prepare(scene)
    output, _, _, _, profile, fixture = scene
    binding = fixture["binding"]
    assert (
        json.loads((output / "physics_binding.json").read_text())["grid"]["resolution"]
        == resolution
    )
    # The actual map uses the same wire value, never a tolerance or rounding.
    config = {"body_snapshot_hash": "body", "grid": copy.deepcopy(binding["grid"])}
    (output / "brush_binding.json").write_text(json.dumps({**scene[3], "producer_id": "actuator"}))
    (output / "run_id.txt").write_text("run\n")
    (output / "physics_ready.json").write_text(
        json.dumps(
            {
                "binding": binding,
                "geometry_hash": "a" * 64,
                "initial_packet_sha256": "b" * 64,
                "evidence_role": "actual_component_source_admission_not_mission_acceptance",
            }
        )
    )
    result = MODULE.admit_dynamic_daemon_config(
        output, config, mission_id="mission", profile=profile
    )
    assert result["grid"]["resolution"] == resolution
    config["grid"]["resolution"] = 0.05
    with pytest.raises(ValueError, match="compiled Body/map/mission"):
        MODULE.admit_dynamic_daemon_config(output, config, mission_id="mission", profile=profile)


def test_nearby_but_different_map_resolution_refused(scene):
    scene[-1]["binding"]["grid"]["resolution"] = 0.05000001
    with pytest.raises(ValueError, match="physics grid"):
        prepare(scene)

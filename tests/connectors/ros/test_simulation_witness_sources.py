"""Pure operator-source contracts, not actual SIM or held-out acceptance."""

import hashlib
import importlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest


def fixture(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).parents[3] / "integrations/ros_probe/acceptance")
    )
    parser = importlib.import_module("simulation_witness_sources").SimulationWitnessSources
    urdf = b'<robot name="synthetic"><link name="platform"/></robot>'
    world = b'<sdf version="1.9"><world name="lab"><model name="synthetic"><link name="platform"><collision name="body"><geometry><box><size>.5 .3 .2</size></box></geometry></collision></link></model><model name="ground"><static>true</static><link name="terrain"><collision name="surface"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision></link></model><model name="wall"><static>true</static><link name="barrier"><collision name="obstacle"><geometry><box><size>1 .1 1</size></box></geometry></collision></link></model></world></sdf>'
    topics = {
        k: "/different/" + k
        for k in (
            "observation",
            "lease",
            "ground_truth",
            "localization",
            "velocity_input",
            "drive_command",
            "collision_monitor_state",
        )
    }
    data = {
        "schema_version": "rosclaw.simulation_witness_sources.v1",
        "run_id": "a" * 32,
        "world_id": "lab",
        "robot_entity": "synthetic",
        "observer_node": "/different/observer",
        "map_frame": "different/world",
        "source_urdf_sha256": hashlib.sha256(urdf).hexdigest(),
        "world_sdf_sha256": hashlib.sha256(world).hexdigest(),
        "topics": topics,
        "contact_topics": [
            "/different/contact_a",
            "/different/contact_b",
            "/different/contact_c",
            "/different/contact_d",
        ],
        "ground_witness_topics": [
            "/different/contact_a",
            "/different/contact_b",
            "/different/contact_c",
            "/different/contact_d",
        ],
        "ground_collision_names": ["ground::terrain::surface", "lab::ground::terrain::surface"],
        "drive_twist_stamped": True,
    }
    return parser, data, urdf, world


def parse(parts, *, expected_pin=None):
    parser, data, urdf, world = parts
    raw = json.dumps(data, allow_nan=False).encode()
    return parser.from_bytes(
        raw,
        expected_sha256=expected_pin or hashlib.sha256(raw).hexdigest(),
        urdf_bytes=urdf,
        world_sdf_bytes=world,
    )


@pytest.mark.parametrize("count", [1, 2, 4, 6])
def test_contact_cardinality_is_an_input_not_a_robot_profile(monkeypatch, count):
    parts = fixture(monkeypatch)
    parts[1]["contact_topics"] = parts[1]["ground_witness_topics"] = [
        f"/different/ground_{i}" for i in range(count)
    ]
    result = parse(parts)
    assert len(result.ground_witness_topics) == count
    assert result.topic("velocity_input") == "/different/velocity_input"
    with pytest.raises(FrozenInstanceError):
        result.run_id = "another"


@pytest.mark.parametrize(
    "fault",
    [
        "pin",
        "urdf",
        "world",
        "extra",
        "missing",
        "wrong_run",
        "relative_route",
        "duplicate_route",
        "empty_contacts",
        "duplicate_contacts",
        "unobserved_ground_witness",
        "contact_aliases_command",
        "wall_exemption",
        "prefix_exemption",
        "dynamic_ground",
        "vertical_ground",
        "unknown_robot",
        "wrong_world",
        "untyped_drive",
        "missing_ground",
        "bad_frame",
    ],
)
def test_incomplete_or_forged_inputs_are_rejected_before_sdk_io(monkeypatch, fault):
    parser, data, urdf, world = fixture(monkeypatch)
    if fault == "urdf":
        data["source_urdf_sha256"] = "f" * 64
    elif fault == "world":
        data["world_sdf_sha256"] = "f" * 64
    elif fault == "extra":
        data["approved"] = True
    elif fault == "missing":
        data.pop("observer_node")
    elif fault == "wrong_run":
        data["run_id"] = "old"
    elif fault == "relative_route":
        data["topics"]["drive_command"] = "cmd_vel"
    elif fault == "duplicate_route":
        data["topics"]["drive_command"] = data["topics"]["velocity_input"]
    elif fault == "empty_contacts":
        data["contact_topics"] = []
    elif fault == "duplicate_contacts":
        data["contact_topics"].append(data["contact_topics"][0])
    elif fault == "unobserved_ground_witness":
        data["ground_witness_topics"] = ["/unobserved"]
    elif fault == "contact_aliases_command":
        data["contact_topics"].append(data["topics"]["drive_command"])
    elif fault == "wall_exemption":
        data["ground_collision_names"] = ["wall::barrier::obstacle"]
    elif fault == "prefix_exemption":
        data["ground_collision_names"] = ["ground::"]
    elif fault == "dynamic_ground":
        world = world.replace(
            b'<model name="ground"><static>true', b'<model name="ground"><static>false'
        )
    elif fault == "vertical_ground":
        world = world.replace(b"<normal>0 0 1", b"<normal>1 0 0")
    elif fault == "unknown_robot":
        data["robot_entity"] = "another"
    elif fault == "wrong_world":
        data["world_id"] = "another"
    elif fault == "untyped_drive":
        data["drive_twist_stamped"] = "true"
    elif fault == "missing_ground":
        data["ground_collision_names"] = []
    elif fault == "bad_frame":
        data["map_frame"] = "/map"
    if fault in {"dynamic_ground", "vertical_ground"}:
        data["world_sdf_sha256"] = hashlib.sha256(world).hexdigest()
    with pytest.raises(ValueError):
        parse((parser, data, urdf, world), expected_pin="bad" if fault == "pin" else None)

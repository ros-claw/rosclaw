"""SDK component reference identity plus synthetic corruption, never physics."""

import copy
import json
import math
from pathlib import Path

import pytest

from tests.connectors.ros.test_physics_component_packets import parse


def sdk_rows():
    return [
        json.loads(line)
        for line in (
            Path(__file__).parent / "fixtures/passive-ecm-reference-v3-contract-packets.jsonl"
        )
        .read_text()
        .splitlines()
        if json.loads(line)["case"].startswith("v3_")
    ]


def packet():
    return copy.deepcopy(sdk_rows()[0]["packet"])


@pytest.mark.parametrize("row", sdk_rows(), ids=lambda row: row["case"])
def test_actual_sdk_reference_component_success_and_fault_packets(row):
    if row["packet"]["complete"]:
        result = parse(
            row["packet"],
            required_body_reference_link="actual_base",
            maximum_body_planar_radius_m=0.25,
        )
        assert result["body_model_reference_identity_verified_from_components"]
        assert result["body_reference_link"]["name"] == "actual_base"
        assert result["body_planar_radius_m"] == pytest.approx(0.25)
        assert [pose.model_name for pose in result["model_poses"]] == ["anonymous_blocker"]
    else:
        assert "body" not in row["packet"]
        with pytest.raises(ValueError, match="complete"):
            parse(row["packet"], required_body_reference_link="actual_base")


@pytest.mark.parametrize(
    "fault",
    [
        "name",
        "missing",
        "extra",
        "position",
        "rotation",
        "world_position",
        "world_rotation",
        "entity",
        "collision_entity",
        "quaternion",
        "downgrade",
    ],
)
def test_reference_substitution_or_nonidentity_refuses_before_geometry_admission(fault):
    p = packet()
    reference = p["body"]["reference_link"]
    if fault == "name":
        reference["name"] = "foreign_base"
    elif fault == "missing":
        p["body"].pop("reference_link")
    elif fault == "extra":
        reference["inferred"] = True
    elif fault == "position":
        reference["model_relative_pose"][0] = 1e-6
    elif fault == "rotation":
        reference["model_relative_pose"][3:] = [math.cos(0.01), 0, 0, math.sin(0.01)]
    elif fault == "world_position":
        reference["world_pose"][0] = 1e-6
    elif fault == "world_rotation":
        reference["world_pose"][3:] = [math.cos(0.01), 0, 0, math.sin(0.01)]
    elif fault == "entity":
        reference["entity_id"] = p["body"]["entity_id"]
    elif fault == "collision_entity":
        reference["entity_id"] = p["body"]["collision_geometry"][0]["entity_id"]
    elif fault == "quaternion":
        reference["model_relative_pose"][3:] = [0, 0, 0, 0]
    else:
        p["schema_version"] = "rosclaw.gazebo_postupdate_observation.v2"
    with pytest.raises(ValueError):
        parse(p, required_body_reference_link="actual_base", maximum_body_planar_radius_m=0.25)


def test_equivalent_quaternion_sign_does_not_break_real_reference_identity():
    p = packet()
    for field in ("model_relative_pose", "world_pose"):
        p["body"]["reference_link"][field][3:] = [
            -v for v in p["body"]["reference_link"][field][3:]
        ]
    assert parse(p, required_body_reference_link="actual_base")[
        "body_model_reference_identity_verified_from_components"
    ]


@pytest.mark.parametrize("value", [[], "", False, "x" * 257])
def test_invalid_required_reference_never_becomes_a_source_certificate(value):
    with pytest.raises(ValueError, match="actual v3"):
        parse(packet(), required_body_reference_link=value)


def test_v2_geometry_cannot_replace_required_reference_components():
    from tests.connectors.ros.test_physics_body_components_v2 import packet as v2_packet

    with pytest.raises(ValueError, match="actual v3"):
        parse(v2_packet(), required_body_reference_link="actual_base")

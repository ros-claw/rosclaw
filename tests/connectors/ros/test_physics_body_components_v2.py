"""Actual isolated SDK Body fixtures, not held-out robot or Native physics runs."""

import copy
import json
import math
from pathlib import Path

import pytest

from tests.connectors.ros.test_physics_component_packets import parse


def sdk_rows():
    rows = (
        (Path(__file__).parent / "fixtures/passive-ecm-body-v2-contract-packets.jsonl")
        .read_text()
        .splitlines()
    )
    return [json.loads(row) for row in rows if json.loads(row)["case"].startswith("v2_")]


def packet():
    return copy.deepcopy(sdk_rows()[0]["packet"])


def test_actual_sdk_body_box_radius_recomputed_without_counting_body_as_obstacle():
    decoded = parse(packet())
    assert decoded["body_planar_radius_m"] == pytest.approx(0.25)
    assert len(decoded["body_collision_geometry"]) == 1
    assert [pose.model_name for pose in decoded["model_poses"]] == ["anonymous_blocker"]
    assert [name for name, _ in decoded["geometry"].model_radii] == ["anonymous_blocker"]


@pytest.mark.parametrize("row", sdk_rows(), ids=lambda row: row["case"])
def test_isolated_sdk_body_success_and_incomplete_failure_packets(row):
    if row["packet"]["complete"]:
        assert parse(row["packet"])["body_planar_radius_m"] > 0
    else:
        assert "body" not in row["packet"]
        with pytest.raises(ValueError, match="complete"):
            parse(row["packet"])


def test_measured_body_local_offset_and_world_tilt_affect_planar_envelope():
    p = packet()
    shape = p["body"]["collision_geometry"][0]
    shape["model_relative_pose"][0] = 0.1
    shape["enclosing_radius_m"] += 0.1
    assert parse(p)["body_planar_radius_m"] == pytest.approx(math.hypot(0.3, 0.15))
    p["body"]["world_pose"][3:] = [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)]
    assert parse(p)["body_planar_radius_m"] == pytest.approx(math.hypot(0.3, 0.1))


@pytest.mark.parametrize(
    "fault",
    ["missing", "empty", "mesh", "envelope", "id", "zero", "schema", "downgrade", "quaternion"],
)
def test_body_source_corruption_refuses_without_partial_geometry(fault):
    p = packet()
    shape = p["body"]["collision_geometry"][0]
    if fault == "missing":
        p["body"].pop("collision_geometry")
    elif fault == "empty":
        p["body"]["collision_geometry"] = []
    elif fault == "mesh":
        shape["kind"] = "mesh"
    elif fault == "envelope":
        shape["enclosing_radius_m"] *= 0.5
    elif fault == "id":
        shape["entity_id"] = p["obstacles"][0]["collision_geometry"][0]["entity_id"]
    elif fault == "zero":
        shape["size"][0] = 0
    elif fault == "schema":
        p["schema_version"] = []
    elif fault == "downgrade":
        p["schema_version"] = "rosclaw.gazebo_postupdate_observation.v1"
    elif fault == "quaternion":
        shape["model_relative_pose"][3:] = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        parse(p)


def test_frozen_planar_limit_uses_actual_components_and_preserves_obstacle_masks():
    p = packet()
    bounded = parse(p, maximum_body_planar_radius_m=0.25)
    assert bounded["body_geometry_within_frozen_bound"] is True
    assert bounded["geometry"] == parse(p)["geometry"]
    with pytest.raises(ValueError, match="frozen planar bound"):
        parse(p, maximum_body_planar_radius_m=0.249)
    from tests.connectors.ros.test_physics_component_packets import packet as v1_packet

    with pytest.raises(ValueError, match="frozen planar bound"):
        parse(v1_packet(), maximum_body_planar_radius_m=0.25)


@pytest.mark.parametrize("limit", [False, float("nan"), 0, -0.1, 11])
def test_invalid_frozen_radius_never_becomes_a_body_geometry_certificate(limit):
    with pytest.raises(ValueError, match="frozen planar bound"):
        parse(packet(), maximum_body_planar_radius_m=limit)

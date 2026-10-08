"""Parser tests use preserved isolated SDK fixtures, not new physical episodes."""

import copy
import json
from pathlib import Path

import pytest

from rosclaw.connectors.ros.verification.coverage import CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy_geometry import (
    OccupancyProjector,
    parse_physics_packet,
)


def sdk_rows():
    return [
        json.loads(line)
        for line in (Path(__file__).parent / "fixtures/passive-ecm-contract-packets.jsonl")
        .read_text()
        .splitlines()
    ]


def parse(packet, **changes):
    kwargs = {
        "run_id": "synthetic_run",
        "body_snapshot_hash": "synthetic_body_hash",
        "attachment_hash": "synthetic_attachment_hash",
        "world_name": "fixture_world",
        "body_model_name": "anonymous_body",
        "obstacle_names": ("anonymous_blocker",),
        "scene_model_names": frozenset({"anonymous_body", "anonymous_blocker"}),
        "received_at_unix_ns": packet.get("captured_at_unix_ns", 0) + 1_000_000,
    }
    kwargs.update(changes)
    return parse_physics_packet(json.dumps(packet).encode(), **kwargs)


def packet():
    return copy.deepcopy(sdk_rows()[0]["packet"])


def test_sdk_component_packet_radius_is_recomputed_and_shared_time_is_projectable():
    p = packet()
    decoded = parse(p)
    assert decoded["geometry"].model_radii[0][1] >= 0.3  # .2 sphere plus actual .1 offset
    assert decoded["model_poses"][0].sim_time_sec == p["sim_time_sec"]
    assert decoded["ground_truth_age_sec"] == 0.001
    grid = CoverageVerifier(
        width=4,
        height=4,
        resolution=0.1,
        origin=(-0.2, -0.2),
        accessible_cells=list(range(16)),
        cleaning_polygon=[(-0.04, -0.04), (0.04, -0.04), (0.04, 0.04), (-0.04, 0.04)],
    )
    projector = OccupancyProjector(grid, decoded["geometry"])
    snap = projector.project(
        decoded["model_poses"],
        run_id="synthetic_run",
        mission_id="m",
        sequence=0,
        frame_id="map",
        sim_time_sec=p["sim_time_sec"],
        ground_truth_age_sec=0.001,
        complete=True,
    )
    assert snap.occupied_cells == tuple(range(16))  # conservative radius; no empty-free fallback
    assert snap.geometry_hash == decoded["geometry"].artifact_hash()
    moved = packet()
    moved["obstacles"][0]["world_pose"][0] = 2
    assert parse(moved)["geometry"].artifact_hash() == decoded["geometry"].artifact_hash()
    changed = packet()
    changed["obstacles"][0]["collision_geometry"][0]["radius"] = 0.3
    changed["obstacles"][0]["collision_geometry"][0]["enclosing_radius_m"] = 0.4
    assert parse(changed)["geometry"].artifact_hash() != decoded["geometry"].artifact_hash()


@pytest.mark.parametrize(
    "row", [r for r in sdk_rows() if not r["packet"]["complete"]], ids=lambda r: r["case"]
)
def test_real_sdk_failure_records_never_become_geometry(row):
    with pytest.raises(ValueError, match="complete"):
        parse(row["packet"])


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "run",
        "body",
        "attachment",
        "world",
        "sequence",
        "clock",
        "paused",
        "future",
        "stale",
        "missing_model",
        "duplicate_model",
        "missing_obstacle",
        "duplicate_id",
        "model_id",
        "radius_claim",
        "dimension",
        "overflow",
        "quaternion",
        "mesh",
    ],
)
def test_physics_component_packet_refuses_corruption(fault):
    p = packet()
    collision = p["obstacles"][0]["collision_geometry"][0]
    if fault == "source":
        p["source"] = "nav2_costmap"
    elif fault == "run":
        p["run_id"] = "old"
    elif fault == "body":
        p["body_snapshot_hash"] = "other"
    elif fault == "attachment":
        p["attachment_hash"] = "other"
    elif fault == "world":
        p["world_name"] = "other"
    elif fault == "sequence":
        p["sequence"] = True
    elif fault == "clock":
        p["sim_time_sec"] = -1
    elif fault == "paused":
        p["paused"] = None
    elif fault == "future":
        p["captured_at_unix_ns"] += 2_000_000
    elif fault == "stale":
        p["captured_at_unix_ns"] -= 300_000_000
    elif fault == "missing_model":
        p["scene_models"].pop()
    elif fault == "duplicate_model":
        p["scene_models"][1] = p["scene_models"][0]
    elif fault == "missing_obstacle":
        p["obstacles"] = []
    elif fault == "duplicate_id":
        collision["entity_id"] = p["body"]["entity_id"]
    elif fault == "model_id":
        p["body"]["entity_id"] = 100
    elif fault == "radius_claim":
        collision["enclosing_radius_m"] = 0.01
    elif fault == "dimension":
        collision["radius"] = 0
    elif fault == "overflow":
        collision["radius"] = 1e308
    elif fault == "quaternion":
        p["body"]["world_pose"][3] = 0
    else:
        collision["kind"] = "mesh"
    # Same actual receipt time as the original SDK fixture; source mutations do
    # not refresh the receiver's clock or disguise a stale/future packet.
    received = packet()["captured_at_unix_ns"] + 1_000_000
    with pytest.raises(ValueError):
        parse(p, received_at_unix_ns=received)

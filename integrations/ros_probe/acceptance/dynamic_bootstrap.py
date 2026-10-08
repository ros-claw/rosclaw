"""Prepare fresh known-scene source bindings before a Gazebo process exists.

The grid is declared fixture policy, never a measured map. After launch,
run.py must compile the actual Body again and admit the measured map and
passive physics source before the daemon may execute any action.
"""

import argparse
import hashlib
import json
import re
import struct
import uuid
from pathlib import Path

from fixture_body import configure_fixture_body
from profiles import profile_for_urdf

from rosclaw.connectors.ros.verification.reachable import cleanable_cells


def prepare_bindings(output, *, urdf_path, library_path, mission_id, profile_name):
    if type(mission_id) is not str or not 1 <= len(mission_id) <= 256:
        raise ValueError("bounded nonempty mission id required")
    profile = profile_for_urdf(urdf_path, profile_name)
    plugin = library_path.read_bytes()
    if not 0 < len(plugin) <= 20_000_000 or not plugin.startswith(b"\x7fELF"):
        raise ValueError("bounded compiled passive plugin required")
    output.mkdir(exist_ok=False)
    body = {
        "body_id": profile.body_id,
        "base_frame": "base_footprint",
        "map_frame": "map",
        "physical_radius_m": profile.physical_radius_m,
        "cleaning_polygon": profile.cleaning_polygon,
    }
    body_hash = configure_fixture_body(output / "home", body, urdf_path)
    run_id = uuid.uuid4().hex
    brush = {
        "run_id": run_id,
        "body_snapshot_hash": body_hash,
        "attachment_hash": hashlib.sha256(
            json.dumps(
                {
                    "body_hash": body_hash,
                    "cleaning_polygon": profile.cleaning_polygon,
                    "kind": "explicit_sim_attachment",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "producer_id": "sim_actuator_" + run_id,
    }
    # Exact ROS wire representation; no live values are manufactured here.
    resolution = struct.unpack("f", struct.pack("f", 0.05))[0]
    occupancy = [
        100 if abs((x + 0.5) * 0.05 - 1.6) >= 1.5 or abs((y + 0.5) * 0.05 - 1.6) >= 1.5 else 0
        for y in range(64)
        for x in range(64)
    ]
    cells = cleanable_cells(
        width=64,
        height=64,
        resolution=resolution,
        occupancy=occupancy,
        start_cell=32 * 64 + 32,
        robot_radius=profile.physical_radius_m,
        cleaning_radius=profile.cleaner_half_width_m,
    )
    name = "ros_expert_temporal_obstacle"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", name):
        raise ValueError("invalid fixture obstacle identifier")
    binding = {
        **{k: brush[k] for k in ("run_id", "body_snapshot_hash", "attachment_hash")},
        "mission_id": mission_id,
        "world_name": "ros_expert",
        "body_model_name": profile.simulation_model,
        "obstacle_names": [name],
        "scene_model_names": [
            "floor",
            "east",
            "west",
            "north",
            "south",
            profile.simulation_model,
            name,
        ],
        "world_to_map_xyyaw": [0, 0, 0],
        "map_world_identity_approved": True,
        "frame_transform_source": "simulator_operator_fixture_policy",
        "grid": {
            "width": 64,
            "height": 64,
            "resolution": resolution,
            "origin": [-1.6, -1.6],
            "frame_id": "map",
            "accessible_cells": cells,
            "cleaning_polygon": profile.cleaning_polygon,
        },
    }
    fixture = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": binding,
        "plugin_sha256": hashlib.sha256(plugin).hexdigest(),
        "obstacles": [{"name": name, "pose": [5, 0, 0.1, 0, 0, 0], "box_size": [0.7, 0.7, 0.2]}],
    }
    for filename, value in (
        ("brush.json", brush),
        ("physics.json", fixture),
        (
            "bootstrap.json",
            {
                "evidence_role": "fixture_policy_and_compiled_body_not_live_evidence",
                "physical_acceptance": "NOT_RUN",
                "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                "compiled_body_hash": body_hash,
                "run_id": run_id,
                "requires_actual_map_and_source_readmission": True,
            },
        ),
    ):
        with (output / filename).open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--profile", choices=["waffle", "burger"], required=True)
    parser.add_argument("--mission-id", default="gazebo-room-cleaning")
    args = parser.parse_args()
    fixture = prepare_bindings(
        args.directory,
        urdf_path=args.urdf,
        library_path=args.plugin,
        mission_id=args.mission_id,
        profile_name=args.profile,
    )
    print(json.dumps({"run_id": fixture["binding"]["run_id"], "physical_acceptance": "NOT_RUN"}))


if __name__ == "__main__":
    main()

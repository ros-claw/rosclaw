"""Synthetic two-blocker source windows; no World or Native physical claim."""

import importlib
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.connectors.ros.test_dynamic_fixture_scenario import binding, policy
from tests.connectors.ros.test_physics_component_packets import packet, parse


def setup(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    module = importlib.import_module("dynamic_two_blocker_replay")
    b = binding()
    b["obstacle_names"].append("second_blocker")
    b["scene_model_names"].append("second_blocker")
    b["grid"] = {
        "width": 20,
        "height": 20,
        "resolution": 0.1,
        "origin": [-1, -1],
        "frame_id": "map",
        "accessible_cells": list(range(400)),
        "cleaning_polygon": [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
    }
    spec = policy()
    spec.update(
        case="D3",
        target_xy=[0.5, 0.5],
        second_obstacle_name="second_blocker",
        second_target_xy=[-0.5, -0.5],
        second_dwell_sim_sec=20,
        gap_sim_sec=5,
    )
    fixture = {
        "binding": b,
        "obstacles": [
            {"name": "anonymous_blocker", "pose": [5, 0, 0.1, 0, 0, 0]},
            {"name": "second_blocker", "pose": [5, 1, 0.1, 0, 0, 0]},
        ],
    }
    events = []
    packets = {}
    for index, (stage, state, name, stamp, first_xy, second_xy) in enumerate(
        [
            (0, "CONFIRMING_INTRODUCTION", "anonymous_blocker", 10, [0.5, 0.5], [5, 1]),
            (0, "CONFIRMING_WITHDRAWAL", "anonymous_blocker", 30, [5, 0], [5, 1]),
            (1, "CONFIRMING_INTRODUCTION", "second_blocker", 35, [5, 0], [-0.5, -0.5]),
            (1, "CONFIRMING_WITHDRAWAL", "second_blocker", 55, [5, 0], [5, 1]),
        ]
    ):
        p = packet()
        p.update(
            sequence=index, paused=False, sim_time_sec=stamp, captured_at_unix_ns=time.time_ns()
        )
        second = deepcopy(p["obstacles"][0])
        second.update(model_name="second_blocker", entity_id=103)
        second["collision_geometry"][0]["entity_id"] = 105
        p["scene_models"].append({"model_name": "second_blocker", "entity_id": 103})
        p["obstacles"].append(second)
        p["obstacles"][0]["world_pose"][:2] = first_xy
        p["obstacles"][1]["world_pose"][:2] = second_xy
        decoded = parse(
            p,
            obstacle_names=tuple(b["obstacle_names"]),
            scene_model_names=frozenset(b["scene_model_names"]),
        )
        sha = decoded["packet_sha256"]
        packets[sha] = decoded
        events.append(
            {
                "blocking_stage": stage,
                "state": state,
                "obstacle_name": name,
                "sim_time_sec": stamp,
                "actual_xy": first_xy if stage == 0 else second_xy,
                "packet_sha256": sha,
                "captured_at": datetime.fromtimestamp(
                    (p["captured_at_unix_ns"] + 1_000_000) / 1e9, UTC
                ).isoformat(),
            }
        )
    return module, fixture, spec, events, packets


def test_confirmations_require_two_actual_nonconcurrent_mask_windows(monkeypatch):
    module, fixture, spec, events, packets = setup(monkeypatch)
    windows = module.verify_confirmations(events, packets, fixture, spec)
    assert len(windows) == 4
    assert windows[0]["occupied_cells"] and windows[2]["occupied_cells"]
    assert not set(windows[0]["occupied_cells"]) & set(windows[2]["occupied_cells"])
    assert all(not row["other_model_occupied_cells"] for row in windows)
    assert windows[1]["occupied_cells"] == windows[3]["occupied_cells"] == []


@pytest.mark.parametrize(
    "fault",
    [
        "missing_confirmation",
        "reorder",
        "foreign_packet",
        "false_pose",
        "false_time",
        "stale_receipt",
        "short_gap",
        "short_dwell",
        "unknown_second",
        "concurrent",
        "changed_geometry",
    ],
)
def test_missing_stale_or_leaking_original_source_windows_refuse(monkeypatch, fault):
    module, fixture, spec, events, packets = setup(monkeypatch)
    if fault == "missing_confirmation":
        events.pop()
    elif fault == "reorder":
        events[1], events[2] = events[2], events[1]
    elif fault == "foreign_packet":
        events[0]["packet_sha256"] = "unknown"
    elif fault == "false_pose":
        events[0]["actual_xy"] = [0, 0]
    elif fault == "false_time":
        events[0]["sim_time_sec"] += 1
    elif fault == "stale_receipt":
        events[0]["captured_at"] = datetime.fromtimestamp(time.time() + 5, UTC).isoformat()
    elif fault == "short_gap":
        spec["gap_sim_sec"] = 6
    elif fault == "short_dwell":
        spec["dwell_sim_sec"] = 21
    elif fault == "unknown_second":
        spec["second_obstacle_name"] = "foreign"
    elif fault == "concurrent":
        from dataclasses import replace

        decoded = packets[events[2]["packet_sha256"]]
        decoded["model_poses"] = tuple(
            replace(pose, x=0.5, y=0.5) if pose.model_name == "anonymous_blocker" else pose
            for pose in decoded["model_poses"]
        )
    else:
        from dataclasses import replace

        decoded = packets[events[2]["packet_sha256"]]
        decoded["geometry"] = replace(decoded["geometry"], component_geometry_hash="changed")
    with pytest.raises(ValueError):
        module.verify_confirmations(events, packets, fixture, spec)

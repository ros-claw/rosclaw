"""Explicit synthetic crossing/source tests; no World, DDS or robot actions."""

import importlib
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from tests.connectors.ros.test_dynamic_fixture_scenario import binding, policy
from tests.connectors.ros.test_physics_component_packets import packet, parse
from tests.connectors.ros.test_qualified_backend_episode import protocol


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("crossing_fixture"), importlib.import_module(
        "crossing_source_replay"
    )


def crossing_policy():
    return {
        "crossing_end_xy": [0, 1],
        "step_m": 0.05,
        "interval_sim_sec": 0.25,
        "maximum_crossing_sim_sec": 30,
    }


def swath_row():
    return {
        "kind": "swaths",
        "sim_time_sec": 1,
        "artifact_sha256": "swath",
        "payload": {
            "topic": "/coverage_server/swaths",
            "frame_id": "map",
            "marker_type": 5,
            "marker_action": 0,
            "points": [[-1, 0], [1, 0]],
        },
    }


def test_source_waypoints_are_bounded_small_steps_and_transversely_cross_actual_swaths(modules):
    source, _ = modules
    points = source.crossing_waypoints([0, -1], crossing_policy())
    assert points[0] == [0, -1] and points[-1] == [0, 1] and len(points) == 41
    assert all(math.dist(a, b) <= 0.050000001 for a, b in zip(points, points[1:], strict=False))
    assert source.crossing_intersects_swaths(points[0], points[-1], swath_row(), frame_id="map")


@pytest.mark.parametrize(
    "fault", ["large_step", "fast_step", "unbounded_time", "nan", "same_point", "extra"]
)
def test_unbounded_or_vacuous_crossing_policy_refuses(modules, fault):
    source, _ = modules
    p = crossing_policy()
    if fault == "large_step":
        p["step_m"] = 0.051
    elif fault == "fast_step":
        p["interval_sim_sec"] = 0.1
    elif fault == "unbounded_time":
        p["maximum_crossing_sim_sec"] = 31
    elif fault == "nan":
        p["crossing_end_xy"][1] = float("nan")
    elif fault == "same_point":
        p["crossing_end_xy"] = [0, -1]
    else:
        p["ignore_main_goal"] = True
    with pytest.raises(ValueError):
        source.crossing_waypoints([0, -1], p)


@pytest.mark.parametrize(
    "fault", ["parallel", "endpoint", "wrong_frame", "delete", "not_line_list"]
)
def test_touching_parallel_or_unbound_debug_path_is_not_a_main_swath_crossing(modules, fault):
    source, _ = modules
    row = swath_row()
    if fault == "parallel":
        row["payload"]["points"] = [[0, -0.5], [0, 0.5]]
    elif fault == "endpoint":
        row["payload"]["points"] = [[0, 0], [1, 0]]
    elif fault == "wrong_frame":
        row["payload"]["frame_id"] = "odom"
    elif fault == "delete":
        row["payload"]["marker_action"] = 2
    else:
        row["payload"]["marker_type"] = 4
    assert not source.crossing_intersects_swaths([0, -1], [0, 1], row, frame_id="map")


def test_crossing_keeps_target_guard_and_also_screens_full_segment(modules):
    source, _ = modules
    fixture = {
        "schema_version": "rosclaw.sim_physics_fixture.v1",
        "binding": {"body_snapshot_hash": "body", "obstacle_names": ["blocker"]},
        "obstacles": [
            {"name": "blocker", "pose": [5, 0, 0.01, 0, 0, 0], "box_size": [0.01, 0.01, 0.02]}
        ],
    }
    body = {"effective_body_hash": "body", "physical_radius_m": 0.1}
    sample = {
        "x": 0,
        "y": 0,
        "observation_complete": True,
        "collision_count": 0,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    assert source.crossing_pose_request(
        fixture, body, sample, name="blocker", previous_xy=[1, 0], target_xy=[1.05, 0]
    )
    # Target passes the original guard, but the start of the short segment does not.
    with pytest.raises(ValueError, match="segment violates"):
        source.crossing_pose_request(
            fixture, body, sample, name="blocker", previous_xy=[0.2, 0], target_xy=[0.25, 0]
        )
    with pytest.raises(ValueError, match="cannot jump"):
        source.crossing_pose_request(
            fixture, body, sample, name="blocker", previous_xy=[1, 0], target_xy=[2, 0]
        )
    with pytest.raises(ValueError, match="finite actual previous"):
        source.crossing_pose_request(
            fixture, body, sample, name="blocker", previous_xy=[float("nan"), 0], target_xy=[1, 0]
        )


def synthetic_confirmations(modules):
    source, _ = modules
    b = {**binding(), "grid": {"frame_id": "map"}}
    spec = {
        **policy(),
        "case": "D1",
        "target_xy": [0, -1],
        "dwell_sim_sec": 10,
        "crossing_source": crossing_policy(),
    }
    fixture = {
        "binding": b,
        "obstacles": [{"name": "anonymous_blocker", "pose": [5, 0, 0.1, 0, 0, 0]}],
    }
    start_wall = time.time() - 60
    context = {
        "run_id": b["run_id"],
        "body_snapshot_hash": b["body_snapshot_hash"],
        "action_id": "action",
    }
    audit = [
        {**context, "kind": "action_admitted", "payload": {"capability_id": "coverage.execute"}},
        {
            **context,
            "kind": "goal_started",
            "artifact_sha256": "main",
            "captured_at": datetime.fromtimestamp(start_wall, UTC).isoformat(),
            "payload": {"stage": "MAIN_COVERAGE", "nav_goal_id": "goal"},
        },
        {
            **context,
            "kind": "goal_ended",
            "captured_at": datetime.fromtimestamp(start_wall + 40, UTC).isoformat(),
            "payload": {"nav_goal_id": "goal"},
        },
    ]
    main = {"action_id": "action", "nav_goal_id": "goal", "audit_event_sha256": "main"}
    points = source.crossing_waypoints(spec["target_xy"], spec["crossing_source"])
    events, packets, trajectory = [], {}, []
    for i, xy in enumerate([*points, [5, 0]]):
        p = packet()
        stamp = 10 + 0.3 * i
        p.update(
            sequence=i,
            sim_time_sec=stamp,
            paused=False,
            captured_at_unix_ns=int((start_wall + 1 + 0.3 * i) * 1e9),
        )
        p["obstacles"][0]["world_pose"][:2] = xy
        decoded = parse(p)
        sha = decoded["packet_sha256"]
        packets[sha] = decoded
        events.append(
            {
                "state": "CONFIRMING_INTRODUCTION"
                if i == 0
                else "CONFIRMING_WITHDRAWAL"
                if i == len(points)
                else "CONFIRMING_CROSSING",
                "crossing_index": min(i, len(points) - 1),
                "packet_sha256": sha,
                "actual_xy": xy,
                "sim_time_sec": stamp,
                "captured_at": datetime.fromtimestamp(
                    p["captured_at_unix_ns"] / 1e9 + 0.001, UTC
                ).isoformat(),
                "main_coverage": main.copy(),
                "swath_event_sha256": "swath",
            }
        )
        trajectory.append({"time_sec": stamp, "cleaning_enabled": True})
    swath = swath_row()
    swath["captured_at"] = datetime.fromtimestamp(start_wall + 0.5, UTC).isoformat()
    return (
        events,
        packets,
        {"swath": swath},
        audit,
        spec,
        fixture,
        {"action_ids": ["action"], "trajectory": trajectory},
    )


def test_every_step_matches_original_component_packet_and_one_canonical_main_interval(modules):
    _, replay = modules
    result = replay.verify_crossing_confirmations(*synthetic_confirmations(modules))
    assert result["confirmed_waypoints"] == 41 and result["physical_acceptance"] == "NOT_VERIFIED"


@pytest.mark.parametrize(
    "fault",
    [
        "skip",
        "reorder",
        "ack_only",
        "different_goal",
        "after_main",
        "before_main",
        "wrong_pose",
        "stale",
        "wrong_frame",
        "not_crossing",
        "fast",
        "deadline",
        "brush_off",
        "foreign_action",
        "geometry",
        "missing_cleaning_interval",
        "old_swath",
    ],
)
def test_missing_unbound_or_outside_main_original_motion_refuses(modules, fault):
    _, replay = modules
    events, packets, swaths, audit, spec, fixture, evidence = synthetic_confirmations(modules)
    if fault == "skip":
        events.pop(3)
    elif fault == "reorder":
        events[3], events[4] = events[4], events[3]
    elif fault == "ack_only":
        events[3]["packet_sha256"] = "service_ack"
    elif fault == "different_goal":
        events[3]["main_coverage"]["nav_goal_id"] = "other"
    elif fault == "after_main":
        audit[-1]["captured_at"] = events[3]["captured_at"]
    elif fault == "before_main":
        audit[1]["captured_at"] = events[3]["captured_at"]
    elif fault == "wrong_pose":
        events[3]["actual_xy"] = [1, 1]
    elif fault == "stale":
        events[3]["captured_at"] = datetime.now(UTC).isoformat()
    elif fault == "wrong_frame":
        swaths["swath"]["payload"]["frame_id"] = "odom"
    elif fault == "not_crossing":
        swaths["swath"]["payload"]["points"] = [[1, -0.5], [1, 0.5]]
    elif fault == "fast":
        spec["crossing_source"]["interval_sim_sec"] = 0.5
    elif fault == "deadline":
        spec["crossing_source"]["maximum_crossing_sim_sec"] = 10
    elif fault == "brush_off":
        evidence["trajectory"][4]["cleaning_enabled"] = False
    elif fault == "foreign_action":
        evidence["action_ids"] = ["other"]
    elif fault == "missing_cleaning_interval":
        evidence["trajectory"] = evidence["trajectory"][10:20]
    elif fault == "old_swath":
        swaths["swath"]["captured_at"] = audit[0].get(
            "captured_at", datetime.fromtimestamp(0, UTC).isoformat()
        )
    else:
        from dataclasses import replace

        decoded = packets[events[3]["packet_sha256"]]
        decoded["geometry"] = replace(decoded["geometry"], component_geometry_hash="changed")
    with pytest.raises(ValueError):
        replay.verify_crossing_confirmations(
            events, packets, swaths, audit, spec, fixture, evidence
        )


def test_d1_qualified_protocol_preserves_merge_model_body_and_budget(modules):
    qualified = importlib.import_module("qualified_backend_episode")
    episode = importlib.import_module("dynamic_native_episode")
    spec = {
        **protocol(),
        "schema_version": "rosclaw.dynamic_native_episode.v6",
        "case": "D1",
        "target_xy": [0, -1],
        "dwell_sim_sec": 10,
        "scenario_source": crossing_policy(),
    }
    base, backend = qualified.validate_qualified_spec(spec, episode.validate_episode_spec)
    assert base["case"] == "D1" and base["mission_timeout_sec"] == spec["mission_timeout_sec"]
    assert base["p0_merge_commit"] == spec["p0_merge_commit"] and backend == spec["backend_source"]


def test_live_main_goal_cursor_does_not_infer_active_from_cleaner_state(modules, tmp_path):
    source, _ = modules
    b = binding()
    window = source.MainCoverageWindow(tmp_path, b)
    assert window.poll() is None
    (tmp_path / "actions").mkdir()
    path = tmp_path / "actions/coverage-audit-synthetic.jsonl"
    context = {
        "schema_version": "rosclaw.coverage_audit_event.v1",
        "run_id": b["run_id"],
        "body_snapshot_hash": b["body_snapshot_hash"],
        "action_id": "action",
    }
    rows = []

    def append(kind, payload):
        row = {
            **context,
            "kind": kind,
            "payload": payload,
            "sequence": len(rows) + 1,
            "previous_hash": rows[-1]["artifact_sha256"] if rows else None,
        }
        row["artifact_sha256"] = digest(row)
        rows.append(row)
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    append("action_admitted", {"capability_id": "coverage.execute"})
    assert window.poll() is None
    append("goal_started", {"stage": "MAIN_COVERAGE", "nav_goal_id": "goal"})
    assert window.poll()["nav_goal_id"] == "goal"
    append("goal_ended", {"nav_goal_id": "goal"})
    assert window.poll() is None

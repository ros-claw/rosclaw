"""Closed-loop endpoint proximity must not collapse both boundary halves."""

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission.boundary_pass import inset_corner_boundary_targets
from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor

STRATEGY = "through_poses_tracking_inset_open_segments"


def fixture(tmp_path):
    bt = b"owned fixture BT bytes"
    (tmp_path / "boundary-through-poses.xml").write_bytes(bt)
    centers = [(x / 10, y / 10) for x in range(-3, 4) for y in range(-3, 4)]
    return RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(fresh=lambda: {"x": 0.29, "y": 0.29}),
        output=tmp_path / "actions",
        body_id="fixture",
        body_snapshot_hash="body",
        grid={"frame_id": "map", "resolution": 0.1},
        recovery_centers=centers,
        boundary_pass=True,
        boundary_strategy=STRATEGY,
        boundary_tracking_bt_sha256=hashlib.sha256(bt).hexdigest(),
    )


def test_two_open_halves_preserve_route_and_share_original_budget(tmp_path, monkeypatch):
    e = fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr("rosclaw.connectors.ros.mission.executor.time.monotonic", lambda: clock[0])
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        clock[0] += 70
        return {"status": 4}

    e._run_goal = run
    result = e._boundary({"status": 4}, "root", 999)
    assert result["status"] == "SUCCEEDED" and result["segment_count"] == 2
    assert result["waypoint_count"] == 9 and len(calls) == 2
    assert [k["goal_timeout_sec"] for _, k in calls] == [180, 110]
    assert [a[3] for a, _ in calls] == ["root:boundary:0", "root:boundary:1"]
    original = inset_corner_boundary_targets(e.boundary_centers, e.witness.fresh(), resolution=0.1)
    dispatched = []
    for args, kwargs in calls:
        assert args[:2] == ("/navigate_through_poses", "nav2_msgs/action/NavigateThroughPoses")
        assert args[4] == 280 and kwargs["stage"] == "BOUNDARY_PASS"
        assert args[2]["behavior_tree"] == "/evidence/boundary-through-poses.xml"
        poses = args[2]["poses"]
        assert len(poses) == 5 and poses[0] != poses[-1]
        dispatched.append([(p["pose"]["position"]["x"], p["pose"]["position"]["y"]) for p in poses])
    assert dispatched[0][-1] == dispatched[1][0]
    assert dispatched[0] + dispatched[1][1:] == [(p["x"], p["y"]) for p in original]
    assert "coverage_ratio" not in result


@pytest.mark.parametrize(
    "terminal", [{"status": 5}, {"status": 6}, {"status": 4, "timed_out": True}]
)
def test_first_failed_or_cancelled_half_prevents_second_dispatch(tmp_path, terminal, monkeypatch):
    e = fixture(tmp_path)
    monkeypatch.setattr("rosclaw.connectors.ros.mission.executor.time.monotonic", lambda: 100.0)
    calls = []
    e._run_goal = lambda *a, **kw: calls.append((a, kw)) or terminal
    result = e._boundary({"status": 4}, "root", 999)
    assert result["status"] == "FAILED" and result["nav_goal_results"] == [terminal]
    assert len(calls) == 1


def test_exhausted_shared_budget_does_not_dispatch_second_half(tmp_path, monkeypatch):
    e = fixture(tmp_path)
    clock = [100.0]
    monkeypatch.setattr("rosclaw.connectors.ros.mission.executor.time.monotonic", lambda: clock[0])
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        clock[0] += 180
        return {"status": 4}

    e._run_goal = run
    result = e._boundary({"status": 4}, "root", 999)
    assert result["status"] == "FAILED" and "budget" in result["reason"]
    assert len(calls) == 1


def test_original_action_deadline_caps_both_halves(tmp_path, monkeypatch):
    e = fixture(tmp_path)
    monkeypatch.setattr("rosclaw.connectors.ros.mission.executor.time.monotonic", lambda: 100.0)
    calls = []
    e._run_goal = lambda *a, **kw: calls.append((a, kw)) or {"status": 4}
    result = e._boundary({"status": 4}, "root", 140)
    assert result["status"] == "SUCCEEDED"
    assert all(a[4] == 140 and k["goal_timeout_sec"] == 40 for a, k in calls)


def test_changed_owned_bt_between_halves_prevents_second_dispatch(tmp_path):
    e = fixture(tmp_path)
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        (tmp_path / "boundary-through-poses.xml").write_bytes(b"changed after first half")
        return {"status": 4}

    e._run_goal = run
    with pytest.raises(RuntimeError, match="changed|hash|SHA|mismatch"):
        e._boundary({"status": 4}, "root", float("inf"))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "body,base",
    [
        ("waffle", "perimeter_stateless_overlap"),
        ("burger", "perimeter_stateless_clearance"),
    ],
)
def test_open_segments_require_exact_registration_and_preserve_body_parameters(
    monkeypatch, body, base
):
    root = Path(__file__).resolve().parents[3]
    runner = root / "integrations/ros_probe/acceptance"
    monkeypatch.syspath_prepend(str(runner))
    import experiments

    spec = importlib.util.spec_from_file_location(
        "open_boundary_pairs", runner / "paired_efficiency.py"
    )
    pairs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pairs)
    preset = base + "_boundary_tracking_open_segments"
    profile = SimpleNamespace(name=body, coverage_width_m=0.5 if body == "waffle" else 0.3)
    assert experiments.planning_parameters(profile, preset) == experiments.planning_parameters(
        profile, base
    )
    assert experiments.controller_parameters(profile, preset) == experiments.controller_parameters(
        profile, base
    )
    assert experiments.continuous_boundary_strategy(preset) == STRATEGY
    valid = {
        "preset": preset,
        "profile": body,
        "boundary_pass": True,
        "boundary_strategy": STRATEGY,
        "boundary_stage_budget_sec": 180,
        "boundary_waypoint_count": 9,
        "boundary_segment_count": 2,
        "precise_through_poses": True,
        "boundary_tracking_prune_radius_m": 0.1,
        "boundary_tracking_bt_sha256": "a" * 64,
        "boundary_corner_inset_cells": 1,
    }
    protocol = {"candidate_" + k: v for k, v in valid.items() if k.startswith("boundary_")}
    protocol["precise_repair_waypoints"] = True
    experiments.validate_continuous_boundary_experiment(valid)
    experiments.validate_boundary_tracking_runtime_registration(valid, protocol)
    pairs.validate_continuous_boundary_registration(protocol, preset, True)
    for bad in [None, True, 1, 3, 2.0, "2"]:
        with pytest.raises(ValueError, match="two registered segments"):
            experiments.validate_continuous_boundary_experiment(
                {**valid, "boundary_segment_count": bad}
            )
        for validator in [
            lambda p: pairs.validate_continuous_boundary_registration(p, preset, True),
            lambda p: experiments.validate_boundary_tracking_runtime_registration(valid, p),
        ]:
            with pytest.raises(ValueError, match="two registered segments"):
                validator({**protocol, "candidate_boundary_segment_count": bad})
    legacy = base + "_boundary_tracking_inset_corners"
    with pytest.raises(ValueError, match="registered candidate"):
        experiments.validate_continuous_boundary_experiment({**valid, "preset": legacy})
    with pytest.raises(ValueError, match="registered candidate"):
        pairs.validate_continuous_boundary_registration(protocol, legacy, True)

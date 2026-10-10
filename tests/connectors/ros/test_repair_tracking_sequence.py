"""Explicit repair tracking predictions/BTs never grant physical coverage."""

import hashlib
import importlib
import math
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw.connectors.ros.mission import executor as executor_module
from rosclaw.connectors.ros.mission import repair_optimizer
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier


@pytest.fixture
def fixture_modules(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("precise_through_poses_bt"), importlib.import_module(
        "experiments"
    )


def test_one_action_overhead_is_shared_only_by_explicit_continuous_candidate():
    grid = {
        "width": 3,
        "height": 1,
        "resolution": 0.1,
        "origin": [0, 0],
        "accessible_cells": [0, 1, 2],
        "cleaning_polygon": [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
    }
    args = (
        grid,
        [(0.05, 0.05), (0.15, 0.05), (0.25, 0.05)],
        {1, 2},
        {"x": 0.05, "y": 0.05, "yaw": 0.0},
    )
    legacy = repair_optimizer.rank_repair_poses(*args, goal_overhead_sec=10)
    candidate = repair_optimizer.rank_repair_poses(
        *args, goal_overhead_sec=10, shared_sequence_overhead=True
    )
    assert legacy.status == candidate.status == "READY"
    assert len(legacy.poses) == 1 and len(candidate.poses) == 2
    assert (
        candidate.cost_model == "STATIC_LEGAL_CENTER_GRID_SHARED_SEQUENCE_OVERHEAD_PREDICTION_ONLY"
    )
    assert legacy.cost_model == "STATIC_LEGAL_CENTER_GRID_PREDICTION_ONLY"
    assert candidate.predicted_dispatch_cost_sec == pytest.approx(11.0)
    assert legacy.predicted_dispatch_cost_sec is None
    assert {c for p in candidate.poses for c in p.predicted_new_cells} == {1, 2}


def test_separate_repair_checkpoint_bt_preserves_global_precise_and_boundary_bytes(
    fixture_modules, tmp_path
):
    bt, _ = fixture_modules
    original = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/ros_expert/nav2/installed-through-poses.xml"
    ).read_bytes()
    bt.prepare_precise_through_poses_bt(tmp_path, original, xy_goal_tolerance=0.025)
    bt.prepare_boundary_tracking_through_poses_bt(
        tmp_path,
        original,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    report = bt.prepare_repair_tracking_through_poses_bt(
        tmp_path,
        original,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )
    assert all((tmp_path / name).read_bytes() == raw for name, raw in before.items())
    assert (tmp_path / "repair-tracking-through-poses.xml").read_bytes() == before[
        "boundary-through-poses.xml"
    ]
    assert report["scope"] == "REPAIR_INTERMEDIATE_CHECKPOINTS_ONLY"
    assert report["unchanged_final_goal_xy_tolerance_m"] == 0.025
    assert report["global_precise_repair_bt_changed"] is False
    assert report["actual_waypoint_reached"] == "NOT_MEASURED"


def test_repair_checkpoint_runtime_gate_requires_exact_frozen_metadata(fixture_modules):
    _, experiments = fixture_modules
    experiment = {
        "preset": "perimeter_stateless_overlap_boundary_tracking_inset_corners",
        "profile": "waffle",
        "precise_through_poses": True,
        "repair_tracking_sequence": True,
        "repair_tracking_prune_radius_m": 0.1,
        "repair_tracking_bt_sha256": "a" * 64,
        "repair_shared_sequence_overhead": True,
    }
    protocol = {
        "precise_repair_waypoints": True,
        "selected_repair_strategies": {"waffle": "pose_aware_robust_tracking_sequence"},
        **{"candidate_" + k: v for k, v in experiment.items() if k.startswith("repair_")},
    }
    experiments.validate_repair_tracking_runtime_registration(experiment, protocol)
    for key in [
        "candidate_repair_tracking_bt_sha256",
        "candidate_repair_shared_sequence_overhead",
        "candidate_repair_tracking_prune_radius_m",
    ]:
        missing = {k: v for k, v in protocol.items() if k != key}
        with pytest.raises(ValueError):
            experiments.validate_repair_tracking_runtime_registration(experiment, missing)
    with pytest.raises(ValueError):
        experiments.validate_repair_tracking_runtime_registration(
            experiment, {**protocol, "candidate_repair_tracking_prune_radius_m": True}
        )
    disabled = {k: v for k, v in experiment.items() if not k.startswith("repair_")}
    with pytest.raises(ValueError, match="missing from generated"):
        experiments.validate_repair_tracking_runtime_registration(disabled, protocol)
    experiments.validate_repair_tracking_runtime_registration({"preset": "baseline"}, protocol)
    with pytest.raises(ValueError, match="explicit repair strategy"):
        experiments.validate_repair_tracking_candidate_registration(
            protocol, experiment["preset"], "pose_aware_robust_sequence", True
        )


@pytest.mark.parametrize("value", [None, 1, 0, "true", 0.1])
def test_shared_overhead_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        repair_optimizer.rank_repair_poses({}, [], [], {}, shared_sequence_overhead=value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("repair_tracking_sequence", 1),
        ("repair_tracking_prune_radius_m", True),
        ("repair_tracking_prune_radius_m", 0.7),
        ("repair_tracking_prune_radius_m", float("nan")),
        ("repair_tracking_bt_sha256", "A" * 64),
        ("repair_shared_sequence_overhead", 1),
        ("preset", "baseline"),
        ("profile", "burger"),
        ("precise_through_poses", False),
    ],
)
def test_invalid_generated_tracking_declarations_are_refused(fixture_modules, field, value):
    _, experiments = fixture_modules
    experiment = {
        "preset": "perimeter_stateless_overlap_boundary_tracking_inset_corners",
        "profile": "waffle",
        "precise_through_poses": True,
        "repair_tracking_sequence": True,
        "repair_tracking_prune_radius_m": 0.1,
        "repair_tracking_bt_sha256": "a" * 64,
        "repair_shared_sequence_overhead": True,
    }
    with pytest.raises(ValueError):
        experiments.validate_repair_tracking_experiment({**experiment, field: value})


def prepared_executor(tmp_path, bt):
    original = (
        Path(__file__).resolve().parents[3]
        / "tests/fixtures/ros_expert/nav2/installed-through-poses.xml"
    ).read_bytes()
    boundary = bt.prepare_boundary_tracking_through_poses_bt(
        tmp_path,
        original,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )
    tracking = bt.prepare_repair_tracking_through_poses_bt(
        tmp_path,
        original,
        xy_goal_tolerance=0.025,
        controller_lookahead_m=0.1,
        tracking_radius_m=0.1,
    )
    grid = {
        "width": 4,
        "height": 4,
        "resolution": 1.0,
        "origin": [0.0, 0.0],
        "frame_id": "map",
        "accessible_cells": list(range(16)),
        "cleaning_polygon": [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]],
    }
    observed = []
    witness = SimpleNamespace(
        fresh=lambda: {"x": 1.5, "y": 1.5, "yaw": 0.0}, since=lambda start: observed[start:]
    )
    kwargs = {
        "owner": "daemon_test",
        "client": None,
        "control": None,
        "witness": witness,
        "output": tmp_path / "actions",
        "body_id": "fixture",
        "body_snapshot_hash": "body",
        "grid": grid,
        "recovery_centers": [(1.5, 1.5), (2.5, 1.5)],
        "boundary_pass": True,
        "boundary_strategy": "through_poses_tracking_inset_corners",
        "boundary_tracking_bt_sha256": boundary["source_output_sha256"],
        "repair_strategy": "pose_aware_robust_tracking_sequence",
        "repair_tracking_bt_sha256": tracking["source_output_sha256"],
    }
    return kwargs, observed


@pytest.mark.parametrize("status,count", [("READY", 2), ("READY", 1), ("BUDGET_EXCEEDED", 0)])
def test_only_two_pose_candidate_dispatch_uses_container_bt_and_observed_credit(
    fixture_modules, tmp_path, monkeypatch, status, count
):
    bt, _ = fixture_modules
    kwargs, observed = prepared_executor(tmp_path, bt)
    driver = executor_module.RosCoverageSimulationExecutor(**kwargs)
    verifier = CoverageVerifier(**kwargs["grid"])
    poses = tuple(
        repair_optimizer.RepairPose(x, 1.5, 0.0, tuple(range(16)), 1.0, 0.0, 5.0, 1.0, int(x))
        for x in [1.5, 2.5]
    )

    def rank(*args, **kw):
        assert kw["shared_sequence_overhead"] is True and kw["robust_footprint"] is True
        return repair_optimizer.RepairSelection(status, poses[:count])

    calls = []
    deadline = time.monotonic() + 30

    def goal(name, action_type, args, goal_id, actual_deadline):
        assert not verifier.visits
        assert actual_deadline == deadline
        if count == 2:
            assert (name, action_type) == (
                "/navigate_through_poses",
                "nav2_msgs/action/NavigateThroughPoses",
            )
            assert args["behavior_tree"] == "/evidence/repair-tracking-through-poses.xml"
            assert len(args["poses"]) == 2
        else:
            assert (name, action_type) == ("/navigate_to_pose", "nav2_msgs/action/NavigateToPose")
            assert "behavior_tree" not in args and "poses" not in args
            if count == 0:
                orientation = args["pose"]["pose"]["orientation"]
                assert 2 * math.atan2(orientation["z"], orientation["w"]) == pytest.approx(
                    math.pi / 4
                )
        calls.append(args)
        observed.append(
            {
                "x": 2.5,
                "y": 1.5,
                "yaw": 0.0,
                "time_sec": 1.0,
                "cleaning_enabled": True,
                "observation_complete": True,
                "collision_count": 0,
            }
        )
        return {"status": 4, "result": {"error_code": 0}}

    monkeypatch.setattr(executor_module, "rank_repair_poses", rank)
    driver._run_goal = goal
    records = driver._repair(verifier, 0, "root", deadline)
    assert len(calls) == len(records) == 1 and records[0]["waypoint_count"] == max(1, count)
    assert verifier.result()["coverage_ratio"] == 1.0


@pytest.mark.parametrize("arm,expected", [("candidate", True), ("baseline", False)])
def test_only_explicit_candidate_stack_gets_repair_tracking_flag(
    fixture_modules, tmp_path, monkeypatch, arm, expected
):
    paired = importlib.import_module("paired_efficiency")
    (tmp_path / "protocol.json").write_text("{}")
    args = SimpleNamespace(
        profile="waffle",
        seed=123,
        candidate="perimeter_stateless_overlap_boundary_tracking_inset_corners",
        candidate_repair_strategy="pose_aware_robust_tracking_sequence",
        precise_repair_waypoints=True,
        port_base=20191,
        domain_base=81,
        mission_timeout=900,
        image="fixture",
    )
    calls = []

    def refuse(argv, **kwargs):
        calls.append(argv)
        raise RuntimeError("instrumented fixture refuses every process launch")

    monkeypatch.setattr(paired, "command", refuse)
    result = paired.run_arm(tmp_path, arm, args, 0, "image", "source")
    assert result["status"] == "FAIL" and len(calls) == 1
    assert ("--repair-tracking-sequence" in calls[0][-1]) is expected


def test_tracking_bt_source_rechecked_and_legacy_override_refused(fixture_modules, tmp_path):
    bt, _ = fixture_modules
    kwargs, _ = prepared_executor(tmp_path, bt)
    driver = executor_module.RosCoverageSimulationExecutor(**kwargs)
    path = tmp_path / "repair-tracking-through-poses.xml"
    path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA256"):
        driver._verified_repair_tracking_bt()
    with pytest.raises(ValueError, match="other repair strategies"):
        executor_module.RosCoverageSimulationExecutor(
            **{**kwargs, "repair_strategy": "pose_aware_robust_sequence"}
        )


@pytest.mark.parametrize("raw", [None, b"x" * 128001, "symlink"])
def test_missing_or_oversized_tracking_bt_refused(fixture_modules, tmp_path, raw):
    bt, _ = fixture_modules
    kwargs, _ = prepared_executor(tmp_path, bt)
    path = tmp_path / "repair-tracking-through-poses.xml"
    if raw is None:
        path.unlink()
    elif raw == "symlink":
        target = tmp_path / "redirected.xml"
        path.rename(target)
        path.symlink_to(target)
    else:
        path.write_bytes(raw)
        kwargs["repair_tracking_bt_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(RuntimeError):
        executor_module.RosCoverageSimulationExecutor(**kwargs)


@pytest.mark.parametrize("digest", [None, True, "a" * 63, "G" * 64])
def test_explicit_tracking_strategy_requires_valid_digest(fixture_modules, tmp_path, digest):
    bt, _ = fixture_modules
    kwargs, _ = prepared_executor(tmp_path, bt)
    with pytest.raises(ValueError, match="source-bound inset fixture"):
        executor_module.RosCoverageSimulationExecutor(
            **{**kwargs, "repair_tracking_bt_sha256": digest}
        )

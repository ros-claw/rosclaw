"""Pose predictions must be bounded, body-aware and separate from measured credit."""

import copy
import math
import time

import pytest

from rosclaw.connectors.ros.mission import executor as executor_module
from rosclaw.connectors.ros.mission import repair_optimizer
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier


def grid(width=12, resolution=0.25, polygon=None):
    return {
        "width": width,
        "height": width,
        "resolution": resolution,
        "origin": [0.0, 0.0],
        "accessible_cells": list(range(width * width)),
        "cleaning_polygon": polygon or [[-0.45, -0.15], [0.45, -0.15], [0.45, 0.15], [-0.45, 0.15]],
        "frame_id": "map",
    }


def test_asymmetric_cleaner_selects_useful_yaw_without_credit_or_mutation():
    specification = grid()
    verifier = CoverageVerifier(**specification)
    remaining = {5 * 12 + 4}
    current = {"x": 1.125, "y": 1.125, "yaw": 0.0}
    before = copy.deepcopy((specification, remaining, current, verifier.result()))
    result = repair_optimizer.rank_repair_poses(specification, [(1.125, 1.125)], remaining, current)
    assert result.status == "READY"
    assert abs(result.poses[0].yaw) == pytest.approx(math.pi / 2)
    assert result.poses[0].predicted_new_cells == (64,)
    assert (specification, remaining, current, verifier.result()) == before


def test_static_cost_does_not_cross_disconnected_diagonal():
    specification = grid(2, 1.0, [[-0.4, -0.4], [0.4, -0.4], [0.4, 0.4], [-0.4, 0.4]])
    current = {"x": 0.5, "y": 0.5, "yaw": 0.0}
    result = repair_optimizer.rank_repair_poses(
        specification, [(0.5, 0.5), (1.5, 1.5)], {3}, current
    )
    assert result.status == "NO_CANDIDATE"
    connected = repair_optimizer.rank_repair_poses(
        specification, [(0.5, 0.5), (1.5, 0.5), (0.5, 1.5), (1.5, 1.5)], {3}, current
    )
    assert connected.status == "READY"
    assert connected.poses[0].predicted_access_distance_m == pytest.approx(math.sqrt(2))


def test_exhausted_retry_cells_have_no_predicted_reward():
    specification = grid()
    attempts = {64: 3}
    result = repair_optimizer.rank_repair_poses(
        specification,
        [(1.125, 1.125)],
        {64},
        {"x": 1.125, "y": 1.125, "yaw": 0.0},
        attempts=attempts,
    )
    assert result.status == "NO_CANDIDATE"
    assert attempts == {64: 3}


def test_search_budget_returns_no_partial_permission(monkeypatch):
    clock = iter([0.0, 0.01, 0.02, 0.03])
    monkeypatch.setattr(repair_optimizer.time, "monotonic", lambda: next(clock))
    result = repair_optimizer.rank_repair_poses(
        grid(), [(1.125, 1.125)], {64}, {"x": 1.125, "y": 1.125, "yaw": 0.0}, budget_ms=5
    )
    assert result.status == "BUDGET_EXCEEDED"
    assert result.poses == ()


def test_map_mismatch_and_invalid_costs_are_rejected():
    for budget in [-1, 0, float("inf")]:
        with pytest.raises(ValueError, match="budget"):
            repair_optimizer.rank_repair_poses(
                grid(),
                [(1.125, 1.125)],
                {64},
                {"x": 1.125, "y": 1.125, "yaw": 0.0},
                budget_ms=budget,
            )
    with pytest.raises(ValueError, match="denominator"):
        repair_optimizer.rank_repair_poses(
            grid(), [(1.125, 1.125)], {999}, {"x": 1.125, "y": 1.125, "yaw": 0.0}
        )


@pytest.mark.parametrize("strategy", ["greedy", "pose_aware"])
def test_budget_fallback_keeps_original_dispatch_and_requires_measured_pose(
    tmp_path, monkeypatch, strategy
):
    specification = grid(4, 1.0, [[-3, -3], [3, -3], [3, 3], [-3, 3]])
    verifier = CoverageVerifier(**specification)
    observed = []
    audit = []
    ranking_calls = []

    class Witness:
        def since(self, start):
            return observed[start:]

        def fresh(self):
            return {"x": 1.5, "y": 1.5, "yaw": 0.0}

    driver = executor_module.RosCoverageSimulationExecutor(
        owner="daemon_mock",
        client=None,
        control=None,
        witness=Witness(),
        output=tmp_path,
        body_id="mock",
        body_snapshot_hash="mock",
        grid=specification,
        recovery_centers=[(1.5, 1.5)],
        repair_strategy=strategy,
    )

    def timeout_rank(*args, **kwargs):
        ranking_calls.append(True)
        return repair_optimizer.RepairSelection("BUDGET_EXCEEDED")

    def goal(name, action_type, args, goal_id, deadline):
        assert not verifier.visits  # Neither a prediction nor dispatch grants credit.
        quaternion = args["pose"]["pose"]["orientation"]
        yaw = 2 * math.atan2(quaternion["z"], quaternion["w"])
        assert yaw == pytest.approx(math.pi / 4)
        observed.append(
            {
                "x": 1.5,
                "y": 1.5,
                "yaw": yaw,
                "time_sec": 1.0,
                "cleaning_enabled": True,
                "observation_complete": True,
                "collision_count": 0,
            }
        )
        return {"status": 4, "result": {}}

    monkeypatch.setattr(executor_module, "rank_repair_poses", timeout_rank)
    monkeypatch.setattr(driver, "_run_goal", goal)
    monkeypatch.setattr(driver, "_audit_event", lambda kind, payload: audit.append((kind, payload)))
    records = driver._repair(verifier, 0, "mock-action", time.monotonic() + 60)
    assert len(records) == 1
    assert verifier.result()["coverage_ratio"] == 1.0
    assert bool(ranking_calls) == (strategy == "pose_aware")
    decisions = [p for k, p in audit if k == "repair_candidate_selection"]
    if strategy == "pose_aware":
        assert decisions[0]["fallback"] is True
        assert decisions[0]["credit_role"] == "prediction_only_never_measured_credit"

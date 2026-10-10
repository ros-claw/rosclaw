"""Pose predictions must be bounded, body-aware and separate from measured credit."""

import copy
import math
import time

import pytest

from rosclaw.connectors.ros.mission import executor as executor_module
from rosclaw.connectors.ros.mission import repair_optimizer
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier


def test_equivalent_yaws_cannot_starve_second_goal_lookahead():
    # Both distant adjacent holes need separate tiny brush placements. Several
    # cheaper headings at the first hole must not hide the second placement.
    specification = {
        "width": 22,
        "height": 1,
        "resolution": 0.1,
        "origin": [-2.0, 0.0],
        "accessible_cells": list(range(22)),
        "cleaning_polygon": [[-0.04, -0.04], [0.04, -0.04], [0.04, 0.04], [-0.04, 0.04]],
    }
    centers = [(-2.0 + (i + 0.5) * 0.1, 0.05) for i in range(22)]
    result = repair_optimizer.rank_repair_poses(
        specification,
        centers,
        {20, 21},
        {"x": -1.95, "y": 0.05, "yaw": 0.0},
        beam_width=2,
        shortlist_size=2,
    )
    assert result.status == "READY"
    assert len(result.poses) == 2
    assert set(result.poses[0].predicted_new_cells) | set(result.poses[1].predicted_new_cells) == {
        20,
        21,
    }
    assert result.poses[0].center_cell != result.poses[1].center_cell


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


@pytest.mark.parametrize(
    "strategy", ["greedy", "pose_aware", "pose_aware_robust", "pose_aware_robust_sequence"]
)
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
    assert bool(ranking_calls) == (strategy != "greedy")
    decisions = [p for k, p in audit if k == "repair_candidate_selection"]
    if strategy != "greedy":
        assert decisions[0]["fallback"] is True
        assert decisions[0]["credit_role"] == "prediction_only_never_measured_credit"


def test_robust_prediction_averages_clipped_translations_without_widening_credit():
    specification = grid(3, 1.0, [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    remaining = {0, 1, 3, 4}
    current = {"x": 0.5, "y": 0.5, "yaw": 0.0}
    original = copy.deepcopy((specification, remaining, current))
    nominal = repair_optimizer.rank_repair_poses(specification, [(0.5, 0.5)], remaining, current)
    robust = repair_optimizer.rank_repair_poses(
        specification, [(0.5, 0.5)], remaining, current, robust_footprint=True
    )
    assert robust.status == "READY"
    assert robust.poses[0].predicted_new_cells == nominal.poses[0].predicted_new_cells == (0,)
    assert robust.poses[0].utility * robust.poses[0].estimated_cost_sec == pytest.approx(12 / 9)
    assert nominal.poses[0].utility * nominal.poses[0].estimated_cost_sec == pytest.approx(3)
    assert robust.reward_model == "NINE_ONE_CELL_TRANSLATIONS_NOT_CALIBRATED_PROBABILITY"
    assert (specification, remaining, current) == original
    verifier = CoverageVerifier(**specification)
    assert not verifier.visits


def test_robust_translation_never_wraps_at_grid_edges_or_resurrects_exhausted_cells():
    specification = grid(3, 1.0, [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    current = {"x": 0.5, "y": 0.5, "yaw": 0.0}
    result = repair_optimizer.rank_repair_poses(
        specification, [(0.5, 0.5)], {0, 2, 5}, current, robust_footprint=True
    )
    assert result.poses[0].utility * result.poses[0].estimated_cost_sec == pytest.approx(3 / 9)
    exhausted = repair_optimizer.rank_repair_poses(
        specification, [(0.5, 0.5)], {0, 1}, current, attempts={0: 3, 1: 3}, robust_footprint=True
    )
    assert exhausted.status == "NO_CANDIDATE"


@pytest.mark.parametrize("value", [1, None, "true"])
def test_robust_mode_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        repair_optimizer.rank_repair_poses(
            grid(),
            [(1.125, 1.125)],
            {64},
            {"x": 1.125, "y": 1.125, "yaw": 0},
            robust_footprint=value,
        )


def test_continuous_sequence_sends_two_typed_targets_without_predicted_credit(
    tmp_path, monkeypatch
):
    specification = grid(4, 1.0, [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]])
    specification["accessible_cells"] = [5, 10]
    verifier = CoverageVerifier(**specification)
    observed, calls = [], []

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
        recovery_centers=[(0.5 + x, 0.5 + y) for y in range(4) for x in range(4)],
        repair_strategy="pose_aware_robust_sequence",
    )
    poses = tuple(
        repair_optimizer.RepairPose(x, y, yaw, (cell,), 0.5, 0.1, 5, 1, cell)
        for x, y, yaw, cell in [(1.5, 1.5, 0.0, 5), (2.5, 2.5, math.pi / 2, 10)]
    )
    monkeypatch.setattr(
        executor_module,
        "rank_repair_poses",
        lambda *a, **k: repair_optimizer.RepairSelection("READY", poses),
    )
    monkeypatch.setattr(driver, "_audit_event", lambda *a: None)

    def goal(name, action_type, args, goal_id, deadline):
        calls.append((name, action_type, args, goal_id))
        if len(calls) > 1:
            # The successful SDK action response did not establish arrival at
            # its second predicted pose; only the first observation is credit.
            assert set(verifier.visits) == {5}
            raise RuntimeError("synthetic next-dispatch boundary")
        assert name == "/navigate_through_poses"
        assert action_type == "nav2_msgs/action/NavigateThroughPoses"
        assert len(args["poses"]) == 2
        assert [p["header"]["frame_id"] for p in args["poses"]] == ["map", "map"]
        assert args["poses"][1]["pose"]["position"] == {"x": 2.5, "y": 2.5, "z": 0.0}
        assert not verifier.visits
        observed.append(
            {
                "x": 1.5,
                "y": 1.5,
                "yaw": 0.0,
                "time_sec": 1.0,
                "cleaning_enabled": True,
                "observation_complete": True,
                "collision_count": 0,
            }
        )
        return {"status": 4, "result": {}}

    monkeypatch.setattr(driver, "_run_goal", goal)
    with pytest.raises(RuntimeError, match="next-dispatch"):
        driver._repair(verifier, 0, "mock-action", time.monotonic() + 60)
    assert set(verifier.visits) == {5}
    assert verifier.result()["coverage_ratio"] == 0.5
    assert len(verifier.accessible) == 2


def test_sequence_retry_bookkeeping_counts_overlaps_without_claiming_credit():
    from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery

    verifier = CoverageVerifier(**grid(3, 1.0))
    recovery = MissedRegionRecovery(verifier)
    recovery.record_attempt_sequence([[1, 1, 2], [2, 3]], action_id="actual-parent-goal")
    assert recovery.attempts == {1: 1, 2: 2, 3: 1}
    recovery.record_attempt_sequence([[1, 1, 2], [2, 3]], action_id="actual-parent-goal")
    assert recovery.attempts == {1: 1, 2: 2, 3: 1}
    assert not verifier.visits


def test_overlapping_pair_cannot_exceed_retry_budget_or_partially_mutate_bookkeeping():
    from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery

    verifier = CoverageVerifier(**grid(3, 1.0))
    recovery = MissedRegionRecovery(verifier)
    recovery.record_attempt_sequence([[1, 2], [2, 3]], action_id="first-parent")
    before = dict(recovery.attempts)
    with pytest.raises(ValueError, match="attempt budget"):
        recovery.record_attempt_sequence([[2, 4], [2, 5]], action_id="refused-parent")
    assert recovery.attempts == before
    assert "refused-parent" not in recovery.seen_action_ids
    assert not verifier.visits

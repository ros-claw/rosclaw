"""Synthetic temporal contracts; these are not Native physical acceptance."""

import pytest

from rosclaw.connectors.ros.mission.temporal_recovery import TimePairedRecovery
from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting, OccupancySnapshot


def setup():
    grid = CoverageVerifier(
        width=4,
        height=1,
        resolution=1,
        accessible_cells=[0, 1, 2, 3],
        cleaning_polygon=[(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)],
    )
    accounting = OccupancyAccounting(
        grid, run_id="run", mission_id="mission", geometry_hash="geometry"
    )
    recovery = TimePairedRecovery(accounting, admission_sim_time=0, deadline_sim_time=30)
    return accounting, recovery


def observe(accounting, time, sequence, blocked):
    s = OccupancySnapshot(
        "run",
        "mission",
        "map",
        time,
        sequence,
        blocked,
        "geometry",
        "independent_gazebo_model_geometry",
        True,
        0.01,
    )
    accounting.observe(CleaningPose(0.5, 0.5, 0, time, False), s, artifact_hash=s.artifact_hash())


def proposal(recovery, time=1, sequence=0, **changes):
    kwargs = {
        "sim_time_sec": time,
        "snapshot_sequence": sequence,
        "reachable_cells": frozenset(range(4)),
    }
    kwargs.update(changes)
    return recovery.propose_at(**kwargs)


def test_partial_obstruction_defers_only_blocked_cells_and_clearance_restores_candidates():
    accounting, recovery = setup()
    observe(accounting, 1, 0, (0,))
    result = proposal(recovery)
    assert result["ready"][0]["cells"] == [1, 2, 3]
    assert result["deferred"][0]["cells"] == [0]
    assert result["status"] == "REPAIR_PENDING"
    assert recovery.attempts == {}
    observe(accounting, 2, 1, ())
    result = proposal(recovery, 2, 1)
    assert result["ready"][0]["cells"] == [0, 1, 2, 3]
    assert result["deferred"] == []
    assert accounting.verifier.visits == {}  # withdrawal is not cleaning credit


def test_paired_route_filters_body_clearance_and_recovers_only_after_withdrawal():
    from rosclaw.connectors.ros.mission.temporal_recovery import paired_route_candidates

    accounting, _ = setup()
    centers = tuple((i + 0.5, 0.5) for i in range(4))
    current = {"x": 0.5, "y": 0.5, "yaw": 0, "time_sec": 1}
    observe(accounting, 1, 0, (2,))
    selected, reachable, proof = paired_route_candidates(
        accounting, centers, current, physical_radius_m=0.05
    )
    assert selected == ((0.5, 0.5),)
    assert reachable == frozenset({0})
    assert proof["snapshot_sequence"] == 0 and proof["prediction_only"]
    observe(accounting, 2, 1, ())
    current["time_sec"] = 2
    selected, reachable, proof = paired_route_candidates(
        accounting, centers, current, physical_radius_m=0.05
    )
    assert selected == centers and reachable == frozenset(range(4))
    assert accounting.verifier.visits == {}
    current["time_sec"] = 1
    with pytest.raises(ValueError, match="paired occupancy"):
        paired_route_candidates(accounting, centers, current, physical_radius_m=0.05)


def test_paired_route_does_not_jump_to_nearest_disconnected_center_or_cut_corners():
    from rosclaw.connectors.ros.mission.temporal_recovery import paired_route_candidates

    grid = CoverageVerifier(
        width=3,
        height=3,
        resolution=1,
        accessible_cells=list(range(9)),
        cleaning_polygon=[(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)],
    )
    accounting = OccupancyAccounting(
        grid, run_id="run", mission_id="mission", geometry_hash="geometry"
    )
    observe(accounting, 1, 0, ())
    current = {"x": 0.5, "y": 0.5, "yaw": 0, "time_sec": 1}
    centers = ((0.5, 0.5), (1.5, 1.5), (2.5, 1.5))
    selected, reachable, _ = paired_route_candidates(
        accounting, centers, current, physical_radius_m=0.05
    )
    assert selected == ((0.5, 0.5),) and reachable == frozenset({0})
    current["x"] = 1.5
    selected, reachable, _ = paired_route_candidates(
        accounting, centers, current, physical_radius_m=0.05
    )
    assert selected == () and not reachable  # current cell has no legal entry


def test_route_budget_expiration_returns_no_partial_candidate(monkeypatch):
    from rosclaw.connectors.ros.mission import temporal_recovery as module

    accounting, _ = setup()
    observe(accounting, 1, 0, ())
    ticks = iter([0, 1])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(ValueError, match="no partial"):
        module.paired_route_candidates(
            accounting,
            ((0.5, 0.5),),
            {"x": 0.5, "y": 0.5, "yaw": 0, "time_sec": 1},
            physical_radius_m=0.05,
        )


def test_blocked_route_waits_without_spending_attempts_and_deadline_removes_all_goals():
    accounting, recovery = setup()
    observe(accounting, 1, 0, (0,))
    result = proposal(recovery, reachable_cells=frozenset())
    assert result["status"] == "WAITING_FOR_OBSTACLE"
    assert recovery.attempts == {}
    observe(accounting, 30, 1, (0,))
    result = proposal(recovery, 30, 1)
    assert result["status"] == "BLOCKED"
    assert result["ready"] == []
    assert result["fixed_denominator_cells"] == 4
    assert result["dispatched"] is False


def test_canonical_attempt_delivery_is_idempotent_and_exhaustion_is_per_cell():
    accounting, recovery = setup()
    observe(accounting, 1, 0, ())
    for i in range(3):
        recovery.record_attempt([0], action_id=f"actual_action_{i}")
    recovery.record_attempt([0], action_id="actual_action_2")
    result = proposal(recovery)
    assert result["ready"][0]["cells"] == [1, 2, 3]
    assert result["exhausted"][0]["cells"] == [0]
    assert recovery.attempts[0] == 3


@pytest.mark.parametrize(
    "changes",
    [
        {"snapshot_sequence": 1},
        {"sim_time_sec": 2},
        {"reachable_cells": frozenset([4])},
        {"reachable_cells": frozenset([True])},
        {"reachable_cells": None},
    ],
)
def test_unpaired_or_missing_route_evidence_cannot_make_a_dispatch_proposal(changes):
    accounting, recovery = setup()
    observe(accounting, 1, 0, ())
    result = proposal(recovery, **changes)
    assert result["status"] == "UNKNOWN"
    assert result["ready"] == []


def test_latched_observer_fault_cannot_be_overridden_by_route_reachability():
    accounting, recovery = setup()
    observe(accounting, 1, 0, ())
    accounting.fault = "geometry missing"
    assert proposal(recovery)["status"] == "UNKNOWN"


@pytest.mark.parametrize("deadline", [float("nan"), 0, 1801])
def test_invalid_or_extended_deadlines_are_rejected(deadline):
    accounting, _ = setup()
    with pytest.raises(ValueError):
        TimePairedRecovery(accounting, admission_sim_time=0, deadline_sim_time=deadline)


def test_legacy_proposal_entry_cannot_bypass_temporal_evidence():
    _, recovery = setup()
    with pytest.raises(ValueError, match="paired"):
        recovery.propose()

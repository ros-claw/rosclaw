"""Temporal credit tests are contract evidence, never new physical acceptance."""

from dataclasses import replace

import pytest

from rosclaw.connectors.ros.verification.coverage import CleaningPose, CoverageVerifier
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting, OccupancySnapshot


def setup():
    verifier = CoverageVerifier(
        width=4,
        height=1,
        resolution=1,
        origin=(0, 0),
        accessible_cells=list(range(4)),
        cleaning_polygon=[(-0.4, -0.4), (0.4, -0.4), (0.4, 0.4), (-0.4, 0.4)],
    )
    return OccupancyAccounting(verifier, run_id="run", mission_id="mission", geometry_hash="geom")


def sample(t, seq, cells=()):
    return OccupancySnapshot(
        run_id="run",
        mission_id="mission",
        frame_id="map",
        sim_time_sec=t,
        sequence=seq,
        occupied_cells=cells,
        geometry_hash="geom",
        source="independent_gazebo_model_geometry",
        complete=True,
        ground_truth_age_sec=0.01,
    )


def observe(accounting, x, snapshot):
    accounting.observe(
        CleaningPose(x, 0.5, 0, snapshot.sim_time_sec, True),
        snapshot,
        artifact_hash=snapshot.artifact_hash(),
    )


def test_occupied_brush_overlap_receives_zero_credit_then_revisit_credits_after_clear():
    accounting = setup()
    observe(accounting, 0.5, sample(1, 1, (0,)))
    assert accounting.verifier.visits == {}
    assert accounting.result()["coverage"]["coverage_ratio"] == 0
    assert accounting.result()["fixed_denominator_cells"] == 4
    observe(accounting, 1.5, sample(2, 2))
    assert set(accounting.verifier.visits) == {1}  # final free mask cannot credit old occupied pose
    observe(accounting, 0.5, sample(3, 3))
    assert set(accounting.verifier.visits) == {0, 1}
    observe(accounting, 0.5, sample(4, 4, (0,)))
    assert set(accounting.verifier.visits) == {
        0,
        1,
    }  # later occupancy does not erase prior cleaning
    assert accounting.result()["complete"] is True


def test_sampled_mode_never_sweeps_an_unobserved_obstacle_interval():
    accounting = setup()
    observe(accounting, 0.5, sample(1, 0))
    observe(accounting, 2.5, sample(2, 1, (1,)))
    assert set(accounting.verifier.visits) == {0, 2}
    assert accounting.verifier.temporary_blocked == {1}
    assert accounting.verifier.result()["trace_gaps"] == 0
    observe(accounting, 3.5, sample(4, 2))
    assert accounting.verifier.result()["trace_gaps"] == 1  # original gap detector preserved


@pytest.mark.parametrize(
    "changes",
    [
        {"complete": False},
        {"frame_id": "odom"},
        {"mission_id": "other"},
        {"run_id": "old"},
        {"geometry_hash": "unknown"},
        {"source": "nav2_costmap"},
        {"evidence_domain": "REAL"},
        {"ground_truth_age_sec": 0.3},
        {"ground_truth_age_sec": float("nan")},
        {"sim_time_sec": 0.9},
        {"sequence": 0},
        {"sequence": 3},
        {"occupied_cells": (True,)},
        {"occupied_cells": (1, 1)},
        {"occupied_cells": (4,)},
    ],
)
def test_bad_occupancy_latches_failure_without_new_credit(changes):
    accounting = setup()
    observe(accounting, 0.5, sample(0, 0))
    bad = replace(sample(1, 1), **changes)
    before = dict(accounting.verifier.visits)
    try:
        submitted_hash = bad.artifact_hash()
    except ValueError:
        submitted_hash = "nonfinite-payload"
    with pytest.raises(ValueError):
        accounting.observe(CleaningPose(1.5, 0.5, 0, 1, True), bad, artifact_hash=submitted_hash)
    assert accounting.verifier.visits == before
    assert accounting.result()["complete"] is False
    with pytest.raises(ValueError, match="latched"):
        observe(accounting, 1.5, sample(2, 2))


def test_integrity_and_denominator_are_checked_before_projection():
    accounting = setup()
    s = sample(1, 0)
    with pytest.raises(ValueError, match="integrity"):
        accounting.observe(CleaningPose(0.5, 0.5, 0, 1, True), s, artifact_hash="tampered")
    accounting = setup()
    accounting.verifier.accessible.remove(0)
    with pytest.raises(ValueError, match="denominator"):
        observe(accounting, 0.5, s)
    assert accounting.verifier.visits == {}

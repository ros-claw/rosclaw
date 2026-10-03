"""Physical trace counterexamples for the openly declared E05 acceptance contract."""

import pytest

from benchmarks.harnessbench.dynamic_oracle import (
    convergence,
    timestep_duration_met,
    timestep_targets_met,
)


def _position_trace(dt=0.002, offset=0):
    return {"states": [{"t": offset + i * dt, "qpos": [0, 0, 0]} for i in range(round(3 / dt) + 1)]}


def test_single_impact_outlier_cannot_hide_behind_low_mean_or_scalar_axis_error():
    coarse, fine = _position_trace(0.004), _position_trace(0.002)
    fine["states"][500]["qpos"] = [0.0004, 0.0004, 0]
    measured = convergence(coarse, fine)
    assert measured == pytest.approx((2 * 0.0004**2) ** 0.5)
    # A 0.4 mm bound on each coordinate, low RMSE and 90% improvement still
    # fail the public 0.5 mm maximum Euclidean-distance requirement.
    assert not timestep_targets_met(measured, 0.9)


@pytest.mark.parametrize(
    "error,improvement,accepted",
    [
        (0.00049, 0.3, True),
        (0.0005, 0.3, True),
        (0.00051, 0.9, False),
        (0.0004, 0.29, False),
        (0.0010791, 0.8, False),
        (float("nan"), 0.9, False),
        (0.0004, float("inf"), False),
    ],
)
def test_public_metre_and_improvement_limits_are_both_required(error, improvement, accepted):
    assert timestep_targets_met(error, improvement) is accepted


def test_equal_trajectories_at_shifted_times_cannot_use_interpolation_as_evidence():
    with pytest.raises(ValueError, match="common recorded physical timestamps"):
        convergence(_position_trace(0.004), _position_trace(0.002, offset=0.0001))


def test_numerically_agreeing_sparse_trajectories_do_not_qualify_contact_convergence():
    with pytest.raises(ValueError, match="undersampled"):
        convergence(_position_trace(0.012), _position_trace(0.006))


def test_nonzero_start_time_does_not_make_a_short_rollout_three_seconds():
    assert not timestep_duration_met({"start_time": 5, "end_time": 6})
    assert timestep_duration_met({"start_time": 5, "end_time": 8})

from __future__ import annotations

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
    g1_ballistic_contact_torque_residual,
)


def test_contact_torque_residual_is_bounded_smooth_and_right_leg_only() -> None:
    config = G1BallisticContactTorqueResidualConfig(
        right_leg_residual_nm=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
        contact_policy_frame=256,
        lead_duration_sec=0.16,
        trail_duration_sec=0.08,
    )

    before, before_active = g1_ballistic_contact_torque_residual(
        policy_frame=247, control_dt_sec=0.02, config=config
    )
    peak, peak_active = g1_ballistic_contact_torque_residual(
        policy_frame=256, control_dt_sec=0.02, config=config
    )
    after, after_active = g1_ballistic_contact_torque_residual(
        policy_frame=261, control_dt_sec=0.02, config=config
    )

    assert not before_active
    assert np.array_equal(before, np.zeros(29))
    assert peak_active
    assert np.array_equal(peak[6:12], np.asarray(config.right_leg_residual_nm))
    assert np.array_equal(peak[:6], np.zeros(6))
    assert np.array_equal(peak[12:], np.zeros(17))
    assert not after_active
    assert np.array_equal(after, np.zeros(29))


def test_contact_torque_residual_fails_closed() -> None:
    with pytest.raises(ValueError, match="six finite"):
        G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=(0.0, 0.0, 0.0, 0.0, 0.0, float("nan"))
        )
    with pytest.raises(ValueError, match="SIM-only limit"):
        G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=(0.0, 0.0, 0.0, 0.0, 12.1, 0.0)
        )
    with pytest.raises(ValueError, match="phase exceeds"):
        G1BallisticContactTorqueResidualConfig(
            right_leg_phase_offset_sec=(0.041, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
    with pytest.raises(ValueError, match="combined pulse"):
        G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=(0.0, 0.0, 0.0, 0.0, 8.0, 0.0),
            right_leg_preload_nm=(0.0, 0.0, 0.0, 0.0, 4.1, 0.0),
        )
    with pytest.raises(ValueError, match="counterbalance torque exceeds"):
        G1BallisticContactTorqueResidualConfig(
            counterbalance_residual_nm=(0.0, 0.0, 0.0, 0.0, 0.0, 6.1)
        )


def test_contact_torque_residual_supports_bounded_joint_phase_synergy() -> None:
    config = G1BallisticContactTorqueResidualConfig(
        right_leg_residual_nm=(-4.0, 0.0, 0.0, 0.0, 3.0, 0.0),
        right_leg_phase_offset_sec=(-0.02, 0.0, 0.0, 0.0, 0.02, 0.0),
        contact_policy_frame=256,
        lead_duration_sec=0.08,
        trail_duration_sec=0.065,
    )

    before, active = g1_ballistic_contact_torque_residual(
        policy_frame=255, control_dt_sec=0.02, config=config
    )
    at_contact, contact_active = g1_ballistic_contact_torque_residual(
        policy_frame=256, control_dt_sec=0.02, config=config
    )
    after, after_active = g1_ballistic_contact_torque_residual(
        policy_frame=257, control_dt_sec=0.02, config=config
    )

    assert active and contact_active and after_active
    assert abs(before[6]) > abs(before[10])
    assert abs(after[10]) > abs(after[6])
    assert abs(at_contact[6]) < 4.0
    assert abs(at_contact[10]) < 3.0


def test_contact_torque_residual_has_zero_endpoint_preload() -> None:
    config = G1BallisticContactTorqueResidualConfig(
        right_leg_preload_nm=(-3.0, 0.0, 0.0, 0.0, 2.0, 0.0),
        contact_policy_frame=256,
        lead_duration_sec=0.08,
        trail_duration_sec=0.065,
    )

    start, start_active = g1_ballistic_contact_torque_residual(
        policy_frame=252, control_dt_sec=0.02, config=config
    )
    middle, middle_active = g1_ballistic_contact_torque_residual(
        policy_frame=254, control_dt_sec=0.02, config=config
    )
    contact, contact_active = g1_ballistic_contact_torque_residual(
        policy_frame=256, control_dt_sec=0.02, config=config
    )

    assert not start_active
    assert middle_active
    assert middle[6] == pytest.approx(-3.0)
    assert middle[10] == pytest.approx(2.0)
    assert not contact_active
    assert np.array_equal(start, np.zeros(29))
    assert np.array_equal(contact, np.zeros(29))


def test_contact_torque_residual_adds_bounded_counterbalance_synergy() -> None:
    config = G1BallisticContactTorqueResidualConfig(
        counterbalance_residual_nm=(1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
        contact_policy_frame=256,
    )

    torque, active = g1_ballistic_contact_torque_residual(
        policy_frame=256, control_dt_sec=0.02, config=config
    )

    assert active
    np.testing.assert_array_equal(
        torque[[0, 1, 2, 4, 12, 14]],
        np.asarray(config.counterbalance_residual_nm),
    )
    untouched = np.delete(torque, [0, 1, 2, 4, 12, 14])
    np.testing.assert_array_equal(untouched, np.zeros(23))

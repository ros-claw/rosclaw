from __future__ import annotations

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_residual import (
    G1BallisticContactResidualConfig,
    blend_g1_ballistic_contact_target,
)


def test_ballistic_contact_residual_is_bounded_and_contact_centred() -> None:
    config = G1BallisticContactResidualConfig(
        right_leg_residual_rad=(0.10, -0.05, 0.02, 0.12, -0.08, 0.0),
    )
    target = np.zeros(29, dtype=np.float64)

    before, before_delta, before_active = blend_g1_ballistic_contact_target(
        target=target,
        policy_frame=240,
        control_dt_sec=0.02,
        config=config,
    )
    at_contact, contact_delta, contact_active = blend_g1_ballistic_contact_target(
        target=target,
        policy_frame=256,
        control_dt_sec=0.02,
        config=config,
    )
    after, after_delta, after_active = blend_g1_ballistic_contact_target(
        target=target,
        policy_frame=261,
        control_dt_sec=0.02,
        config=config,
    )

    np.testing.assert_array_equal(before, target)
    np.testing.assert_array_equal(before_delta, np.zeros(29))
    assert not before_active
    np.testing.assert_allclose(contact_delta[6:12], config.right_leg_residual_rad)
    np.testing.assert_allclose(at_contact, contact_delta)
    assert contact_active
    np.testing.assert_array_equal(after, target)
    np.testing.assert_array_equal(after_delta, np.zeros(29))
    assert not after_active


def test_ballistic_contact_residual_rejects_out_of_support_actions() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        G1BallisticContactResidualConfig(
            right_leg_residual_rad=(0.0, 0.0, 0.0, 0.251, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1BallisticContactResidualConfig(activation_ceiling="HARDWARE")

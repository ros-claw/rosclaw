from __future__ import annotations

import numpy as np
import pytest

from rosclaw.simforge.g1_loft_teacher import (
    G1LoftTeacherConfig,
    g1_loft_teacher_effect,
    project_g1_vertical_foot_force,
)


def test_loft_teacher_is_disabled_by_default_and_bounded() -> None:
    config = G1LoftTeacherConfig()

    assert not config.enabled
    assert config.config_hash.startswith("sha256:")
    with pytest.raises(ValueError, match="target speed"):
        G1LoftTeacherConfig(target_vertical_speed_mps=2.9)
    with pytest.raises(ValueError, match="target speed"):
        G1LoftTeacherConfig(target_vertical_speed_mps=-0.4)
    with pytest.raises(ValueError, match="force limit"):
        G1LoftTeacherConfig(maximum_vertical_force_n=251.0)
    with pytest.raises(ValueError, match="forward speed"):
        G1LoftTeacherConfig(target_forward_speed_mps=4.9)
    with pytest.raises(ValueError, match="lateral speed"):
        G1LoftTeacherConfig(target_lateral_speed_mps=0.9)
    with pytest.raises(ValueError, match="foot envelope"):
        G1LoftTeacherConfig(foot_strike_point_offset_m=(0.20, 0.0, 0.0))
    with pytest.raises(ValueError, match="foot-ball distance"):
        G1LoftTeacherConfig(maximum_foot_ball_distance_m=0.10)


def test_loft_teacher_proximity_gate_fails_closed_and_rejects_distant_ball() -> None:
    class _Model:
        nv = 35

    class _Data:
        xmat = np.eye(3, dtype=np.float64)[None, ...]
        xpos = np.zeros((1, 3), dtype=np.float64)
        qvel = np.zeros(35, dtype=np.float64)

    config = G1LoftTeacherConfig(
        target_vertical_speed_mps=5.0,
        maximum_foot_ball_distance_m=0.30,
        start_policy_frame=200,
        end_policy_frame=300,
    )
    with pytest.raises(ValueError, match="finite ball position"):
        g1_loft_teacher_effect(
            model=_Model(),
            data=_Data(),
            right_ankle_body_id=0,
            config=config,
            policy_frame=240,
            contact_observed=False,
        )

    effect = g1_loft_teacher_effect(
        model=_Model(),
        data=_Data(),
        right_ankle_body_id=0,
        config=config,
        policy_frame=240,
        contact_observed=False,
        ball_position=np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert not effect.active
    assert np.array_equal(effect.torque, np.zeros(29))


def test_loft_teacher_projects_a_bounded_vertical_force() -> None:
    jacobian = np.zeros((3, 41), dtype=np.float64)
    jacobian[2, 6] = 0.50
    jacobian[2, 9] = -0.25
    velocity = np.zeros(41, dtype=np.float64)
    velocity[6] = 4.0
    config = G1LoftTeacherConfig(
        target_vertical_speed_mps=5.0,
        velocity_gain_n_per_mps=24.0,
        maximum_vertical_force_n=20.0,
    )

    effect = project_g1_vertical_foot_force(
        jacobian_position=jacobian,
        generalized_velocity=velocity,
        config=config,
    )

    assert effect.active
    assert effect.foot_vertical_speed_mps == pytest.approx(2.0)
    assert effect.foot_forward_speed_mps == 0.0
    assert effect.vertical_force_n == 20.0
    assert effect.forward_force_n == 0.0
    assert effect.torque[0] == 10.0
    assert effect.torque[3] == -5.0
    assert effect.torque.shape == (29,)


def test_loft_teacher_preserves_forward_foot_velocity_in_the_contact_neighborhood() -> None:
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[0, 6] = 0.25
    jacobian[2, 6] = 0.50
    config = G1LoftTeacherConfig(
        target_vertical_speed_mps=5.0,
        velocity_gain_n_per_mps=10.0,
        maximum_vertical_force_n=30.0,
        target_forward_speed_mps=12.0,
        forward_velocity_gain_n_per_mps=20.0,
        maximum_forward_force_n=40.0,
    )

    effect = project_g1_vertical_foot_force(
        jacobian_position=jacobian,
        generalized_velocity=np.zeros(35),
        config=config,
    )

    assert effect.vertical_force_n == 30.0
    assert effect.forward_force_n == 40.0
    assert effect.torque[0] == 25.0


def test_loft_teacher_projects_a_bounded_downward_cut() -> None:
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[2, 6] = 0.50
    jacobian[2, 9] = -0.25
    config = G1LoftTeacherConfig(
        target_vertical_speed_mps=-2.0,
        velocity_gain_n_per_mps=24.0,
        maximum_vertical_force_n=30.0,
    )

    effect = project_g1_vertical_foot_force(
        jacobian_position=jacobian,
        generalized_velocity=np.zeros(35),
        config=config,
    )

    assert effect.active
    assert effect.vertical_force_n == -30.0
    assert effect.torque[0] == -15.0
    assert effect.torque[3] == 7.5


def test_loft_teacher_projects_a_signed_lateral_foot_velocity() -> None:
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[1, 7] = 0.4
    jacobian[1, 8] = -0.2
    config = G1LoftTeacherConfig(
        target_lateral_speed_mps=-7.0,
        lateral_velocity_gain_n_per_mps=20.0,
        maximum_lateral_force_n=50.0,
    )

    effect = project_g1_vertical_foot_force(
        jacobian_position=jacobian,
        generalized_velocity=np.zeros(35),
        config=config,
    )

    assert effect.active
    assert effect.lateral_force_n == -50.0
    assert effect.foot_lateral_speed_mps == 0.0
    assert effect.torque[1] == -20.0
    assert effect.torque[2] == 10.0


def test_loft_teacher_fails_closed_on_bad_shape_or_non_finite_input() -> None:
    config = G1LoftTeacherConfig(target_vertical_speed_mps=5.0)
    with pytest.raises(ValueError, match="Jacobian"):
        project_g1_vertical_foot_force(
            jacobian_position=np.zeros((3, 34)),
            generalized_velocity=np.zeros(35),
            config=config,
        )
    bad_velocity = np.zeros(35)
    bad_velocity[0] = np.nan
    with pytest.raises(FloatingPointError, match="finite"):
        project_g1_vertical_foot_force(
            jacobian_position=np.zeros((3, 35)),
            generalized_velocity=bad_velocity,
            config=config,
        )

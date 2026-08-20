from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_coupled_relay import G1CoupledRelayResult
from rosclaw.simforge.g1_three_player_showcase import (
    G1ThreePlayerShowcaseEvidence,
    run_g1_three_player_showcase,
    three_player_goal_spec,
    three_player_goalkeeper_config,
    three_player_simulation_kwargs,
)


def _result(**overrides: object) -> G1CoupledRelayResult:
    values: dict[str, object] = {
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "pass_contact_time_sec": 5.6,
        "shot_contact_time_sec": 7.9,
        "pass_peak_ball_speed_mps": 1.3,
        "shot_peak_ball_speed_mps": 8.1,
        "goal_crossed": True,
        "goal_crossing_y_m": 0.891,
        "goal_crossing_z_m": 0.115,
        "target_error_m": 0.001,
        "passer_min_pelvis_height_m": 0.68,
        "shooter_min_pelvis_height_m": 0.67,
        "passer_roll_peak_rad": 0.2,
        "passer_pitch_peak_rad": 0.3,
        "shooter_roll_peak_rad": 0.3,
        "shooter_pitch_peak_rad": 0.3,
        "passer_tail_wobble_index": 0.0,
        "shooter_tail_wobble_index": 0.0,
        "receiver_phase_hold_frames": 0,
        "receiver_phase_advance_frames": 0,
        "receiver_max_ball_phase_error_m": 0.0,
        "robot_robot_contact_count": 0,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "physics_steps": 7500,
        "pass_delivery_position_m": (0.97, -0.02, 0.115),
        "pass_delivery_error_m": 0.036,
        "pass_delivery_lateral_error_m": 0.02,
        "goalkeeper_enabled": True,
        "goalkeeper_reaction_active_fraction": 0.45,
        "goalkeeper_lateral_displacement_m": 1.2,
        "goalkeeper_peak_lateral_speed_mps": 0.47,
        "goalkeeper_min_pelvis_height_m": 0.75,
    }
    values.update(overrides)
    return G1CoupledRelayResult(**values)  # type: ignore[arg-type]


def _evidence(**overrides: object) -> G1ThreePlayerShowcaseEvidence:
    values: dict[str, object] = {
        "body_hash": "sha256:" + "1" * 64,
        "kick_prior_hash": "sha256:" + "2" * 64,
        "standby_policy_hash": "sha256:" + "3" * 64,
        "backend_commit": "abc123",
        "implementation_hash": "sha256:" + "4" * 64,
        "request_hash": "sha256:" + "5" * 64,
        "trajectory_hash": "sha256:" + "6" * 64,
        "trajectory_digest": "sha256:" + "7" * 64,
        "strict_replay": True,
        "result": _result(),
        "pass_distance_m": 2.9,
        "shot_distance_m": 6.5,
        "pass_speed_start_mps": 1.3,
        "pass_speed_end_mps": 0.9,
        "pass_speed_max_positive_step_mps": 0.0,
        "pass_speed_positive_step_count": 0,
    }
    values.update(overrides)
    return G1ThreePlayerShowcaseEvidence(**values)  # type: ignore[arg-type]


def test_three_player_candidate_binds_distance_goal_and_keeper() -> None:
    goal = three_player_goal_spec()
    keeper = three_player_goalkeeper_config()
    values = three_player_simulation_kwargs()

    assert goal.plane_x_m == pytest.approx(7.5)
    assert goal.precision_radius_m == pytest.approx(0.10)
    assert goal.ball_free_joint_damping_n_s_m == pytest.approx(0.02)
    assert values["passer_origin"][0] == pytest.approx(5.1)
    assert values["receiver_phase_sync_enabled"] is False
    assert values["goalkeeper_config"] == keeper
    assert values["unified_stadium_scene"] is True
    assert values["shooter_parameter_overrides"] == {
        "foot_yaw_offset": 0.085,
        "foot_pitch_offset": 0.010,
    }


def test_three_player_evidence_requires_motion_safety_and_strict_replay() -> None:
    assert _evidence().passed
    assert not _evidence(strict_replay=False).passed
    assert not _evidence(pass_distance_m=2.74).passed
    assert not _evidence(shot_distance_m=6.24).passed
    assert not _evidence(pass_speed_max_positive_step_mps=0.031).passed
    assert not _evidence(result=_result(goalkeeper_lateral_displacement_m=0.74)).passed
    assert not _evidence(result=_result(goalkeeper_joint_limit_violation=True)).passed


def test_three_player_evidence_refuses_source_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_three_player_showcase(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )

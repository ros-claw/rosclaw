from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.simforge.g1_coupled_relay import (
    G1CoupledRelayEvidence,
    G1CoupledRelayResult,
    G1CoupledRelayRobustnessCase,
    G1GoalkeeperConfig,
    G1JointGuardConfig,
    _normalized_locomotion_command,
    _normalized_zero_locomotion_command,
    _project_joint_safe_torque,
    _select_joint_guard_config,
    _smooth_policy_handoff,
    run_g1_coupled_relay,
    shared_post_impact_recovery_config,
    shared_post_impact_simulation_kwargs,
    trained_coupled_skill_simulation_kwargs,
)
from rosclaw.simforge.g1_coupled_relay_video import (
    render_g1_coupled_relay_video,
)


def _passing_result(**overrides: object) -> G1CoupledRelayResult:
    values: dict[str, object] = {
        "finite_state": True,
        "pass_contact_observed": True,
        "shot_contact_observed": True,
        "pass_contact_time_sec": 5.6,
        "shot_contact_time_sec": 7.4,
        "pass_peak_ball_speed_mps": 1.5,
        "shot_peak_ball_speed_mps": 10.2,
        "goal_crossed": True,
        "goal_crossing_y_m": 0.91,
        "goal_crossing_z_m": 1.03,
        "target_error_m": 0.20,
        "passer_min_pelvis_height_m": 0.68,
        "shooter_min_pelvis_height_m": 0.67,
        "passer_roll_peak_rad": 0.20,
        "passer_pitch_peak_rad": 0.30,
        "shooter_roll_peak_rad": 0.33,
        "shooter_pitch_peak_rad": 0.28,
        "passer_tail_wobble_index": 0.00005,
        "shooter_tail_wobble_index": 0.00010,
        "receiver_phase_hold_frames": 0,
        "receiver_phase_advance_frames": 0,
        "receiver_max_ball_phase_error_m": 0.01,
        "robot_robot_contact_count": 0,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "physics_steps": 7500,
        "pass_delivery_position_m": (1.0, 0.0, 0.115),
        "pass_delivery_error_m": 0.01,
        "pass_delivery_lateral_error_m": 0.005,
    }
    values.update(overrides)
    return G1CoupledRelayResult(**values)  # type: ignore[arg-type]


def test_coupled_relay_result_requires_ordered_contacts_and_high_finish() -> None:
    assert _passing_result().passed
    assert not _passing_result(shot_contact_time_sec=5.5).passed
    assert not _passing_result(shot_peak_ball_speed_mps=5.9).passed
    assert not _passing_result(target_error_m=0.481).passed
    assert not _passing_result(pass_delivery_error_m=0.051).passed
    assert not _passing_result(pass_delivery_lateral_error_m=0.031).passed
    assert not _passing_result(passer_min_pelvis_height_m=0.54).passed


def test_post_policy_handoff_is_continuous_and_reaches_destination() -> None:
    zeros = np.zeros(29)
    ones = np.ones(29)
    first = _smooth_policy_handoff(
        origin_target=zeros,
        origin_kp=zeros,
        origin_kd=zeros,
        destination_target=ones,
        destination_kp=ones * 2.0,
        destination_kd=ones * 3.0,
        transition_step=0,
        blend_frames=20,
    )
    final = _smooth_policy_handoff(
        origin_target=zeros,
        origin_kp=zeros,
        origin_kd=zeros,
        destination_target=ones,
        destination_kp=ones * 2.0,
        destination_kd=ones * 3.0,
        transition_step=19,
        blend_frames=20,
    )

    assert 0.0 < first[3] < 0.01
    assert np.all(first[0] > zeros)
    assert final[3] == 1.0
    assert np.array_equal(final[0], ones)
    assert np.array_equal(final[1], ones * 2.0)
    assert np.array_equal(final[2], ones * 3.0)


def test_post_policy_handoff_rejects_bad_contracts() -> None:
    vector = np.zeros(29)
    with pytest.raises(ValueError, match="inside a positive blend window"):
        _smooth_policy_handoff(
            origin_target=vector,
            origin_kp=vector,
            origin_kd=vector,
            destination_target=vector,
            destination_kp=vector,
            destination_kd=vector,
            transition_step=2,
            blend_frames=2,
        )
    with pytest.raises(ValueError, match="finite 29-joint"):
        _smooth_policy_handoff(
            origin_target=np.zeros(28),
            origin_kp=vector,
            origin_kd=vector,
            destination_target=vector,
            destination_kp=vector,
            destination_kd=vector,
            transition_step=0,
            blend_frames=2,
        )


def test_joint_guard_brakes_only_threatened_outward_motion() -> None:
    position = np.zeros(29)
    velocity = np.zeros(29)
    command = np.zeros(29)
    ranges = np.tile(np.asarray((-1.0, 1.0)), (29, 1))
    limited = np.ones(29, dtype=bool)
    position[3] = -0.94
    velocity[3] = -1.0
    command[3] = -20.0

    projected, active = _project_joint_safe_torque(
        joint_position=position,
        joint_velocity=velocity,
        commanded_torque=command,
        joint_ranges=ranges,
        limited=limited,
    )

    assert active
    assert projected[3] > command[3]
    assert np.array_equal(projected[:3], command[:3])
    assert np.array_equal(projected[4:], command[4:])


def test_joint_guard_is_noop_inside_safe_envelope() -> None:
    vector = np.zeros(29)
    projected, active = _project_joint_safe_torque(
        joint_position=vector,
        joint_velocity=vector,
        commanded_torque=np.ones(29),
        joint_ranges=np.tile(np.asarray((-1.0, 1.0)), (29, 1)),
        limited=np.ones(29, dtype=bool),
    )

    assert not active
    assert np.array_equal(projected, np.ones(29))


def test_joint_guard_config_is_bounded() -> None:
    assert G1JointGuardConfig().margin_rad == 0.04
    with pytest.raises(ValueError, match="margin"):
        G1JointGuardConfig(margin_rad=0.001)
    with pytest.raises(ValueError, match="horizon"):
        G1JointGuardConfig(prediction_horizon_sec=0.01)


def test_joint_guard_router_uses_only_latched_phase_signal() -> None:
    standard = G1JointGuardConfig(margin_rad=0.02, prediction_horizon_sec=0.06)
    late = G1JointGuardConfig()

    assert _select_joint_guard_config(
        standard=standard,
        late_arrival=late,
        phase_advance_count=0,
    ) == (standard, "standard")
    assert _select_joint_guard_config(
        standard=standard,
        late_arrival=late,
        phase_advance_count=2,
    ) == (late, "late_arrival")
    with pytest.raises(ValueError, match="non-negative"):
        _select_joint_guard_config(
            standard=standard,
            late_arrival=late,
            phase_advance_count=-1,
        )


def test_neutral_locomotion_command_inverts_asymmetric_scaling() -> None:
    policy = SimpleNamespace(
        range_velx=np.asarray((-0.4, 0.7)),
        range_vely=np.asarray((-0.4, 0.4)),
        range_velz=np.asarray((-1.57, 1.57)),
    )

    command = _normalized_zero_locomotion_command(policy)
    ranges = np.asarray((policy.range_velx, policy.range_vely, policy.range_velz))
    scaled = (command + 1.0) * (ranges[:, 1] - ranges[:, 0]) / 2.0 + ranges[:, 0]

    assert command[0] == pytest.approx(-0.2727272727)
    assert np.allclose(scaled, 0.0)


def test_physical_locomotion_command_round_trips_asymmetric_scaling() -> None:
    policy = SimpleNamespace(
        range_velx=np.asarray((-0.4, 0.7)),
        range_vely=np.asarray((-0.4, 0.4)),
        range_velz=np.asarray((-1.57, 1.57)),
    )
    physical = np.asarray((0.25, -0.18, 0.31))

    command = _normalized_locomotion_command(policy, physical)
    ranges = np.asarray((policy.range_velx, policy.range_vely, policy.range_velz))
    round_trip = (command + 1.0) * (ranges[:, 1] - ranges[:, 0]) / 2.0 + ranges[:, 0]

    assert np.allclose(round_trip, physical)
    with pytest.raises(ValueError, match="outside"):
        _normalized_locomotion_command(policy, np.asarray((0.8, 0.0, 0.0)))


def test_goalkeeper_config_is_strictly_bounded() -> None:
    assert G1GoalkeeperConfig().maximum_lateral_speed_mps == pytest.approx(0.38)
    with pytest.raises(ValueError, match="reaction delay"):
        G1GoalkeeperConfig(reaction_delay_sec=0.01)
    with pytest.raises(ValueError, match="lateral speed"):
        G1GoalkeeperConfig(maximum_lateral_speed_mps=0.45)
    with pytest.raises(ValueError, match="finite"):
        G1GoalkeeperConfig(arm_spread_rad=float("nan"))


def test_shared_post_impact_controller_is_symmetric_and_physical_zero() -> None:
    recovery = shared_post_impact_recovery_config()
    values = shared_post_impact_simulation_kwargs()

    assert values["passer_recovery_config"] == recovery
    assert values["shooter_recovery_config"] == recovery
    assert values["passer_post_policy_neutral_velocity_enabled"] is True
    assert values["shooter_post_policy_neutral_velocity_enabled"] is True
    assert values["passer_post_policy_recovery_enabled"] is True
    assert values["shooter_post_policy_recovery_enabled"] is True
    assert values["passer_joint_guard_enabled"] is True
    assert values["shooter_joint_guard_enabled"] is True
    assert recovery.standing_pose_blend == pytest.approx(0.02)
    assert recovery.target_smoothing_alpha == pytest.approx(0.60)


def test_trained_coupled_skill_has_causal_early_arrival_aim_expert() -> None:
    values = trained_coupled_skill_simulation_kwargs()

    assert values["shooter_parameter_overrides"] == {
        "foot_yaw_offset": 0.085,
        "foot_pitch_offset": 0.010,
    }
    assert values["shooter_early_arrival_parameter_overrides"] == {
        "foot_yaw_offset": 0.115,
        "foot_pitch_offset": 0.025,
    }


def test_coupled_evidence_is_strict_replay_and_sim_only() -> None:
    cases = tuple(
        G1CoupledRelayRobustnessCase(
            shooter_start_offset_sec=offset,
            passed=True,
            shot_contact_time_sec=7.4,
            goal_crossing_y_m=0.91,
            goal_crossing_z_m=1.03,
            target_error_m=0.20,
            phase_hold_frames=0,
            phase_advance_frames=0,
            minimum_pelvis_height_m=0.67,
            joint_limit_violation=False,
            torque_limit_violation=False,
        )
        for offset in (-0.04, -0.02, 0.0, 0.02, 0.04)
    )
    parent_cases = tuple(
        G1CoupledRelayRobustnessCase(
            shooter_start_offset_sec=offset,
            passed=offset == 0.0,
            shot_contact_time_sec=7.4,
            goal_crossing_y_m=0.91,
            goal_crossing_z_m=1.03,
            target_error_m=0.20 if offset == 0.0 else 0.80,
            phase_hold_frames=0,
            phase_advance_frames=0,
            minimum_pelvis_height_m=0.67,
            joint_limit_violation=False,
            torque_limit_violation=False,
        )
        for offset in (-0.04, -0.02, 0.0, 0.02, 0.04)
    )
    evidence = G1CoupledRelayEvidence(
        body_hash="sha256:" + "1" * 64,
        kick_prior_hash="sha256:" + "2" * 64,
        standby_policy_hash="sha256:" + "3" * 64,
        backend_commit="abc123",
        implementation_hash="sha256:" + "7" * 64,
        request_hash="sha256:" + "4" * 64,
        trajectory_hash="sha256:" + "5" * 64,
        trajectory_digest="sha256:" + "6" * 64,
        strict_replay=True,
        result=_passing_result(),
        receiver_timing_parent=parent_cases,
        receiver_timing_robustness=cases,
    )

    assert evidence.passed
    value = evidence.to_dict()
    assert value["claims"]["simultaneous_two_body_physics"] is True
    assert value["claims"]["single_shared_ball"] is True
    assert value["claims"]["pixels_used_for_promotion"] is False


def test_coupled_relay_evidence_cannot_be_written_inside_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_coupled_relay(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


def test_coupled_relay_video_cannot_be_written_inside_checkout(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_coupled_relay_video(
            evidence_path=tmp_path / "evidence.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_real_g1_coupled_relay_is_strict_high_and_stable(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")

    evidence = run_g1_coupled_relay(
        asset_root=Path(asset_root),
        output_dir=tmp_path / "coupled-relay",
        source_checkout=Path(__file__).resolve().parents[2],
    )

    assert evidence.passed
    assert evidence.strict_replay
    assert evidence.simultaneous_two_body_physics
    assert evidence.shared_ball_state
    assert sum(case.passed for case in evidence.receiver_timing_parent) == 3
    assert len(evidence.receiver_timing_robustness) == 5
    assert all(case.passed for case in evidence.receiver_timing_robustness)
    assert evidence.result.pass_contact_time_sec == pytest.approx(5.602)
    assert evidence.result.shot_contact_time_sec == pytest.approx(7.412)
    assert evidence.result.goal_crossing_z_m is not None
    assert evidence.result.goal_crossing_z_m >= 1.0
    assert evidence.result.target_error_m is not None
    assert evidence.result.target_error_m <= 0.25
    assert evidence.result.passer_min_pelvis_height_m >= 0.65
    assert evidence.result.shooter_min_pelvis_height_m >= 0.65
    assert not evidence.result.joint_limit_violation
    assert not evidence.result.torque_limit_violation

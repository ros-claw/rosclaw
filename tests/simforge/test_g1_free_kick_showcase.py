from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw.simforge.backends.unitree_mujoco_backend import (
    _adapt_target,
    _contact_observation,
)
from rosclaw.simforge.g1_free_kick_showcase import (
    G1FootballEventPhase,
    G1FreeKickFlowConfig,
    G1FreeKickResult,
    _deepest_goal_mouth_point,
    _select_contextual_phase,
    run_g1_free_kick_showcase,
)
from rosclaw.simforge.g1_free_kick_showcase_video import (
    _duration_text,
    _metric_text,
    _optional_finite_float,
    render_g1_free_kick_showcase_video,
)
from rosclaw.simforge.g1_learned_runup import (
    G1LearnedRunupConfig,
    qualify_g1_learned_gait,
)
from rosclaw.simforge.g1_sonic_runup import (
    ISAACLAB_TO_MUJOCO,
    MUJOCO_TO_ISAACLAB,
    G1SonicRunupConfig,
    G1SonicRunupController,
    qualify_g1_sonic,
)
from rosclaw.simforge.g1_stadium_scene import (
    G1TrainingGoalSpec,
    build_g1_stadium_model,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import ShotParameters


def _passing_result(**changes: object) -> G1FreeKickResult:
    values: dict[str, object] = {
        "finite_state": True,
        "learned_runup_executed": True,
        "learned_approach_strike_residual_executed": False,
        "residual_accepted_frames": 0,
        "residual_rejected_frames": 0,
        "residual_peak_nm": 0.0,
        "residual_rms_nm": 0.0,
        "residual_effect_fraction": 0.0,
        "loft_teacher_executed": False,
        "loft_teacher_active_frames": 0,
        "loft_teacher_peak_torque_nm": 0.0,
        "loft_teacher_peak_force_n": 0.0,
        "joint_boundary_guard_active_steps": 0,
        "joint_boundary_guard_peak_correction_nm": 0.0,
        "continuous_single_world": True,
        "state_reset_after_start": False,
        "initial_ball_distance_m": 4.5,
        "shot_distance_m": 4.0,
        "runup_distance_m": 3.4,
        "runup_peak_speed_mps": 1.3,
        "runup_min_pelvis_height_m": 0.74,
        "runup_peak_tilt_rad": 0.12,
        "runup_terminal_speed_mps": 0.20,
        "handoff_yaw_rad": 0.01,
        "handoff_roll_rad": 0.01,
        "handoff_pitch_rad": 0.02,
        "handoff_pelvis_x_m": -0.1,
        "handoff_pelvis_y_m": 0.0,
        "handoff_joint_velocity_rms_rad_s": 0.5,
        "selected_kick_phase_start_frame": 214,
        "contextual_phase_expert_executed": False,
        "proprioceptive_router_executed": False,
        "proprioceptive_router_fallback": False,
        "proprioceptive_router_nearest_distance": None,
        "proprioceptive_router_distance_margin": None,
        "handoff_to_contact_sec": 2.5,
        "pre_contact_motion_pause_sec": 0.0,
        "handoff_min_forward_speed_mps": 0.18,
        "handoff_low_forward_speed_duration_sec": 0.08,
        "handoff_forward_speed_retention_ratio": 0.50,
        "skill_bridge_max_joint_delta_rad": 1.0,
        "skill_bridge_rms_joint_delta_rad": 0.30,
        "skill_bridge_entry_velocity_rms_rad_s": 0.25,
        "skill_bridge_target_exit_velocity_rms_rad_s": 0.10,
        "skill_bridge_exit_velocity_error_rms_rad_s": 0.08,
        "skill_bridge_peak_target_acceleration_rms_rad_s2": 8.0,
        "kick_contact_observed": True,
        "contact_time_sec": 11.8,
        "kick_contact_point_xyz_m": (1.0, 0.0, 0.10),
        "kick_contact_height_relative_ball_center_m": -0.015,
        "ball_launch_velocity_xyz_mps": (7.5, 1.0, 2.5),
        "ball_apex_height_m": 0.85,
        "ball_speed_peak_mps": 8.0,
        "goal_crossed": True,
        "goal_crossing_xyz_m": (5.0, 1.0, 0.115),
        "goal_mouth_hit": True,
        "goal_plane_target_error_m": 0.01,
        "net_capture_xyz_m": (5.2, 1.04, 0.115),
        "net_capture_target_error_m": 0.04,
        "final_ball_xyz_m": (5.1, 1.08, 0.115),
        "final_ball_yz_target_error_m": 0.08,
        "ball_retained_in_goal": True,
        "precision_radius_m": 0.16,
        "declared_target_corner": "left_lower",
        "declared_corner_distance_m": 0.20,
        "upper_corner_distance_m": 1.20,
        "lower_corner_distance_m": 0.20,
        "kick_min_pelvis_height_m": 0.70,
        "kick_peak_tilt_rad": 0.30,
        "final_pelvis_height_m": 0.78,
        "final_speed_mps": 0.01,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "actuator_saturation_steps": 0,
        "actuator_saturation_fraction": 0.0,
        "actuator_peak_demand_ratio": 0.95,
        "physics_steps": 10_000,
        "ballistic_contact_residual_executed": False,
        "ballistic_contact_residual_active_frames": 0,
        "ballistic_contact_residual_peak_target_delta_rad": 0.0,
    }
    values.update(changes)
    return G1FreeKickResult(**values)  # type: ignore[arg-type]


def test_free_kick_contract_is_strict_about_precision_and_continuity() -> None:
    assert _passing_result().passed
    assert not _passing_result(goal_plane_target_error_m=0.161).passed
    assert not _passing_result(state_reset_after_start=True).passed
    assert not _passing_result(runup_terminal_speed_mps=0.05).passed
    assert not _passing_result(pre_contact_motion_pause_sec=0.26).passed
    assert not _passing_result(ball_retained_in_goal=False).passed
    assert not _passing_result(declared_corner_distance_m=0.251).passed
    assert not _passing_result(actuator_saturation=True).passed
    assert not _passing_result(loft_teacher_executed=True).passed
    assert not _passing_result(post_contact_backward_displacement_m=0.201).passed
    assert not _passing_result(post_contact_forward_velocity_reversals=13).passed
    assert not _passing_result(post_contact_settling_time_sec=5.01).passed
    assert not _passing_result(post_contact_final_joint_velocity_rms_rad_s=0.101).passed
    assert not _passing_result(post_contact_mean_pelvis_speed_mps=0.121).passed
    assert not _passing_result(post_contact_mean_joint_velocity_rms_rad_s=0.251).passed
    assert _passing_result(
        declared_target_corner="left_upper",
        declared_corner_distance_m=0.20,
        upper_corner_distance_m=0.20,
        lower_corner_distance_m=1.20,
    ).passed
    assert not _passing_result().perceptual_continuity_passed
    assert _passing_result(handoff_to_contact_sec=1.0).perceptual_continuity_passed
    assert not _passing_result(
        handoff_to_contact_sec=1.0,
        handoff_low_forward_speed_duration_sec=0.12,
    ).perceptual_continuity_passed


def test_retry_recovery_requires_a_bound_outcome_model() -> None:
    with pytest.raises(ValueError, match="requires an outcome model"):
        G1FreeKickFlowConfig(football_retry_recovery_duration_sec=1.0)

    flow = G1FreeKickFlowConfig(
        football_outcome_model_hash="sha256:" + "1" * 64,
        football_retry_recovery_duration_sec=1.0,
    )
    assert flow.football_retry_recovery_duration_sec == 1.0


def test_vertical_aim_bias_is_bounded_and_separate_from_the_scoring_target() -> None:
    assert G1FreeKickFlowConfig(aim_bias_z_m=0.70).aim_bias_z_m == 0.70
    with pytest.raises(ValueError, match="vertical aim bias"):
        G1FreeKickFlowConfig(aim_bias_z_m=1.21)


def test_post_contact_damping_is_bounded() -> None:
    assert G1FreeKickFlowConfig(post_contact_damping_scale=1.8).post_contact_damping_scale == 1.8
    with pytest.raises(ValueError, match="post-contact damping"):
        G1FreeKickFlowConfig(post_contact_damping_scale=2.51)


def test_ballistic_skill_memory_binding_is_paired_and_sonic_only() -> None:
    memory_hash = "sha256:" + "1" * 64
    with pytest.raises(ValueError, match="must be paired"):
        G1FreeKickFlowConfig(ballistic_skill_memory_hash=memory_hash)
    with pytest.raises(ValueError, match="SONIC"):
        G1FreeKickFlowConfig(
            ballistic_skill_memory_hash=memory_hash,
            ballistic_skill_id="sonic-seed-0",
        )

    flow = G1FreeKickFlowConfig(
        approach_provider="sonic_fullbody",
        ballistic_skill_memory_hash=memory_hash,
        ballistic_skill_id="sonic-seed-0",
    )
    assert flow.ballistic_skill_id == "sonic-seed-0"


@pytest.mark.parametrize(
    ("geom1", "geom2", "direction"),
    ((1, 2, 1.0), (2, 1, -1.0)),
)
def test_ball_contact_frame_is_normalized_foot_to_ball_in_world(
    monkeypatch: pytest.MonkeyPatch,
    geom1: int,
    geom2: int,
    direction: float,
) -> None:
    frame = np.asarray(((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)))
    contact = SimpleNamespace(
        geom1=geom1,
        geom2=geom2,
        frame=frame.reshape(-1),
        pos=np.asarray((0.8, 0.1, 0.2)),
    )
    fake_mujoco = SimpleNamespace(
        mjtObj=SimpleNamespace(mjOBJ_GEOM=0),
        mj_id2name=lambda _model, _kind, geom: {
            1: "right_foot_ball",
            2: "ball_geom",
        }[geom],
        mj_contactForce=lambda _model, _data, _index, output: output.__setitem__(
            slice(None), np.asarray((10.0, 2.0, 0.0, 0.0, 0.0, 0.0))
        ),
    )
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)

    observed = _contact_observation(
        object(),
        SimpleNamespace(ncon=1, contact=[contact]),
        SimpleNamespace(ball_geom=2),
    )

    assert observed.ball_right
    assert observed.ball_force_n == pytest.approx(np.hypot(10.0, 2.0))
    assert observed.ball_contact_normal_xyz == pytest.approx(
        direction * frame[0]
    )
    assert observed.ball_contact_force_world_xyz_n == pytest.approx(
        direction * (frame.T @ np.asarray((10.0, 2.0, 0.0)))
    )


def test_rejected_video_metrics_render_missing_physics_as_na() -> None:
    assert _optional_finite_float(None) is None
    assert _optional_finite_float(float("nan")) is None
    assert _metric_text(None) == "N/A"
    assert _metric_text(0.0392) == "0.039 m"
    assert _duration_text(None) == "N/A"
    assert _duration_text(0.978) == "0.98 s"


def test_foot_pitch_loft_adapter_is_bounded_to_the_swing_phase() -> None:
    target = np.zeros(29)
    parameters = ShotParameters(foot_pitch_offset=-0.12, loft_synergy=0.10)

    before = _adapt_target(
        target=target,
        default=np.zeros(29),
        parameters=parameters,
        policy_frame=170,
    )
    swing = _adapt_target(
        target=target,
        default=np.zeros(29),
        parameters=parameters,
        policy_frame=250,
    )

    assert before[10] == 0.0
    assert swing[6] == pytest.approx(-0.10)
    assert swing[8] == pytest.approx(-0.06)
    assert swing[9] == pytest.approx(0.10)
    assert swing[10] == pytest.approx(-0.145)


def test_training_goal_and_runup_specs_fail_closed() -> None:
    goal = G1TrainingGoalSpec()
    runup = G1LearnedRunupConfig()

    assert goal.target_y_m < goal.width_m / 2.0
    assert goal.precision_radius_m == 0.16
    assert goal.plane_x_m == 5.0
    assert G1TrainingGoalSpec(plane_x_m=7.0).plane_x_m == 7.0
    with pytest.raises(ValueError, match="goal plane"):
        G1TrainingGoalSpec(plane_x_m=12.1)
    assert goal.target_corner == "left_lower"
    assert G1TrainingGoalSpec(target_y_m=1.0, target_z_m=1.35).target_corner == "left_upper"
    assert G1TrainingGoalSpec(target_y_m=-1.0, target_z_m=1.35).target_corner == "right_upper"
    assert G1FreeKickFlowConfig(kick_phase_start_frame=240).kick_phase_start_frame == 240
    assert G1FreeKickFlowConfig(aim_bias_y_m=2.0).aim_bias_y_m == 2.0
    assert runup.total_duration_sec == 4.7
    with pytest.raises(ValueError, match="gain schedule start"):
        G1FreeKickFlowConfig(strike_gain_schedule_start_policy_frame=180)
    with pytest.raises(ValueError, match="aim bias"):
        G1FreeKickFlowConfig(aim_bias_y_m=2.01)
    with pytest.raises(ValueError, match="yaw threshold"):
        G1FreeKickFlowConfig(contextual_phase_yaw_threshold_rad=0.04)
    with pytest.raises(ValueError, match="requires enabled routing"):
        G1FreeKickFlowConfig(contextual_phase_calibration_hash="sha256:" + "1" * 64)
    with pytest.raises(ValueError, match="must be SHA-256"):
        G1FreeKickFlowConfig(
            contextual_phase_yaw_threshold_rad=0.15,
            contextual_phase_calibration_hash="unsigned",
        )
    with pytest.raises(ValueError, match="inside the goal posts"):
        G1TrainingGoalSpec(target_y_m=1.18)
    with pytest.raises(ValueError, match="integer multiple"):
        G1LearnedRunupConfig(control_dt_sec=0.019)
    with pytest.raises(ValueError, match="kick phase start"):
        G1FreeKickFlowConfig(kick_phase_start_frame=241)


def test_contextual_phase_expert_uses_proprioceptive_yaw_not_planner_identity() -> None:
    flow = G1FreeKickFlowConfig(
        kick_phase_start_frame=214,
        contextual_phase_yaw_threshold_rad=0.15,
        contextual_high_yaw_kick_phase_start_frame=190,
    )

    assert _select_contextual_phase(flow, 0.02) == (214, False)
    assert _select_contextual_phase(flow, 0.237) == (190, True)
    assert _select_contextual_phase(flow, -0.16) == (190, True)


def test_loft_teacher_pitch_bonus_is_teacher_only() -> None:
    with pytest.raises(ValueError, match="requires the teacher"):
        G1FreeKickFlowConfig(shot_loft_teacher_foot_pitch_bonus_rad=0.01)
    with pytest.raises(ValueError, match="pitch bonus"):
        G1FreeKickFlowConfig(
            shot_loft_teacher_target_vz_mps=5.0,
            shot_loft_teacher_foot_pitch_bonus_rad=0.121,
        )
    assert (
        G1FreeKickFlowConfig(
            shot_loft_teacher_target_vz_mps=5.0,
            shot_loft_teacher_foot_pitch_bonus_rad=0.08,
        ).shot_loft_teacher_foot_pitch_bonus_rad
        == 0.08
    )
    with pytest.raises(ValueError, match="shot COM shift"):
        G1FreeKickFlowConfig(shot_com_shift_y_m=-0.081)


def test_net_capture_uses_deepest_measured_point_inside_goal_mouth() -> None:
    goal = G1TrainingGoalSpec()

    deepest = _deepest_goal_mouth_point(None, np.asarray((5.04, 0.98, 0.115)), goal)
    deepest = _deepest_goal_mouth_point(deepest, np.asarray((5.16, 0.97, 0.116)), goal)
    shallower = _deepest_goal_mouth_point(deepest, np.asarray((5.12, 0.96, 0.115)), goal)
    outside = _deepest_goal_mouth_point(deepest, np.asarray((5.25, 1.20, 0.115)), goal)

    assert deepest == (5.16, 0.97, 0.116)
    assert shallower == deepest
    assert outside == deepest


def test_missing_learned_gait_assets_are_ineligible(tmp_path: Path) -> None:
    qualification = qualify_g1_learned_gait(tmp_path)

    assert not qualification.eligible
    assert len(qualification.errors) == 2
    with pytest.raises(ValueError, match="not eligible"):
        qualification.require_eligible()


def test_sonic_contract_is_full_body_and_fails_closed(tmp_path: Path) -> None:
    config = G1SonicRunupConfig()
    qualification = qualify_g1_sonic(tmp_path)

    assert config.execution_frames == 170
    assert sorted(ISAACLAB_TO_MUJOCO.tolist()) == list(range(29))
    assert sorted(MUJOCO_TO_ISAACLAB.tolist()) == list(range(29))
    assert not qualification.eligible
    assert len(qualification.errors) == 4
    with pytest.raises(ValueError, match="SONIC assets are not eligible"):
        qualification.require_eligible()
    with pytest.raises(ValueError, match="gain scale"):
        G1SonicRunupConfig(gain_scale=0.49)


def test_sonic_stationary_recovery_padding_is_explicit_and_digest_bound() -> None:
    controller = object.__new__(G1SonicRunupController)
    controller.config = G1SonicRunupConfig()
    controller.reference = np.arange(180 * 36, dtype=np.float64).reshape(180, 36)
    controller.reference_digest = "sha256:before"
    terminal = controller.reference[-1].copy()

    controller.extend_stationary_recovery(50)

    assert controller.recovery_extension_frames == 50
    np.testing.assert_allclose(controller.reference[-1], terminal)
    assert controller.reference_digest.startswith("sha256:")
    assert controller.reference_digest != "sha256:before"
    with pytest.raises(ValueError, match="non-negative"):
        controller.extend_stationary_recovery(-1)


def test_free_kick_outputs_must_remain_external(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_free_kick_showcase(
            asset_root=tmp_path / "assets",
            gait_policy_root=tmp_path / "policies",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )
    with pytest.raises(ValueError, match="outside the source checkout"):
        render_g1_free_kick_showcase_video(
            evidence_path=tmp_path / "evidence.json",
            asset_root=tmp_path / "assets",
            output_path=tmp_path / "video.mp4",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_real_free_kick_is_continuous_precise_and_strict(tmp_path: Path) -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    gait_root = os.environ.get("ROSCLAW_G1_GAIT_POLICY_ROOT")
    if not asset_root or not gait_root:
        pytest.skip("G1 body and learned gait policy roots are not configured")

    evidence = run_g1_free_kick_showcase(
        asset_root=Path(asset_root),
        gait_policy_root=Path(gait_root),
        output_dir=tmp_path / "g1-free-kick",
        source_checkout=Path(__file__).resolve().parents[2],
    )

    assert evidence.passed
    assert evidence.strict_replay
    assert evidence.result.runup_distance_m >= 3.0
    assert evidence.result.runup_peak_speed_mps >= 1.0
    assert evidence.result.pre_contact_motion_pause_sec == 0.0
    assert evidence.result.goal_plane_target_error_m is not None
    assert evidence.result.goal_plane_target_error_m <= 0.16
    assert evidence.result.ball_retained_in_goal
    assert evidence.result.goal_mouth_hit
    assert not evidence.result.state_reset_after_start
    assert not evidence.result.post_kick_fall
    with np.load(evidence.trajectory_path, allow_pickle=False) as trajectory:
        assert np.any(trajectory["event_phase"] == int(G1FootballEventPhase.CONTACT))


@pytest.mark.integration
def test_real_stadium_replaces_wall_with_collision_goal() -> None:
    asset_root = os.environ.get("ROSCLAW_G1_ASSET_ROOT")
    if not asset_root:
        pytest.skip("ROSCLAW_G1_ASSET_ROOT is not configured")
    import mujoco

    model = build_g1_stadium_model(Path(asset_root))

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box") == -1
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_crossbar") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_back_net") >= 0

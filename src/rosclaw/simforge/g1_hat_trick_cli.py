"""Product CLI for GoalForge Hat Trick evidence and visualization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def dispatch_hat_trick_argv(argv: list[str]) -> int | None:
    if len(argv) >= 3 and argv[:2] == ["goalforge", "readiness-recovery"]:
        return _dispatch_readiness_recovery_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "free-kick-showcase"]:
        return _dispatch_free_kick_showcase_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "self-aware-showcase"]:
        return _dispatch_self_aware_showcase_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "coupled-showcase"]:
        return _dispatch_coupled_showcase_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "coupled-relay"]:
        return _dispatch_coupled_relay_argv(argv)
    if len(argv) >= 3 and argv[:2] == ["goalforge", "relay"]:
        return _dispatch_relay_argv(argv)
    if len(argv) < 3 or argv[:2] != ["goalforge", "hat-trick"]:
        return None
    parser = argparse.ArgumentParser(prog="rosclaw goalforge hat-trick")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute three strictly replayed MuJoCo shots")
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser("export", help="render a passing Hat Trick report")
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_hat_trick import run_goalforge_hat_trick

        result = run_goalforge_hat_trick(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_hat_trick_video import render_goalforge_hat_trick_video

    result = render_goalforge_hat_trick_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_free_kick_showcase_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge free-kick-showcase")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strict continuous learned run-up and precision free kick",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--gait-policy-root", type=Path, required=True)
    run.add_argument(
        "--approach-provider",
        choices=("groot_history", "sonic_fullbody"),
        default="groot_history",
    )
    run.add_argument(
        "--sonic-model-root",
        type=Path,
        help="GEAR-SONIC model root; required for sonic_fullbody",
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    run.add_argument("--start-x-m", type=float)
    run.add_argument("--start-y-m", type=float, default=0.0)
    run.add_argument("--run-velocity-mps", type=float, default=1.50)
    run.add_argument("--brake-velocity-mps", type=float, default=0.55)
    run.add_argument("--sonic-duration-sec", type=float, default=3.40)
    run.add_argument("--sonic-gain-scale", type=float, default=1.0)
    run.add_argument("--sonic-authority-calibration", type=Path)
    run.add_argument("--planner-seed", type=int, default=0)
    run.add_argument(
        "--direct-strike",
        action="store_true",
        help="use the short velocity-matched run-to-strike transition profile",
    )
    run.add_argument("--bridge-duration-sec", type=float)
    run.add_argument("--bridge-entry-velocity-scale", type=float)
    run.add_argument("--bridge-exit-velocity-scale", type=float)
    run.add_argument("--bridge-boundary-velocity-limit-rad-s", type=float, default=2.0)
    run.add_argument("--kick-phase-start-frame", type=int)
    run.add_argument("--contextual-phase-calibration", type=Path)
    run.add_argument("--proprioceptive-expert-router", type=Path)
    run.add_argument("--football-outcome-model", type=Path)
    run.add_argument(
        "--ballistic-skill-memory",
        type=Path,
        help="full-state SIM_ONLY memory of qualified SONIC ballistic skill islands",
    )
    run.add_argument(
        "--select-best-registered-skill",
        action="store_true",
        help="retrieve the lowest-error registered skill before the mandatory shot",
    )
    run.add_argument(
        "--football-motion-prior",
        type=Path,
        help="train-only SIM_ONLY football contact-motion prior artifact",
    )
    run.add_argument("--football-motion-prior-blend", type=float, default=0.0)
    run.add_argument(
        "--football-motion-prior-contact-policy-frame",
        type=int,
        default=265,
    )
    run.add_argument(
        "--ballistic-contact-residual-rad",
        type=float,
        nargs=6,
        default=(0.0,) * 6,
        metavar=("HIP_P", "HIP_R", "HIP_Y", "KNEE", "ANKLE_P", "ANKLE_R"),
        help="bounded SIM curriculum residual for six right-leg joint targets",
    )
    run.add_argument("--ballistic-contact-policy-frame", type=int, default=256)
    run.add_argument(
        "--ballistic-contact-lead-duration-sec",
        type=float,
        default=0.16,
        help="bounded pre-contact pulse duration for SIM_ONLY contact learning",
    )
    run.add_argument(
        "--ballistic-contact-trail-duration-sec",
        type=float,
        default=0.08,
        help="bounded post-contact pulse duration for SIM_ONLY contact learning",
    )
    run.add_argument(
        "--post-contact-damping-scale",
        type=float,
        default=1.0,
        help="bounded recovery velocity-feedback multiplier after follow-through",
    )
    run.add_argument("--football-retry-recovery-sec", type=float, default=0.0)
    run.add_argument("--football-retry-follow-through-gain-scale", type=float, default=1.0)
    run.add_argument("--contextual-phase-yaw-threshold-rad", type=float, default=0.0)
    run.add_argument("--contextual-high-yaw-kick-phase-start-frame", type=int, default=190)
    run.add_argument("--aim-bias-y-m", type=float, default=0.30)
    run.add_argument(
        "--aim-bias-z-m",
        type=float,
        default=0.0,
        help="bounded policy aim correction for learned ballistic drop compensation",
    )
    run.add_argument("--shot-pelvis-yaw-rad", type=float, default=0.10)
    run.add_argument("--shot-foot-yaw-rad", type=float, default=0.01)
    run.add_argument("--shot-foot-pitch-rad", type=float, default=0.0)
    run.add_argument("--shot-loft-synergy-rad", type=float, default=0.0)
    run.add_argument("--shot-loft-teacher-target-vz-mps", type=float, default=0.0)
    run.add_argument("--shot-loft-teacher-gain-n-per-mps", type=float, default=24.0)
    run.add_argument("--shot-loft-teacher-max-force-n", type=float, default=60.0)
    run.add_argument("--shot-loft-teacher-target-vx-mps", type=float, default=0.0)
    run.add_argument("--shot-loft-teacher-forward-gain-n-per-mps", type=float, default=20.0)
    run.add_argument("--shot-loft-teacher-max-forward-force-n", type=float, default=80.0)
    run.add_argument("--shot-loft-teacher-start-policy-frame", type=int, default=230)
    run.add_argument("--shot-loft-teacher-end-policy-frame", type=int, default=335)
    run.add_argument("--shot-loft-teacher-foot-pitch-bonus-rad", type=float, default=0.0)
    run.add_argument(
        "--shot-loft-teacher-max-foot-ball-distance-m",
        type=float,
        default=0.0,
        help="SIM teacher proximity gate; zero preserves the full phase window",
    )
    run.add_argument("--shot-com-shift-y-m", type=float, default=-0.065)
    run.add_argument("--shot-swing-amplitude", type=float, default=0.85)
    run.add_argument("--shot-swing-speed-scale", type=float)
    run.add_argument(
        "--shot-load-speed-scale",
        type=float,
        default=1.0,
        help="compress learned pre-swing loading without accelerating contact",
    )
    run.add_argument("--shot-contact-phase-offset", type=float, default=-0.015)
    run.add_argument(
        "--strike-gain-schedule-start-policy-frame",
        type=int,
        default=0,
        help="apply calibrated stance/follow-through gains before contact; zero disables",
    )
    run.add_argument(
        "--target-zone",
        choices=("custom", "left-upper", "right-upper", "left-lower", "right-lower"),
        default="custom",
    )
    run.add_argument(
        "--goal-plane-x-m",
        type=float,
        default=5.0,
        help="goal line x position; the ball starts at x=1 m",
    )
    run.add_argument("--target-y-m", type=float)
    run.add_argument("--target-z-m", type=float)
    run.add_argument("--approach-strike-candidate", type=Path)
    run.add_argument("--residual-fraction", type=float, default=0.20)
    run.add_argument("--maximum-residual-nm", type=float, default=5.0)
    export = commands.add_parser(
        "export",
        help="render passing free-kick evidence in the native training goal",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    export.add_argument(
        "--allow-rejected-candidate",
        action="store_true",
        help="render a strict-replay failed candidate with a diagnostic watermark",
    )
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.growth.approach_strike_residual import (
            G1ApproachStrikeResidualConfig,
        )
        from rosclaw.growth.ballistic_skill_memory import (
            load_g1_ballistic_skill_memory,
        )
        from rosclaw.growth.contextual_phase_calibration import (
            load_g1_contextual_phase_calibration,
        )
        from rosclaw.growth.football_motion_prior import (
            load_g1_football_motion_prior,
        )
        from rosclaw.growth.football_outcome_model import (
            load_g1_football_outcome_model,
        )
        from rosclaw.growth.proprioceptive_expert_router import (
            load_g1_proprioceptive_expert_router,
        )
        from rosclaw.growth.sonic_authority_calibration import (
            load_g1_sonic_authority_calibration,
        )
        from rosclaw.simforge.g1_free_kick_showcase import (
            G1FreeKickFlowConfig,
            run_g1_free_kick_showcase,
        )
        from rosclaw.simforge.g1_learned_runup import G1LearnedRunupConfig
        from rosclaw.simforge.g1_sonic_runup import G1SonicRunupConfig
        from rosclaw.simforge.g1_stadium_scene import G1TrainingGoalSpec

        sonic_enabled = args.approach_provider == "sonic_fullbody"
        if sonic_enabled and args.sonic_model_root is None:
            parser.error("--sonic-model-root is required for sonic_fullbody")
        start_x = args.start_x_m
        if start_x is None:
            start_x = -3.40 if sonic_enabled else G1LearnedRunupConfig().start_x_m
        bridge_duration = args.bridge_duration_sec
        if bridge_duration is None:
            bridge_duration = (
                0.16 if sonic_enabled and args.direct_strike else 0.35 if sonic_enabled else 0.60
            )
        bridge_entry_velocity_scale = args.bridge_entry_velocity_scale
        if bridge_entry_velocity_scale is None:
            bridge_entry_velocity_scale = (
                0.65 if sonic_enabled and args.direct_strike else 0.45 if sonic_enabled else 0.0
            )
        bridge_exit_velocity_scale = args.bridge_exit_velocity_scale
        if bridge_exit_velocity_scale is None:
            bridge_exit_velocity_scale = 1.0 if sonic_enabled else 0.0
        contextual_calibration = (
            load_g1_contextual_phase_calibration(args.contextual_phase_calibration)
            if args.contextual_phase_calibration is not None
            else None
        )
        proprioceptive_router = (
            load_g1_proprioceptive_expert_router(args.proprioceptive_expert_router)
            if args.proprioceptive_expert_router is not None
            else None
        )
        football_outcome_model = (
            load_g1_football_outcome_model(args.football_outcome_model)
            if args.football_outcome_model is not None
            else None
        )
        ballistic_skill_memory = (
            load_g1_ballistic_skill_memory(args.ballistic_skill_memory)
            if args.ballistic_skill_memory is not None
            else None
        )
        if args.select_best_registered_skill and ballistic_skill_memory is None:
            parser.error("--select-best-registered-skill requires --ballistic-skill-memory")
        football_motion_prior = (
            load_g1_football_motion_prior(args.football_motion_prior)
            if args.football_motion_prior is not None
            else None
        )
        if (football_motion_prior is None) != (args.football_motion_prior_blend == 0.0):
            parser.error(
                "--football-motion-prior and non-zero --football-motion-prior-blend "
                "must be provided together"
            )
        routing_modes = sum(
            item is not None
            for item in (
                contextual_calibration,
                proprioceptive_router,
                football_outcome_model,
            )
        )
        if routing_modes > 1:
            parser.error("contextual router, expert router, and outcome model are exclusive")
        if ballistic_skill_memory is not None and routing_modes:
            parser.error("ballistic skill memory and legacy routing models are exclusive")
        if ballistic_skill_memory is not None and not sonic_enabled:
            parser.error("ballistic skill memory requires --approach-provider sonic_fullbody")
        if args.football_retry_recovery_sec > 0.0 and football_outcome_model is None:
            parser.error("--football-retry-recovery-sec requires --football-outcome-model")
        if football_outcome_model is not None:
            if args.contextual_phase_yaw_threshold_rad != 0.0:
                parser.error("football outcome model cannot be combined with a yaw threshold")
            if (
                args.kick_phase_start_frame is not None
                and args.kick_phase_start_frame != football_outcome_model.baseline_phase
            ):
                parser.error("--kick-phase-start-frame disagrees with outcome model")
            kick_phase_start_frame = football_outcome_model.baseline_phase
            contextual_phase_yaw_threshold_rad = 0.0
            contextual_high_yaw_kick_phase_start_frame = 190
        elif proprioceptive_router is not None:
            if args.contextual_phase_yaw_threshold_rad != 0.0:
                parser.error("proprioceptive router cannot be combined with a yaw threshold")
            if (
                args.kick_phase_start_frame is not None
                and args.kick_phase_start_frame != proprioceptive_router.baseline_phase
            ):
                parser.error("--kick-phase-start-frame disagrees with expert router")
            kick_phase_start_frame = proprioceptive_router.baseline_phase
            contextual_phase_yaw_threshold_rad = 0.0
            contextual_high_yaw_kick_phase_start_frame = 190
        elif contextual_calibration is not None:
            if args.contextual_phase_yaw_threshold_rad != 0.0:
                parser.error("--contextual-phase-yaw-threshold-rad cannot override a calibration")
            if (
                args.kick_phase_start_frame is not None
                and args.kick_phase_start_frame != contextual_calibration.normal_phase_start_frame
            ):
                parser.error("--kick-phase-start-frame disagrees with calibration")
            if (
                args.contextual_high_yaw_kick_phase_start_frame != 190
                and args.contextual_high_yaw_kick_phase_start_frame
                != contextual_calibration.high_yaw_phase_start_frame
            ):
                parser.error(
                    "--contextual-high-yaw-kick-phase-start-frame disagrees with calibration"
                )
            kick_phase_start_frame = contextual_calibration.normal_phase_start_frame
            contextual_phase_yaw_threshold_rad = contextual_calibration.yaw_threshold_rad
            contextual_high_yaw_kick_phase_start_frame = (
                contextual_calibration.high_yaw_phase_start_frame
            )
        else:
            kick_phase_start_frame = args.kick_phase_start_frame
            if kick_phase_start_frame is None:
                kick_phase_start_frame = 180 if sonic_enabled and args.direct_strike else 150
            contextual_phase_yaw_threshold_rad = args.contextual_phase_yaw_threshold_rad
            contextual_high_yaw_kick_phase_start_frame = (
                args.contextual_high_yaw_kick_phase_start_frame
            )
        shot_swing_speed_scale = args.shot_swing_speed_scale
        if shot_swing_speed_scale is None:
            shot_swing_speed_scale = 1.10 if args.direct_strike else 0.90
        preset_targets = {
            "left-upper": (1.0, 1.35),
            "right-upper": (-1.0, 1.35),
            "left-lower": (1.0, 0.115),
            "right-lower": (-1.0, 0.115),
        }
        if args.target_zone != "custom" and (
            args.target_y_m is not None or args.target_z_m is not None
        ):
            parser.error("--target-y-m/--target-z-m cannot override --target-zone")
        if args.target_zone == "custom":
            target_y_m = 1.0 if args.target_y_m is None else args.target_y_m
            target_z_m = 0.115 if args.target_z_m is None else args.target_z_m
        else:
            target_y_m, target_z_m = preset_targets[args.target_zone]
        authority_calibration = (
            load_g1_sonic_authority_calibration(args.sonic_authority_calibration)
            if args.sonic_authority_calibration is not None
            else None
        )
        planner_seed = args.planner_seed
        ballistic_contact_residual_rad = tuple(args.ballistic_contact_residual_rad)
        ballistic_contact_policy_frame = args.ballistic_contact_policy_frame
        post_contact_damping_scale = args.post_contact_damping_scale
        ballistic_skill_id = None
        if ballistic_skill_memory is not None:
            try:
                prototype = (
                    ballistic_skill_memory.best_prototype
                    if args.select_best_registered_skill
                    else ballistic_skill_memory.prototype_for_seed(planner_seed)
                )
            except ValueError as error:
                parser.error(str(error))
            planner_seed = prototype.planner_seed
            ballistic_contact_residual_rad = prototype.action_rad
            ballistic_contact_policy_frame = prototype.contact_policy_frame
            post_contact_damping_scale = prototype.post_contact_damping_scale
            ballistic_skill_id = prototype.skill_id

        result = run_g1_free_kick_showcase(
            asset_root=args.asset_root,
            gait_policy_root=args.gait_policy_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
            runup_config=G1LearnedRunupConfig(
                start_x_m=start_x,
                start_y_m=args.start_y_m,
            ),
            flow_config=G1FreeKickFlowConfig(
                approach_provider=args.approach_provider,
                bridge_duration_sec=bridge_duration,
                bridge_entry_velocity_scale=bridge_entry_velocity_scale,
                bridge_exit_velocity_scale=bridge_exit_velocity_scale,
                bridge_boundary_velocity_limit_rad_s=(args.bridge_boundary_velocity_limit_rad_s),
                kick_phase_start_frame=kick_phase_start_frame,
                contextual_phase_yaw_threshold_rad=(contextual_phase_yaw_threshold_rad),
                contextual_high_yaw_kick_phase_start_frame=(
                    contextual_high_yaw_kick_phase_start_frame
                ),
                contextual_phase_calibration_hash=(
                    None
                    if contextual_calibration is None
                    else contextual_calibration.calibration_hash
                ),
                proprioceptive_router_hash=(
                    None if proprioceptive_router is None else proprioceptive_router.router_hash
                ),
                football_outcome_model_hash=(
                    None
                    if football_outcome_model is None
                    else football_outcome_model.model_hash
                ),
                football_motion_prior_hash=(
                    None if football_motion_prior is None else football_motion_prior.prior_hash
                ),
                football_motion_prior_blend=args.football_motion_prior_blend,
                football_motion_prior_contact_policy_frame=(
                    args.football_motion_prior_contact_policy_frame
                ),
                ballistic_contact_residual_rad=ballistic_contact_residual_rad,
                ballistic_contact_policy_frame=ballistic_contact_policy_frame,
                ballistic_contact_lead_duration_sec=(args.ballistic_contact_lead_duration_sec),
                ballistic_contact_trail_duration_sec=(args.ballistic_contact_trail_duration_sec),
                post_contact_damping_scale=post_contact_damping_scale,
                ballistic_skill_memory_hash=(
                    None
                    if ballistic_skill_memory is None
                    else ballistic_skill_memory.memory_hash
                ),
                ballistic_skill_id=ballistic_skill_id,
                football_retry_recovery_duration_sec=(
                    args.football_retry_recovery_sec
                ),
                football_retry_follow_through_gain_scale=(
                    args.football_retry_follow_through_gain_scale
                ),
                aim_bias_y_m=args.aim_bias_y_m,
                aim_bias_z_m=args.aim_bias_z_m,
                shot_pelvis_yaw_offset_rad=args.shot_pelvis_yaw_rad,
                shot_foot_yaw_offset_rad=args.shot_foot_yaw_rad,
                shot_foot_pitch_offset_rad=args.shot_foot_pitch_rad,
                shot_loft_synergy_rad=args.shot_loft_synergy_rad,
                shot_loft_teacher_target_vz_mps=(args.shot_loft_teacher_target_vz_mps),
                shot_loft_teacher_gain_n_per_mps=(args.shot_loft_teacher_gain_n_per_mps),
                shot_loft_teacher_max_force_n=args.shot_loft_teacher_max_force_n,
                shot_loft_teacher_target_vx_mps=args.shot_loft_teacher_target_vx_mps,
                shot_loft_teacher_forward_gain_n_per_mps=(
                    args.shot_loft_teacher_forward_gain_n_per_mps
                ),
                shot_loft_teacher_max_forward_force_n=(args.shot_loft_teacher_max_forward_force_n),
                shot_loft_teacher_start_policy_frame=(args.shot_loft_teacher_start_policy_frame),
                shot_loft_teacher_end_policy_frame=(args.shot_loft_teacher_end_policy_frame),
                shot_loft_teacher_foot_pitch_bonus_rad=(
                    args.shot_loft_teacher_foot_pitch_bonus_rad
                ),
                shot_loft_teacher_max_foot_ball_distance_m=(
                    args.shot_loft_teacher_max_foot_ball_distance_m
                ),
                shot_com_shift_y_m=args.shot_com_shift_y_m,
                shot_swing_amplitude=args.shot_swing_amplitude,
                shot_swing_speed_scale=shot_swing_speed_scale,
                shot_load_speed_scale=args.shot_load_speed_scale,
                shot_contact_phase_offset=args.shot_contact_phase_offset,
                strike_gain_schedule_start_policy_frame=(
                    args.strike_gain_schedule_start_policy_frame
                ),
                strike_gain_scales=(
                    (1.0,) * 29
                    if authority_calibration is None
                    else authority_calibration.strike_gain_scales
                ),
                follow_through_gain_scales=(
                    (1.0,) * 29
                    if authority_calibration is None
                    else authority_calibration.follow_through_gain_scales
                ),
                authority_calibration_hash=(
                    None
                    if authority_calibration is None
                    else authority_calibration.calibration_hash
                ),
            ),
            goal_spec=G1TrainingGoalSpec(
                plane_x_m=args.goal_plane_x_m,
                target_y_m=target_y_m,
                target_z_m=target_z_m,
            ),
            sonic_model_root=args.sonic_model_root,
            sonic_runup_config=(
                G1SonicRunupConfig(
                    run_velocity_mps=args.run_velocity_mps,
                    brake_velocity_mps=args.brake_velocity_mps,
                    execution_duration_sec=args.sonic_duration_sec,
                    gain_scale=args.sonic_gain_scale,
                    joint_gain_scales=(
                        (1.0,) * 29
                        if authority_calibration is None
                        else authority_calibration.joint_gain_scales
                    ),
                    authority_calibration_hash=(
                        None
                        if authority_calibration is None
                        else authority_calibration.calibration_hash
                    ),
                    planner_seed=planner_seed,
                )
                if sonic_enabled
                else None
            ),
            approach_strike_candidate_path=args.approach_strike_candidate,
            approach_strike_residual_config=(
                G1ApproachStrikeResidualConfig(
                    residual_fraction=args.residual_fraction,
                    maximum_residual_nm=args.maximum_residual_nm,
                )
                if args.approach_strike_candidate is not None
                else None
            ),
            proprioceptive_expert_router=proprioceptive_router,
            football_outcome_model=football_outcome_model,
            football_motion_prior=football_motion_prior,
            ballistic_skill_memory=ballistic_skill_memory,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.passed else 2
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rosclaw.simforge.g1_free_kick_showcase_video import (
        render_g1_free_kick_showcase_video,
    )

    result = render_g1_free_kick_showcase_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
        allow_rejected_candidate=args.allow_rejected_candidate,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_readiness_recovery_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge readiness-recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strict continuous-world recovery after readiness abstention",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--sonic-model-root", type=Path, required=True)
    run.add_argument("--sonic-authority-calibration", type=Path, required=True)
    run.add_argument("--proprioceptive-expert-router", type=Path, required=True)
    run.add_argument("--proprioceptive-readiness-gate", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    run.add_argument("--planner-seed", type=int, required=True)
    run.add_argument("--run-velocity-mps", type=float, default=1.50)
    run.add_argument("--brake-velocity-mps", type=float, default=0.55)
    run.add_argument("--sonic-duration-sec", type=float, default=3.40)
    run.add_argument("--neural-deceleration-duration-sec", type=float, default=1.80)
    run.add_argument("--hold-duration-sec", type=float, default=1.20)
    run.add_argument("--recovery-gain-scale", type=float, default=0.75)
    run.add_argument(
        "--evidence-domain",
        choices=(
            "DEVELOPMENT_READINESS_RECOVERY",
            "FROZEN_READINESS_RECOVERY_VALIDATION",
        ),
        default="DEVELOPMENT_READINESS_RECOVERY",
    )
    export = commands.add_parser(
        "export",
        help="render frozen multi-seed abstention recovery evidence",
    )
    export.add_argument("evaluation", type=Path)
    export.add_argument("--evidence-json", type=Path, action="append", required=True)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])

    if args.command == "export":
        os.environ.setdefault("MUJOCO_GL", "egl")
        from rosclaw.simforge.g1_readiness_recovery_video import (
            render_g1_readiness_recovery_video,
        )

        result = render_g1_readiness_recovery_video(
            evaluation_path=args.evaluation,
            evidence_paths=tuple(args.evidence_json),
            asset_root=args.asset_root,
            output_path=args.output,
            source_checkout=args.source_checkout,
            fps=args.fps,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0

    from rosclaw.growth.proprioceptive_expert_router import (
        load_g1_proprioceptive_expert_router,
    )
    from rosclaw.growth.proprioceptive_readiness_gate import (
        load_g1_proprioceptive_readiness_gate,
    )
    from rosclaw.growth.sonic_authority_calibration import (
        load_g1_sonic_authority_calibration,
    )
    from rosclaw.simforge.g1_readiness_recovery import (
        G1ReadinessRecoveryConfig,
        run_g1_readiness_recovery,
    )
    from rosclaw.simforge.g1_sonic_runup import G1SonicRunupConfig

    authority = load_g1_sonic_authority_calibration(args.sonic_authority_calibration)
    router = load_g1_proprioceptive_expert_router(args.proprioceptive_expert_router)
    readiness = load_g1_proprioceptive_readiness_gate(args.proprioceptive_readiness_gate)
    result = run_g1_readiness_recovery(
        asset_root=args.asset_root,
        sonic_model_root=args.sonic_model_root,
        output_dir=args.output_dir,
        source_checkout=args.source_checkout,
        router=router,
        readiness_gate=readiness,
        sonic_config=G1SonicRunupConfig(
            run_velocity_mps=args.run_velocity_mps,
            brake_velocity_mps=args.brake_velocity_mps,
            execution_duration_sec=args.sonic_duration_sec,
            joint_gain_scales=authority.joint_gain_scales,
            authority_calibration_hash=authority.calibration_hash,
            planner_seed=args.planner_seed,
        ),
        recovery_config=G1ReadinessRecoveryConfig(
            neural_deceleration_duration_sec=(
                args.neural_deceleration_duration_sec
            ),
            hold_duration_sec=args.hold_duration_sec,
            gain_scale=args.recovery_gain_scale,
        ),
        evidence_domain=args.evidence_domain,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.passed else 2


def _dispatch_self_aware_showcase_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge self-aware-showcase")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute three strict moving-ball self-awareness challenges",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render the challenge evidence and audited v2/v3 safety contrast",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--rejected-v2-evidence", type=Path, required=True)
    export.add_argument("--self-aware-v3-evidence", type=Path, required=True)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_self_aware_showcase import (
            run_g1_self_aware_showcase,
        )

        result = run_g1_self_aware_showcase(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.passed else 2
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rosclaw.simforge.g1_self_aware_showcase_video import (
        render_g1_self_aware_showcase_video,
    )

    result = render_g1_self_aware_showcase_video(
        showcase_evidence_path=args.evidence,
        rejected_v2_evidence_path=args.rejected_v2_evidence,
        self_aware_v3_evidence_path=args.self_aware_v3_evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_relay_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge relay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strictly replayed G1 pass-to-high-shot relay",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render a passing G1 relay evidence report",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_two_player_relay import run_g1_two_player_relay

        result = run_g1_two_player_relay(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_two_player_relay_video import (
        render_g1_two_player_relay_video,
    )

    result = render_g1_two_player_relay_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_coupled_relay_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge coupled-relay")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute a strictly replayed two-G1 relay in one MuJoCo world",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render passing coupled two-G1 evidence",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_coupled_relay import run_g1_coupled_relay

        result = run_g1_coupled_relay(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    from rosclaw.simforge.g1_coupled_relay_video import (
        render_g1_coupled_relay_video,
    )

    result = render_g1_coupled_relay_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _dispatch_coupled_showcase_argv(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="rosclaw goalforge coupled-showcase")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help="execute five strict two-G1 physics challenges",
    )
    run.add_argument("--asset-root", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export = commands.add_parser(
        "export",
        help="render a passing five-challenge showcase",
    )
    export.add_argument("evidence", type=Path)
    export.add_argument("--asset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--source-checkout", type=Path, default=Path.cwd())
    export.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv[2:])
    if args.command == "run":
        from rosclaw.simforge.g1_coupled_relay_showcase import (
            run_g1_coupled_showcase,
        )

        result = run_g1_coupled_showcase(
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            source_checkout=args.source_checkout,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    # MuJoCo selects its OpenGL backend during import, so headless CLI export
    # must declare EGL before importing the renderer dependency graph.
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rosclaw.simforge.g1_coupled_showcase_video import (
        render_g1_coupled_showcase_video,
    )

    result = render_g1_coupled_showcase_video(
        evidence_path=args.evidence,
        asset_root=args.asset_root,
        output_path=args.output,
        source_checkout=args.source_checkout,
        fps=args.fps,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


__all__ = ["dispatch_hat_trick_argv"]

from __future__ import annotations

import numpy as np
import pytest

from rosclaw.simforge.backends.unitree_mujoco_backend import GoalForgeEpisode
from rosclaw.simforge.g1_failure_curriculum_validation import (
    G1EmpiricalRiskCritic,
    audit_g1_trajectory,
    build_g1_failure_curriculum,
    calibrate_g1_regime_belief,
    evaluate_g1_failure_curriculum_gate,
    select_g1_boundary_guard,
    select_g1_contextual_skill,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import (
    GoalForgeResult,
    GoalForgeStatus,
    ShotParameters,
)


def _result(**changes: object) -> GoalForgeResult:
    values: dict[str, object] = {
        "status": GoalForgeStatus.SUCCESS,
        "success": True,
        "physics_executed": True,
        "contact_observed": True,
        "kick_foot_contacted": True,
        "goal_crossed": True,
        "target_zone_hit": True,
        "target_error_m": 0.2,
        "ball_speed_mps": 8.0,
        "ball_contact_time_sec": 2.0,
        "contact_impulse_ns": 4.0,
        "support_foot_slip_m": 0.01,
        "com_margin_min_m": 0.01,
        "torso_roll_peak_rad": 0.2,
        "torso_pitch_peak_rad": 0.2,
        "peak_torque_scale": 0.5,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "post_kick_fall": False,
        "post_kick_stability_time_sec": 0.5,
        "final_pelvis_height_m": 0.8,
        "physics_steps": 100,
        "finite_state": True,
        "robustness": 0.8,
    }
    values.update(changes)
    return GoalForgeResult(**values)  # type: ignore[arg-type]


def _episode(*, result: GoalForgeResult, speed: float, height: float) -> GoalForgeEpisode:
    scenario = build_g1_failure_curriculum()[0].scenario
    return GoalForgeEpisode(
        scenario=scenario,
        parameters=ShotParameters(),
        result=result,
        receipt=None,
        artifact_root=None,
        trajectory={
            "ball_velocity": np.asarray(((0.0, 0.0, 0.0), (speed, 0.0, 0.0))),
            "ball_pose": np.asarray(
                ((1.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0),
                 (1.0, 0.0, height, 1.0, 0.0, 0.0, 0.0))
            ),
        },
    )


def test_quality_audit_quarantines_noncausal_ball_explosion() -> None:
    episode = _episode(
        result=_result(
            success=False,
            kick_foot_contacted=False,
            status=GoalForgeStatus.BALL_NOT_CONTACTED,
        ),
        speed=31.0,
        height=20.0,
    )

    audit = audit_g1_trajectory(episode)

    assert not audit.accepted_for_learning
    assert audit.reasons == ("NONCAUSAL_BALL_MOTION_WITHOUT_KICK_CONTACT",)


def test_quality_audit_accepts_fast_ball_after_real_contact() -> None:
    audit = audit_g1_trajectory(_episode(result=_result(), speed=20.0, height=3.0))

    assert audit.accepted_for_learning


def test_empirical_risk_critic_excludes_quarantined_labels() -> None:
    clean = audit_g1_trajectory(_episode(result=_result(), speed=8.0, height=0.5))
    quarantined = audit_g1_trajectory(
        _episode(
            result=_result(kick_foot_contacted=False),
            speed=30.0,
            height=10.0,
        )
    )
    context = {"body_calibration_state": 0.02, "target_z": 0.55}
    critic = G1EmpiricalRiskCritic()

    assert critic.observe(observed_context=context, critical=False, audit=clean)
    assert not critic.observe(observed_context=context, critical=True, audit=quarantined)

    estimate = critic.estimate(context)
    assert estimate.sample_count == 1
    assert estimate.critical_count == 0
    assert estimate.wilson_upper_95 > 0.5
    assert critic.quarantined_count == 1


def test_curriculum_is_disjoint_and_sealed_cases_hide_physical_truth() -> None:
    curriculum = build_g1_failure_curriculum()
    commitments = [case.scenario.scenario_commitment for case in curriculum]

    assert len(commitments) == len(set(commitments))
    assert {case.scenario.partition for case in curriculum} == {
        Partition.DEVELOPMENT,
        Partition.VALIDATION,
        Partition.HOLDOUT,
    }
    sealed = [case for case in curriculum if case.purpose != "development"]
    assert all("support_ground_friction" not in case.public_dict()["scenario"] for case in sealed)


def test_contextual_routes_use_observed_calibration_and_remain_bounded() -> None:
    calibrated = {"body_calibration_state": 0.01}
    moderate = {"body_calibration_state": 0.021}
    biased = {"body_calibration_state": 0.025}
    high_bias = {"body_calibration_state": 0.03}

    assert select_g1_contextual_skill(calibrated).pelvis_yaw_offset == 0.10
    assert select_g1_contextual_skill(calibrated).com_shift_y == -0.065
    assert select_g1_contextual_skill(moderate).pelvis_yaw_offset == pytest.approx(
        0.04797899070127244
    )
    assert len(select_g1_boundary_guard(moderate).protected_joint_names) == 3
    assert len(select_g1_boundary_guard(biased).protected_joint_names) == 2
    assert len(select_g1_boundary_guard(high_bias).protected_joint_names) == 3
    assert select_g1_boundary_guard(high_bias).activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="missing"):
        select_g1_contextual_skill({})


def test_regime_belief_executes_calibrated_case_and_abstains_unsafe_case() -> None:
    curriculum = build_g1_failure_curriculum()
    safe = next(case.scenario for case in curriculum if case.case_id == "development-neighbour-easy")
    unsafe = next(
        case.scenario
        for case in curriculum
        if case.case_id == "private-holdout-v3-unsafe-regime"
    )

    safe_receipt = calibrate_g1_regime_belief(safe)
    unsafe_receipt = calibrate_g1_regime_belief(unsafe)

    assert safe_receipt.safe_to_execute
    assert not unsafe_receipt.safe_to_execute
    assert unsafe_receipt.reasons
    assert safe_receipt.receipt_hash.startswith("sha256:")
    assert "support_ground_friction" not in unsafe_receipt.to_dict()


def test_gate_requires_success_no_critical_and_strict_replay() -> None:
    safe_success = {
        "success": True,
        "critical": False,
        "abstained": False,
        "strict_replay": True,
        "quality_accepted": True,
    }
    validation = (safe_success, {**safe_success, "success": False})
    holdout = ({**safe_success, "success": False},)

    decision, reasons = evaluate_g1_failure_curriculum_gate(validation, holdout)

    assert decision == "SIM_CANDIDATE"
    assert not reasons
    rejected, rejected_reasons = evaluate_g1_failure_curriculum_gate(
        ({**safe_success, "critical": True},),
        holdout,
    )
    assert rejected == "REJECTED"
    assert "new_critical_failure" in rejected_reasons


def test_case_rejects_partition_purpose_mismatch() -> None:
    case = build_g1_failure_curriculum()[0]
    with pytest.raises(ValueError, match="partition disagree"):
        type(case)("bad", case.scenario, "holdout")

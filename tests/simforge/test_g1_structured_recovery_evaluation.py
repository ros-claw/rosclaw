from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.simforge.g1_structured_recovery_evaluation import (
    G1StructuredRecoveryCandidate,
    G1StructuredRecoveryCase,
    G1StructuredRecoveryEvidence,
    _evaluate_capture_step_gate,
    run_g1_structured_recovery_evaluation,
    structured_recovery_specs,
)


def _case(*, passed: bool = True) -> G1StructuredRecoveryCase:
    spec = structured_recovery_specs()[0]
    gate = {"passed": passed}
    return G1StructuredRecoveryCase(
        spec=spec,
        parent_result={"passed": True},
        candidate_result={"passed": passed},
        parent_quality={},
        candidate_quality={},
        naturalness_gate=gate,
        capture_step_gate=gate,
        absolute_gate=gate,
        parent_strict_replay=passed,
        candidate_strict_replay=passed,
        parent_trajectory_hash="sha256:" + "1" * 64,
        candidate_trajectory_hash="sha256:" + "2" * 64,
    )


def test_structured_recovery_partition_is_frozen_and_disjoint() -> None:
    specs = structured_recovery_specs()

    assert len(specs) == 8
    assert len({spec.case_id for spec in specs}) == 8
    assert [spec.partition for spec in specs].count("development") == 3
    assert [spec.partition for spec in specs].count("validation") == 1
    assert [spec.partition for spec in specs].count("reserved") == 1
    assert [spec.partition for spec in specs].count("generalization") == 3


def test_structured_recovery_candidate_is_sim_only_and_hashed() -> None:
    candidate = G1StructuredRecoveryCandidate()

    assert candidate.candidate_hash.startswith("sha256:")
    assert len(candidate.candidate_hash) == 71
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1StructuredRecoveryCandidate(activation_ceiling="HARDWARE")
    with pytest.raises(ValueError, match="handoff frame"):
        G1StructuredRecoveryCandidate(post_policy_frame=269)


def test_structured_recovery_evidence_never_authorizes_activation() -> None:
    candidate = G1StructuredRecoveryCandidate()
    cases = tuple(_case() for _ in range(8))
    evidence = G1StructuredRecoveryEvidence(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        environment_hash="sha256:" + "3" * 64,
        implementation_hash="sha256:" + "4" * 64,
        request_hash="sha256:" + "5" * 64,
        cases=cases,
        status="SIM_GATE_PASS",
    )

    assert evidence.passed
    value = evidence.to_dict()
    assert value["promotion_authorized"] is False
    assert value["activation_authorized"] is False
    assert value["hardware_authorized"] is False
    assert value["claims"]["holdout_used_for_selection"] is False


def test_structured_recovery_evidence_fails_closed() -> None:
    candidate = G1StructuredRecoveryCandidate()
    evidence = G1StructuredRecoveryEvidence(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        environment_hash="sha256:" + "3" * 64,
        implementation_hash="sha256:" + "4" * 64,
        request_hash="sha256:" + "5" * 64,
        cases=tuple([_case() for _ in range(7)] + [_case(passed=False)]),
        status="REJECTED_BY_SIM_GATE",
    )

    assert not evidence.passed


def test_structured_recovery_output_must_be_outside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_structured_recovery_evaluation(
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )


def test_capture_step_gate_allows_one_bounded_forward_step() -> None:
    parent = {
        "post_contact_pelvis_path_length_m": 1.0,
        "post_contact_backward_reversal_m": 0.5,
        "post_contact_joint_jerk_rms_rad_s3": 500.0,
        "post_contact_leg_joint_jerk_rms_rad_s3": 600.0,
        "post_contact_waist_joint_jerk_rms_rad_s3": 400.0,
        "post_contact_arm_joint_jerk_rms_rad_s3": 300.0,
    }
    candidate = {
        **{key: value * 0.4 for key, value in parent.items()},
        "post_contact_pelvis_displacement_m": 0.25,
        "post_contact_forward_peak_advance_m": 0.26,
        "post_contact_lateral_peak_return_m": 0.05,
        "tail_wobble_index": 0.005,
        "terminal_bilateral_support": True,
    }
    result = {
        "success": True,
        "goal_crossed": True,
        "ball_speed_mps": 10.0,
        "target_error_m": 0.2,
        "post_kick_fall": False,
        "joint_limit_violation": False,
        "torque_limit_violation": False,
        "actuator_saturation": False,
        "support_foot_slip_m": 0.03,
    }

    gate = _evaluate_capture_step_gate(
        parent_quality=parent,
        candidate_quality=candidate,
        parent_result=result,
        candidate_result=result,
        parent_strict_replay=True,
        candidate_strict_replay=True,
    )

    assert gate["passed"] is True
    candidate["post_contact_pelvis_displacement_m"] = 0.31
    rejected = _evaluate_capture_step_gate(
        parent_quality=parent,
        candidate_quality=candidate,
        parent_result=result,
        candidate_result=result,
        parent_strict_replay=True,
        candidate_strict_replay=True,
    )
    assert rejected["passed"] is False
    assert "pelvis_displacement_m_above_gate" in rejected["reasons"]

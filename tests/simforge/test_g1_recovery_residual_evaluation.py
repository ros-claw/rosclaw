from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.growth.learners import IQLResidualGuardConfig
from rosclaw.simforge.g1_recovery_residual_evaluation import (
    G1ResidualRecoveryCandidate,
    _evaluate_non_regression_gate,
    run_g1_residual_recovery_evaluation,
)


def test_residual_candidate_is_content_addressed_and_sim_only() -> None:
    candidate = G1ResidualRecoveryCandidate(
        actor_candidate_hash="sha256:" + "1" * 64,
        actor_candidate_file_hash="sha256:" + "2" * 64,
    )

    assert candidate.candidate_hash.startswith("sha256:")
    assert candidate.activation_ceiling == "SIM_ONLY"
    with pytest.raises(ValueError, match="SIM_ONLY"):
        G1ResidualRecoveryCandidate(
            actor_candidate_hash="sha256:" + "1" * 64,
            actor_candidate_file_hash="sha256:" + "2" * 64,
            activation_ceiling="REAL",
        )


def test_residual_non_regression_gate_accepts_bounded_contribution() -> None:
    quality = {
        "post_contact_pelvis_path_length_m": 0.35,
        "post_contact_joint_jerk_rms_rad_s3": 110.0,
        "post_contact_backward_reversal_m": 0.002,
        "settling_time_sec": 1.2,
    }
    result = {
        "passed": True,
        "shooter_learned_torque_fraction": 0.90,
        "shooter_learned_torque_fallback_fraction": 0.05,
        "shooter_learned_torque_mean_confidence": 0.80,
        "shooter_learned_torque_peak_residual_nm": 0.09,
        "shooter_post_kick_fall": False,
    }
    contract = {
        "success": True,
        "goal_crossed": True,
        "ball_speed_mps": 10.0,
        "target_error_m": 0.2,
    }

    gate = _evaluate_non_regression_gate(
        parent_quality=quality,
        candidate_quality={**quality, "post_contact_joint_jerk_rms_rad_s3": 105.0},
        parent_result={"passed": True},
        candidate_result=result,
        parent_contract=contract,
        candidate_contract=contract,
        parent_strict_replay=True,
        candidate_strict_replay=True,
        residual_guard=IQLResidualGuardConfig(),
    )

    assert gate["passed"] is True
    assert gate["peak_residual_nm"] == 0.09


def test_residual_non_regression_gate_rejects_token_participation() -> None:
    quality = {
        "post_contact_pelvis_path_length_m": 0.35,
        "post_contact_joint_jerk_rms_rad_s3": 110.0,
        "post_contact_backward_reversal_m": 0.002,
        "settling_time_sec": 1.2,
    }
    contract = {
        "success": True,
        "goal_crossed": True,
        "ball_speed_mps": 10.0,
        "target_error_m": 0.2,
    }

    gate = _evaluate_non_regression_gate(
        parent_quality=quality,
        candidate_quality=quality,
        parent_result={"passed": True},
        candidate_result={
            "passed": True,
            "shooter_learned_torque_fraction": 0.01,
            "shooter_learned_torque_fallback_fraction": 0.99,
            "shooter_learned_torque_mean_confidence": 0.0,
            "shooter_learned_torque_peak_residual_nm": 0.0,
            "shooter_post_kick_fall": False,
        },
        parent_contract=contract,
        candidate_contract=contract,
        parent_strict_replay=True,
        candidate_strict_replay=True,
        residual_guard=IQLResidualGuardConfig(),
    )

    assert gate["passed"] is False
    assert "learned_residual_participation_below_50_percent" in gate["reasons"]
    assert "support_envelope_fallback_above_50_percent" in gate["reasons"]


def test_residual_evidence_must_be_outside_checkout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        run_g1_residual_recovery_evaluation(
            actor_candidate_path=tmp_path / "candidate.json",
            asset_root=tmp_path / "assets",
            output_dir=tmp_path / "evidence",
            source_checkout=tmp_path,
        )

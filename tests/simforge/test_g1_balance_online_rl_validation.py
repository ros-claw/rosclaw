from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw.simforge.g1_balance_online_rl_validation import (
    _balance_action_indices,
    _balance_update_gate,
    _critic_convergence,
    _external_root,
    _fresh_balance_validation_scenarios,
    _matched_evidence_equal,
    _quarantine_actor_replay,
    _value_convergence,
)
from rosclaw.simforge.g1_neural_torque import G1_NEURAL_TORQUE_OBSERVATIONS
from rosclaw.simforge.g1_neural_torque_learning import G1NeuralTorqueReplay
from rosclaw.simforge.g1_neural_torque_validation import (
    G1NeuralTorqueRolloutEvidence,
)
from rosclaw.simforge.models import Partition
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_DDS_JOINT_NAMES


def _evidence(**changes: object) -> G1NeuralTorqueRolloutEvidence:
    value = G1NeuralTorqueRolloutEvidence(
        stage="development",
        partition="DEVELOPMENT",
        scenario_id="scenario-0",
        scenario_commitment="sha256:" + "1" * 64,
        status="SUCCESS",
        success=True,
        contact=True,
        target_error_m=0.1,
        ball_speed_mps=4.0,
        support_slip_m=0.01,
        com_margin_min_m=0.03,
        torso_roll_peak_rad=0.20,
        torso_pitch_peak_rad=0.20,
        fall=False,
        joint_violation=False,
        torque_violation=False,
        finite=True,
        score=10.0,
        direct_torque_inferences=100,
        learned_output_fraction=0.8,
        fallback_fraction=0.2,
        artifact_hash="sha256:" + "2" * 64,
        activation_ceiling="SIM_ONLY",
        hardware_authorized=False,
        strict_replay=False,
        trace_hash="sha256:" + "3" * 64,
    )
    return replace(value, **changes)


def test_balance_validation_requires_external_evidence_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside the source checkout"):
        _external_root(tmp_path / "evidence", tmp_path)
    assert (
        _external_root(tmp_path.parent / "evidence", tmp_path)
        == (tmp_path.parent / "evidence").resolve()
    )


def test_balance_gate_rejects_hidden_com_and_ball_speed_regressions() -> None:
    parent = (_evidence(),)
    candidate = (
        _evidence(
            com_margin_min_m=0.02,
            ball_speed_mps=3.0,
        ),
    )

    accepted, reasons = _balance_update_gate(parent=parent, candidate=candidate)

    assert not accepted
    assert "matched_com_margin_regression" in reasons
    assert "matched_ball_speed_regression_gt_5pct" in reasons


def test_balance_gate_accepts_matched_stability_improvement() -> None:
    parent = (_evidence(),)
    candidate = (
        _evidence(
            support_slip_m=0.008,
            torso_roll_peak_rad=0.18,
            torso_pitch_peak_rad=0.18,
            com_margin_min_m=0.04,
        ),
    )

    accepted, reasons = _balance_update_gate(parent=parent, candidate=candidate)

    assert accepted
    assert not reasons


def test_balance_validation_round_is_fresh_and_reproducible() -> None:
    round_two = _fresh_balance_validation_scenarios(2)
    repeated = _fresh_balance_validation_scenarios(2)
    round_three = _fresh_balance_validation_scenarios(3)

    assert round_two == repeated
    assert {item.seed_commitment for item in round_two}.isdisjoint(
        item.seed_commitment for item in round_three
    )
    assert all(item.partition is Partition.VALIDATION for item in round_two)


def test_exact_parent_match_ignores_only_identity_fields() -> None:
    parent = (_evidence(stage="parent", artifact_hash="sha256:" + "4" * 64),)
    clone = (_evidence(stage="clone", artifact_hash="sha256:" + "5" * 64),)

    assert _matched_evidence_equal(parent, clone)
    assert not _matched_evidence_equal(parent, (_evidence(score=9.9),))


def test_balance_action_subspace_excludes_kick_leg() -> None:
    selected = {G1_DDS_JOINT_NAMES[index] for index in _balance_action_indices()}

    assert any(name.startswith("left_") for name in selected)
    assert "waist_roll_joint" in selected
    assert not any(name.startswith("right_hip") for name in selected)
    assert not any(name.startswith("right_knee") for name in selected)
    assert not any(name.startswith("right_ankle") for name in selected)


def test_critic_convergence_requires_all_value_heads_to_improve() -> None:
    converged = tuple(
        {
            "reward_critic_loss": 1.0 / (index + 1),
            "fall_critic_loss": 0.5 / (index + 1),
            "constraint_critic_loss": 0.8 / (index + 1),
        }
        for index in range(12)
    )
    stalled = tuple({**value, "constraint_critic_loss": 0.8} for value in converged)

    assert _critic_convergence(converged)["converged"] is True
    assert _critic_convergence(stalled)["converged"] is False
    assert _critic_convergence(converged[:7])["converged"] is False


def test_awr_value_convergence_and_non_elite_quarantine() -> None:
    converged = tuple({"value_loss": 1.0 / (index + 1)} for index in range(12))
    assert _value_convergence(converged)["converged"] is True
    assert _value_convergence(converged[:7])["converged"] is False

    count = 3
    observation_dim = len(G1_NEURAL_TORQUE_OBSERVATIONS)
    replay = G1NeuralTorqueReplay(
        observations=np.zeros((count, 2, observation_dim), dtype=np.float32),
        actions=np.zeros((count, 29), dtype=np.float32),
        next_observations=np.zeros((count, 2, observation_dim), dtype=np.float32),
        rewards=np.zeros((count, 1), dtype=np.float32),
        fall_costs=np.zeros((count, 1), dtype=np.float32),
        constraint_costs=np.zeros((count, 1), dtype=np.float32),
        terminals=np.asarray([[0.0], [0.0], [1.0]], dtype=np.float32),
        parent_actions=np.zeros((count, 29), dtype=np.float32),
        partitions=np.asarray([0, 2, 0], dtype=np.int8),
        policy_lags=np.zeros(count, dtype=np.int64),
    )

    quarantined = _quarantine_actor_replay(replay)

    assert np.all(quarantined.partitions == 2)
    assert np.array_equal(quarantined.rewards, replay.rewards)

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_active_sampler import (
    derive_g1_ballistic_contact_active_sample,
)
from rosclaw.growth.ballistic_contact_actor_critic import (
    derive_g1_ballistic_contact_actor_critic,
)
from rosclaw.growth.ballistic_contact_candidate_evaluation import (
    evaluate_g1_ballistic_contact_candidate,
)
from rosclaw.growth.cli import dispatch_growth_argv


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(root: Path, index: int, ankle_action: float) -> Path:
    trajectory = root / f"trajectory-{index}.npz"
    np.savez_compressed(trajectory, time=np.arange(12, dtype=np.float64) * 0.02)
    evidence = root / f"evidence-{index}.json"
    error = 1.0 - 2.0 * ankle_action
    evidence.write_text(
        json.dumps(
            {
                "body_hash": "sha256:" + "1" * 64,
                "implementation_hash": "sha256:" + "2" * 64,
                "strict_replay": True,
                "trajectory_path": str(trajectory.resolve()),
                "trajectory_hash": _hash(trajectory),
                "approach_strike_candidate_hash": "sha256:" + "3" * 64,
                "flow_config": {
                    "schema_version": "flow.v1",
                    "ballistic_contact_residual_rad": [0, 0, 0, 0, ankle_action, 0],
                    "football_motion_prior_hash": "sha256:" + "4" * 64,
                },
                "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
                "runup_config": {"start_x_m": -3.4},
                "sonic_runup_config": {"planner_seed": 0},
                "result": {
                    "finite_state": True,
                    "post_kick_fall": False,
                    "joint_limit_violation": False,
                    "torque_limit_violation": False,
                    "perceptual_continuity_passed": True,
                    "goal_plane_target_error_m": error,
                    "ball_launch_velocity_xyz_mps": [9.0, 1.0, 4.0 + ankle_action],
                    "kick_contact_height_relative_ball_center_m": -0.05,
                    "actuator_saturation_steps": 2,
                    "actuator_peak_demand_ratio": 1.02,
                    "post_contact_peak_joint_velocity_rms_rad_s": 4.0,
                    "post_contact_settling_time_sec": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence


def test_ballistic_actor_critic_uses_strict_replay_and_proposes_new_action(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )

    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.best_observed_action_rad[4] == pytest.approx(0.16)
    assert candidate.proposed_action_rad[4] > candidate.best_observed_action_rad[4]
    assert candidate.proposed_action_rad[4] <= 0.25
    assert candidate.active_action_dimensions == (4,)
    assert candidate.frozen_action_dimensions == (0, 1, 2, 3, 5)
    assert candidate.action_unique_counts == (1, 1, 1, 1, 8, 1)
    assert candidate.proposed_action_rad[:4] == (0.0, 0.0, 0.0, 0.0)
    assert candidate.proposed_action_rad[5] == 0.0
    assert candidate.proposed_action_dimensions == (4,)
    assert candidate.predicted_improvement_over_anchor > 0.0
    assert candidate.sim_replay_recommended
    assert candidate.replay_anchor_count == 8
    assert not candidate.promotion_authorized
    assert json.loads((tmp_path / "candidate.json").read_text())["candidate_hash"]


def test_active_sampler_fills_one_local_gap_without_activation(tmp_path: Path) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    candidate_path = tmp_path / "candidate.json"
    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )

    sample = derive_g1_ballistic_contact_active_sample(
        actor_critic_path=candidate_path,
        output_path=tmp_path / "sample.json",
        source_checkout=tmp_path / "checkout",
        maximum_step_rad=0.03,
    )

    assert sample.source_candidate_hash == candidate.candidate_hash
    assert sample.proposed_action_dimension == 4
    assert sample.anchor_action_rad[4] == pytest.approx(0.16)
    assert 0.13 < sample.proposed_action_rad[4] < 0.19
    assert sample.proposed_action_rad[:4] == sample.anchor_action_rad[:4]
    assert sample.proposed_action_rad[5] == sample.anchor_action_rad[5]
    assert sample.sim_replay_required
    assert not sample.promotion_authorized
    assert not sample.hardware_authorized
    stored = json.loads((tmp_path / "sample.json").read_text())
    assert stored["sample_hash"] == sample.sample_hash


def test_active_sampler_rejects_tampered_actor_critic(tmp_path: Path) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    candidate_path = tmp_path / "candidate.json"
    derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )
    payload = json.loads(candidate_path.read_text())
    payload["best_observed_action_rad"][4] = 0.15
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        derive_g1_ballistic_contact_active_sample(
            actor_critic_path=candidate_path,
            output_path=tmp_path / "sample.json",
            source_checkout=tmp_path / "checkout",
        )


def test_candidate_evaluation_requires_measured_safe_improvement(tmp_path: Path) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    candidate_path = tmp_path / "candidate.json"
    actor = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )
    replay = _probe(tmp_path, 8, actor.proposed_action_rad[4])

    evaluation = evaluate_g1_ballistic_contact_candidate(
        actor_critic_path=candidate_path,
        anchor_evidence_path=paths[-1],
        candidate_evidence_path=replay,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert evaluation.accepted
    assert evaluation.decision == "ACCEPTED"
    assert evaluation.measured_error_improvement_m >= 0.005
    assert evaluation.stability_preserved
    assert evaluation.hard_safe
    assert not evaluation.promotion_authorized


def test_candidate_evaluation_rejects_prediction_without_measured_gain(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    candidate_path = tmp_path / "candidate.json"
    actor = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )
    replay = _probe(tmp_path, 8, actor.proposed_action_rad[4])
    payload = json.loads(replay.read_text())
    payload["result"]["goal_plane_target_error_m"] = 2.0
    replay.write_text(json.dumps(payload), encoding="utf-8")

    evaluation = evaluate_g1_ballistic_contact_candidate(
        actor_critic_path=candidate_path,
        anchor_evidence_path=paths[-1],
        candidate_evidence_path=replay,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert not evaluation.accepted
    assert evaluation.decision == "REJECTED"
    assert not evaluation.precision_improved


def test_candidate_evaluation_fails_closed_without_saturation_metric(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    candidate_path = tmp_path / "candidate.json"
    actor = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )
    replay = _probe(tmp_path, 8, actor.proposed_action_rad[4])
    payload = json.loads(replay.read_text())
    payload["result"].pop("actuator_saturation_steps")
    replay.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="actuator_saturation_steps"):
        evaluate_g1_ballistic_contact_candidate(
            actor_critic_path=candidate_path,
            anchor_evidence_path=paths[-1],
            candidate_evidence_path=replay,
            output_path=tmp_path / "evaluation.json",
            source_checkout=tmp_path / "checkout",
        )


def test_ballistic_actor_critic_rejects_context_mixing(tmp_path: Path) -> None:
    paths = [
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    ]
    mixed = json.loads(paths[-1].read_text(encoding="utf-8"))
    mixed["goal_spec"]["target_z_m"] = 0.65
    paths[-1].write_text(json.dumps(mixed), encoding="utf-8")

    with pytest.raises(ValueError, match="contexts disagree"):
        derive_g1_ballistic_contact_actor_critic(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "candidate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_ballistic_actor_critic_freezes_correlated_low_support_axes(
    tmp_path: Path,
) -> None:
    paths = []
    ankle_actions = (-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16)
    for index, ankle_action in enumerate(ankle_actions):
        path = _probe(tmp_path, index, ankle_action)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        correlated = -0.06 if index >= 4 else 0.0
        evidence["flow_config"]["ballistic_contact_residual_rad"][1] = correlated
        evidence["flow_config"]["ballistic_contact_residual_rad"][2] = correlated
        path.write_text(json.dumps(evidence), encoding="utf-8")
        paths.append(path)

    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=tuple(paths),
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.active_action_dimensions == (4,)
    assert candidate.proposed_action_rad[1] == candidate.best_observed_action_rad[1]
    assert candidate.proposed_action_rad[2] == candidate.best_observed_action_rad[2]


def test_ballistic_actor_critic_changes_only_one_independently_probed_axis(
    tmp_path: Path,
) -> None:
    paths = []
    ankle_actions = (-0.16, -0.08, 0.0, 0.08)
    knee_actions = (-0.15, -0.05, 0.05, 0.15)
    for index, ankle_action in enumerate(ankle_actions):
        paths.append(_probe(tmp_path, index, ankle_action))
    for offset, knee_action in enumerate(knee_actions, start=4):
        path = _probe(tmp_path, offset, 0.02)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["flow_config"]["ballistic_contact_residual_rad"][3] = knee_action
        evidence["result"]["goal_plane_target_error_m"] = 0.9 - knee_action
        path.write_text(json.dumps(evidence), encoding="utf-8")
        paths.append(path)

    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=tuple(paths),
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    changed = tuple(
        index
        for index, (best, proposed) in enumerate(
            zip(
                candidate.best_observed_action_rad,
                candidate.proposed_action_rad,
                strict=True,
            )
        )
        if proposed != pytest.approx(best)
    )
    assert candidate.active_action_dimensions == (3, 4)
    assert changed == candidate.proposed_action_dimensions
    assert len(changed) == 1


def test_ballistic_actor_critic_abstains_when_cross_validation_is_poor(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _probe(tmp_path, index, action)
        for index, action in enumerate((-0.16, -0.10, -0.04, 0.0, 0.04, 0.08, 0.12, 0.16))
    )
    for index, path in enumerate(paths):
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["result"]["goal_plane_target_error_m"] = float(index % 2) * 2.0
        path.write_text(json.dumps(evidence), encoding="utf-8")

    candidate = derive_g1_ballistic_contact_actor_critic(
        evidence_paths=paths,
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.critic_leave_one_out_rmse > 0.15
    assert candidate.maximum_critic_leave_one_out_rmse == 0.15
    assert not candidate.sim_replay_recommended
    assert (
        dispatch_growth_argv(
            [
                "growth",
                "ballistic-contact-actor-critic",
                *[item for path in paths for item in ("--evidence-json", str(path))],
                "--output",
                str(tmp_path / "cli-candidate.json"),
                "--source-checkout",
                str(tmp_path / "checkout"),
            ]
        )
        == 3
    )

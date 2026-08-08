from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_actor_critic import (
    derive_g1_ballistic_contact_actor_critic,
)


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
    assert candidate.replay_anchor_count == 8
    assert not candidate.promotion_authorized
    assert json.loads((tmp_path / "candidate.json").read_text())["candidate_hash"]


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

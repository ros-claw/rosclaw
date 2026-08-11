from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_candidate_evaluation import (
    evaluate_g1_ballistic_contact_candidate,
)
from rosclaw.growth.ballistic_contact_context_holdout_evaluation import (
    evaluate_g1_ballistic_contact_context_holdout,
)
from rosclaw.growth.ballistic_contact_coupled_actor_critic import (
    derive_g1_ballistic_contact_coupled_actor_critic,
)
from rosclaw.growth.cli import dispatch_growth_argv


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(root: Path, index: int, hip_yaw: float, ankle_pitch: float) -> Path:
    trajectory = root / f"trajectory-{index}.npz"
    np.savez_compressed(trajectory, time=np.arange(12, dtype=np.float64) * 0.02)
    error = (
        0.5
        + 100.0 * (hip_yaw + 0.01) ** 2
        + 80.0 * (ankle_pitch - 0.19) ** 2
        + 20.0 * (hip_yaw + 0.01) * (ankle_pitch - 0.19)
    )
    evidence = root / f"evidence-{index}.json"
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
                    "ballistic_contact_residual_rad": [
                        -0.02,
                        -0.01,
                        hip_yaw,
                        -0.06,
                        ankle_pitch,
                        0.03,
                    ],
                    "football_motion_prior_hash": "sha256:" + "4" * 64,
                },
                "goal_spec": {"target_y_m": 3.4, "target_z_m": 2.18},
                "runup_config": {"start_x_m": -3.4},
                "sonic_runup_config": {"planner_seed": 0},
                "result": {
                    "finite_state": True,
                    "post_kick_fall": False,
                    "joint_limit_violation": False,
                    "torque_limit_violation": False,
                    "perceptual_continuity_passed": True,
                    "goal_plane_target_error_m": error,
                    "ball_launch_velocity_xyz_mps": [9.0, 2.0, 4.5],
                    "kick_contact_height_relative_ball_center_m": -0.03,
                    "actuator_saturation_steps": 1,
                    "actuator_saturation": False,
                    "actuator_peak_demand_ratio": 1.01,
                    "post_contact_peak_joint_velocity_rms_rad_s": 4.0,
                    "post_contact_settling_time_sec": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence


def _grid(tmp_path: Path) -> tuple[Path, ...]:
    values: list[Path] = []
    for index, (yaw_index, ankle_index) in enumerate(np.ndindex(3, 3)):
        values.append(
            _probe(
                tmp_path,
                index,
                (-0.02, 0.0, 0.02)[yaw_index],
                (0.18, 0.20, 0.22)[ankle_index],
            )
        )
    return tuple(values)


def test_coupled_critic_requires_full_grid_and_proposes_two_axis_replay(
    tmp_path: Path,
) -> None:
    paths = _grid(tmp_path)
    candidate = derive_g1_ballistic_contact_coupled_actor_critic(
        evidence_paths=paths,
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.coupled_action_dimensions == (2, 4)
    assert candidate.frozen_action_dimensions == (0, 1, 3, 5)
    assert candidate.action_grid_values_rad == (
        (-0.02, 0.0, 0.02),
        (0.18, 0.2, 0.22),
    )
    assert candidate.proposed_action_dimensions == (2, 4)
    assert candidate.proposed_action_rad[2] == pytest.approx(-0.01)
    assert candidate.proposed_action_rad[4] == pytest.approx(0.19)
    assert candidate.proposal_support_mode == "INTERPOLATION"
    assert candidate.maximum_per_axis_extrapolation_rad == 0.0
    assert candidate.critic_leave_one_out_rmse < 0.15
    assert candidate.predicted_improvement_over_anchor >= 0.005
    assert candidate.sim_replay_recommended
    assert not candidate.direct_torque_output
    assert not candidate.online_hot_swap_allowed


def test_coupled_candidate_reenters_strict_physics_gate(tmp_path: Path) -> None:
    paths = _grid(tmp_path)
    candidate_path = tmp_path / "candidate.json"
    candidate = derive_g1_ballistic_contact_coupled_actor_critic(
        evidence_paths=paths,
        output_path=candidate_path,
        source_checkout=tmp_path / "checkout",
    )
    anchor = next(
        path
        for path in paths
        if json.loads(path.read_text())["flow_config"]["ballistic_contact_residual_rad"]
        == list(candidate.best_observed_action_rad)
    )
    replay = _probe(
        tmp_path,
        9,
        candidate.proposed_action_rad[2],
        candidate.proposed_action_rad[4],
    )
    replay_payload = json.loads(replay.read_text())
    replay_payload["flow_config"]["ballistic_contact_residual_rad"][4] = round(
        candidate.proposed_action_rad[4], 2
    )
    replay.write_text(json.dumps(replay_payload), encoding="utf-8")

    evaluation = evaluate_g1_ballistic_contact_candidate(
        actor_critic_path=candidate_path,
        anchor_evidence_path=anchor,
        candidate_evidence_path=replay,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert evaluation.accepted
    assert evaluation.precision_improved
    assert evaluation.stability_preserved
    assert evaluation.hard_safe


def test_coupled_critic_rejects_incomplete_cartesian_support(tmp_path: Path) -> None:
    paths = list(_grid(tmp_path))
    payload = json.loads(paths[-1].read_text())
    payload["flow_config"]["ballistic_contact_residual_rad"][4] = 0.23
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="complete Cartesian"):
        derive_g1_ballistic_contact_coupled_actor_critic(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "candidate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_coupled_critic_rejects_one_unsafe_grid_point(tmp_path: Path) -> None:
    paths = _grid(tmp_path)
    payload = json.loads(paths[-1].read_text())
    payload["result"]["post_kick_fall"] = True
    paths[-1].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="hard-safe continuous grid"):
        derive_g1_ballistic_contact_coupled_actor_critic(
            evidence_paths=paths,
            output_path=tmp_path / "candidate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_coupled_critic_cli_reports_replay_candidate(tmp_path: Path) -> None:
    paths = _grid(tmp_path)
    output = tmp_path / "candidate.json"
    argv = ["growth", "ballistic-contact-coupled-actor-critic"]
    for path in paths:
        argv.extend(("--evidence-json", str(path)))
    argv.extend(("--output", str(output), "--source-checkout", str(tmp_path / "checkout")))

    assert dispatch_growth_argv(argv) == 0
    assert json.loads(output.read_text())["sim_replay_recommended"] is True


def test_coupled_critic_can_request_bounded_next_ring_simulation(tmp_path: Path) -> None:
    paths = _grid(tmp_path)
    for path in paths:
        payload = json.loads(path.read_text())
        action = payload["flow_config"]["ballistic_contact_residual_rad"]
        payload["result"]["goal_plane_target_error_m"] = 1.0 + action[2] - action[4]
        path.write_text(json.dumps(payload), encoding="utf-8")

    candidate = derive_g1_ballistic_contact_coupled_actor_critic(
        evidence_paths=paths,
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.proposal_support_mode == "NEXT_RING_HALF_STEP"
    assert candidate.proposed_action_rad[2] == pytest.approx(-0.03)
    assert candidate.proposed_action_rad[4] == pytest.approx(0.23)
    assert candidate.maximum_per_axis_extrapolation_rad == pytest.approx(0.01)
    assert candidate.sim_replay_recommended
    assert not candidate.promotion_authorized
    assert not candidate.hardware_authorized


def _context_pairs(
    tmp_path: Path, *, regress: bool = False
) -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    grid_root = tmp_path / "grid"
    grid_root.mkdir()
    grid = _grid(grid_root)
    actor_path = tmp_path / "candidate.json"
    actor = derive_g1_ballistic_contact_coupled_actor_critic(
        evidence_paths=grid,
        output_path=actor_path,
        source_checkout=tmp_path / "checkout",
    )
    anchors: list[Path] = []
    candidates: list[Path] = []
    for index, seed in enumerate((4, 7, 21)):
        root = tmp_path / f"seed-{seed}"
        root.mkdir()
        anchor = _probe(
            root,
            0,
            actor.best_observed_action_rad[2],
            actor.best_observed_action_rad[4],
        )
        candidate = _probe(
            root,
            1,
            actor.proposed_action_rad[2],
            actor.proposed_action_rad[4],
        )
        anchor_payload = json.loads(anchor.read_text())
        candidate_payload = json.loads(candidate.read_text())
        anchor_payload["sonic_runup_config"]["planner_seed"] = seed
        candidate_payload["sonic_runup_config"]["planner_seed"] = seed
        anchor_error = 0.60 + 0.02 * index
        improvement = (-0.06, 0.10, -0.002)[index] if regress else 0.02
        anchor_payload["result"]["goal_plane_target_error_m"] = anchor_error
        candidate_payload["result"]["goal_plane_target_error_m"] = anchor_error - improvement
        if regress and seed == 21:
            candidate_payload["result"]["perceptual_continuity_passed"] = False
        anchor.write_text(json.dumps(anchor_payload), encoding="utf-8")
        candidate.write_text(json.dumps(candidate_payload), encoding="utf-8")
        anchors.append(anchor)
        candidates.append(candidate)
    return actor_path, tuple(anchors), tuple(candidates)


def test_context_holdout_accepts_only_consistent_safe_improvement(tmp_path: Path) -> None:
    actor, anchors, candidates = _context_pairs(tmp_path)

    evaluation = evaluate_g1_ballistic_contact_context_holdout(
        actor_critic_path=actor,
        anchor_evidence_paths=anchors,
        candidate_evidence_paths=candidates,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert evaluation.accepted
    assert evaluation.improved_context_count == 3
    assert evaluation.required_improved_context_count == 2
    assert evaluation.worst_context_improvement_m == pytest.approx(0.02)
    assert evaluation.lower_tail_cvar_improvement_m == pytest.approx(0.02)
    assert evaluation.all_contexts_hard_safe


def test_context_holdout_rejects_mean_gain_with_tail_regression(tmp_path: Path) -> None:
    actor, anchors, candidates = _context_pairs(tmp_path, regress=True)

    evaluation = evaluate_g1_ballistic_contact_context_holdout(
        actor_critic_path=actor,
        anchor_evidence_paths=anchors,
        candidate_evidence_paths=candidates,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert evaluation.mean_error_improvement_m > 0.005
    assert evaluation.worst_context_improvement_m == pytest.approx(-0.06)
    assert evaluation.lower_tail_cvar_improvement_m == pytest.approx(-0.06)
    assert not evaluation.all_contexts_hard_safe
    assert not evaluation.accepted
    assert evaluation.decision == "REJECTED"

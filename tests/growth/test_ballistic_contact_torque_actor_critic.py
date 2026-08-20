from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.ballistic_contact_torque_actor_critic import (
    derive_g1_ballistic_contact_torque_actor_critic,
)
from rosclaw.growth.cli import dispatch_growth_argv


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(root: Path, index: int, ankle_pitch_nm: float) -> Path:
    trajectory = root / f"trajectory-{index}.npz"
    np.savez_compressed(trajectory, time=np.arange(12, dtype=np.float64) * 0.02)
    qualified = ankle_pitch_nm <= 3.0
    error = 0.34 + 0.035 * (ankle_pitch_nm - 2.55) ** 2
    if not qualified:
        error = 1.20
    crossing_height = 1.05 if qualified else 0.20
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
                "claims": {"bounded_sim_only_ballistic_contact_torque_residual": True},
                "flow_config": {
                    "schema_version": "flow.v1",
                    "ballistic_contact_residual_rad": [0, 0, 0, 0, 0.25, 0],
                    "ballistic_contact_torque_residual_nm": [
                        0,
                        0,
                        0,
                        0,
                        ankle_pitch_nm,
                        0,
                    ],
                    "ballistic_contact_torque_policy_frame": 256,
                    "ballistic_contact_torque_lead_duration_sec": 0.08,
                    "ballistic_contact_torque_trail_duration_sec": 0.065,
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
                    "kick_contact_observed": True,
                    "goal_crossed": True,
                    "goal_plane_target_error_m": error,
                    "goal_crossing_xyz_m": [6.0, 0.9, crossing_height],
                    "ball_launch_velocity_xyz_mps": [9.0, 1.5, 5.3],
                    "actuator_saturation_steps": 20,
                    "actuator_peak_demand_ratio": 1.15,
                },
            }
        ),
        encoding="utf-8",
    )
    return evidence


def _dataset(root: Path) -> tuple[Path, ...]:
    actions = (0.5, 1.0, 1.5, 2.0, 2.4, 2.8, 3.5, 4.0)
    return tuple(_probe(root, index, action) for index, action in enumerate(actions))


def test_contact_torque_actor_learns_inside_qualified_island(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    candidate = derive_g1_ballistic_contact_torque_actor_critic(
        evidence_paths=paths,
        output_path=tmp_path / "candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.qualified_probe_count == 6
    assert candidate.rejected_probe_count == 2
    assert candidate.active_action_dimensions == (4,)
    assert candidate.frozen_action_dimensions == (0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11)
    assert candidate.best_observed_action_nm[4] == pytest.approx(2.4)
    assert candidate.proposed_action_nm[4] != pytest.approx(2.4)
    assert candidate.proposed_action_dimensions == (4,)
    assert candidate.qualified_island_margin > 0.0
    assert candidate.sim_replay_recommended
    saved = json.loads((tmp_path / "candidate.json").read_text())
    assert saved["candidate_hash"]
    assert saved["direct_torque_output"] is True
    assert saved["activation_ceiling"] == "SIM_ONLY"
    assert saved["promotion_authorized"] is False
    assert saved["hardware_authorized"] is False
    with pytest.raises(ValueError, match="must remain SIM_ONLY"):
        replace(candidate, promotion_authorized=True)


def test_contact_torque_actor_rejects_context_mixing(tmp_path: Path) -> None:
    paths = list(_dataset(tmp_path))
    mixed = json.loads(paths[-1].read_text(encoding="utf-8"))
    mixed["flow_config"]["ballistic_contact_torque_policy_frame"] = 255
    paths[-1].write_text(json.dumps(mixed), encoding="utf-8")

    with pytest.raises(ValueError, match="contexts disagree"):
        derive_g1_ballistic_contact_torque_actor_critic(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "candidate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_contact_torque_actor_requires_rejected_island_examples(
    tmp_path: Path,
) -> None:
    paths = list(_dataset(tmp_path))
    for path in paths[-2:]:
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["result"]["goal_plane_target_error_m"] = 0.4
        evidence["result"]["goal_crossing_xyz_m"] = [6.0, 0.9, 1.0]
        path.write_text(json.dumps(evidence), encoding="utf-8")

    with pytest.raises(ValueError, match="five qualified and two rejected"):
        derive_g1_ballistic_contact_torque_actor_critic(
            evidence_paths=tuple(paths),
            output_path=tmp_path / "candidate.json",
            source_checkout=tmp_path / "checkout",
        )


def test_contact_torque_actor_cli_is_replay_only(tmp_path: Path) -> None:
    paths = _dataset(tmp_path)
    code = dispatch_growth_argv(
        [
            "growth",
            "ballistic-contact-torque-actor-critic",
            *[item for path in paths for item in ("--evidence-json", str(path))],
            "--output",
            str(tmp_path / "candidate.json"),
            "--source-checkout",
            str(tmp_path / "checkout"),
        ]
    )

    assert code == 0
    candidate = json.loads((tmp_path / "candidate.json").read_text())
    assert candidate["online_hot_swap_allowed"] is False
    assert candidate["sealed_generalization_evidence"] is False


def test_contact_torque_actor_learns_counterbalance_axis(tmp_path: Path) -> None:
    actions = (-1.0, 0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0)
    paths: list[Path] = []
    for index, action in enumerate(actions):
        path = _probe(tmp_path, index, 2.0 + index * 0.01)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        evidence["flow_config"]["ballistic_contact_torque_residual_nm"] = [
            0,
            0,
            0,
            0,
            3.0,
            0,
        ]
        evidence["flow_config"]["ballistic_counterbalance_torque_residual_nm"] = [
            action,
            0,
            0,
            5.0,
            0,
            0,
        ]
        qualified = action <= 3.0
        evidence["result"]["goal_plane_target_error_m"] = (
            0.30 + 0.02 * (action - 2.7) ** 2 if qualified else 1.1
        )
        evidence["result"]["goal_crossing_xyz_m"] = [
            6.0,
            0.9,
            1.1 if qualified else 0.2,
        ]
        path.write_text(json.dumps(evidence), encoding="utf-8")
        paths.append(path)

    candidate = derive_g1_ballistic_contact_torque_actor_critic(
        evidence_paths=tuple(paths),
        output_path=tmp_path / "counter-candidate.json",
        source_checkout=tmp_path / "checkout",
    )

    assert candidate.active_action_dimensions == (6,)
    assert candidate.proposed_action_dimensions == (6,)
    assert candidate.proposed_action_nm[:6] == candidate.best_observed_action_nm[:6]
    assert candidate.proposed_action_nm[7:] == candidate.best_observed_action_nm[7:]

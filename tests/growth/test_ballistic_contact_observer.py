from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_observer import (
    derive_g1_ballistic_contact_observer,
    load_g1_ballistic_contact_observer,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root: Path, *, index: int) -> Path:
    qualified = index < 6
    directory = root / f"sample-{index}"
    directory.mkdir()
    force = 850.0 + index if qualified else 420.0 + index
    normal_z = 0.80 + 0.005 * index if qualified else 0.30 + 0.005 * index
    normal_x = math.sqrt(1.0 - normal_z * normal_z)
    foot_velocity = [2.0 + 0.1 * index, -4.0 + 0.02 * index, 0.3]
    normal = [normal_x, 0.0, normal_z]
    launch = [6.0 + foot_velocity[0], 1.0, 1.0 + 4.0 * normal_z]
    trajectory = directory / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        right_foot_linear_velocity=np.asarray(
            [[0.0, 0.0, 0.0], foot_velocity, foot_velocity], dtype=np.float64
        ),
        ball_contact_force_peak_n=np.asarray([0.0, force, 0.0]),
        ball_contact_normal=np.asarray([[0.0, 0.0, 0.0], normal, normal]),
        ball_contact_force_world=np.asarray(
            [[0.0, 0.0, 0.0], [force, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
    )
    crossing = [6.0, 1.0, 0.9 if qualified else 0.2]
    value = {
        "strict_replay": True,
        "claims": {"contact_dynamics_observed_from_physics": True},
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "2" * 64,
        "approach_strike_candidate_hash": "sha256:" + "3" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "control": "frozen",
            "ballistic_contact_residual_rad": [0.01 * index] * 6,
            "ballistic_contact_policy_frame": 250 + index,
            "post_contact_damping_scale": 2.5,
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": index,
            "speed": 1.5,
        },
        "runup_config": {"start_x_m": -3.4},
        "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
        "result": {
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "perceptual_continuity_passed": True,
            "goal_crossed": True,
            "goal_crossing_xyz_m": crossing,
            "goal_plane_target_error_m": 0.4 if qualified else 1.2,
            "kick_contact_foot_velocity_xyz_mps": foot_velocity,
            "kick_contact_normal_xyz": normal,
            "kick_contact_peak_force_n": force,
            "kick_contact_height_relative_ball_center_m": (
                0.02 if qualified else -0.08
            ),
            "ball_launch_velocity_xyz_mps": launch,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _observer(tmp_path: Path):
    paths = tuple(_evidence(tmp_path, index=index) for index in range(12))
    return derive_g1_ballistic_contact_observer(
        evidence_paths=paths,
        output_path=tmp_path / "observer.json",
        source_checkout=tmp_path / "checkout",
    )


def test_contact_observer_fits_strict_replay_and_predicts(tmp_path: Path) -> None:
    observer = _observer(tmp_path)
    loaded = load_g1_ballistic_contact_observer(tmp_path / "observer.json")

    positive = loaded.predict(loaded.samples[0].features)
    negative = loaded.predict(loaded.samples[-1].features)

    assert observer.training_ready
    assert observer.positive_count == observer.negative_count == 6
    assert observer.leave_one_out_brier_score < 0.1
    assert observer.leave_one_out_launch_rmse_mps < 0.2
    assert observer.leave_one_seed_out_brier_score < 0.1
    assert observer.leave_one_seed_out_launch_rmse_mps < 0.2
    assert positive.qualified_skill_probability > negative.qualified_skill_probability
    assert positive.launch_velocity_xyz_mps == pytest.approx(
        loaded.samples[0].launch_velocity_xyz_mps, abs=0.2
    )


def test_contact_observer_tamper_fails_closed(tmp_path: Path) -> None:
    _observer(tmp_path)
    path = tmp_path / "observer.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["feature_scale"][0] = 100.0
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_ballistic_contact_observer(path)


def test_contact_observer_rejects_recomputed_safety_claim(tmp_path: Path) -> None:
    _observer(tmp_path)
    path = tmp_path / "observer.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["direct_torque_output"] = True
    value.pop("observer_hash")
    value["observer_hash"] = canonical_hash(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="safety boundary"):
        load_g1_ballistic_contact_observer(path)

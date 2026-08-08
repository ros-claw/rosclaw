from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_skill_memory import (
    ballistic_handoff_distance,
    ballistic_skill_experiment_context_hash,
    derive_g1_ballistic_skill_memory,
    load_g1_ballistic_skill_memory,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root: Path, *, seed: int, qualified: bool) -> Path:
    directory = root / f"seed-{seed}"
    directory.mkdir()
    trajectory = directory / "trajectory.npz"
    size = 6
    offset = 0.0 if seed == 0 else 0.04 if seed == 32 else 0.7 + seed * 0.001
    np.savez_compressed(
        trajectory,
        controller_mode=np.full(size, 5, dtype=np.int64),
        pelvis_pose=np.tile(np.asarray((offset, 0.0, 0.78, 1.0, 0.0, 0.0, 0.0)), (size, 1)),
        joint_position=np.full((size, 29), offset, dtype=np.float64),
        pelvis_velocity=np.full((size, 6), offset, dtype=np.float64),
        joint_velocity=np.full((size, 29), offset, dtype=np.float64),
    )
    crossing = [6.0, 1.0, 0.90] if qualified else [6.0, 0.0, 0.115]
    evidence = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "2" * 64,
        "approach_strike_candidate_hash": "sha256:" + "3" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "control": "frozen",
            "ballistic_contact_residual_rad": [-0.1, 0.0, 0.0, 0.0, 0.2, 0.0],
            "ballistic_contact_policy_frame": 252,
            "post_contact_damping_scale": 2.5,
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "speed": 1.5,
        },
        "runup_config": {"start_x_m": -3.4},
        "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
        "result": {
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "perceptual_continuity_passed": qualified,
            "goal_crossed": True,
            "goal_crossing_xyz_m": crossing,
            "goal_plane_target_error_m": 0.40 if qualified else 1.4,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _memory(tmp_path: Path):
    skills = (_evidence(tmp_path, seed=0, qualified=True), _evidence(tmp_path, seed=32, qualified=True))
    rejected = tuple(_evidence(tmp_path, seed=seed, qualified=False) for seed in range(33, 37))
    return derive_g1_ballistic_skill_memory(
        skill_evidence_paths=skills,
        rejected_evidence_paths=rejected,
        output_path=tmp_path / "memory.json",
        source_checkout=tmp_path / "checkout",
    )


def test_ballistic_skill_memory_selects_supported_state_and_rejects_ood(
    tmp_path: Path,
) -> None:
    memory = _memory(tmp_path)

    selected = memory.select(memory.prototypes[0].state)
    rejected_state = load_g1_ballistic_skill_memory(tmp_path / "memory.json")
    far = rejected_state.prototypes[0].state
    far_value = type(far)(
        pelvis_pose_xyz_rpy=tuple(value + 2.0 for value in far.pelvis_pose_xyz_rpy),
        joint_position=far.joint_position,
        pelvis_velocity_linear_angular=far.pelvis_velocity_linear_angular,
        joint_velocity=far.joint_velocity,
    )
    abstained = memory.select(far_value)

    assert selected.selected_skill_id == "sonic-seed-0"
    assert not selected.abstained
    assert selected.nearest_distance == pytest.approx(0.0)
    assert abstained.abstained
    assert abstained.failure_code == "OUT_OF_DISTRIBUTION_HANDOFF"
    assert memory.best_prototype.planner_seed == 0
    assert memory.minimum_rejected_distance > memory.maximum_support_distance
    assert ballistic_handoff_distance(memory.prototypes[0].state, far_value) > 1.0


def test_ballistic_skill_memory_tamper_fails_closed(tmp_path: Path) -> None:
    _memory(tmp_path)
    path = tmp_path / "memory.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["maximum_support_distance"] = 0.74
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_ballistic_skill_memory(path)


def test_ballistic_skill_memory_rejects_boolean_coercion_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    _memory(tmp_path)
    path = tmp_path / "memory.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["accepted"] = "true"
    value.pop("memory_hash")
    value["memory_hash"] = canonical_hash(value)
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="decision types"):
        load_g1_ballistic_skill_memory(path)


def test_ballistic_skill_memory_rejects_unqualified_positive(tmp_path: Path) -> None:
    skills = (
        _evidence(tmp_path, seed=0, qualified=True),
        _evidence(tmp_path, seed=32, qualified=False),
    )
    rejected = tuple(_evidence(tmp_path, seed=seed, qualified=False) for seed in range(33, 37))

    with pytest.raises(ValueError, match="not a qualified skill island"):
        derive_g1_ballistic_skill_memory(
            skill_evidence_paths=skills,
            rejected_evidence_paths=rejected,
            output_path=tmp_path / "memory.json",
            source_checkout=tmp_path / "checkout",
        )


def test_ballistic_skill_context_ignores_selection_but_binds_task() -> None:
    inputs = {
        "flow_config": {
            "schema_version": "flow.v2",
            "control": "frozen",
            "ballistic_contact_residual_rad": [0.0] * 6,
            "ballistic_contact_policy_frame": 252,
            "post_contact_damping_scale": 2.5,
            "ballistic_skill_memory_hash": "sha256:" + "1" * 64,
            "ballistic_skill_id": "sonic-seed-0",
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v2",
            "planner_seed": 0,
            "speed": 1.5,
        },
        "runup_config": {"start_x_m": -3.4},
        "goal_spec": {"target_y_m": 1.0, "target_z_m": 1.35},
        "approach_strike_candidate_hash": "sha256:" + "2" * 64,
    }
    baseline = ballistic_skill_experiment_context_hash(**inputs)  # type: ignore[arg-type]
    inputs["flow_config"]["ballistic_skill_id"] = "sonic-seed-32"  # type: ignore[index]
    inputs["sonic_runup_config"]["planner_seed"] = 32  # type: ignore[index]
    assert ballistic_skill_experiment_context_hash(**inputs) == baseline  # type: ignore[arg-type]

    inputs["goal_spec"]["target_z_m"] = 1.45  # type: ignore[index]
    assert ballistic_skill_experiment_context_hash(**inputs) != baseline  # type: ignore[arg-type]

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.proprioceptive_expert_router import (
    derive_g1_proprioceptive_expert_router,
    load_g1_proprioceptive_expert_router,
    strike_handoff_features,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(root: Path, *, seed: int, phase: int, pelvis_x: float, error: float) -> Path:
    directory = root / f"seed-{seed}-phase-{phase}"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    pose = np.asarray(((pelvis_x, 0.02, 0.79, 1.0, 0.0, 0.0, 0.0),), dtype=np.float64)
    np.savez_compressed(
        trajectory,
        controller_mode=np.asarray((5,), dtype=np.int64),
        pelvis_pose=pose,
        joint_velocity=np.zeros((1, 29), dtype=np.float64),
    )
    evidence = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "2" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "kick_phase_start_frame": phase,
            "contextual_phase_yaw_threshold_rad": 0.0,
            "contextual_high_yaw_kick_phase_start_frame": 190,
            "contextual_phase_calibration_hash": None,
            "aim_bias_y_m": 0.8,
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "run_velocity_mps": 1.5,
        },
        "runup_config": {"schema_version": "runup.v1"},
        "goal_spec": {"schema_version": "goal.v1", "precision_radius_m": 0.16},
        "result": {
            "selected_kick_phase_start_frame": phase,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "handoff_to_contact_sec": 1.0,
            "actuator_saturation_steps": 0,
            "precision_radius_m": 0.16,
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_three_expert_router_is_cross_validated_and_fails_back_ood(tmp_path: Path) -> None:
    phases = (190, 205, 214)
    paths: list[Path] = []
    for seed in range(12):
        winner = phases[seed // 4]
        pelvis_x = (-2.0, 0.0, 2.0)[seed // 4] + 0.01 * (seed % 4)
        for phase in phases:
            paths.append(
                _probe(
                    tmp_path,
                    seed=seed,
                    phase=phase,
                    pelvis_x=pelvis_x,
                    error=0.02 if phase == winner else 0.8,
                )
            )
    output = tmp_path / "router.json"

    router = derive_g1_proprioceptive_expert_router(
        evidence_paths=tuple(paths),
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )

    assert router.accepted is True
    assert router.cross_validation_selected_precision_hits == 12
    assert router.cross_validation_baseline_precision_hits == 4
    assert router.cross_validation_selected_unsafe_episodes == 0
    assert load_g1_proprioceptive_expert_router(output) == router
    near_middle = strike_handoff_features(
        np.asarray((0.01, 0.02, 0.79, 1.0, 0.0, 0.0, 0.0)), np.zeros(29)
    )
    assert router.select(near_middle).phase_start_frame == 205
    far_away = strike_handoff_features(
        np.asarray((100.0, 0.02, 0.79, 1.0, 0.0, 0.0, 0.0)), np.zeros(29)
    )
    selection = router.select(far_away)
    assert selection.phase_start_frame == 190
    assert selection.used_fallback is True


def test_router_loader_rejects_tampering(tmp_path: Path) -> None:
    phases = (190, 205, 214)
    paths = tuple(
        _probe(
            tmp_path,
            seed=seed,
            phase=phase,
            pelvis_x=(-2.0, 0.0, 2.0)[seed // 4],
            error=0.02 if phase == phases[seed // 4] else 0.8,
        )
        for seed in range(12)
        for phase in phases
    )
    output = tmp_path / "router.json"
    derive_g1_proprioceptive_expert_router(
        evidence_paths=paths,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["fallback_phase"] = 214
    output.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_proprioceptive_expert_router(output)

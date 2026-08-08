from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.proprioceptive_expert_router import (
    derive_g1_proprioceptive_expert_router,
    strike_handoff_features,
)
from rosclaw.growth.proprioceptive_readiness_gate import (
    derive_g1_proprioceptive_readiness_gate,
    load_g1_proprioceptive_readiness_gate,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(
    root: Path,
    *,
    seed: int,
    phase: int,
    pelvis_x: float,
    error: float,
    safe: bool,
) -> Path:
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
            "proprioceptive_router_hash": None,
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
            "proprioceptive_router_executed": False,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "handoff_to_contact_sec": 1.0,
            "actuator_saturation_steps": 0,
            "precision_radius_m": 0.16,
            "finite_state": safe,
            "post_kick_fall": not safe,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def _artifacts(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    phases = (190, 205, 214)
    router_paths: list[Path] = []
    for seed in range(18):
        group = seed // 6
        winner = phases[group]
        pelvis_x = (-2.0, 0.0, 2.0)[group] + 0.01 * (seed % 6)
        for phase in phases:
            router_paths.append(
                _probe(
                    tmp_path,
                    seed=seed,
                    phase=phase,
                    pelvis_x=pelvis_x,
                    error=0.02 if phase == winner else 0.8,
                    safe=phase == winner,
                )
            )
    router_path = tmp_path / "router.json"
    router = derive_g1_proprioceptive_expert_router(
        evidence_paths=tuple(router_paths),
        output_path=router_path,
        source_checkout=tmp_path / "checkout",
    )
    assert router.accepted is True
    readiness_paths = list(router_paths)
    for seed, pelvis_x in ((18, 5.00), (19, 5.01)):
        for phase in phases:
            readiness_paths.append(
                _probe(
                    tmp_path,
                    seed=seed,
                    phase=phase,
                    pelvis_x=pelvis_x,
                    error=2.0,
                    safe=False,
                )
            )
    return router_path, tuple(readiness_paths)


def test_readiness_gate_abstains_without_safe_support(tmp_path: Path) -> None:
    router_path, evidence_paths = _artifacts(tmp_path)
    output = tmp_path / "readiness.json"
    gate = derive_g1_proprioceptive_readiness_gate(
        evidence_paths=evidence_paths,
        router_path=router_path,
        output_path=output,
        source_checkout=tmp_path / "checkout",
        maximum_support_distance=2.0,
    )

    assert gate.accepted is True
    assert gate.cross_validation_unsafe_attempts == 0
    assert gate.cross_validation_all_unsafe_states == 2
    assert gate.cross_validation_all_unsafe_abstained == 2
    assert load_g1_proprioceptive_readiness_gate(output) == gate
    from rosclaw.growth.proprioceptive_expert_router import (
        load_g1_proprioceptive_expert_router,
    )

    router = load_g1_proprioceptive_expert_router(router_path)
    ready = strike_handoff_features(
        np.asarray((0.01, 0.02, 0.79, 1.0, 0.0, 0.0, 0.0)), np.zeros(29)
    )
    assert gate.decide(ready, router).selected_phase_start_frame == 205
    unsafe = strike_handoff_features(
        np.asarray((5.005, 0.02, 0.79, 1.0, 0.0, 0.0, 0.0)), np.zeros(29)
    )
    decision = gate.decide(unsafe, router)
    assert decision.abstained is True
    assert decision.selected_phase_start_frame is None


def test_readiness_gate_loader_rejects_tampering(tmp_path: Path) -> None:
    router_path, evidence_paths = _artifacts(tmp_path)
    output = tmp_path / "readiness.json"
    derive_g1_proprioceptive_readiness_gate(
        evidence_paths=evidence_paths,
        router_path=router_path,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )
    value = json.loads(output.read_text(encoding="utf-8"))
    value["maximum_support_distance"] = 9.0
    output.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_proprioceptive_readiness_gate(output)

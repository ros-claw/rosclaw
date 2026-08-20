from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import rosclaw.growth.proprioceptive_readiness_evaluation as module


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(
    root: Path,
    *,
    seed: int,
    phase: int,
    pelvis_x: float,
    safe: bool,
    error: float,
) -> Path:
    directory = root / f"seed-{seed}-phase-{phase}"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    np.savez_compressed(
        trajectory,
        controller_mode=np.asarray((5,), dtype=np.int64),
        pelvis_pose=np.asarray(
            ((pelvis_x, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0),), dtype=np.float64
        ),
        joint_velocity=np.zeros((1, 29), dtype=np.float64),
    )
    value = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "kick_phase_start_frame": phase,
            "contextual_phase_yaw_threshold_rad": 0.0,
            "proprioceptive_router_hash": None,
            "aim_bias_y_m": 0.8,
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "run_velocity_mps": 1.5,
        },
        "runup_config": {"schema_version": "runup.v1"},
        "goal_spec": {"schema_version": "goal.v1"},
        "result": {
            "selected_kick_phase_start_frame": phase,
            "proprioceptive_router_executed": False,
            "finite_state": safe,
            "post_kick_fall": not safe,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "precision_radius_m": 0.16,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class _Router:
    router_hash = "sha256:" + "2" * 64
    body_hash = "sha256:" + "1" * 64

    def select(self, _features):
        return SimpleNamespace(phase_start_frame=205)


class _Gate:
    gate_hash = "sha256:" + "3" * 64
    router_hash = _Router.router_hash
    body_hash = _Router.body_hash
    accepted = True
    expert_phases = (190, 205, 214)
    development_seeds = tuple(range(20))
    minimum_attempt_coverage = 0.5

    def decide(self, features, _router):
        abstained = features.pelvis_x_m >= 8.0
        return SimpleNamespace(
            abstained=abstained,
            selected_phase_start_frame=None if abstained else 205,
        )


def test_sealed_readiness_accepts_safe_attempts_and_necessary_abstention(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "load_g1_proprioceptive_expert_router", lambda _: _Router())
    monkeypatch.setattr(module, "load_g1_proprioceptive_readiness_gate", lambda _: _Gate())
    paths: list[Path] = []
    for seed in range(20, 28):
        all_unsafe = seed >= 26
        for phase in _Gate.expert_phases:
            safe = not all_unsafe and phase == 205
            paths.append(
                _evidence(
                    tmp_path,
                    seed=seed,
                    phase=phase,
                    pelvis_x=10.0 if all_unsafe else 0.01 * seed,
                    safe=safe,
                    error=0.05 if safe else 1.0,
                )
            )

    report = module.evaluate_g1_proprioceptive_readiness_holdout(
        evidence_paths=tuple(paths),
        router_path=tmp_path / "router.json",
        gate_path=tmp_path / "gate.json",
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert report.accepted is True
    assert report.attempted == 6
    assert report.abstained == 2
    assert report.all_unsafe_abstained == 2
    assert report.gated_unsafe_attempts == 0
    assert report.gated_precision_hits == 6
    assert report.report_hash.startswith("sha256:")

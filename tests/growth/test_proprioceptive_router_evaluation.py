from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rosclaw.growth.proprioceptive_router_evaluation as module


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(
    root: Path,
    *,
    seed: int,
    routed: bool,
    error: float,
    safe: bool = True,
) -> Path:
    directory = root / f"{'routed' if routed else 'baseline'}-{seed}"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(f"trajectory-{seed}-{routed}".encode())
    evidence = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "2" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "kick_phase_start_frame": 214,
            "proprioceptive_router_hash": "sha256:router" if routed else None,
        },
        "sonic_runup_config": {"schema_version": "sonic.v1", "planner_seed": seed},
        "runup_config": {"schema_version": "runup.v1"},
        "goal_spec": {"schema_version": "goal.v1"},
        "result": {
            "proprioceptive_router_executed": routed,
            "proprioceptive_router_fallback": False,
            "selected_kick_phase_start_frame": 205 if routed else 214,
            "finite_state": safe,
            "post_kick_fall": not safe,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "precision_radius_m": 0.16,
            "actuator_saturation_steps": 5 if routed else 20,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_sealed_router_evaluation_accepts_safe_measurable_gain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    router = SimpleNamespace(
        router_hash="sha256:router",
        body_hash="sha256:" + "1" * 64,
        baseline_phase=214,
        expert_phases=(190, 205, 214),
        development_seeds=tuple(range(16)),
    )
    monkeypatch.setattr(module, "load_g1_proprioceptive_expert_router", lambda _: router)
    baseline = tuple(
        _evidence(tmp_path, seed=seed, routed=False, error=0.5) for seed in range(16, 24)
    )
    routed = tuple(
        _evidence(tmp_path, seed=seed, routed=True, error=0.1) for seed in range(16, 24)
    )

    report = module.evaluate_g1_proprioceptive_router_holdout(
        baseline_paths=baseline,
        routed_paths=routed,
        router_path=tmp_path / "router.json",
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert report.accepted is True
    assert report.measurable_improvement is True
    assert report.routed_precision_hits == 8
    assert report.routed_unsafe_episodes == 0
    assert report.report_hash.startswith("sha256:")

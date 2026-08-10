from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import rosclaw.growth.sonic_authority_evaluation as module


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(
    root: Path,
    *,
    seed: int,
    candidate: bool,
    calibration_hash: str,
) -> Path:
    directory = root / f"{'candidate' if candidate else 'baseline'}-{seed}"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(f"trajectory-{seed}-{candidate}".encode())
    value = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "4" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "authority_calibration_hash": calibration_hash,
            "strike_gain_scales": [0.8 if candidate else 1.0] * 29,
            "follow_through_gain_scales": [0.8 if candidate else 1.0] * 29,
            "kick_phase_start_frame": 205,
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "authority_calibration_hash": calibration_hash,
            "joint_gain_scales": [0.8 if candidate else 1.0] * 29,
        },
        "runup_config": {"schema_version": "runup.v1"},
        "goal_spec": {"schema_version": "goal.v1"},
        "result": {
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "actuator_saturation_steps": 0 if candidate else 10,
            "runup_peak_tilt_rad": 0.25 if candidate else 0.30,
            "kick_peak_tilt_rad": 0.30 if candidate else 0.35,
            "final_pelvis_height_m": 0.78,
            "final_speed_mps": 0.10,
            "goal_crossed": True,
            "goal_plane_target_error_m": 0.10 if candidate else 0.20,
            "precision_radius_m": 0.16,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_sonic_authority_evaluation_accepts_paired_stability_gain(
    tmp_path: Path, monkeypatch
) -> None:
    baseline_hash = "sha256:" + "2" * 64
    candidate_hash = "sha256:" + "3" * 64
    baseline_calibration = SimpleNamespace(
        calibration_hash=baseline_hash,
        base_calibration_hash=None,
        approach_gain_frozen=False,
        body_hash="sha256:" + "1" * 64,
    )
    candidate_calibration = SimpleNamespace(
        calibration_hash=candidate_hash,
        base_calibration_hash=baseline_hash,
        approach_gain_frozen=False,
        body_hash="sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        module,
        "load_g1_sonic_authority_calibration",
        lambda path: candidate_calibration if "candidate" in path.name else baseline_calibration,
    )
    baseline = tuple(
        _evidence(
            tmp_path,
            seed=seed,
            candidate=False,
            calibration_hash=baseline_hash,
        )
        for seed in range(8)
    )
    candidate = tuple(
        _evidence(
            tmp_path,
            seed=seed,
            candidate=True,
            calibration_hash=candidate_hash,
        )
        for seed in range(8)
    )

    report = module.evaluate_g1_sonic_authority_holdout(
        baseline_paths=baseline,
        candidate_paths=candidate,
        baseline_calibration_path=tmp_path / "baseline.json",
        candidate_calibration_path=tmp_path / "candidate.json",
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert report.accepted is True
    assert report.baseline_saturation_steps == 80
    assert report.candidate_saturation_steps == 0
    assert report.candidate_stability_qualified_episodes == 8
    assert report.candidate_precision_hits == 8

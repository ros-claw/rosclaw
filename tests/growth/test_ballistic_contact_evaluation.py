from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rosclaw.growth.ballistic_contact_evaluation import (
    evaluate_g1_ballistic_contact_holdout,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(
    root: Path,
    *,
    seed: int,
    action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.2, 0.0),
    error: float = 0.20,
    height: float = 0.90,
    continuous: bool = True,
) -> Path:
    directory = root / f"seed-{seed}"
    directory.mkdir()
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(f"trajectory-{seed}".encode())
    evidence = {
        "strict_replay": True,
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_hash": _hash(trajectory),
        "body_hash": "sha256:" + "1" * 64,
        "implementation_hash": "sha256:" + "2" * 64,
        "flow_config": {
            "schema_version": "flow.v1",
            "control": "frozen",
            "ballistic_contact_residual_rad": list(action),
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "speed": 1.5,
        },
        "runup_config": {"schema_version": "runup.v1", "start": -3.4},
        "goal_spec": {"schema_version": "goal.v1", "target_z_m": 1.35},
        "approach_strike_candidate_hash": "sha256:" + "3" * 64,
        "football_motion_prior_hash": "sha256:" + "4" * 64,
        "result": {
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "perceptual_continuity_passed": continuous,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "goal_crossing_xyz_m": [6.0, 1.0, height],
            "actuator_saturation_steps": 10,
            "actuator_peak_demand_ratio": 1.1,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_ballistic_holdout_accepts_consistent_safe_multi_seed_action(tmp_path: Path) -> None:
    paths = tuple(_evidence(tmp_path, seed=seed) for seed in range(4))

    report = evaluate_g1_ballistic_contact_holdout(
        evidence_paths=paths,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert report.accepted
    assert report.hard_safe_episodes == 4
    assert report.continuous_episodes == 4
    assert report.failure_codes == ()
    assert report.report_hash.startswith("sha256:")


def test_ballistic_holdout_rejects_height_and_continuity_regressions(
    tmp_path: Path,
) -> None:
    paths = tuple(
        _evidence(
            tmp_path,
            seed=seed,
            error=1.2 if seed == 3 else 0.2,
            height=0.115 if seed == 3 else 0.9,
            continuous=seed != 2,
        )
        for seed in range(4)
    )

    report = evaluate_g1_ballistic_contact_holdout(
        evidence_paths=paths,
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert not report.accepted
    assert "CONTINUITY_GENERALIZATION_FAILED" in report.failure_codes
    assert "WORST_CASE_TARGET_ERROR_FAILED" in report.failure_codes
    assert "HIGH_TARGET_GENERALIZATION_FAILED" in report.failure_codes

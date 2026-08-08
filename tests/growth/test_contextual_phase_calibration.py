from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.contextual_phase_calibration import (
    derive_g1_contextual_phase_calibration,
    load_g1_contextual_phase_calibration,
)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(
    root: Path,
    *,
    seed: int,
    phase: int,
    yaw: float,
    error: float,
) -> Path:
    trajectory = root / f"trajectory-seed{seed}-phase{phase}.npz"
    np.savez_compressed(trajectory, marker=np.asarray((seed, phase), dtype=np.int64))
    evidence_path = root / f"evidence-seed{seed}-phase{phase}.json"
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
            "contextual_phase_expert_executed": False,
            "handoff_yaw_rad": yaw,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "handoff_to_contact_sec": 0.9 if phase == 214 else 1.2,
            "actuator_saturation_steps": 2,
            "precision_radius_m": 0.16,
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
        },
    }
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path


def test_contextual_phase_calibration_fits_paired_counterfactuals(tmp_path: Path) -> None:
    errors = {
        0: (0.10, 0.40, 0.01),
        1: (1.00, 0.10, 0.25),
        2: (0.08, 0.30, -0.02),
        3: (0.09, 0.35, -0.03),
    }
    paths = tuple(
        _probe(tmp_path, seed=seed, phase=phase, yaw=yaw, error=error)
        for seed, (normal_error, expert_error, yaw) in errors.items()
        for phase, error in ((214, normal_error), (190, expert_error))
    )
    output = tmp_path / "calibration.json"

    calibration = derive_g1_contextual_phase_calibration(
        evidence_paths=paths,
        output_path=output,
        source_checkout=tmp_path / "checkout",
        normal_phase_start_frame=214,
        high_yaw_phase_start_frame=190,
        holdout_seeds=(3,),
    )

    assert calibration.accepted is True
    assert 0.03 < calibration.yaw_threshold_rad < 0.25
    assert calibration.development_selected_precision_hits == 3
    assert calibration.development_baseline_precision_hits == 2
    assert calibration.holdout_selected_mean_penalized_error_m == pytest.approx(0.09)
    assert load_g1_contextual_phase_calibration(output) == calibration

    value = json.loads(output.read_text(encoding="utf-8"))
    value["yaw_threshold_rad"] = 0.30
    output.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_g1_contextual_phase_calibration(output)


def test_contextual_phase_calibration_rejects_unpaired_probe(tmp_path: Path) -> None:
    paths = tuple(
        _probe(tmp_path, seed=seed, phase=phase, yaw=0.01 * seed, error=0.1)
        for seed in range(5)
        for phase in ((214, 190) if seed != 2 else (214,))
    )

    with pytest.raises(ValueError, match="paired phase probes"):
        derive_g1_contextual_phase_calibration(
            evidence_paths=paths,
            output_path=tmp_path / "calibration.json",
            source_checkout=tmp_path / "checkout",
            normal_phase_start_frame=214,
            high_yaw_phase_start_frame=190,
            holdout_seeds=(4,),
        )

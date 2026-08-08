from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import rosclaw.growth.football_outcome_evaluation as module
from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.proprioceptive_expert_router import G1StrikeHandoffFeatures


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(root: Path, *, seed: int, phase: int, error: float) -> Path:
    directory = root / f"seed{seed}-phase{phase}"
    directory.mkdir(parents=True)
    trajectory = directory / "trajectory.npz"
    trajectory.write_bytes(f"trajectory-{seed}-{phase}".encode())
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
            "football_outcome_model_hash": None,
            "control": "frozen",
        },
        "sonic_runup_config": {
            "schema_version": "sonic.v1",
            "planner_seed": seed,
            "speed": 1.5,
        },
        "runup_config": {"schema_version": "runup.v1", "start": -3.4},
        "goal_spec": {"schema_version": "goal.v1", "target": 1.0},
        "result": {
            "selected_kick_phase_start_frame": phase,
            "finite_state": True,
            "post_kick_fall": False,
            "joint_limit_violation": False,
            "torque_limit_violation": False,
            "goal_crossed": True,
            "goal_plane_target_error_m": error,
            "precision_radius_m": 0.16,
            "actuator_saturation_steps": 0,
            "runup_peak_tilt_rad": 0.2,
            "kick_peak_tilt_rad": 0.2,
            "final_pelvis_height_m": 0.78,
            "final_speed_mps": 0.01,
        },
    }
    path = directory / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    return path


def test_sealed_outcome_evaluation_requires_and_scores_every_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context_hash = canonical_hash(
        {
            "flow_config": {"control": "frozen"},
            "sonic_runup_config": {"speed": 1.5},
            "runup_config": {"start": -3.4},
            "goal_spec": {"target": 1.0},
        }
    )
    model = SimpleNamespace(
        model_hash="sha256:" + "3" * 64,
        body_hash="sha256:" + "1" * 64,
        baseline_phase=190,
        expert_phases=(190, 205, 214),
        development_seeds=tuple(range(48)),
        experiment_context_hash=context_hash,
        decide=lambda _: SimpleNamespace(
            selected_phase_start_frame=205,
            retry_recommended=False,
        ),
    )
    monkeypatch.setattr(module, "load_g1_football_outcome_model", lambda _: model)
    monkeypatch.setattr(
        module,
        "_trajectory_features",
        lambda _: G1StrikeHandoffFeatures(0.1, 0.02, 0.03, -0.1, 0.0, 0.5),
    )
    paths = tuple(
        _evidence(
            tmp_path,
            seed=seed,
            phase=phase,
            error=0.10 if phase == 205 else 0.50,
        )
        for seed in range(56, 64)
        for phase in (190, 205, 214)
    )

    report = module.evaluate_g1_football_outcome_model(
        evidence_paths=paths,
        model_path=tmp_path / "model.json",
        output_path=tmp_path / "evaluation.json",
        source_checkout=tmp_path / "checkout",
    )

    assert report.accepted is True
    assert report.mandatory_attempts == 8
    assert report.terminal_abstentions == 0
    assert report.baseline_precision_hits == 0
    assert report.selected_precision_hits == 8
    assert report.selected_phase_counts == {190: 0, 205: 8, 214: 0}
    assert report.report_hash.startswith("sha256:")

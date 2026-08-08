from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw.growth.adapters.g1_free_kick import (
    triage_g1_free_kick_trajectory,
    write_g1_free_kick_triage,
)
from rosclaw.growth.approach_strike_dataset import (
    MINIMUM_TRAINING_EPISODES,
    build_g1_approach_strike_dataset,
)
from rosclaw.growth.cli import dispatch_growth_argv
from rosclaw.growth.learners.iql import IQLTrainingConfig, train_recovery_iql


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_episode(root: Path, index: int = 0) -> tuple[Path, Path]:
    count = 16
    trajectory_path = root / f"trajectory-{index}.npz"
    np.savez_compressed(
        trajectory_path,
        time=np.arange(count, dtype=np.float64) * 0.02,
        joint_position=np.full((count, 29), index * 1e-3, dtype=np.float64),
        joint_velocity=np.zeros((count, 29), dtype=np.float64),
        joint_torque=np.ones((count, 29), dtype=np.float64),
        commanded_torque=np.full((count, 29), 2.0 + index * 1e-3, dtype=np.float64),
        safety_projected_torque=np.ones((count, 29), dtype=np.float64),
        executed_torque=np.ones((count, 29), dtype=np.float64),
        torque_projection_applied=np.ones(count, dtype=bool),
        policy_action=np.full((count, 29), index * 1e-3, dtype=np.float64),
        pelvis_pose=np.tile(
            np.asarray((index * 1e-3, 0.0, 0.79, 1.0, 0.0, 0.0, 0.0)),
            (count, 1),
        ),
        torso_quaternion=np.tile(np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64), (count, 1)),
        ball_pose=np.tile(
            np.asarray((1.0, 0.0, 0.115, 1.0, 0.0, 0.0, 0.0), dtype=np.float64),
            (count, 1),
        ),
        ball_velocity=np.zeros((count, 6), dtype=np.float64),
        # Legacy collectors returned to SWING after the contact marker.
        event_phase=np.asarray(
            (0, 0, 1, 1, 2, 3, 4, 4, 5, 4, 6, 6, 7, 7, 8, 8),
            dtype=np.int64,
        ),
    )
    evidence_path = root / f"evidence-{index}.json"
    evidence_path.write_text(
        json.dumps(
            {
                "body_hash": "sha256:" + "1" * 64,
                "implementation_hash": "sha256:" + "2" * 64,
                "trajectory_path": str(trajectory_path.resolve()),
                "trajectory_hash": _hash(trajectory_path),
                "strict_replay": True,
                "passed": False,
                "result": {
                    "goal_plane_target_error_m": 0.20 + index * 1e-3,
                    "precision_radius_m": 0.16,
                    "lower_corner_distance_m": 0.40,
                    "actuator_saturation": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return trajectory_path, evidence_path


def _synthetic_teacher_episode(root: Path, index: int) -> tuple[Path, Path]:
    trajectory_path, evidence_path = _synthetic_episode(root, index)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    teacher_active = np.zeros(len(values["time"]), dtype=bool)
    teacher_active[6:10] = True
    teacher_torque = np.zeros((len(teacher_active), 29), dtype=np.float64)
    teacher_torque[teacher_active, 10] = 4.0 + index * 0.01
    values["loft_teacher_active"] = teacher_active
    values["loft_teacher_torque"] = teacher_torque
    np.savez_compressed(trajectory_path, **values)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["trajectory_hash"] = _hash(trajectory_path)
    evidence["result"]["loft_teacher_executed"] = True
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return trajectory_path, evidence_path


def test_free_kick_adapter_uses_measured_events_and_routes_failures(tmp_path: Path) -> None:
    trajectory_path, evidence_path = _synthetic_episode(tmp_path)

    report = triage_g1_free_kick_trajectory(
        trajectory_path=trajectory_path, evidence_path=evidence_path
    )

    assert [phase.phase.value for phase in report.phases] == [
        "approach",
        "align",
        "load",
        "swing",
        "contact",
        "follow_through",
        "recovery",
        "ready",
    ]
    assert [item.primary_type for item in report.failure_signatures] == [
        "contact_mode_precision",
        "declared_corner_miss",
        "authority_projection_required",
    ]
    assert not report.promotion_ready
    assert report.data_profile.has_executed_action
    assert not report.data_profile.offline_rl_ready
    assert report.learner_route.disposition.value == "selected"
    assert report.learner_route.learner_ids
    receipt = build_g1_approach_strike_dataset(
        trajectory_paths=(trajectory_path,),
        evidence_paths=(evidence_path,),
        output_dir=tmp_path / "dataset",
        source_checkout=Path(__file__).resolve().parents[2],
    )
    assert receipt.transition_count > 0
    assert receipt.episode_count == 1
    assert receipt.minimum_training_episodes == MINIMUM_TRAINING_EPISODES
    assert not receipt.training_eligible
    assert "iql" in receipt.learner_ids
    with np.load(receipt.array_path, allow_pickle=False) as archive:
        assert archive["teacher_residual_action"].shape[1] == 29
        assert not np.any(archive["teacher_active"])
    cli_triage_path = tmp_path / "cli-triage.json"
    assert (
        dispatch_growth_argv(
            [
                "growth",
                "free-kick-triage",
                "--trajectory",
                str(trajectory_path),
                "--evidence-json",
                str(evidence_path),
                "--output",
                str(cli_triage_path),
                "--source-checkout",
                str(tmp_path / "checkout"),
            ]
        )
        == 0
    )
    assert json.loads(cli_triage_path.read_text(encoding="utf-8"))["report_hash"]
    assert (
        dispatch_growth_argv(
            [
                "growth",
                "approach-strike-dataset",
                "--trajectory",
                str(trajectory_path),
                "--evidence-json",
                str(evidence_path),
                "--output-dir",
                str(tmp_path / "cli-dataset"),
                "--source-checkout",
                str(tmp_path / "checkout"),
            ]
        )
        == 0
    )
    assert (tmp_path / "cli-dataset" / "manifest.json").is_file()
    with pytest.raises(ValueError, match="outside the source checkout"):
        write_g1_free_kick_triage(
            report=report,
            output_path=tmp_path / "triage.json",
            source_checkout=tmp_path,
        )


def test_free_kick_adapter_routes_measured_perceptual_stall(tmp_path: Path) -> None:
    trajectory_path, evidence_path = _synthetic_episode(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["result"]["perceptual_continuity_passed"] = False
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    report = triage_g1_free_kick_trajectory(
        trajectory_path=trajectory_path,
        evidence_path=evidence_path,
    )

    assert report.failure_signatures[0].primary_type == "perceptual_handoff_stall"
    assert "continuous_run_to_strike" in report.failure_signatures[0].affected_capability_ids


def test_free_kick_adapter_routes_upper_corner_vertical_deficit(tmp_path: Path) -> None:
    trajectory_path, evidence_path = _synthetic_episode(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["result"].update(
        {
            "declared_target_corner": "left_upper",
            "goal_crossing_xyz_m": [6.0, 1.0, 0.60],
            "declared_corner_distance_m": 0.90,
        }
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    report = triage_g1_free_kick_trajectory(
        trajectory_path=trajectory_path,
        evidence_path=evidence_path,
    )

    signatures = {item.primary_type: item for item in report.failure_signatures}
    assert "insufficient_ballistic_loft" in signatures
    assert "upper_corner_ballistics" in signatures[
        "insufficient_ballistic_loft"
    ].affected_capability_ids


def test_approach_strike_dataset_trains_manifest_driven_iql(tmp_path: Path) -> None:
    sources = [_synthetic_episode(tmp_path, index) for index in range(8)]
    receipt = build_g1_approach_strike_dataset(
        trajectory_paths=tuple(item[0] for item in sources),
        evidence_paths=tuple(item[1] for item in sources),
        output_dir=tmp_path / "eligible-dataset",
        source_checkout=tmp_path / "checkout",
    )

    assert receipt.training_eligible
    training = train_recovery_iql(
        dataset_manifest_path=Path(receipt.manifest_path),
        output_dir=tmp_path / "iql-candidate",
        source_checkout=tmp_path / "checkout",
        config=IQLTrainingConfig(steps=2, batch_size=8, hidden_size=16, seed=7),
    )

    candidate = json.loads(Path(training.candidate_path).read_text(encoding="utf-8"))
    assert candidate["task_id"] == "g1_approach_strike_transition"
    assert len(candidate["artifact"]["state_features"]) == 110
    assert candidate["promotion_authorized"] is False


def test_approach_strike_dataset_distills_teacher_residual_iql(tmp_path: Path) -> None:
    sources = [_synthetic_teacher_episode(tmp_path, index) for index in range(8)]
    receipt = build_g1_approach_strike_dataset(
        trajectory_paths=tuple(item[0] for item in sources),
        evidence_paths=tuple(item[1] for item in sources),
        output_dir=tmp_path / "teacher-dataset",
        source_checkout=tmp_path / "checkout",
    )

    training = train_recovery_iql(
        dataset_manifest_path=Path(receipt.manifest_path),
        output_dir=tmp_path / "teacher-iql-candidate",
        source_checkout=tmp_path / "checkout",
        config=IQLTrainingConfig(
            steps=2,
            batch_size=8,
            hidden_size=16,
            seed=7,
            action_source="teacher_residual_action",
        ),
    )

    candidate = json.loads(Path(training.candidate_path).read_text(encoding="utf-8"))
    assert training.action_source == "teacher_residual_action"
    assert candidate["artifact"]["actor_output"] == "sim_teacher_residual_torque_nm"
    assert candidate["artifact"]["training_action_source"] == "teacher_residual_action"


def test_free_kick_adapter_rejects_unbound_evidence(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.npz"
    np.savez_compressed(trajectory_path, time=np.arange(10, dtype=np.float64))
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "trajectory_path": str(tmp_path / "other.npz"),
                "trajectory_hash": _hash(trajectory_path),
                "strict_replay": True,
                "result": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not bind"):
        triage_g1_free_kick_trajectory(trajectory_path=trajectory_path, evidence_path=evidence_path)


def test_free_kick_adapter_routes_teacher_rollout_to_distillation(tmp_path: Path) -> None:
    trajectory_path, evidence_path = _synthetic_episode(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["result"]["loft_teacher_executed"] = True
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    report = triage_g1_free_kick_trajectory(
        trajectory_path=trajectory_path,
        evidence_path=evidence_path,
    )

    assert report.failure_signatures[0].primary_type == "sim_teacher_distillation_required"
    assert not report.promotion_ready

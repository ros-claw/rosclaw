from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.adapters import (
    FieldTruthStatus,
    FootballPhase,
    triage_g1_coupled_trajectory,
    verified_coupled_evidence_context,
)
from rosclaw.growth.cli import dispatch_growth_argv
from rosclaw.growth.recovery_dataset import (
    COST_NAMES,
    REWARD_NAMES,
    STATE_FEATURES,
    build_g1_recovery_dataset,
)
from rosclaw.growth.routing import RouteDisposition


def _trace(path: Path, *, unstable: bool = False, enriched: bool = False) -> None:
    count = 350
    time = np.arange(1, count + 1, dtype=np.float64) * 0.02
    contact = 170
    pelvis = np.zeros((count, 7), dtype=np.float64)
    pelvis[:, 2] = 0.78
    pelvis[:, 3] = 1.0
    joints = np.zeros((count, 29), dtype=np.float64)
    if unstable:
        elapsed = time[contact:] - time[contact]
        pelvis[contact:, 0] = -0.45 * (1.0 - np.exp(-elapsed / 0.8))
        pelvis[contact:, 1] = 0.15 * np.sin(elapsed * 4.0) * np.exp(-elapsed / 3.0)
        joints[contact:] = np.sin(elapsed[:, None] * 5.0) * np.exp(-elapsed[:, None] / 3.0)
    ball_velocity = np.zeros((count, 6), dtype=np.float64)
    ball_velocity[contact:, 0] = 10.0
    contact_role = np.zeros(count, dtype=np.int64)
    contact_role[contact] = 2
    foot_contact = np.ones((count, 2), dtype=bool)
    zeros29 = np.zeros((count, 29), dtype=np.float64)
    values = {
        "time": time,
        "ball_velocity": ball_velocity,
        "ball_contact_role": contact_role,
        "shooter_pelvis_pose": pelvis,
        "shooter_torso_quaternion": np.tile((1.0, 0.0, 0.0, 0.0), (count, 1)),
        "shooter_joint_velocity": joints,
        "shooter_foot_contact": foot_contact,
        "shooter_policy_action": zeros29,
        "shooter_joint_torque": zeros29,
    }
    if enriched:
        ball_pose = np.zeros((count, 7), dtype=np.float64)
        ball_pose[:, 3] = 1.0
        ball_pose[:, 0] = 1.0
        left_foot = np.zeros((count, 3), dtype=np.float64)
        right_foot = np.zeros((count, 3), dtype=np.float64)
        left_foot[:, 1] = 0.1
        right_foot[:, 1] = -0.1
        values.update(
            {
                "ball_pose": ball_pose,
                "shooter_joint_position": zeros29,
                "shooter_com_position": pelvis[:, :3].copy(),
                "shooter_left_foot_position": left_foot,
                "shooter_right_foot_position": right_foot,
                "shooter_support_foot_slip": np.zeros(count),
                "shooter_contact_impulse": (contact_role == 2).astype(np.float64),
                "shooter_commanded_torque": zeros29,
                "shooter_safety_projected_torque": zeros29,
                "shooter_executed_torque": zeros29,
            }
        )
    np.savez_compressed(path, **values)


def _hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_adapter_segments_all_physical_phases_and_marks_missing_truth(tmp_path: Path) -> None:
    path = tmp_path / "trace.npz"
    _trace(path)

    report = triage_g1_coupled_trajectory(path)

    assert tuple(item.phase for item in report.phases) == tuple(FootballPhase)
    statuses = {item.field_id: item.status for item in report.field_provenance}
    assert statuses["contact_event"] is FieldTruthStatus.MEASURED
    assert statuses["whole_body_com"] is FieldTruthStatus.MISSING
    assert statuses["com_lateral_proxy"] is FieldTruthStatus.PROXY
    assert not report.data_profile.offline_rl_ready
    assert report.learner_route.disposition is RouteDisposition.BLOCKED_SAFETY_MODEL
    assert not report.absolute_recovery.passed


def test_adapter_creates_specific_recovery_failures(tmp_path: Path) -> None:
    path = tmp_path / "unstable.npz"
    _trace(path, unstable=True)

    report = triage_g1_coupled_trajectory(path)
    failure_types = {item.primary_type for item in report.failure_signatures}

    assert "excessive_capture_steps" in failure_types
    assert "recovery_oscillation" in failure_types
    assert report.recovery_quality.post_contact_backward_reversal_m > 0.25


def test_evidence_context_checks_trajectory_hash(tmp_path: Path) -> None:
    trajectory = tmp_path / "trace.npz"
    _trace(trajectory)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "trajectory_path": str(trajectory),
                        "trajectory_hash": _hash(trajectory),
                        "strict_replay": True,
                        "result": {
                            "goal_crossed": True,
                            "shot_peak_ball_speed_mps": 10.0,
                            "target_error_m": 0.1,
                            "joint_limit_violation": False,
                            "torque_limit_violation": False,
                            "actuator_saturation": False,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    context = verified_coupled_evidence_context(evidence, trajectory)

    assert context.strict_replay
    assert context.result["goal_crossed"] is True


def test_growth_triage_cli_writes_governed_report(tmp_path: Path) -> None:
    trajectory = tmp_path / "trace.npz"
    _trace(trajectory)
    output = tmp_path / "evidence" / "triage.json"

    code = dispatch_growth_argv(
        [
            "growth",
            "triage",
            "--skill",
            "g1_football",
            "--trajectory",
            str(trajectory),
            "--output",
            str(output),
            "--source-checkout",
            str(tmp_path / "checkout"),
        ]
    )

    assert code == 0
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["claims"]["parc_event_segmentation_completed"] is True
    assert value["claims"]["promotion_ready"] is False


def test_recovery_dataset_contains_action_triplet_rewards_costs_and_runtime(
    tmp_path: Path,
) -> None:
    trajectory = tmp_path / "enriched.npz"
    _trace(trajectory, enriched=True)
    runtime = {"python": "3.12.13", "mujoco": "3.10.0", "numpy": np.__version__}
    environment_hash = canonical_hash(runtime)
    request = {
        "runtime": runtime,
        "environment_hash": environment_hash,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "request_hash": _hash(request_path),
                "environment_hash": environment_hash,
                "cases": [
                    {
                        "trajectory_path": str(trajectory),
                        "trajectory_hash": _hash(trajectory),
                        "strict_replay": True,
                        "result": {
                            "goal_crossed": True,
                            "shot_peak_ball_speed_mps": 10.0,
                            "target_error_m": 0.1,
                            "joint_limit_violation": False,
                            "torque_limit_violation": False,
                            "actuator_saturation": False,
                            "shooter_post_kick_fall": False,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    receipt = build_g1_recovery_dataset(
        trajectory_paths=(trajectory,),
        evidence_path=evidence,
        output_dir=tmp_path / "dataset",
        source_checkout=tmp_path / "checkout",
    )

    assert receipt.training_eligible
    assert "iql" in receipt.learner_ids
    with np.load(receipt.array_path, allow_pickle=False) as arrays:
        assert arrays["state"].shape[1] == len(STATE_FEATURES)
        assert arrays["reward_vector"].shape[1] == len(REWARD_NAMES)
        assert arrays["cost_vector"].shape[1] == len(COST_NAMES)
        assert arrays["executed_action"].shape[1] == 29

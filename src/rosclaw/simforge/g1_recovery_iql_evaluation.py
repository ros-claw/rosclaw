"""Fail-fast closed-loop evaluation for an offline IQL recovery candidate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.adapters import measure_g1_coupled_recovery_quality
from rosclaw.growth.learners import NumpyIQLActor
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import (
    _simulate,
    coupled_runtime_manifest,
)
from rosclaw.simforge.g1_recovery_quality import evaluate_g1_absolute_recovery_gate
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1RecoveryIQLEvaluation:
    candidate_hash: str
    environment_hash: str
    request_hash: str
    parent_trajectory_hash: str
    candidate_trajectory_hash: str
    parent_strict_replay: bool
    candidate_strict_replay: bool
    parent_result: dict[str, Any]
    candidate_result: dict[str, Any]
    parent_quality: dict[str, Any]
    candidate_quality: dict[str, Any]
    absolute_gate: dict[str, Any]
    reasons: tuple[str, ...]
    status: str
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.growth.g1_recovery_iql_evaluation.v1"

    @property
    def passed(self) -> bool:
        return self.status == "CANDIDATE_EVALUATED_PASS"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["passed"] = self.passed
        value["promotion_authorized"] = False
        value["activation_authorized"] = False
        value["hardware_authorized"] = False
        return value


def evaluate_g1_recovery_iql_candidate(
    *,
    candidate_path: Path,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
) -> G1RecoveryIQLEvaluation:
    """Evaluate one reserved scenario and stop immediately on safety failure."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("IQL evaluation evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    actor = NumpyIQLActor.load(candidate_path)
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    runtime = coupled_runtime_manifest()
    request = {
        "schema_version": "rosclaw.growth.g1_recovery_iql_evaluation_request.v1",
        "candidate_hash": actor.candidate_hash,
        "scenario_id": "reserved-late-arrival-acceleration",
        "shooter_start_sec": 2.06,
        "ball_ground_friction": 0.10,
        "runtime": runtime,
        "environment_hash": hash_json(runtime),
        "implementation_hash": hash_bytes(Path(__file__).read_bytes()),
        "evaluation_policy": "fail_fast_on_first_reserved_safety_failure",
        "activation_ceiling": "SIM_ONLY",
    }
    request_path = root / "request.json"
    _write_json(request_path, request)
    parameters = {
        "shooter_start_sec": 2.06,
        "ball_ground_friction": 0.10,
    }
    parent_result, parent_trace = _simulate(asset_root, backend, **parameters)
    parent_replay_result, parent_replay_trace = _simulate(asset_root, backend, **parameters)
    candidate_result, candidate_trace = _simulate(
        asset_root,
        backend,
        **parameters,
        shooter_recovery_candidate_path=candidate_path,
    )
    candidate_replay_result, candidate_replay_trace = _simulate(
        asset_root,
        backend,
        **parameters,
        shooter_recovery_candidate_path=candidate_path,
    )
    parent_strict = bool(
        parent_result.to_dict() == parent_replay_result.to_dict()
        and trajectory_digest(parent_trace) == trajectory_digest(parent_replay_trace)
    )
    candidate_strict = bool(
        candidate_result.to_dict() == candidate_replay_result.to_dict()
        and trajectory_digest(candidate_trace) == trajectory_digest(candidate_replay_trace)
    )
    parent_path = root / "parent.npz"
    candidate_trace_path = root / "candidate.npz"
    np.savez_compressed(parent_path, **parent_trace)
    np.savez_compressed(candidate_trace_path, **candidate_trace)
    parent_quality = measure_g1_coupled_recovery_quality(parent_trace)
    candidate_quality = measure_g1_coupled_recovery_quality(candidate_trace)
    normalized_result = {
        "success": bool(
            candidate_result.goal_crossed
            and candidate_result.shot_peak_ball_speed_mps >= 9.5
            and candidate_result.target_error_m is not None
            and candidate_result.target_error_m <= 0.25
        ),
        "goal_crossed": candidate_result.goal_crossed,
        "target_zone_hit": bool(
            candidate_result.target_error_m is not None and candidate_result.target_error_m <= 0.25
        ),
        "ball_speed_mps": candidate_result.shot_peak_ball_speed_mps,
        "target_error_m": candidate_result.target_error_m,
        "post_kick_fall": candidate_result.shooter_post_kick_fall,
        "joint_limit_violation": candidate_result.joint_limit_violation,
        "torque_limit_violation": candidate_result.torque_limit_violation,
        "actuator_saturation": candidate_result.actuator_saturation,
        "support_foot_slip_m": candidate_result.shooter_post_contact_support_foot_slip_m,
    }
    absolute_gate = evaluate_g1_absolute_recovery_gate(
        quality=candidate_quality,
        result=normalized_result,
        strict_replay=candidate_strict,
    )
    reasons: list[str] = []
    if not parent_strict:
        reasons.append("parent_strict_replay_failed")
    if not candidate_strict:
        reasons.append("candidate_strict_replay_failed")
    if candidate_result.shooter_post_kick_fall:
        reasons.append("candidate_post_kick_fall")
    if not candidate_result.passed:
        reasons.append("candidate_task_or_safety_gate_failed")
    if candidate_result.shooter_learned_torque_fraction < 0.95:
        reasons.append("learned_torque_contribution_below_95_percent")
    if candidate_quality.post_contact_pelvis_path_length_m >= (
        parent_quality.post_contact_pelvis_path_length_m * 0.95
    ):
        reasons.append("pelvis_path_not_improved_by_5_percent")
    parent_settling = parent_quality.settling_time_sec
    candidate_settling = candidate_quality.settling_time_sec
    if candidate_settling is None or (
        parent_settling is not None and candidate_settling >= parent_settling * 0.95
    ):
        reasons.append("settling_time_not_improved_by_5_percent")
    if not absolute_gate.passed:
        reasons.append("absolute_recovery_gate_failed")
    status = "CANDIDATE_EVALUATED_PASS" if not reasons else "REJECTED_CLOSED_LOOP"
    evaluation = G1RecoveryIQLEvaluation(
        candidate_hash=actor.candidate_hash,
        environment_hash=str(request["environment_hash"]),
        request_hash=_file_hash(request_path),
        parent_trajectory_hash=_file_hash(parent_path),
        candidate_trajectory_hash=_file_hash(candidate_trace_path),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
        parent_result=parent_result.to_dict(),
        candidate_result=candidate_result.to_dict(),
        parent_quality=parent_quality.to_dict(),
        candidate_quality=candidate_quality.to_dict(),
        absolute_gate=absolute_gate.to_dict(),
        reasons=tuple(reasons),
        status=status,
    )
    _write_json(root / "evaluation.json", evaluation.to_dict())
    return evaluation


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = ["G1RecoveryIQLEvaluation", "evaluate_g1_recovery_iql_candidate"]

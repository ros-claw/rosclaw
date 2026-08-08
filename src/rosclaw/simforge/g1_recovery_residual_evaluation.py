"""Holdout-aware evaluation of a support-bound IQL residual recovery actor."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.adapters import measure_g1_coupled_recovery_quality
from rosclaw.growth.learners import (
    IQLResidualGuardConfig,
    NumpyIQLActor,
)
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import (
    G1CoupledRelayResult,
    _simulate,
    coupled_runtime_manifest,
)
from rosclaw.simforge.g1_recovery_quality import (
    G1AbsoluteRecoveryThresholds,
    evaluate_g1_absolute_recovery_gate,
)
from rosclaw.simforge.g1_structured_recovery_evaluation import (
    G1StructuredRecoveryCandidate,
    G1StructuredRecoveryCaseSpec,
    structured_recovery_specs,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1ResidualRecoveryCandidate:
    actor_candidate_hash: str
    actor_candidate_file_hash: str
    residual_guard: IQLResidualGuardConfig = field(default_factory=IQLResidualGuardConfig)
    structured_parent: G1StructuredRecoveryCandidate = field(
        default_factory=G1StructuredRecoveryCandidate
    )
    selection_method: str = "default_bounded_residual_selected_on_development_partition"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.growth.g1_residual_recovery_candidate.v1"

    def __post_init__(self) -> None:
        if not self.actor_candidate_hash.startswith("sha256:"):
            raise ValueError("residual recovery candidate requires an actor hash")
        if not self.actor_candidate_file_hash.startswith("sha256:"):
            raise ValueError("residual recovery candidate requires an actor file hash")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("residual recovery candidate must remain SIM_ONLY")

    @property
    def candidate_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1ResidualRecoveryCase:
    spec: G1StructuredRecoveryCaseSpec
    parent_result: dict[str, Any]
    candidate_result: dict[str, Any]
    parent_quality: dict[str, Any]
    candidate_quality: dict[str, Any]
    non_regression_gate: dict[str, Any]
    absolute_gate: dict[str, Any]
    parent_strict_replay: bool
    candidate_strict_replay: bool
    parent_trajectory_hash: str
    candidate_trajectory_hash: str
    schema_version: str = "rosclaw.growth.g1_residual_recovery_case.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.parent_strict_replay
            and self.candidate_strict_replay
            and self.candidate_result.get("passed")
            and self.non_regression_gate.get("passed")
            and self.absolute_gate.get("passed")
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class G1ResidualRecoveryEvidence:
    candidate: G1ResidualRecoveryCandidate
    candidate_hash: str
    environment_hash: str
    implementation_hash: str
    request_hash: str
    cases: tuple[G1ResidualRecoveryCase, ...]
    development_aggregate_gate: dict[str, Any]
    status: str
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.growth.g1_residual_recovery_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.cases) == 8
            and all(case.passed for case in self.cases)
            and self.development_aggregate_gate.get("passed")
            and self.status == "SIM_GATE_PASS"
            and not self.hardware_command_sent
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cases"] = [case.to_dict() for case in self.cases]
        value["passed"] = self.passed
        value["promotion_authorized"] = False
        value["activation_authorized"] = False
        value["hardware_authorized"] = False
        value["claims"] = {
            "direct_torque_takeover": False,
            "structured_parent_retained": True,
            "standardized_support_envelope_is_calibrated_ood_probability": False,
            "holdout_used_for_selection": False,
            "strict_replay_every_parent_and_candidate": all(
                case.parent_strict_replay and case.candidate_strict_replay
                for case in self.cases
            ),
            "real_hardware": False,
        }
        return value


def run_g1_residual_recovery_evaluation(
    *,
    actor_candidate_path: Path,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    residual_guard: IQLResidualGuardConfig | None = None,
) -> G1ResidualRecoveryEvidence:
    """Freeze and strictly replay one residual candidate across all partitions."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("residual recovery evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    actor_path = actor_candidate_path.expanduser().resolve()
    actor = NumpyIQLActor.load(actor_path)
    candidate = G1ResidualRecoveryCandidate(
        actor_candidate_hash=actor.candidate_hash,
        actor_candidate_file_hash=_file_hash(actor_path),
        residual_guard=residual_guard or IQLResidualGuardConfig(),
    )
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    runtime = coupled_runtime_manifest()
    environment_hash = hash_json(runtime)
    implementation_hash = hash_json(
        {
            "evaluation": hash_bytes(Path(__file__).read_bytes()),
            "runtime": hash_bytes(Path(__file__).with_name("g1_coupled_relay.py").read_bytes()),
            "learner": hash_bytes(
                Path(__file__).parents[1].joinpath("growth/learners/iql.py").read_bytes()
            ),
        }
    )
    specs = structured_recovery_specs()
    request = {
        "schema_version": "rosclaw.growth.g1_residual_recovery_request.v1",
        "candidate": {**asdict(candidate), "candidate_hash": candidate.candidate_hash},
        "partitions": [asdict(spec) for spec in specs],
        "selection_policy": {
            "development_case_ids": [
                spec.case_id for spec in specs if spec.partition == "development"
            ],
            "holdout_can_reject_but_cannot_retune": True,
            "candidate_frozen_before_holdout": True,
        },
        "runtime": runtime,
        "environment_hash": environment_hash,
        "implementation_hash": implementation_hash,
        "activation_ceiling": "SIM_ONLY",
    }
    _write_json(root / "candidate.json", request["candidate"])
    request_path = root / "request.json"
    _write_json(request_path, request)
    thresholds = G1AbsoluteRecoveryThresholds()
    cases = tuple(
        _evaluate_case(
            spec=spec,
            candidate=candidate,
            actor_path=actor_path,
            asset_root=asset_root,
            backend=backend,
            output_dir=root,
            thresholds=thresholds,
        )
        for spec in specs
    )
    development_gate = _development_aggregate_gate(cases)
    status = (
        "SIM_GATE_PASS"
        if all(case.passed for case in cases) and development_gate["passed"]
        else "REJECTED_BY_SIM_GATE"
    )
    evidence = G1ResidualRecoveryEvidence(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        environment_hash=environment_hash,
        implementation_hash=implementation_hash,
        request_hash=_file_hash(request_path),
        cases=cases,
        development_aggregate_gate=development_gate,
        status=status,
    )
    _write_json(root / "evaluation.json", evidence.to_dict())
    return evidence


def _evaluate_case(
    *,
    spec: G1StructuredRecoveryCaseSpec,
    candidate: G1ResidualRecoveryCandidate,
    actor_path: Path,
    asset_root: Path,
    backend: G1MuJoCoBackend,
    output_dir: Path,
    thresholds: G1AbsoluteRecoveryThresholds,
) -> G1ResidualRecoveryCase:
    structured = candidate.structured_parent
    parameters = {
        "shooter_start_sec": spec.shooter_start_sec,
        "ball_ground_friction": spec.ball_ground_friction,
        "shooter_post_policy_frame": structured.post_policy_frame,
        "shooter_post_policy_blend_frames": structured.post_policy_blend_frames,
        "shooter_joint_guard_enabled": structured.post_contact_joint_guard_enabled,
        "shooter_post_policy_neutral_velocity_enabled": (
            structured.post_policy_neutral_velocity_enabled
        ),
        "shooter_joint_guard_config": structured.joint_guard_config,
        "shooter_joint_guard_late_config": structured.late_arrival_joint_guard_config,
    }
    parent_result, parent_trace = _simulate(asset_root, backend, **parameters)
    parent_replay_result, parent_replay_trace = _simulate(asset_root, backend, **parameters)
    candidate_parameters = {
        **parameters,
        "shooter_recovery_candidate_path": actor_path,
        "shooter_recovery_residual_config": candidate.residual_guard,
    }
    candidate_result, candidate_trace = _simulate(
        asset_root, backend, **candidate_parameters
    )
    candidate_replay_result, candidate_replay_trace = _simulate(
        asset_root, backend, **candidate_parameters
    )
    parent_strict = _strict_replay(
        parent_result, parent_trace, parent_replay_result, parent_replay_trace
    )
    candidate_strict = _strict_replay(
        candidate_result,
        candidate_trace,
        candidate_replay_result,
        candidate_replay_trace,
    )
    parent_path = output_dir / f"{spec.case_id}-structured-parent.npz"
    candidate_path = output_dir / f"{spec.case_id}-residual-candidate.npz"
    np.savez_compressed(parent_path, **parent_trace)
    np.savez_compressed(candidate_path, **candidate_trace)
    parent_quality = measure_g1_coupled_recovery_quality(parent_trace)
    candidate_quality = measure_g1_coupled_recovery_quality(candidate_trace)
    parent_contract = _result_contract(parent_result)
    candidate_contract = _result_contract(candidate_result)
    absolute = evaluate_g1_absolute_recovery_gate(
        quality=candidate_quality,
        result=candidate_contract,
        strict_replay=candidate_strict,
        thresholds=thresholds,
    )
    non_regression = _evaluate_non_regression_gate(
        parent_quality=parent_quality.to_dict(),
        candidate_quality=candidate_quality.to_dict(),
        parent_result=parent_result.to_dict(),
        candidate_result=candidate_result.to_dict(),
        parent_contract=parent_contract,
        candidate_contract=candidate_contract,
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
        residual_guard=candidate.residual_guard,
    )
    return G1ResidualRecoveryCase(
        spec=spec,
        parent_result=parent_result.to_dict(),
        candidate_result=candidate_result.to_dict(),
        parent_quality=parent_quality.to_dict(),
        candidate_quality=candidate_quality.to_dict(),
        non_regression_gate=non_regression,
        absolute_gate=absolute.to_dict(),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
        parent_trajectory_hash=_file_hash(parent_path),
        candidate_trajectory_hash=_file_hash(candidate_path),
    )


def _evaluate_non_regression_gate(
    *,
    parent_quality: dict[str, Any],
    candidate_quality: dict[str, Any],
    parent_result: dict[str, Any],
    candidate_result: dict[str, Any],
    parent_contract: dict[str, Any],
    candidate_contract: dict[str, Any],
    parent_strict_replay: bool,
    candidate_strict_replay: bool,
    residual_guard: IQLResidualGuardConfig,
) -> dict[str, Any]:
    """Require meaningful learned participation without destabilizing champion."""

    path_ratio = float(candidate_quality["post_contact_pelvis_path_length_m"]) / max(
        float(parent_quality["post_contact_pelvis_path_length_m"]), 1e-12
    )
    jerk_ratio = float(candidate_quality["post_contact_joint_jerk_rms_rad_s3"]) / max(
        float(parent_quality["post_contact_joint_jerk_rms_rad_s3"]), 1e-12
    )
    parent_settle = parent_quality.get("settling_time_sec")
    candidate_settle = candidate_quality.get("settling_time_sec")
    task_preserved = bool(
        parent_contract.get("success")
        and candidate_contract.get("success")
        and candidate_contract.get("goal_crossed") == parent_contract.get("goal_crossed")
        and float(candidate_contract.get("ball_speed_mps", 0.0))
        >= float(parent_contract.get("ball_speed_mps", 0.0)) - 1e-9
        and float(candidate_contract.get("target_error_m", 99.0))
        <= float(parent_contract.get("target_error_m", 99.0)) + 1e-9
    )
    active_fraction = float(candidate_result.get("shooter_learned_torque_fraction", 0.0))
    fallback_fraction = float(
        candidate_result.get("shooter_learned_torque_fallback_fraction", 1.0)
    )
    confidence = float(candidate_result.get("shooter_learned_torque_mean_confidence", 0.0))
    peak_residual = float(
        candidate_result.get("shooter_learned_torque_peak_residual_nm", np.inf)
    )
    reasons: list[str] = []
    if not task_preserved:
        reasons.append("upstream_task_outcome_not_preserved")
    if not parent_strict_replay or not candidate_strict_replay:
        reasons.append("strict_replay_missing")
    if path_ratio > 1.03 + 1e-9:
        reasons.append("pelvis_path_regression_above_3_percent")
    if jerk_ratio > 1.10 + 1e-9:
        reasons.append("joint_jerk_regression_above_10_percent")
    if float(candidate_quality["post_contact_backward_reversal_m"]) > 0.01 + 1e-9:
        reasons.append("backward_reversal_above_residual_gate")
    if (
        candidate_settle is None
        or parent_settle is None
        or float(candidate_settle) > float(parent_settle) + 0.15 + 1e-9
    ):
        reasons.append("settling_time_regression_above_0.15_sec")
    if active_fraction < 0.50:
        reasons.append("learned_residual_participation_below_50_percent")
    if fallback_fraction > 0.50:
        reasons.append("support_envelope_fallback_above_50_percent")
    if confidence <= 0.0:
        reasons.append("learned_residual_confidence_missing")
    maximum_effective_residual = (
        residual_guard.maximum_residual_nm * residual_guard.residual_fraction
    )
    if peak_residual > maximum_effective_residual + 1e-9:
        reasons.append("learned_residual_amplitude_contract_violated")
    if candidate_result.get("shooter_post_kick_fall") is not False:
        reasons.append("candidate_post_kick_fall")
    return {
        "schema_version": "rosclaw.growth.g1_residual_non_regression_gate.v1",
        "passed": not reasons,
        "task_preserved": task_preserved,
        "strict_replay_preserved": bool(parent_strict_replay and candidate_strict_replay),
        "path_ratio": path_ratio,
        "joint_jerk_ratio": jerk_ratio,
        "settling_time_delta_sec": (
            None
            if parent_settle is None or candidate_settle is None
            else float(candidate_settle) - float(parent_settle)
        ),
        "active_fraction": active_fraction,
        "fallback_fraction": fallback_fraction,
        "mean_confidence": confidence,
        "peak_residual_nm": peak_residual,
        "maximum_effective_residual_nm": maximum_effective_residual,
        "reasons": reasons,
    }


def _development_aggregate_gate(
    cases: tuple[G1ResidualRecoveryCase, ...],
) -> dict[str, Any]:
    development = tuple(case for case in cases if case.spec.partition == "development")
    if len(development) != 3:
        return {
            "schema_version": "rosclaw.growth.g1_residual_development_gate.v1",
            "passed": False,
            "reasons": ["development_partition_incomplete"],
        }

    def mean(side: str, key: str) -> float:
        return float(
            np.mean(
                [float(getattr(case, side)[key]) for case in development],
                dtype=np.float64,
            )
        )

    parent_path = mean("parent_quality", "post_contact_pelvis_path_length_m")
    candidate_path = mean("candidate_quality", "post_contact_pelvis_path_length_m")
    parent_jerk = mean("parent_quality", "post_contact_joint_jerk_rms_rad_s3")
    candidate_jerk = mean("candidate_quality", "post_contact_joint_jerk_rms_rad_s3")
    parent_settle = mean("parent_quality", "settling_time_sec")
    candidate_settle = mean("candidate_quality", "settling_time_sec")
    reasons: list[str] = []
    if candidate_path > parent_path * 1.01 + 1e-9:
        reasons.append("development_mean_path_regressed_above_1_percent")
    if candidate_jerk > parent_jerk + 1e-9:
        reasons.append("development_mean_jerk_not_improved")
    if candidate_settle > parent_settle + 0.05 + 1e-9:
        reasons.append("development_mean_settling_regressed_above_0.05_sec")
    return {
        "schema_version": "rosclaw.growth.g1_residual_development_gate.v1",
        "passed": not reasons,
        "parent_mean_path_m": parent_path,
        "candidate_mean_path_m": candidate_path,
        "parent_mean_jerk_rad_s3": parent_jerk,
        "candidate_mean_jerk_rad_s3": candidate_jerk,
        "parent_mean_settling_sec": parent_settle,
        "candidate_mean_settling_sec": candidate_settle,
        "reasons": reasons,
    }


def _result_contract(result: G1CoupledRelayResult) -> dict[str, Any]:
    target_hit = bool(result.target_error_m is not None and result.target_error_m <= 0.25)
    return {
        "success": bool(
            result.goal_crossed and result.shot_peak_ball_speed_mps >= 9.5 and target_hit
        ),
        "goal_crossed": result.goal_crossed,
        "target_zone_hit": target_hit,
        "ball_speed_mps": result.shot_peak_ball_speed_mps,
        "target_error_m": result.target_error_m,
        "post_kick_fall": result.shooter_post_kick_fall,
        "joint_limit_violation": result.joint_limit_violation,
        "torque_limit_violation": result.torque_limit_violation,
        "actuator_saturation": result.actuator_saturation,
        "support_foot_slip_m": result.shooter_post_contact_support_foot_slip_m,
    }


def _strict_replay(
    result: G1CoupledRelayResult,
    trace: dict[str, np.ndarray],
    replay_result: G1CoupledRelayResult,
    replay_trace: dict[str, np.ndarray],
) -> bool:
    return bool(
        result.to_dict() == replay_result.to_dict()
        and trajectory_digest(trace) == trajectory_digest(replay_trace)
    )


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


__all__ = [
    "G1ResidualRecoveryCandidate",
    "G1ResidualRecoveryCase",
    "G1ResidualRecoveryEvidence",
    "run_g1_residual_recovery_evaluation",
]

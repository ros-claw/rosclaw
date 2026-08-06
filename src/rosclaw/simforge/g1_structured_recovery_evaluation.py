"""Runtime-bound, holdout-aware evaluation of structured G1 recovery.

The candidate is deliberately small: a measured-contact-gated handoff to the
retained locomotion policy plus a velocity-aware joint boundary projector.  It
does not change the kick prior and it can never authorize hardware or promote
itself.  Development, validation, reserved, and generalization cases are
declared and committed before any rollout is executed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.growth.adapters import measure_g1_coupled_recovery_quality
from rosclaw.simforge.backends.unitree_mujoco_backend import (
    G1MuJoCoBackend,
    trajectory_digest,
)
from rosclaw.simforge.g1_coupled_relay import (
    G1CoupledRelayResult,
    G1JointGuardConfig,
    _simulate,
    coupled_runtime_manifest,
)
from rosclaw.simforge.g1_recovery_quality import (
    G1AbsoluteRecoveryThresholds,
    compare_g1_naturalness,
    evaluate_g1_absolute_recovery_gate,
)
from rosclaw.simforge.tasks.g1_goalforge.concepts import hash_bytes, hash_json


@dataclass(frozen=True)
class G1StructuredRecoveryCandidate:
    post_policy_frame: int = 275
    post_policy_blend_frames: int = 0
    post_contact_joint_guard_enabled: bool = True
    post_policy_neutral_velocity_enabled: bool = True
    joint_guard_config: G1JointGuardConfig = field(
        default_factory=lambda: G1JointGuardConfig(
            margin_rad=0.02,
            prediction_horizon_sec=0.06,
            boundary_kp=60.0,
            boundary_kd=4.0,
        )
    )
    late_arrival_joint_guard_config: G1JointGuardConfig = field(
        default_factory=G1JointGuardConfig
    )
    selection_method: str = "bounded_discrete_search_on_development_partition"
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw.growth.g1_structured_recovery_candidate.v3"

    def __post_init__(self) -> None:
        if not 270 <= self.post_policy_frame <= 430:
            raise ValueError("structured recovery handoff frame must be in [270, 430]")
        if not 0 <= self.post_policy_blend_frames <= 200:
            raise ValueError("structured recovery blend frames must be in [0, 200]")
        if not self.post_contact_joint_guard_enabled:
            raise ValueError("structured recovery candidate requires its joint guard")
        if not self.post_policy_neutral_velocity_enabled:
            raise ValueError("structured recovery candidate requires a true zero-velocity handoff")
        if self.activation_ceiling != "SIM_ONLY":
            raise ValueError("structured recovery candidate must remain SIM_ONLY")

    @property
    def candidate_hash(self) -> str:
        return hash_json(asdict(self))


@dataclass(frozen=True)
class G1StructuredRecoveryCaseSpec:
    case_id: str
    partition: str
    shooter_start_sec: float
    ball_ground_friction: float
    schema_version: str = "rosclaw.growth.g1_structured_recovery_case_spec.v1"

    def __post_init__(self) -> None:
        if self.partition not in {"development", "validation", "reserved", "generalization"}:
            raise ValueError("unknown structured recovery partition")
        if not 1.90 <= self.shooter_start_sec <= 2.10:
            raise ValueError("structured recovery receiver timing is outside the SIM envelope")
        if not 0.05 <= self.ball_ground_friction <= 0.15:
            raise ValueError("structured recovery friction is outside the SIM envelope")


@dataclass(frozen=True)
class G1StructuredRecoveryCase:
    spec: G1StructuredRecoveryCaseSpec
    parent_result: dict[str, Any]
    candidate_result: dict[str, Any]
    parent_quality: dict[str, Any]
    candidate_quality: dict[str, Any]
    naturalness_gate: dict[str, Any]
    capture_step_gate: dict[str, Any]
    absolute_gate: dict[str, Any]
    parent_strict_replay: bool
    candidate_strict_replay: bool
    parent_trajectory_hash: str
    candidate_trajectory_hash: str
    schema_version: str = "rosclaw.growth.g1_structured_recovery_case.v1"

    @property
    def passed(self) -> bool:
        return bool(
            self.parent_strict_replay
            and self.candidate_strict_replay
            and self.candidate_result.get("passed")
            and self.capture_step_gate.get("passed")
            and self.absolute_gate.get("passed")
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["passed"] = self.passed
        return value


@dataclass(frozen=True)
class G1StructuredRecoveryEvidence:
    candidate: G1StructuredRecoveryCandidate
    candidate_hash: str
    environment_hash: str
    implementation_hash: str
    request_hash: str
    cases: tuple[G1StructuredRecoveryCase, ...]
    status: str
    activation_ceiling: str = "SIM_ONLY"
    evidence_domain: str = "SIM"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw.growth.g1_structured_recovery_evidence.v1"

    @property
    def passed(self) -> bool:
        return bool(
            len(self.cases) == 8
            and all(case.passed for case in self.cases)
            and self.status == "SIM_GATE_PASS"
            and self.activation_ceiling == "SIM_ONLY"
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
            "kick_prior_changed": False,
            "contact_gated_recovery": True,
            "same_world_two_g1_physics": True,
            "strict_replay_every_parent_and_candidate": all(
                case.parent_strict_replay and case.candidate_strict_replay for case in self.cases
            ),
            "holdout_used_for_selection": False,
            "pixels_used_for_evaluation": False,
            "real_hardware": False,
        }
        return value


def structured_recovery_specs() -> tuple[G1StructuredRecoveryCaseSpec, ...]:
    """Return the frozen partition used by the first structured campaign."""

    return (
        G1StructuredRecoveryCaseSpec("01-early-arrival", "development", 1.98, 0.10),
        G1StructuredRecoveryCaseSpec("02-slick-pitch", "development", 2.02, 0.05),
        G1StructuredRecoveryCaseSpec("03-high-target", "development", 2.02, 0.10),
        G1StructuredRecoveryCaseSpec("04-grippy-pitch", "validation", 2.02, 0.15),
        G1StructuredRecoveryCaseSpec("05-late-arrival", "reserved", 2.06, 0.10),
        G1StructuredRecoveryCaseSpec("06-early-mid-slick", "generalization", 1.98, 0.09),
        G1StructuredRecoveryCaseSpec("07-center-high-grip", "generalization", 2.02, 0.13),
        G1StructuredRecoveryCaseSpec("08-late-mid-grip", "generalization", 2.06, 0.11),
    )


def run_g1_structured_recovery_evaluation(
    *,
    asset_root: Path,
    output_dir: Path,
    source_checkout: Path,
    candidate: G1StructuredRecoveryCandidate | None = None,
) -> G1StructuredRecoveryEvidence:
    """Commit, evaluate, and strictly replay one frozen SIM-only candidate."""

    root = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if root == checkout or checkout in root.parents:
        raise ValueError("structured recovery evidence must be outside the source checkout")
    root.mkdir(parents=True, exist_ok=False)
    candidate = candidate or G1StructuredRecoveryCandidate()
    backend = G1MuJoCoBackend(asset_root=asset_root, trace_stride=1)
    runtime = coupled_runtime_manifest()
    environment_hash = hash_json(runtime)
    thresholds = G1AbsoluteRecoveryThresholds()
    specs = structured_recovery_specs()
    implementation_hash = hash_json(
        {
            "evaluation": hash_bytes(Path(__file__).read_bytes()),
            "runtime": hash_bytes(Path(__file__).with_name("g1_coupled_relay.py").read_bytes()),
            "quality": hash_bytes(Path(__file__).with_name("g1_recovery_quality.py").read_bytes()),
        }
    )
    candidate_payload = {
        **asdict(candidate),
        "candidate_hash": candidate.candidate_hash,
        "promotion_authorized": False,
        "activation_authorized": False,
        "hardware_authorized": False,
    }
    _write_json(root / "candidate.json", candidate_payload)
    request = {
        "schema_version": "rosclaw.growth.g1_structured_recovery_request.v1",
        "candidate": candidate_payload,
        "candidate_hash": candidate.candidate_hash,
        "partitions": [asdict(spec) for spec in specs],
        "absolute_thresholds": thresholds.to_dict(),
        "runtime": runtime,
        "environment_hash": environment_hash,
        "implementation_hash": implementation_hash,
        "body_hash": backend.qualification.body_hash,
        "kick_prior_hash": backend.qualification.kick_prior_hash,
        "selection_policy": {
            "development_case_ids": [spec.case_id for spec in specs if spec.partition == "development"],
            "validation_case_ids": [spec.case_id for spec in specs if spec.partition == "validation"],
            "reserved_case_ids": [spec.case_id for spec in specs if spec.partition == "reserved"],
            "generalization_case_ids": [
                spec.case_id for spec in specs if spec.partition == "generalization"
            ],
            "candidate_frozen_before_this_request": True,
            "holdout_can_reject_but_cannot_retune": True,
        },
        "activation_ceiling": "SIM_ONLY",
    }
    request_path = root / "request.json"
    _write_json(request_path, request)

    cases = tuple(
        _evaluate_case(
            spec=spec,
            candidate=candidate,
            backend=backend,
            asset_root=asset_root,
            output_dir=root,
            thresholds=thresholds,
        )
        for spec in specs
    )
    status = "SIM_GATE_PASS" if all(case.passed for case in cases) else "REJECTED_BY_SIM_GATE"
    evidence = G1StructuredRecoveryEvidence(
        candidate=candidate,
        candidate_hash=candidate.candidate_hash,
        environment_hash=environment_hash,
        implementation_hash=implementation_hash,
        request_hash=_file_hash(request_path),
        cases=cases,
        status=status,
    )
    _write_json(root / "evaluation.json", evidence.to_dict())
    return evidence


def _evaluate_case(
    *,
    spec: G1StructuredRecoveryCaseSpec,
    candidate: G1StructuredRecoveryCandidate,
    backend: G1MuJoCoBackend,
    asset_root: Path,
    output_dir: Path,
    thresholds: G1AbsoluteRecoveryThresholds,
) -> G1StructuredRecoveryCase:
    parameters = {
        "shooter_start_sec": spec.shooter_start_sec,
        "ball_ground_friction": spec.ball_ground_friction,
    }
    parent_result, parent_trace = _simulate(asset_root, backend, **parameters)
    parent_replay_result, parent_replay_trace = _simulate(asset_root, backend, **parameters)
    candidate_parameters = {
        **parameters,
        "shooter_post_policy_frame": candidate.post_policy_frame,
        "shooter_post_policy_blend_frames": candidate.post_policy_blend_frames,
        "shooter_joint_guard_enabled": candidate.post_contact_joint_guard_enabled,
        "shooter_post_policy_neutral_velocity_enabled": (
            candidate.post_policy_neutral_velocity_enabled
        ),
        "shooter_joint_guard_config": candidate.joint_guard_config,
        "shooter_joint_guard_late_config": candidate.late_arrival_joint_guard_config,
    }
    candidate_result, candidate_trace = _simulate(asset_root, backend, **candidate_parameters)
    candidate_replay_result, candidate_replay_trace = _simulate(
        asset_root, backend, **candidate_parameters
    )
    parent_strict = _strict_replay(
        parent_result,
        parent_trace,
        parent_replay_result,
        parent_replay_trace,
    )
    candidate_strict = _strict_replay(
        candidate_result,
        candidate_trace,
        candidate_replay_result,
        candidate_replay_trace,
    )
    parent_path = output_dir / f"{spec.case_id}-parent.npz"
    candidate_path = output_dir / f"{spec.case_id}-candidate.npz"
    np.savez_compressed(parent_path, **parent_trace)
    np.savez_compressed(candidate_path, **candidate_trace)
    parent_quality = measure_g1_coupled_recovery_quality(parent_trace)
    candidate_quality = measure_g1_coupled_recovery_quality(candidate_trace)
    parent_contract = _result_contract(parent_result)
    candidate_contract = _result_contract(candidate_result)
    naturalness = compare_g1_naturalness(
        parent=parent_quality,
        candidate=candidate_quality,
        parent_result=parent_contract,
        candidate_result=candidate_contract,
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
    )
    absolute = evaluate_g1_absolute_recovery_gate(
        quality=candidate_quality,
        result=candidate_contract,
        strict_replay=candidate_strict,
        thresholds=thresholds,
    )
    capture_step = _evaluate_capture_step_gate(
        parent_quality=parent_quality.to_dict(),
        candidate_quality=candidate_quality.to_dict(),
        parent_result=parent_contract,
        candidate_result=candidate_contract,
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
    )
    return G1StructuredRecoveryCase(
        spec=spec,
        parent_result=parent_result.to_dict(),
        candidate_result=candidate_result.to_dict(),
        parent_quality=parent_quality.to_dict(),
        candidate_quality=candidate_quality.to_dict(),
        naturalness_gate=naturalness.to_dict(),
        capture_step_gate=capture_step,
        absolute_gate=absolute.to_dict(),
        parent_strict_replay=parent_strict,
        candidate_strict_replay=candidate_strict,
        parent_trajectory_hash=_file_hash(parent_path),
        candidate_trajectory_hash=_file_hash(candidate_path),
    )


def _evaluate_capture_step_gate(
    *,
    parent_quality: dict[str, Any],
    candidate_quality: dict[str, Any],
    parent_result: dict[str, Any],
    candidate_result: dict[str, Any],
    parent_strict_replay: bool,
    candidate_strict_replay: bool,
) -> dict[str, Any]:
    """Gate a bounded forward capture step without requiring zero displacement."""

    def reduction(key: str) -> float:
        parent = float(parent_quality[key])
        candidate = float(candidate_quality[key])
        return (parent - candidate) / parent if parent > 1e-12 else 0.0

    reductions = {
        "pelvis_path_reduction": reduction("post_contact_pelvis_path_length_m"),
        "backward_reversal_reduction": reduction("post_contact_backward_reversal_m"),
        "joint_jerk_reduction": reduction("post_contact_joint_jerk_rms_rad_s3"),
        "leg_jerk_reduction": reduction("post_contact_leg_joint_jerk_rms_rad_s3"),
        "waist_jerk_reduction": reduction("post_contact_waist_joint_jerk_rms_rad_s3"),
        "arm_jerk_reduction": reduction("post_contact_arm_joint_jerk_rms_rad_s3"),
    }
    task_preserved = bool(
        parent_result.get("success")
        and candidate_result.get("success")
        and candidate_result.get("goal_crossed") == parent_result.get("goal_crossed")
        and float(candidate_result.get("ball_speed_mps", 0.0))
        >= float(parent_result.get("ball_speed_mps", 0.0)) - 1e-9
        and float(candidate_result.get("target_error_m", 99.0))
        <= float(parent_result.get("target_error_m", 99.0)) + 1e-9
    )
    safety_preserved = bool(
        candidate_result.get("post_kick_fall") is False
        and candidate_result.get("joint_limit_violation") is False
        and candidate_result.get("torque_limit_violation") is False
        and candidate_result.get("actuator_saturation") is False
        and float(candidate_result.get("support_foot_slip_m", 99.0)) <= 0.04 + 1e-9
        and candidate_quality.get("terminal_bilateral_support") is True
    )
    strict = bool(parent_strict_replay and candidate_strict_replay)
    reasons: list[str] = []
    if not task_preserved:
        reasons.append("upstream_task_outcome_not_preserved")
    if not safety_preserved:
        reasons.append("capture_step_safety_failed")
    if not strict:
        reasons.append("strict_replay_missing")
    for key, minimum in (
        ("pelvis_path_reduction", 0.30),
        ("backward_reversal_reduction", 0.50),
        ("joint_jerk_reduction", 0.50),
        ("leg_jerk_reduction", 0.50),
        ("waist_jerk_reduction", 0.50),
        ("arm_jerk_reduction", 0.50),
    ):
        if reductions[key] < minimum:
            reasons.append(f"{key}_below_gate")
    absolute_bounds = {
        "pelvis_displacement_m": (
            float(candidate_quality["post_contact_pelvis_displacement_m"]),
            0.30,
        ),
        "forward_peak_advance_m": (
            float(candidate_quality["post_contact_forward_peak_advance_m"]),
            0.30,
        ),
        "lateral_peak_return_m": (
            float(candidate_quality["post_contact_lateral_peak_return_m"]),
            0.12,
        ),
        "tail_wobble_index": (float(candidate_quality["tail_wobble_index"]), 0.015),
    }
    for key, (actual, maximum) in absolute_bounds.items():
        if not np.isfinite(actual) or actual > maximum + 1e-9:
            reasons.append(f"{key}_above_gate")
    return {
        "schema_version": "rosclaw.growth.g1_capture_step_gate.v1",
        "passed": not reasons,
        "task_preserved": task_preserved,
        "safety_preserved": safety_preserved,
        "strict_replay_preserved": strict,
        "reductions": reductions,
        "absolute_bounds": {
            key: {"actual": actual, "maximum": maximum}
            for key, (actual, maximum) in absolute_bounds.items()
        },
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
    "G1StructuredRecoveryCandidate",
    "G1StructuredRecoveryCase",
    "G1StructuredRecoveryCaseSpec",
    "G1StructuredRecoveryEvidence",
    "run_g1_structured_recovery_evaluation",
    "structured_recovery_specs",
]

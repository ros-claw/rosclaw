"""PARC triage adapter for the simultaneous two-G1 MuJoCo trace.

The adapter deliberately distinguishes measured signals, derived quantities,
proxies, and missing fields.  In particular, pelvis lateral position is not
silently represented as whole-body COM and a policy joint target is not
silently treated as an executed torque action.  Those distinctions keep an
attractive video from becoming invalid offline-RL or promotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.experience import FailureSignature
from rosclaw.growth.routing import DataProfile, GrowthProblemSignals, LearnerRoute, route_learners
from rosclaw.simforge.g1_recovery_quality import (
    G1AbsoluteRecoveryThresholds,
    G1RecoveryQuality,
    measure_g1_recovery_quality,
)


class FieldTruthStatus(StrEnum):
    MEASURED = "measured"
    DERIVED = "derived"
    PROXY = "proxy"
    MISSING = "missing"


@dataclass(frozen=True)
class FieldProvenance:
    field_id: str
    status: FieldTruthStatus
    source: str
    note: str = ""
    schema_version: str = "rosclaw.growth.field_provenance.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value


class FootballPhase(StrEnum):
    APPROACH = "approach"
    ALIGN = "align"
    LOAD = "load"
    SWING = "swing"
    CONTACT = "contact"
    FOLLOW_THROUGH = "follow_through"
    RECOVERY = "recovery"
    READY = "ready"


@dataclass(frozen=True)
class PhaseSegment:
    phase: FootballPhase
    start_index: int
    end_index_exclusive: int
    start_time_sec: float
    end_time_sec: float
    boundary_evidence: str
    confidence: float
    schema_version: str = "rosclaw.growth.football_phase_segment.v1"

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index_exclusive <= self.start_index:
            raise ValueError("phase segment indices must satisfy 0 <= start < end")
        if not all(math.isfinite(value) for value in (self.start_time_sec, self.end_time_sec)):
            raise ValueError("phase segment times must be finite")
        if self.end_time_sec < self.start_time_sec:
            raise ValueError("phase segment time must be ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("phase segment confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value


@dataclass(frozen=True)
class AbsoluteRecoveryAssessment:
    passed: bool
    reasons: tuple[str, ...]
    missing_promotion_fields: tuple[str, ...]
    thresholds: G1AbsoluteRecoveryThresholds
    strict_replay_verified: bool
    schema_version: str = "rosclaw.growth.g1_absolute_recovery_assessment.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["missing_promotion_fields"] = list(self.missing_promotion_fields)
        value["evidence_domain"] = "SIM_ONLY"
        value["activation_authorized"] = False
        value["hardware_authorized"] = False
        return value


@dataclass(frozen=True)
class G1CoupledTriageReport:
    source_path: str
    source_hash: str
    source_evidence_path: str | None
    source_evidence_hash: str | None
    role: str
    frame_count: int
    control_dt_sec: float
    contact_index: int
    contact_time_sec: float
    field_provenance: tuple[FieldProvenance, ...]
    phases: tuple[PhaseSegment, ...]
    recovery_quality: G1RecoveryQuality
    failure_signatures: tuple[FailureSignature, ...]
    data_profile: DataProfile
    learner_route: LearnerRoute
    absolute_recovery: AbsoluteRecoveryAssessment
    schema_version: str = "rosclaw.growth.g1_coupled_triage.v1"

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "source": {
                "trajectory_path": self.source_path,
                "trajectory_hash": self.source_hash,
                "evidence_path": self.source_evidence_path,
                "evidence_hash": self.source_evidence_hash,
            },
            "role": self.role,
            "frame_count": self.frame_count,
            "control_dt_sec": self.control_dt_sec,
            "events": {
                "contact_index": self.contact_index,
                "contact_time_sec": self.contact_time_sec,
            },
            "field_provenance": [item.to_dict() for item in self.field_provenance],
            "phases": [item.to_dict() for item in self.phases],
            "recovery_quality": self.recovery_quality.to_dict(),
            "failure_signatures": [item.to_dict() for item in self.failure_signatures],
            "data_profile": self.data_profile.to_dict(),
            "learner_route": self.learner_route.to_dict(),
            "absolute_recovery": self.absolute_recovery.to_dict(),
            "claims": {
                "parc_event_segmentation_completed": True,
                "offline_rl_transition_ready": self.data_profile.offline_rl_ready,
                "promotion_ready": self.absolute_recovery.passed,
                "real_hardware": False,
            },
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        }
        if include_hash:
            value["report_hash"] = self.report_hash
        return value


@dataclass(frozen=True)
class VerifiedCoupledEvidenceContext:
    evidence_path: str
    evidence_hash: str
    strict_replay: bool
    result: Mapping[str, Any]
    environment_hash: str | None = None
    runtime: Mapping[str, str] | None = None


def triage_g1_coupled_trajectory(
    trajectory_path: Path,
    *,
    role: str = "shooter",
    evidence_context: VerifiedCoupledEvidenceContext | None = None,
    thresholds: G1AbsoluteRecoveryThresholds | None = None,
) -> G1CoupledTriageReport:
    """Segment and diagnose one two-G1 trace without authorizing training."""

    if role not in {"passer", "shooter"}:
        raise ValueError("coupled G1 role must be passer or shooter")
    path = trajectory_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"coupled trajectory does not exist: {path}")
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    time, contact_index = _validate_and_contact(trajectory, role)
    quality_input = _quality_input(trajectory, role, contact_index)
    quality = measure_g1_recovery_quality(quality_input)
    phases = _segment_phases(time, contact_index, quality)
    failures = _failure_signatures(quality)
    provenance = _field_provenance(trajectory, role)
    action_triplet_ready = all(
        f"{role}_{suffix}" in trajectory
        for suffix in (
            "commanded_torque",
            "safety_projected_torque",
            "executed_torque",
        )
    )
    data_profile = DataProfile(
        has_state=True,
        has_executed_action=action_triplet_ready,
        has_next_state=True,
        has_reward_vector=False,
        has_cost_vector=False,
        has_kinematic_reference=True,
        has_chunk_feedback=False,
        fixed_dataset=True,
        online_rollout_allowed=False,
    )
    missing_safety = any(
        item.status is FieldTruthStatus.MISSING
        for item in provenance
        if item.field_id
        in {
            "whole_body_com",
            "support_foot_slip",
            "action_triplet_contract",
            "cost_vector",
        }
    )
    signals = GrowthProblemSignals(
        repeated_error=0.8 if failures else 0.0,
        local_physics_residual=0.8 if failures else 0.0,
        safety_model_complete=not missing_safety,
    )
    route = route_learners(signals, data_profile)
    assessment = _absolute_assessment(
        quality,
        result=evidence_context.result if evidence_context else None,
        strict_replay=evidence_context.strict_replay if evidence_context else False,
        support_slip=(
            float(np.max(np.asarray(trajectory[f"{role}_support_foot_slip"])))
            if f"{role}_support_foot_slip" in trajectory
            else None
        ),
        role=role,
        thresholds=thresholds or G1AbsoluteRecoveryThresholds(),
    )
    return G1CoupledTriageReport(
        source_path=str(path),
        source_hash=_file_hash(path),
        source_evidence_path=evidence_context.evidence_path if evidence_context else None,
        source_evidence_hash=evidence_context.evidence_hash if evidence_context else None,
        role=role,
        frame_count=len(time),
        control_dt_sec=float(np.median(np.diff(time))),
        contact_index=contact_index,
        contact_time_sec=float(time[contact_index]),
        field_provenance=provenance,
        phases=phases,
        recovery_quality=quality,
        failure_signatures=failures,
        data_profile=data_profile,
        learner_route=route,
        absolute_recovery=assessment,
    )


def measure_g1_coupled_recovery_quality(
    trajectory: Mapping[str, np.ndarray],
    *,
    role: str = "shooter",
) -> G1RecoveryQuality:
    """Measure a live coupled trace while retaining its explicit provenance gap."""

    if role not in {"passer", "shooter"}:
        raise ValueError("coupled G1 role must be passer or shooter")
    _time, contact_index = _validate_and_contact(trajectory, role)
    return measure_g1_recovery_quality(_quality_input(trajectory, role, contact_index))


def verified_coupled_evidence_context(
    evidence_path: Path,
    trajectory_path: Path,
) -> VerifiedCoupledEvidenceContext:
    """Bind a trajectory to a strict replay receipt and verify its file hash."""

    evidence = evidence_path.expanduser().resolve()
    trajectory = trajectory_path.expanduser().resolve()
    value = json.loads(evidence.read_text(encoding="utf-8"))
    candidates = value.get("cases")
    records = candidates if isinstance(candidates, list) else [value]
    matched: Mapping[str, Any] | None = None
    for record in records:
        if not isinstance(record, Mapping):
            continue
        candidate_path = record.get("trajectory_path", value.get("trajectory_path"))
        if candidate_path and Path(str(candidate_path)).expanduser().resolve() == trajectory:
            matched = record
            break
    if matched is None:
        raise ValueError("evidence receipt does not bind the requested trajectory")
    expected_hash = matched.get("trajectory_hash", value.get("trajectory_hash"))
    actual_hash = _file_hash(trajectory)
    if expected_hash != actual_hash:
        raise ValueError("trajectory hash does not match its evidence receipt")
    result = matched.get("result", value.get("result"))
    if not isinstance(result, Mapping):
        raise ValueError("evidence receipt does not contain a result mapping")
    runtime: Mapping[str, str] | None = None
    environment_hash = value.get("environment_hash")
    request_path = evidence.with_name("request.json")
    expected_request_hash = value.get("request_hash")
    if request_path.is_file() and _file_hash(request_path) == expected_request_hash:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request_runtime = request.get("runtime")
        if isinstance(request_runtime, Mapping):
            runtime = {str(key): str(item) for key, item in request_runtime.items()}
            expected_environment = canonical_hash(runtime)
            if request.get("environment_hash") != expected_environment:
                raise ValueError("request runtime does not match its environment hash")
            if environment_hash not in {None, expected_environment}:
                raise ValueError("evidence and request environment hashes disagree")
            environment_hash = expected_environment
    return VerifiedCoupledEvidenceContext(
        evidence_path=str(evidence),
        evidence_hash=_file_hash(evidence),
        strict_replay=matched.get("strict_replay", value.get("strict_replay")) is True,
        result=dict(result),
        environment_hash=str(environment_hash) if environment_hash else None,
        runtime=runtime,
    )


def _validate_and_contact(
    trajectory: Mapping[str, np.ndarray], role: str
) -> tuple[np.ndarray, int]:
    required = {
        "time",
        "ball_velocity",
        "ball_contact_role",
        f"{role}_pelvis_pose",
        f"{role}_torso_quaternion",
        f"{role}_joint_velocity",
        f"{role}_foot_contact",
    }
    missing = sorted(required.difference(trajectory))
    if missing:
        raise ValueError(f"coupled trajectory is missing fields: {missing}")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    if time.ndim != 1 or len(time) < 3 or not np.all(np.isfinite(time)):
        raise ValueError("coupled trajectory time must be a finite 1-D array")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("coupled trajectory time must be strictly increasing")
    count = len(time)
    expected = {
        "ball_velocity": (count, 6),
        "ball_contact_role": (count,),
        f"{role}_pelvis_pose": (count, 7),
        f"{role}_torso_quaternion": (count, 4),
        f"{role}_joint_velocity": (count, 29),
        f"{role}_foot_contact": (count, 2),
    }
    invalid = [name for name, shape in expected.items() if trajectory[name].shape != shape]
    if invalid:
        raise ValueError(f"coupled trajectory shapes are invalid: {invalid}")
    numeric = [trajectory[name] for name in expected if name != f"{role}_foot_contact"]
    if not all(np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("coupled trajectory contains non-finite values")
    role_code = 2 if role == "shooter" else 1
    contacts = np.flatnonzero(np.asarray(trajectory["ball_contact_role"]) == role_code)
    if not len(contacts):
        raise ValueError(f"coupled trajectory has no measured {role} ball contact")
    contact_index = int(contacts[0])
    if contact_index < 2 or contact_index >= count - 2:
        raise ValueError("ball contact is too close to a trajectory boundary")
    return time, contact_index


def _quality_input(
    trajectory: Mapping[str, np.ndarray], role: str, contact_index: int
) -> dict[str, np.ndarray]:
    pelvis = np.asarray(trajectory[f"{role}_pelvis_pose"], dtype=np.float64)
    contacts = np.asarray(trajectory[f"{role}_foot_contact"], dtype=bool)
    if f"{role}_contact_impulse" in trajectory:
        event = np.asarray(trajectory[f"{role}_contact_impulse"], dtype=np.float64)
    else:
        event = np.zeros(len(pelvis), dtype=np.float64)
        event[contact_index] = 1.0
    com_name = f"{role}_com_position"
    left_name = f"{role}_left_foot_position"
    right_name = f"{role}_right_foot_position"
    if all(name in trajectory for name in (com_name, left_name, right_name)):
        com = np.asarray(trajectory[com_name], dtype=np.float64)
        left = np.asarray(trajectory[left_name], dtype=np.float64)
        right = np.asarray(trajectory[right_name], dtype=np.float64)
        support_count = np.maximum(np.sum(contacts, axis=1), 1)
        support_y = (left[:, 1] * contacts[:, 0] + right[:, 1] * contacts[:, 1]) / support_count
        com_y_relative = com[:, 1] - support_y
    else:
        # Explicitly a pelvis-based lateral proxy; provenance records that COM
        # is absent, so this field cannot satisfy a COM or safety gate.
        com_y_relative = pelvis[:, 1] - pelvis[contact_index, 1]
    return {
        "time": np.asarray(trajectory["time"], dtype=np.float64),
        "torso_quaternion": np.asarray(trajectory[f"{role}_torso_quaternion"], dtype=np.float64),
        "pelvis_pose": pelvis,
        "joint_velocity": np.asarray(trajectory[f"{role}_joint_velocity"], dtype=np.float64),
        "com_y_relative": com_y_relative,
        "left_foot_contact": contacts[:, 0],
        "right_foot_contact": contacts[:, 1],
        # The quality metric needs only the contact edge.  No impulse magnitude
        # is invented and the missing physical impulse remains explicit below.
        "contact_impulse": event,
        "ball_velocity": np.asarray(trajectory["ball_velocity"], dtype=np.float64)[:, :3],
    }


def _segment_phases(
    time: np.ndarray,
    contact_index: int,
    quality: G1RecoveryQuality,
) -> tuple[PhaseSegment, ...]:
    contact_time = float(time[contact_index])
    if contact_time - float(time[0]) < 2.05 or float(time[-1]) - contact_time < 1.1:
        raise ValueError("trajectory is too short for the frozen football phase profile")

    def index_at(value: float) -> int:
        return int(np.searchsorted(time, value, side="left"))

    boundaries = [
        0,
        index_at(contact_time - 2.0),
        index_at(contact_time - 1.2),
        index_at(contact_time - 0.65),
        contact_index,
        contact_index + 1,
        index_at(contact_time + 0.60),
        (
            index_at(contact_time + quality.settling_time_sec)
            if quality.settling_time_sec is not None
            else len(time)
        ),
        len(time),
    ]
    boundaries = [min(max(value, 0), len(time)) for value in boundaries]
    if any(right <= left for left, right in zip(boundaries, boundaries[1:], strict=False)):
        raise ValueError("physical phase boundaries collapsed for this trajectory")
    phases = tuple(FootballPhase)
    evidence = (
        "contact_relative_profile",
        "contact_relative_profile",
        "contact_relative_profile",
        "pre_contact_motion_window",
        "measured_foot_ball_contact",
        "post_contact_follow_through_window",
        "derived_terminal_stability",
        "derived_terminal_stability",
    )
    confidence = (0.60, 0.65, 0.70, 0.75, 1.0, 0.75, 0.90, 0.90)
    return tuple(
        PhaseSegment(
            phase=phase,
            start_index=start,
            end_index_exclusive=end,
            start_time_sec=float(time[start]),
            end_time_sec=float(time[end - 1]),
            boundary_evidence=source,
            confidence=score,
        )
        for phase, start, end, source, score in zip(
            phases,
            boundaries[:-1],
            boundaries[1:],
            evidence,
            confidence,
            strict=True,
        )
    )


def _field_provenance(
    trajectory: Mapping[str, np.ndarray], role: str
) -> tuple[FieldProvenance, ...]:
    has_com = f"{role}_com_position" in trajectory
    has_feet = all(f"{role}_{side}_foot_position" in trajectory for side in ("left", "right"))
    has_slip = f"{role}_support_foot_slip" in trajectory
    has_impulse = f"{role}_contact_impulse" in trajectory
    has_action_triplet = all(
        f"{role}_{suffix}" in trajectory
        for suffix in (
            "commanded_torque",
            "safety_projected_torque",
            "executed_torque",
        )
    )
    return (
        FieldProvenance("time", FieldTruthStatus.MEASURED, "time"),
        FieldProvenance("pelvis_pose", FieldTruthStatus.MEASURED, f"{role}_pelvis_pose"),
        FieldProvenance("torso_orientation", FieldTruthStatus.MEASURED, f"{role}_torso_quaternion"),
        FieldProvenance("joint_velocity", FieldTruthStatus.MEASURED, f"{role}_joint_velocity"),
        FieldProvenance("foot_contact", FieldTruthStatus.MEASURED, f"{role}_foot_contact"),
        FieldProvenance("ball_velocity", FieldTruthStatus.MEASURED, "ball_velocity"),
        FieldProvenance("contact_event", FieldTruthStatus.MEASURED, "ball_contact_role"),
        FieldProvenance(
            "pelvis_planar_velocity",
            FieldTruthStatus.DERIVED,
            f"gradient({role}_pelvis_pose)",
        ),
        FieldProvenance(
            "whole_body_com",
            FieldTruthStatus.MEASURED if has_com else FieldTruthStatus.MISSING,
            f"{role}_com_position" if has_com else "",
            "" if has_com else "pelvis position is retained only as a declared proxy",
        ),
        FieldProvenance(
            "com_lateral_proxy",
            FieldTruthStatus.PROXY,
            f"{role}_pelvis_pose[:,1]",
        ),
        FieldProvenance(
            "contact_impulse_magnitude",
            FieldTruthStatus.MEASURED if has_impulse else FieldTruthStatus.MISSING,
            f"{role}_contact_impulse" if has_impulse else "",
            "" if has_impulse else "the trace contains an event but no impulse magnitude",
        ),
        FieldProvenance(
            "support_foot_slip",
            FieldTruthStatus.DERIVED if has_slip and has_feet else FieldTruthStatus.MISSING,
            f"{role}_support_foot_slip" if has_slip and has_feet else "",
            "" if has_slip and has_feet else "foot positions are absent from this trace",
        ),
        FieldProvenance(
            "action_triplet_contract",
            FieldTruthStatus.MEASURED if has_action_triplet else FieldTruthStatus.MISSING,
            (f"{role}_[commanded|safety_projected|executed]_torque" if has_action_triplet else ""),
            (
                ""
                if has_action_triplet
                else "joint targets and guarded torques differ; raw commanded torque is absent"
            ),
        ),
        FieldProvenance(
            "policy_joint_target",
            (
                FieldTruthStatus.MEASURED
                if f"{role}_policy_action" in trajectory
                else FieldTruthStatus.MISSING
            ),
            f"{role}_policy_action" if f"{role}_policy_action" in trajectory else "",
        ),
        FieldProvenance(
            "applied_guarded_torque",
            (
                FieldTruthStatus.MEASURED
                if f"{role}_joint_torque" in trajectory
                else FieldTruthStatus.MISSING
            ),
            f"{role}_joint_torque" if f"{role}_joint_torque" in trajectory else "",
        ),
        FieldProvenance("reward_vector", FieldTruthStatus.MISSING, ""),
        FieldProvenance("cost_vector", FieldTruthStatus.MISSING, ""),
    )


def _failure_signatures(quality: G1RecoveryQuality) -> tuple[FailureSignature, ...]:
    failures: list[FailureSignature] = []
    if quality.post_contact_backward_reversal_m > 0.25:
        failures.append(
            FailureSignature(
                primary_type="excessive_capture_steps",
                contributors=("backward_reversal", "pelvis_path_excess"),
                confidence=0.90,
                affected_capability_ids=("post_kick_recovery",),
                reusable_evidence_ids=("contact", "shot_outcome"),
                recommended_learner_ids=("adaptive_mpc", "residual_sac"),
            )
        )
    if quality.post_contact_pelvis_path_length_m > 0.70:
        failures.append(
            FailureSignature(
                primary_type="angular_momentum_residual",
                contributors=("pelvis_path_excess", "follow_through_unloading"),
                confidence=0.82,
                affected_capability_ids=("post_kick_recovery", "motion_naturalness"),
                reusable_evidence_ids=("contact",),
                recommended_learner_ids=("system_identification", "residual_sac"),
            )
        )
    if quality.settling_time_sec is None or quality.settling_time_sec > 1.50:
        failures.append(
            FailureSignature(
                primary_type="recovery_oscillation",
                contributors=("late_settling", "joint_velocity_residual"),
                confidence=0.92,
                affected_capability_ids=("post_kick_recovery", "ready_transition"),
                reusable_evidence_ids=("contact", "follow_through"),
                recommended_learner_ids=("motion_tracking", "residual_sac"),
            )
        )
    return tuple(failures)


def _absolute_assessment(
    quality: G1RecoveryQuality,
    *,
    result: Mapping[str, Any] | None,
    strict_replay: bool,
    support_slip: float | None,
    role: str,
    thresholds: G1AbsoluteRecoveryThresholds,
) -> AbsoluteRecoveryAssessment:
    reasons: list[str] = []
    missing: list[str] = []
    if support_slip is None or not math.isfinite(support_slip):
        missing.append("support_foot_slip_m")
    elif support_slip > thresholds.maximum_support_foot_slip_m:
        reasons.append("support_foot_slip_above_absolute_gate")
    if result is None:
        missing.extend(
            (
                "goal_crossed",
                "shot_peak_ball_speed_mps",
                "target_error_m",
                "post_kick_fall",
                "joint_limit_violation",
                "torque_limit_violation",
                "actuator_saturation",
            )
        )
    if quality.post_contact_backward_reversal_m > thresholds.maximum_backward_reversal_m:
        reasons.append("backward_reversal_above_absolute_gate")
    if quality.post_contact_pelvis_path_length_m > thresholds.maximum_pelvis_path_m:
        reasons.append("pelvis_path_above_absolute_gate")
    if quality.settling_time_sec is None:
        reasons.append("settling_time_missing")
    elif quality.settling_time_sec > thresholds.maximum_settling_time_sec:
        reasons.append("settling_time_above_absolute_gate")
    if quality.terminal_stable_duration_sec < thresholds.minimum_terminal_stable_duration_sec:
        reasons.append("terminal_stable_duration_below_gate")
    if not quality.terminal_bilateral_support:
        reasons.append("terminal_bilateral_support_missing")
    if result is not None:
        speed = result.get("shot_peak_ball_speed_mps", result.get("ball_speed_mps"))
        error = result.get("target_error_m")
        if not _finite_at_least(speed, thresholds.minimum_ball_speed_mps):
            reasons.append("ball_speed_below_absolute_gate")
        if not _finite_at_most(error, thresholds.maximum_target_error_m):
            reasons.append("target_error_above_absolute_gate")
        if result.get("goal_crossed") is not True:
            reasons.append("goal_not_crossed")
        for field in (
            "joint_limit_violation",
            "torque_limit_violation",
            "actuator_saturation",
        ):
            if result.get(field) is not False:
                reasons.append(f"{field}_not_cleared")
        # Older coupled traces do not record a dedicated post-kick-fall flag;
        # minimum pelvis height is useful diagnostic evidence but not equivalent.
        role_fall = result.get(f"{role}_post_kick_fall", result.get("post_kick_fall"))
        if role_fall is None:
            missing.append("post_kick_fall")
        elif role_fall is not False:
            reasons.append("post_kick_fall")
    if missing:
        reasons.append("promotion_fields_missing")
    if not strict_replay:
        reasons.append("strict_replay_missing")
    unique_reasons = tuple(dict.fromkeys(reasons))
    unique_missing = tuple(dict.fromkeys(missing))
    return AbsoluteRecoveryAssessment(
        passed=not unique_reasons,
        reasons=unique_reasons,
        missing_promotion_fields=unique_missing,
        thresholds=thresholds,
        strict_replay_verified=strict_replay,
    )


def _finite_at_least(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= threshold


def _finite_at_most(value: Any, threshold: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and value <= threshold


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "AbsoluteRecoveryAssessment",
    "FieldProvenance",
    "FieldTruthStatus",
    "FootballPhase",
    "G1CoupledTriageReport",
    "PhaseSegment",
    "VerifiedCoupledEvidenceContext",
    "triage_g1_coupled_trajectory",
    "measure_g1_coupled_recovery_quality",
    "verified_coupled_evidence_context",
]

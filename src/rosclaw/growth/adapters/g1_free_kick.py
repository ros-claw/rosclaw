"""P8 event-bound triage for the continuous G1 free-kick trajectory."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.adapters.g1_coupled import FootballPhase
from rosclaw.growth.experience import FailureSignature
from rosclaw.growth.routing import (
    DataProfile,
    GrowthProblemSignals,
    LearnerRoute,
    route_learners,
)
from rosclaw.simforge.g1_free_kick_showcase import G1FootballEventPhase
from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

_PHASE_MAP = {
    G1FootballEventPhase.APPROACH: FootballPhase.APPROACH,
    G1FootballEventPhase.ALIGN_BRAKE: FootballPhase.ALIGN,
    G1FootballEventPhase.PLANT_BRIDGE: FootballPhase.LOAD,
    G1FootballEventPhase.LOAD: FootballPhase.LOAD,
    G1FootballEventPhase.SWING: FootballPhase.SWING,
    G1FootballEventPhase.CONTACT: FootballPhase.CONTACT,
    G1FootballEventPhase.FOLLOW_THROUGH: FootballPhase.FOLLOW_THROUGH,
    G1FootballEventPhase.RECOVERY: FootballPhase.RECOVERY,
    G1FootballEventPhase.READY: FootballPhase.READY,
}


@dataclass(frozen=True)
class G1FreeKickPhaseMetrics:
    phase: FootballPhase
    start_index: int
    end_index_exclusive: int
    duration_sec: float
    peak_base_speed_mps: float
    joint_speed_rms_rad_s: float
    action_jerk_rms_rad_s3: float
    torque_rms_nm: float
    schema_version: str = "rosclaw.growth.g1_free_kick_phase_metrics.v1"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value


@dataclass(frozen=True)
class G1FreeKickTriageReport:
    source_path: str
    source_hash: str
    evidence_path: str
    evidence_hash: str
    strict_replay: bool
    evidence_passed: bool
    frame_count: int
    control_dt_sec: float
    phases: tuple[G1FreeKickPhaseMetrics, ...]
    failure_signatures: tuple[FailureSignature, ...]
    data_profile: DataProfile
    learner_route: LearnerRoute
    promotion_ready: bool
    schema_version: str = "rosclaw.growth.g1_free_kick_triage.v1"

    @property
    def report_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "source": {
                "trajectory_path": self.source_path,
                "trajectory_hash": self.source_hash,
                "evidence_path": self.evidence_path,
                "evidence_hash": self.evidence_hash,
            },
            "strict_replay": self.strict_replay,
            "evidence_passed": self.evidence_passed,
            "frame_count": self.frame_count,
            "control_dt_sec": self.control_dt_sec,
            "phases": [phase.to_dict() for phase in self.phases],
            "failure_signatures": [item.to_dict() for item in self.failure_signatures],
            "data_profile": self.data_profile.to_dict(),
            "learner_route": self.learner_route.to_dict(),
            "promotion_ready": self.promotion_ready,
            "claims": {
                "event_boundaries_measured": True,
                "offline_rl_transition_ready": self.data_profile.offline_rl_ready,
                "rendered_pixels_used": False,
                "activation_authorized": False,
                "real_hardware": False,
            },
            "activation_ceiling": "SIM_ONLY",
            "hardware_command_sent": False,
        }
        if include_hash:
            value["report_hash"] = self.report_hash
        return value


def triage_g1_free_kick_trajectory(
    *, trajectory_path: Path, evidence_path: Path
) -> G1FreeKickTriageReport:
    """Verify, segment and route one free-kick trace without promoting it."""

    trajectory_file = trajectory_path.expanduser().resolve()
    evidence_file = evidence_path.expanduser().resolve()
    evidence = _verified_evidence(evidence_file, trajectory_file)
    with np.load(trajectory_file, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    time = _validate_trajectory(trajectory)
    phases = _phase_metrics(trajectory, time)
    result = evidence["result"]
    assert isinstance(result, Mapping)
    failures = _failure_signatures(result)
    action_triplet_ready = _action_triplet_ready(trajectory, len(time))
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
    route = route_learners(
        GrowthProblemSignals(
            repeated_error=0.9 if failures else 0.0,
            local_physics_residual=0.9 if failures else 0.0,
            safety_model_complete=action_triplet_ready,
        ),
        data_profile,
    )
    strict_replay = evidence.get("strict_replay") is True
    evidence_passed = evidence.get("passed") is True
    return G1FreeKickTriageReport(
        source_path=str(trajectory_file),
        source_hash=_file_hash(trajectory_file),
        evidence_path=str(evidence_file),
        evidence_hash=_file_hash(evidence_file),
        strict_replay=strict_replay,
        evidence_passed=evidence_passed,
        frame_count=len(time),
        control_dt_sec=float(np.median(np.diff(time))),
        phases=phases,
        failure_signatures=failures,
        data_profile=data_profile,
        learner_route=route,
        promotion_ready=bool(strict_replay and evidence_passed and not failures),
    )


def write_g1_free_kick_triage(
    *, report: G1FreeKickTriageReport, output_path: Path, source_checkout: Path
) -> Path:
    """Write a canonical review artifact outside the source checkout."""

    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("free-kick triage evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("free-kick triage output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _verified_evidence(evidence_path: Path, trajectory_path: Path) -> Mapping[str, Any]:
    if not evidence_path.is_file():
        raise ValueError("free-kick evidence does not exist")
    if not trajectory_path.is_file():
        raise ValueError("free-kick trajectory does not exist")
    value = json.loads(evidence_path.read_text(encoding="utf-8"))
    bound = Path(str(value.get("trajectory_path", ""))).expanduser().resolve()
    if bound != trajectory_path:
        raise ValueError("free-kick evidence does not bind the requested trajectory")
    if value.get("trajectory_hash") != _file_hash(trajectory_path):
        raise ValueError("free-kick trajectory hash does not match its evidence")
    if value.get("strict_replay") is not True:
        raise ValueError("free-kick triage requires strict replay evidence")
    if not isinstance(value.get("result"), Mapping):
        raise ValueError("free-kick evidence does not contain a result mapping")
    return value


def _validate_trajectory(trajectory: Mapping[str, np.ndarray]) -> np.ndarray:
    required = {
        "time",
        "joint_position",
        "joint_velocity",
        "joint_torque",
        "policy_action",
        "pelvis_pose",
        "event_phase",
    }
    missing = sorted(required.difference(trajectory))
    if missing:
        raise ValueError(f"free-kick trajectory is missing fields: {missing}")
    time = np.asarray(trajectory["time"], dtype=np.float64)
    if time.ndim != 1 or len(time) < 10 or not np.all(np.isfinite(time)):
        raise ValueError("free-kick time must be a finite 1-D array")
    if not np.all(np.diff(time) > 0.0):
        raise ValueError("free-kick time must be strictly increasing")
    count = len(time)
    shapes = {
        "joint_position": (count, 29),
        "joint_velocity": (count, 29),
        "joint_torque": (count, 29),
        "policy_action": (count, 29),
        "pelvis_pose": (count, 7),
        "event_phase": (count,),
    }
    invalid = [name for name, shape in shapes.items() if trajectory[name].shape != shape]
    if invalid:
        raise ValueError(f"free-kick trajectory shapes are invalid: {invalid}")
    if not all(np.all(np.isfinite(trajectory[name])) for name in shapes):
        raise ValueError("free-kick trajectory contains non-finite values")
    phase_ids = np.asarray(trajectory["event_phase"], dtype=np.int64)
    if not set(np.unique(phase_ids)).issubset({int(item) for item in G1FootballEventPhase}):
        raise ValueError("free-kick trajectory contains an unknown event phase")
    if int(G1FootballEventPhase.CONTACT) not in phase_ids:
        raise ValueError("free-kick trajectory is missing the measured contact phase")
    return time


def _phase_metrics(
    trajectory: Mapping[str, np.ndarray], time: np.ndarray
) -> tuple[G1FreeKickPhaseMetrics, ...]:
    phase_ids = np.asarray(trajectory["event_phase"], dtype=np.int64)
    mapped_values: list[str] = []
    contact_seen = False
    for value in phase_ids:
        source_phase = G1FootballEventPhase(int(value))
        contact_seen = contact_seen or source_phase is G1FootballEventPhase.CONTACT
        mapped_phase = _PHASE_MAP[source_phase]
        # v1 collectors returned to SWING after the one-frame CONTACT marker.
        # Canonical learning phases are monotonic, so that legacy tail is
        # explicitly reclassified as follow-through after observed contact.
        if contact_seen and mapped_phase is FootballPhase.SWING:
            mapped_phase = FootballPhase.FOLLOW_THROUGH
        mapped_values.append(mapped_phase.value)
    mapped = np.asarray(mapped_values, dtype="U32")
    # Merge adjacent source phases that intentionally map to the same learning phase.
    starts = np.r_[0, np.flatnonzero(mapped[1:] != mapped[:-1]) + 1]
    ends = np.r_[starts[1:], len(mapped)]
    pelvis = np.asarray(trajectory["pelvis_pose"], dtype=np.float64)
    base_velocity = np.gradient(pelvis[:, :2], time, axis=0)
    joint_velocity = np.asarray(trajectory["joint_velocity"], dtype=np.float64)
    action = np.asarray(trajectory["policy_action"], dtype=np.float64)
    torque = np.asarray(trajectory["joint_torque"], dtype=np.float64)
    action_velocity = np.gradient(action, time, axis=0)
    action_acceleration = np.gradient(action_velocity, time, axis=0)
    action_jerk = np.gradient(action_acceleration, time, axis=0)
    records: list[G1FreeKickPhaseMetrics] = []
    seen: set[str] = set()
    for start, end in zip(starts, ends, strict=True):
        phase_value = str(mapped[start])
        # CONTACT is intentionally a one-frame event; every other phase must
        # appear only once in a monotonic rollout.
        if phase_value in seen and phase_value != FootballPhase.CONTACT.value:
            raise ValueError(f"free-kick phase is non-monotonic: {phase_value}")
        seen.add(phase_value)
        span = slice(int(start), int(end))
        records.append(
            G1FreeKickPhaseMetrics(
                phase=FootballPhase(phase_value),
                start_index=int(start),
                end_index_exclusive=int(end),
                duration_sec=max(0.0, float(time[end - 1] - time[start])),
                peak_base_speed_mps=float(np.max(np.linalg.norm(base_velocity[span], axis=1))),
                joint_speed_rms_rad_s=float(np.sqrt(np.mean(np.square(joint_velocity[span])))),
                action_jerk_rms_rad_s3=float(np.sqrt(np.mean(np.square(action_jerk[span])))),
                torque_rms_nm=float(np.sqrt(np.mean(np.square(torque[span])))),
            )
        )
    return tuple(records)


def _failure_signatures(result: Mapping[str, Any]) -> tuple[FailureSignature, ...]:
    failures: list[FailureSignature] = []
    if result.get("perceptual_continuity_passed") is False:
        failures.append(
            FailureSignature(
                primary_type="perceptual_handoff_stall",
                contributors=("pelvis_forward_speed_loss", "approach_strike_phase_alignment"),
                confidence=0.98,
                affected_capability_ids=("continuous_run_to_strike",),
                reusable_evidence_ids=("align_brake", "plant_bridge", "load", "swing"),
                recommended_learner_ids=("motion_tracking", "residual_sac"),
            )
        )
    if result.get("loft_teacher_executed") is True:
        failures.append(
            FailureSignature(
                primary_type="sim_teacher_distillation_required",
                contributors=("operational_space_teacher", "policy_capability_gap"),
                confidence=1.0,
                affected_capability_ids=("lofted_shot_control",),
                reusable_evidence_ids=("swing", "contact", "follow_through"),
                recommended_learner_ids=("motion_tracking", "iql"),
            )
        )
    error = _finite_float(result.get("goal_plane_target_error_m"))
    radius = _finite_float(result.get("precision_radius_m"))
    corner = _finite_float(
        result.get("declared_corner_distance_m", result.get("lower_corner_distance_m"))
    )
    declared_corner = str(result.get("declared_target_corner", ""))
    crossing = result.get("goal_crossing_xyz_m")
    crossing_height = (
        _finite_float(crossing[2])
        if isinstance(crossing, (list, tuple)) and len(crossing) == 3
        else None
    )
    if "upper" in declared_corner and crossing_height is not None:
        # An upper-corner miss is not merely generic target noise.  A low
        # launch needs contact-height/velocity learning, while lateral-only
        # error can remain with the precision learner.
        target_height = 1.35
        if crossing_height < target_height - 0.16:
            failures.append(
                FailureSignature(
                    primary_type="insufficient_ballistic_loft",
                    contributors=(
                        "ball_launch_vertical_velocity",
                        "foot_ball_contact_height",
                        "support_leg_coordination",
                    ),
                    confidence=0.99,
                    affected_capability_ids=("upper_corner_ballistics",),
                    reusable_evidence_ids=("swing", "contact", "follow_through"),
                    recommended_learner_ids=("motion_tracking", "residual_sac"),
                )
            )
    if error is None or radius is None or error > radius:
        failures.append(
            FailureSignature(
                primary_type="contact_mode_precision",
                contributors=("hybrid_contact_branch", "terminal_approach_state"),
                confidence=0.95,
                affected_capability_ids=("shot_precision", "approach_strike_transition"),
                reusable_evidence_ids=("align_brake", "load", "swing", "contact"),
                recommended_learner_ids=("residual_sac", "motion_tracking"),
            )
        )
    if corner is None or corner > 0.25:
        failures.append(
            FailureSignature(
                primary_type="declared_corner_miss",
                contributors=("foot_ball_contact_normal", "support_phase_alignment"),
                confidence=0.95,
                affected_capability_ids=("dead_corner_targeting",),
                reusable_evidence_ids=("swing", "contact"),
                recommended_learner_ids=("residual_sac", "adaptive_mpc"),
            )
        )
    if result.get("actuator_saturation") is not False:
        failures.append(
            FailureSignature(
                primary_type="authority_projection_required",
                contributors=("upstream_torque_demand", "body_model_transfer"),
                confidence=1.0,
                affected_capability_ids=("safe_fullbody_control",),
                reusable_evidence_ids=("approach", "align_brake", "plant_bridge"),
                recommended_learner_ids=("system_identification", "motion_tracking"),
            )
        )
    return tuple(failures)


def _action_triplet_ready(trajectory: Mapping[str, np.ndarray], count: int) -> bool:
    names = (
        "commanded_torque",
        "safety_projected_torque",
        "executed_torque",
        "torque_projection_applied",
    )
    if not all(name in trajectory for name in names):
        return False
    commanded = np.asarray(trajectory["commanded_torque"], dtype=np.float64)
    projected = np.asarray(trajectory["safety_projected_torque"], dtype=np.float64)
    executed = np.asarray(trajectory["executed_torque"], dtype=np.float64)
    projection = np.asarray(trajectory["torque_projection_applied"], dtype=bool)
    if any(array.shape != (count, 29) for array in (commanded, projected, executed)):
        raise ValueError("free-kick action triplet arrays must be [T, 29]")
    if projection.shape != (count,):
        raise ValueError("free-kick torque projection flags must be [T]")
    if not all(np.all(np.isfinite(array)) for array in (commanded, projected, executed)):
        raise ValueError("free-kick action triplet contains non-finite values")
    limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    if np.any(np.abs(projected) > limits[None, :] + 1e-9):
        raise ValueError("free-kick projected torque exceeds the hard authority limit")
    expected_projection = np.any(np.abs(commanded - projected) > 1e-12, axis=1)
    if not np.array_equal(projection, expected_projection):
        raise ValueError("free-kick projection flags disagree with the action triplet")
    if not np.allclose(executed, projected, rtol=1e-9, atol=1e-9):
        raise ValueError("free-kick executed torque does not match safety-projected torque")
    return True


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


__all__ = [
    "G1FreeKickPhaseMetrics",
    "G1FreeKickTriageReport",
    "triage_g1_free_kick_trajectory",
    "write_g1_free_kick_triage",
]

"""State-routed episodic contact dynamics for SIM-only G1 football.

The memory keeps one locally identified force-to-launch model per qualified
pre-contact state.  Runtime routing is proprioceptive: planner seeds are audit
labels only and are never used to select an action.  States outside the
rehearsed support radius fail closed with zero additive torque.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_impulse_actor import (
    g1_ballistic_contact_impulse_context_hash,
)

_SCHEMA = "rosclaw.growth.g1_episodic_contact_memory.v1"
_PROTOTYPE_SCHEMA = "rosclaw.growth.g1_episodic_contact_prototype.v1"
_OBSERVATION_NAMES = (
    "ankle_ball_dx_m",
    "ankle_ball_dy_m",
    "ankle_ball_dz_m",
    "ankle_vx_mps",
    "ankle_vy_mps",
    "ankle_vz_mps",
    "pelvis_vx_mps",
    "pelvis_vy_mps",
    "pelvis_vz_mps",
    "torso_roll_rad",
    "torso_pitch_rad",
)
_OBSERVATION_SCALES = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.3, 0.3, 0.3, 0.15, 0.15)
_MAX_TRAJECTORY_BYTES = 2 * 1024 * 1024 * 1024


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _roll_pitch(quaternion_wxyz: np.ndarray) -> tuple[float, float]:
    w, x, y, z = map(float, quaternion_wxyz)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch


def g1_episodic_contact_observation(
    *,
    data: Any,
    right_ankle_body_id: int,
    torso_body_id: int,
    ball_position: np.ndarray,
) -> np.ndarray:
    """Read the causal pre-action state shared by training and runtime."""

    ball = np.asarray(ball_position, dtype=np.float64)
    if ball.shape != (3,) or not np.all(np.isfinite(ball)):
        raise ValueError("episodic contact observation requires a finite ball position")
    roll, pitch = _roll_pitch(np.asarray(data.xquat[torso_body_id], dtype=np.float64))
    observation = np.concatenate(
        (
            np.asarray(data.xpos[right_ankle_body_id], dtype=np.float64) - ball,
            np.asarray(data.cvel[right_ankle_body_id][3:6], dtype=np.float64),
            np.asarray(data.qvel[:3], dtype=np.float64),
            (roll, pitch),
        )
    )
    if observation.shape != (len(_OBSERVATION_NAMES),) or not np.all(np.isfinite(observation)):
        raise ValueError("episodic contact observation must contain 11 finite values")
    return observation


def g1_episodic_contact_context_hash(
    *,
    flow_config: dict[str, Any],
    goal_spec: dict[str, Any],
    runup_config: dict[str, Any],
    sonic_runup_config: dict[str, Any] | None,
    approach_strike_candidate_hash: str | None,
) -> str:
    """Bind invariant task/controller state while leaving target and seed routable."""

    flow = dict(flow_config)
    flow.pop("episodic_contact_memory_hash", None)
    sonic = None if sonic_runup_config is None else dict(sonic_runup_config)
    if sonic is not None:
        sonic.pop("planner_seed", None)
    return g1_ballistic_contact_impulse_context_hash(
        flow_config=flow,
        goal_spec=goal_spec,
        runup_config=runup_config,
        sonic_runup_config=sonic,
        approach_strike_candidate_hash=approach_strike_candidate_hash,
        target_conditioned=True,
    )


@dataclass(frozen=True)
class G1EpisodicContactPrototype:
    planner_seed_audit_label: int
    observation: tuple[float, ...]
    forward_dynamics_weight_matrix: tuple[tuple[float, float], ...]
    source_evidence_hashes: tuple[str, ...]
    reference_forward_ball_speed_mps: float
    minimum_lateral_force_n: float
    maximum_lateral_force_n: float
    minimum_vertical_force_n: float
    maximum_vertical_force_n: float
    minimum_supported_lateral_launch_speed_mps: float
    maximum_supported_lateral_launch_speed_mps: float
    minimum_supported_vertical_launch_speed_mps: float
    maximum_supported_vertical_launch_speed_mps: float
    forward_dynamics_fit_rmse_mps: float
    schema_version: str = _PROTOTYPE_SCHEMA

    def __post_init__(self) -> None:
        if self.planner_seed_audit_label < 0:
            raise ValueError("episodic contact prototype seed label must be non-negative")
        observation = np.asarray(self.observation, dtype=np.float64)
        weights = np.asarray(self.forward_dynamics_weight_matrix, dtype=np.float64)
        if observation.shape != (len(_OBSERVATION_NAMES),) or not np.all(np.isfinite(observation)):
            raise ValueError("episodic contact observation must contain 11 finite values")
        if weights.shape != (3, 2) or not np.all(np.isfinite(weights)):
            raise ValueError("episodic contact forward dynamics must have finite shape (3, 2)")
        if len(self.source_evidence_hashes) < 6 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("episodic contact prototype requires six bound safe probes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("episodic contact prototype evidence hashes must be unique")
        if not (
            2.0 <= self.reference_forward_ball_speed_mps <= 20.0
            and -250.0 <= self.minimum_lateral_force_n < self.maximum_lateral_force_n <= 250.0
            and -250.0 <= self.minimum_vertical_force_n < self.maximum_vertical_force_n <= 250.0
            and self.minimum_supported_lateral_launch_speed_mps
            < self.maximum_supported_lateral_launch_speed_mps
            and self.minimum_supported_vertical_launch_speed_mps
            < self.maximum_supported_vertical_launch_speed_mps
            and 0.0 <= self.forward_dynamics_fit_rmse_mps <= 5.0
        ):
            raise ValueError("episodic contact prototype support envelope is invalid")
        if self.schema_version != _PROTOTYPE_SCHEMA:
            raise ValueError("episodic contact prototype schema is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation": list(self.observation),
            "forward_dynamics_weight_matrix": [
                list(row) for row in self.forward_dynamics_weight_matrix
            ],
            "source_evidence_hashes": list(self.source_evidence_hashes),
        }


@dataclass(frozen=True)
class G1EpisodicContactMemory:
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    prototypes: tuple[G1EpisodicContactPrototype, ...]
    rejected_context_seed_labels: tuple[int, ...]
    observation_feature_scales: tuple[float, ...]
    maximum_context_distance: float
    minimum_prototype_distance: float
    maximum_foot_ball_distance_m: float
    start_policy_frame: int
    end_policy_frame: int
    foot_strike_point_offset_m: tuple[float, float, float]
    ridge_regularization: float
    safe_probe_count: int
    rejected_probe_count: int
    training_target_count: int
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        hashes = (self.body_hash, self.implementation_hash, self.experiment_context_hash)
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("episodic contact memory requires SHA-256 provenance")
        if len(self.source_evidence_hashes) < 24 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("episodic contact memory requires 24 bound evidence hashes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("episodic contact memory evidence hashes must be unique")
        if len(self.prototypes) < 2 or len(
            {item.planner_seed_audit_label for item in self.prototypes}
        ) != len(self.prototypes):
            raise ValueError("episodic contact memory requires distinct state prototypes")
        prototype_hashes = {
            value for item in self.prototypes for value in item.source_evidence_hashes
        }
        if not prototype_hashes.issubset(set(self.source_evidence_hashes)):
            raise ValueError("episodic contact prototype evidence is not memory-bound")
        scales = np.asarray(self.observation_feature_scales, dtype=np.float64)
        if (
            scales.shape != (len(_OBSERVATION_NAMES),)
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            raise ValueError("episodic contact observation scales are invalid")
        if not 0.05 <= self.maximum_context_distance <= 1.0:
            raise ValueError("episodic contact support radius must be in [0.05, 1.0]")
        if not self.minimum_prototype_distance > self.maximum_context_distance:
            raise ValueError("episodic contact prototypes overlap the support radius")
        if not self.rejected_context_seed_labels:
            raise ValueError("episodic contact memory requires a rejected context anchor")
        if set(self.rejected_context_seed_labels) & {
            item.planner_seed_audit_label for item in self.prototypes
        }:
            raise ValueError("episodic contact context cannot be both supported and rejected")
        if not 0.15 <= self.maximum_foot_ball_distance_m <= 0.30:
            raise ValueError("episodic contact proximity gate is invalid")
        if not 150 <= self.start_policy_frame < self.end_policy_frame <= 430:
            raise ValueError("episodic contact policy window is invalid")
        if len(self.foot_strike_point_offset_m) != 3 or not all(
            math.isfinite(value) for value in self.foot_strike_point_offset_m
        ):
            raise ValueError("episodic contact strike point is invalid")
        if not 0.0 < self.ridge_regularization <= 10.0:
            raise ValueError("episodic contact ridge regularization is invalid")
        if self.safe_probe_count < 12 or self.rejected_probe_count < 2:
            raise ValueError("episodic contact memory lacks safe/rejected rehearsal")
        if self.training_target_count < 1:
            raise ValueError("episodic contact memory has no bound training target")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.online_hot_swap_allowed
            or self.schema_version != _SCHEMA
        ):
            raise ValueError("episodic contact memory must remain unpromoted SIM_ONLY")

    @property
    def memory_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "prototypes": [item.to_dict() for item in self.prototypes],
            "rejected_context_seed_labels": list(self.rejected_context_seed_labels),
            "observation_feature_scales": list(self.observation_feature_scales),
            "foot_strike_point_offset_m": list(self.foot_strike_point_offset_m),
            "observation_feature_names": list(_OBSERVATION_NAMES),
            "routing": "nearest_pre_action_state_normalized_rms_fail_closed",
            "controller": "local_ridge_forward_dynamics_regularized_inverse",
            "direct_joint_torque_output": True,
            "sealed_generalization_evidence": False,
        }
        if include_hash:
            value["memory_hash"] = self.memory_hash
        return value


@dataclass(frozen=True)
class G1EpisodicContactEffect:
    torque: NDArray[np.float64]
    lateral_force_n: float
    vertical_force_n: float
    foot_lateral_speed_mps: float
    foot_vertical_speed_mps: float
    desired_lateral_launch_speed_mps: float
    desired_vertical_launch_speed_mps: float
    selected_context_seed_label: int | None
    context_distance: float
    active: bool
    context_supported: bool = True
    launch_envelope_supported: bool = True


def load_g1_episodic_contact_memory(path: Path) -> G1EpisodicContactMemory:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = str(payload.pop("memory_hash", ""))
    for name in (
        "observation_feature_names",
        "routing",
        "controller",
        "direct_joint_torque_output",
        "sealed_generalization_evidence",
    ):
        payload.pop(name, None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["rejected_context_seed_labels"] = tuple(payload["rejected_context_seed_labels"])
    payload["observation_feature_scales"] = tuple(payload["observation_feature_scales"])
    payload["foot_strike_point_offset_m"] = tuple(payload["foot_strike_point_offset_m"])
    prototypes = []
    for raw in payload["prototypes"]:
        value = dict(raw)
        value["observation"] = tuple(value["observation"])
        value["forward_dynamics_weight_matrix"] = tuple(
            tuple(float(item) for item in row) for row in value["forward_dynamics_weight_matrix"]
        )
        value["source_evidence_hashes"] = tuple(value["source_evidence_hashes"])
        prototypes.append(G1EpisodicContactPrototype(**value))
    payload["prototypes"] = tuple(prototypes)
    memory = G1EpisodicContactMemory(**payload)
    if memory.memory_hash != expected:
        raise ValueError("episodic contact memory hash mismatch")
    return memory


def _pre_action_observation(trace: Any, *, maximum_distance_m: float) -> np.ndarray:
    required = {
        "loft_teacher_active",
        "loft_teacher_pre_action_observation",
        "loft_teacher_pre_action_observation_valid",
    }
    if not required.issubset(trace.files):
        raise ValueError("episodic contact trajectory lacks pre-action observation channels")
    active = np.asarray(trace["loft_teacher_active"], dtype=np.bool_)
    valid = np.asarray(trace["loft_teacher_pre_action_observation_valid"], dtype=np.bool_)
    observations = np.asarray(trace["loft_teacher_pre_action_observation"], dtype=np.float64)
    if active.shape != valid.shape or observations.shape != (
        len(active),
        len(_OBSERVATION_NAMES),
    ):
        raise ValueError("episodic contact exact pre-action trace shapes are invalid")
    indices = np.flatnonzero(active & valid)
    if indices.size == 0:
        raise ValueError("episodic contact teacher probe has no exact pre-action state")
    observation = observations[int(indices[0])]
    if float(np.linalg.norm(observation[:3])) > maximum_distance_m + 0.20:
        raise ValueError("episodic contact pre-action frame is outside the strike neighborhood")
    if observation.shape != (len(_OBSERVATION_NAMES),) or not np.all(np.isfinite(observation)):
        raise ValueError("episodic contact pre-action observation is invalid")
    return observation


def _fit_forward_dynamics(
    rows: list[dict[str, Any]], ridge_regularization: float
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    launch = np.asarray(
        [[float(row["launch"][1]), float(row["launch"][2])] for row in rows],
        dtype=np.float64,
    )
    force = np.asarray(
        [[float(row["lateral_force"]), float(row["vertical_force"])] for row in rows],
        dtype=np.float64,
    )
    if np.linalg.matrix_rank(np.column_stack((np.ones(len(rows)), force))) < 3:
        raise ValueError("episodic contact force probes lack two-axis coverage")
    if np.ptp(force[:, 0]) < 5.0 or np.ptp(force[:, 1]) < 5.0:
        raise ValueError("episodic contact force support is degenerate")
    mean = np.mean(force, axis=0)
    scale = np.std(force, axis=0)
    if np.any(scale < 1e-4):
        raise ValueError("episodic contact force distribution is degenerate")
    design = np.column_stack((np.ones(len(rows)), (force - mean) / scale))
    penalty = np.diag((0.0, ridge_regularization, ridge_regularization))
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ launch)
    slopes = coefficients[1:, :] / scale[:, None]
    intercept = coefficients[0, :] - mean @ slopes
    weights = np.vstack((intercept, slopes))
    predictions = np.column_stack((np.ones(len(rows)), force)) @ weights
    rmse = float(np.sqrt(np.mean(np.square(predictions - launch))))
    condition = float(np.linalg.cond(slopes))
    if not math.isfinite(rmse) or rmse > 5.0 or not math.isfinite(condition) or condition > 1e4:
        raise ValueError("episodic contact local dynamics are ill-conditioned")
    return weights, rmse, launch, force


def derive_g1_episodic_contact_memory(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    maximum_context_distance: float = 0.50,
    ridge_regularization: float = 0.05,
) -> G1EpisodicContactMemory:
    """Build local contact experts from strict multi-context probe islands."""

    if len(evidence_paths) < 24:
        raise ValueError("episodic contact memory requires at least 24 probes")
    if not 0.05 <= maximum_context_distance <= 1.0:
        raise ValueError("episodic contact support radius must be in [0.05, 1.0]")
    if not 0.0 < ridge_regularization <= 10.0:
        raise ValueError("episodic contact ridge regularization must be in (0, 10]")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("episodic contact memory must remain outside source checkout")
    if output.exists():
        raise FileExistsError("episodic contact memory output already exists")

    rows: list[dict[str, Any]] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: list[str] = []
    policy_starts: list[int] = []
    policy_ends: list[int] = []
    distance_gates: list[float] = []
    targets: set[tuple[float, float]] = set()
    paths: set[Path] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        if path in paths:
            raise ValueError("episodic contact evidence paths must be unique")
        paths.add(path)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("episodic contact memory requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if (
            not trajectory.is_file()
            or not 1 <= trajectory.stat().st_size <= _MAX_TRAJECTORY_BYTES
            or evidence.get("trajectory_hash") != _file_hash(trajectory)
        ):
            raise ValueError("episodic contact trajectory binding is invalid")
        evidence_hash = _file_hash(path)
        if evidence_hash in source_hashes:
            raise ValueError("episodic contact evidence contents must be unique")
        source_hashes.append(evidence_hash)
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        goal = dict(evidence.get("goal_spec", {}))
        sonic = dict(evidence.get("sonic_runup_config", {}))
        seed = int(sonic.get("planner_seed", -1))
        if seed < 0:
            raise ValueError("episodic contact evidence requires a planner seed audit label")
        context_hashes.add(
            g1_episodic_contact_context_hash(
                flow_config=flow,
                goal_spec=goal,
                runup_config=dict(evidence.get("runup_config", {})),
                sonic_runup_config=sonic,
                approach_strike_candidate_hash=evidence.get("approach_strike_candidate_hash"),
            )
        )
        targets.add((float(goal["target_y_m"]), float(goal["target_z_m"])))
        result = dict(evidence.get("result", {}))
        claims = dict(evidence.get("claims", {}))
        teacher = claims.get("sim_only_operational_space_loft_teacher") is True
        baseline = bool(
            not teacher
            and result.get("loft_teacher_executed") is False
            and result.get("ballistic_contact_impulse_actor_executed") is False
        )
        if not teacher and not baseline:
            raise ValueError("episodic contact rows must be teacher probes or zero-force baselines")
        launch = result.get("ball_launch_velocity_xyz_mps")
        if (
            not isinstance(launch, list)
            or len(launch) != 3
            or not all(
                isinstance(item, (int, float)) and math.isfinite(float(item)) for item in launch
            )
        ):
            raise ValueError("episodic contact evidence requires measured launch velocity")
        authority = float(result.get("contact_task_authority_scale_min", 1.0))
        hard_safe = bool(
            result.get("kick_contact_observed") is True
            and result.get("perceptual_continuity_passed") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
            and result.get("actuator_saturation") is not True
            and result.get("torque_authority_projection_qualified", True) is True
            and math.isfinite(authority)
            and authority >= 0.95
        )
        lateral_force = 0.0
        vertical_force = 0.0
        observation: np.ndarray | None = None
        maximum_distance = float(flow["shot_loft_teacher_max_foot_ball_distance_m"])
        with np.load(trajectory, allow_pickle=False) as trace:
            active = np.asarray(trace["loft_teacher_active"], dtype=np.bool_)
            lateral_trace = np.asarray(trace["loft_teacher_lateral_force_n"], dtype=np.float64)
            vertical_trace = np.asarray(trace["loft_teacher_force_n"], dtype=np.float64)
            if teacher:
                if not np.any(active):
                    raise ValueError("episodic contact teacher probe never activated")
                lateral_force = float(
                    lateral_trace[active][np.argmax(np.abs(lateral_trace[active]))]
                )
                vertical_force = float(
                    vertical_trace[active][np.argmax(np.abs(vertical_trace[active]))]
                )
                observation = _pre_action_observation(trace, maximum_distance_m=maximum_distance)
            elif np.any(active) or np.any(lateral_trace) or np.any(vertical_trace):
                raise ValueError("episodic contact zero-force baseline is contaminated")
        # A zero-force baseline intentionally has the teacher disabled, so its
        # serialized teacher window/proximity gate are not part of the learned
        # controller contract.  Bind those values only from active probes.
        if teacher:
            policy_starts.append(int(flow["shot_loft_teacher_start_policy_frame"]))
            policy_ends.append(int(flow["shot_loft_teacher_end_policy_frame"]))
            distance_gates.append(maximum_distance)
        rows.append(
            {
                "seed": seed,
                "teacher": teacher,
                "hard_safe": hard_safe,
                "evidence_hash": evidence_hash,
                "launch": launch,
                "lateral_force": lateral_force,
                "vertical_force": vertical_force,
                "observation": observation,
            }
        )
    if len(body_hashes) != 1 or len(implementation_hashes) != 1 or len(context_hashes) != 1:
        raise ValueError("episodic contact probe provenance or invariant contexts disagree")
    if not policy_starts or not policy_ends or not distance_gates:
        raise ValueError("episodic contact memory has no active teacher contract")
    if max(policy_starts) != min(policy_starts) or max(policy_ends) != min(policy_ends):
        raise ValueError("episodic contact policy windows disagree")
    if not math.isclose(max(distance_gates), min(distance_gates), abs_tol=1e-12):
        raise ValueError("episodic contact proximity gates disagree")

    prototypes: list[G1EpisodicContactPrototype] = []
    rejected_seeds: list[int] = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        group = [row for row in rows if row["seed"] == seed]
        safe = [row for row in group if row["hard_safe"]]
        baselines = [row for row in safe if not row["teacher"]]
        teacher_safe = [row for row in safe if row["teacher"]]
        if len(baselines) != 1 or len(teacher_safe) < 5:
            rejected_seeds.append(seed)
            continue
        try:
            weights, rmse, launch, force = _fit_forward_dynamics(safe, ridge_regularization)
        except ValueError:
            rejected_seeds.append(seed)
            continue
        observations = np.asarray([row["observation"] for row in teacher_safe], dtype=np.float64)
        if observations.shape != (len(teacher_safe), len(_OBSERVATION_NAMES)):
            raise ValueError("episodic contact prototype observations are invalid")
        margin = np.maximum(0.05, 0.10 * np.ptp(launch, axis=0))
        prototypes.append(
            G1EpisodicContactPrototype(
                planner_seed_audit_label=seed,
                observation=tuple(float(item) for item in np.median(observations, axis=0)),
                forward_dynamics_weight_matrix=tuple(
                    tuple(float(item) for item in row) for row in weights
                ),
                source_evidence_hashes=tuple(row["evidence_hash"] for row in safe),
                reference_forward_ball_speed_mps=float(
                    np.median([float(row["launch"][0]) for row in safe])
                ),
                minimum_lateral_force_n=float(np.min(force[:, 0])),
                maximum_lateral_force_n=float(np.max(force[:, 0])),
                minimum_vertical_force_n=float(np.min(force[:, 1])),
                maximum_vertical_force_n=float(np.max(force[:, 1])),
                minimum_supported_lateral_launch_speed_mps=float(np.min(launch[:, 0]) - margin[0]),
                maximum_supported_lateral_launch_speed_mps=float(np.max(launch[:, 0]) + margin[0]),
                minimum_supported_vertical_launch_speed_mps=float(np.min(launch[:, 1]) - margin[1]),
                maximum_supported_vertical_launch_speed_mps=float(np.max(launch[:, 1]) + margin[1]),
                forward_dynamics_fit_rmse_mps=rmse,
            )
        )
    if len(prototypes) < 2 or not rejected_seeds:
        raise ValueError("episodic contact memory needs two supported and one rejected context")
    scales = np.asarray(_OBSERVATION_SCALES, dtype=np.float64)
    minimum_distance = min(
        float(
            np.sqrt(
                np.mean(
                    np.square(
                        (np.asarray(left.observation) - np.asarray(right.observation)) / scales
                    )
                )
            )
        )
        for index, left in enumerate(prototypes)
        for right in prototypes[index + 1 :]
    )
    memory = G1EpisodicContactMemory(
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(source_hashes),
        prototypes=tuple(prototypes),
        rejected_context_seed_labels=tuple(rejected_seeds),
        observation_feature_scales=_OBSERVATION_SCALES,
        maximum_context_distance=maximum_context_distance,
        minimum_prototype_distance=minimum_distance,
        maximum_foot_ball_distance_m=float(np.median(distance_gates)),
        start_policy_frame=int(np.median(policy_starts)),
        end_policy_frame=int(np.median(policy_ends)),
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        ridge_regularization=ridge_regularization,
        safe_probe_count=sum(int(row["hard_safe"]) for row in rows),
        rejected_probe_count=sum(int(not row["hard_safe"]) for row in rows),
        training_target_count=len(targets),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(memory.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return memory


def g1_episodic_contact_effect(
    *,
    model: Any,
    data: Any,
    right_ankle_body_id: int,
    torso_body_id: int,
    memory: G1EpisodicContactMemory,
    policy_frame: int,
    contact_observed: bool,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray,
    goal_plane_x_m: float,
    target_y_m: float,
    target_z_m: float,
) -> G1EpisodicContactEffect:
    """Route to a local memory island and decode its bounded task force."""

    import mujoco

    zero: NDArray[np.float64] = np.zeros(29, dtype=np.float64)

    def inactive(
        *,
        seed: int | None = None,
        distance: float = 0.0,
        context_supported: bool = True,
        launch_supported: bool = True,
        desired_vy: float = 0.0,
        desired_vz: float = 0.0,
        foot_vy: float = 0.0,
        foot_vz: float = 0.0,
    ) -> G1EpisodicContactEffect:
        return G1EpisodicContactEffect(
            torque=zero,
            lateral_force_n=0.0,
            vertical_force_n=0.0,
            foot_lateral_speed_mps=foot_vy,
            foot_vertical_speed_mps=foot_vz,
            desired_lateral_launch_speed_mps=desired_vy,
            desired_vertical_launch_speed_mps=desired_vz,
            selected_context_seed_label=seed,
            context_distance=distance,
            active=False,
            context_supported=context_supported,
            launch_envelope_supported=launch_supported,
        )

    if contact_observed or not memory.start_policy_frame <= policy_frame <= memory.end_policy_frame:
        return inactive()
    ball = np.asarray(ball_position, dtype=np.float64)
    velocity = np.asarray(ball_velocity, dtype=np.float64)
    if (
        ball.shape != (3,)
        or velocity.shape != (3,)
        or not np.all(np.isfinite(np.concatenate((ball, velocity))))
    ):
        raise ValueError("episodic contact memory requires finite ball state")
    foot_rotation = np.asarray(data.xmat[right_ankle_body_id], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(
        data.xpos[right_ankle_body_id], dtype=np.float64
    ) + foot_rotation @ np.asarray(memory.foot_strike_point_offset_m, dtype=np.float64)
    if float(np.linalg.norm(foot_point - ball)) > memory.maximum_foot_ball_distance_m:
        return inactive()
    jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation_jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(model, data, jacobian, rotation_jacobian, foot_point, right_ankle_body_id)
    observation = g1_episodic_contact_observation(
        data=data,
        right_ankle_body_id=right_ankle_body_id,
        torso_body_id=torso_body_id,
        ball_position=ball,
    )
    scales = np.asarray(memory.observation_feature_scales, dtype=np.float64)
    distances = np.asarray(
        [
            np.sqrt(
                np.mean(
                    np.square(
                        (observation - np.asarray(item.observation, dtype=np.float64)) / scales
                    )
                )
            )
            for item in memory.prototypes
        ],
        dtype=np.float64,
    )
    selected_index = int(np.argmin(distances))
    prototype = memory.prototypes[selected_index]
    distance = float(distances[selected_index])
    foot_vy = float(jacobian[1] @ data.qvel)
    foot_vz = float(jacobian[2] @ data.qvel)
    if distance > memory.maximum_context_distance:
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            context_supported=False,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    goal_values = (goal_plane_x_m, target_y_m, target_z_m)
    if not all(math.isfinite(value) for value in goal_values):
        raise ValueError("episodic contact memory requires a finite goal target")
    remaining_x = goal_plane_x_m - float(ball[0])
    if remaining_x <= 0.0:
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    flight_time = remaining_x / prototype.reference_forward_ball_speed_mps
    desired_vy = (target_y_m - float(ball[1])) / flight_time - float(velocity[1])
    desired_vz = (target_z_m - float(ball[2]) + 0.5 * 9.81 * flight_time**2) / flight_time - float(
        velocity[2]
    )
    launch_supported = bool(
        prototype.minimum_supported_lateral_launch_speed_mps
        <= desired_vy
        <= prototype.maximum_supported_lateral_launch_speed_mps
        and prototype.minimum_supported_vertical_launch_speed_mps
        <= desired_vz
        <= prototype.maximum_supported_vertical_launch_speed_mps
    )
    if not launch_supported:
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            launch_supported=False,
            desired_vy=desired_vy,
            desired_vz=desired_vz,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    weights = np.asarray(prototype.forward_dynamics_weight_matrix, dtype=np.float64)
    slopes = weights[1:, :]
    force = (np.asarray((desired_vy, desired_vz)) - weights[0, :]) @ np.linalg.pinv(
        slopes, rcond=1e-4
    )
    lateral = float(
        np.clip(
            force[0],
            prototype.minimum_lateral_force_n,
            prototype.maximum_lateral_force_n,
        )
    )
    vertical = float(
        np.clip(
            force[1],
            prototype.minimum_vertical_force_n,
            prototype.maximum_vertical_force_n,
        )
    )
    torque = jacobian[1, 6:35] * lateral + jacobian[2, 6:35] * vertical
    if torque.shape != (29,) or not np.all(np.isfinite(torque)):
        raise FloatingPointError("episodic contact memory emitted invalid joint torque")
    return G1EpisodicContactEffect(
        torque=torque,
        lateral_force_n=lateral,
        vertical_force_n=vertical,
        foot_lateral_speed_mps=foot_vy,
        foot_vertical_speed_mps=foot_vz,
        desired_lateral_launch_speed_mps=desired_vy,
        desired_vertical_launch_speed_mps=desired_vz,
        selected_context_seed_label=prototype.planner_seed_audit_label,
        context_distance=distance,
        active=bool(abs(lateral) > 0.0 or abs(vertical) > 0.0),
    )


__all__ = [
    "G1EpisodicContactEffect",
    "G1EpisodicContactMemory",
    "G1EpisodicContactPrototype",
    "derive_g1_episodic_contact_memory",
    "g1_episodic_contact_context_hash",
    "g1_episodic_contact_effect",
    "g1_episodic_contact_observation",
    "load_g1_episodic_contact_memory",
]

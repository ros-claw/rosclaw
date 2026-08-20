"""State-routed episodic goal-plane dynamics for SIM-only G1 football.

The memory keeps one locally identified force-to-goal-plane model per
qualified pre-contact state.  Unlike a single-flight ballistic approximation,
the learned outcome includes any measured flight, bounce, and roll before the
ball reaches the goal plane.  Runtime routing is proprioceptive: planner seeds
are audit labels only and are never used to select an action.  States, targets,
or forces outside rehearsed support fail closed with zero additive torque.
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

_SCHEMA = "rosclaw.growth.g1_episodic_contact_memory.v2"
_PROTOTYPE_SCHEMA = "rosclaw.growth.g1_episodic_contact_prototype.v2"
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


def _signed_polygon_area(polygon: np.ndarray) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    return 0.5 * float(
        np.sum(points[:, 0] * np.roll(points[:, 1], -1))
        - np.sum(points[:, 1] * np.roll(points[:, 0], -1))
    )


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a deterministic counter-clockwise 2-D convex hull."""

    values = sorted({(float(row[0]), float(row[1])) for row in np.asarray(points)})
    if len(values) < 3:
        raise ValueError("episodic goal-plane support requires three distinct outcomes")

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3 or abs(_signed_polygon_area(hull)) <= 1e-10:
        raise ValueError("episodic goal-plane support outcomes are collinear")
    return hull


def _inside_convex_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    target = np.asarray(point, dtype=np.float64)
    hull = np.asarray(polygon, dtype=np.float64)
    edges = np.roll(hull, -1, axis=0) - hull
    offsets = target - hull
    crosses = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    tolerance = 1e-9
    return bool(np.all(crosses >= -tolerance) or np.all(crosses <= tolerance))


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
    goal_plane_dynamics_weight_matrix: tuple[tuple[float, float, float], ...]
    supported_goal_plane_polygon_yz_m: tuple[tuple[float, float], ...]
    source_evidence_hashes: tuple[str, ...]
    contact_regime: str
    minimum_lateral_force_n: float
    maximum_lateral_force_n: float
    minimum_vertical_force_n: float
    maximum_vertical_force_n: float
    goal_plane_fit_rmse_m: float
    arrival_time_fit_rmse_sec: float
    maximum_target_prediction_error_m: float
    schema_version: str = _PROTOTYPE_SCHEMA

    def __post_init__(self) -> None:
        if self.planner_seed_audit_label < 0:
            raise ValueError("episodic contact prototype seed label must be non-negative")
        observation = np.asarray(self.observation, dtype=np.float64)
        weights = np.asarray(self.goal_plane_dynamics_weight_matrix, dtype=np.float64)
        polygon = np.asarray(self.supported_goal_plane_polygon_yz_m, dtype=np.float64)
        if observation.shape != (len(_OBSERVATION_NAMES),) or not np.all(np.isfinite(observation)):
            raise ValueError("episodic contact observation must contain 11 finite values")
        if weights.shape != (3, 3) or not np.all(np.isfinite(weights)):
            raise ValueError("episodic goal-plane dynamics must have finite shape (3, 3)")
        if (
            polygon.ndim != 2
            or polygon.shape[1:] != (2,)
            or len(polygon) < 3
            or not np.all(np.isfinite(polygon))
            or abs(_signed_polygon_area(polygon)) <= 1e-10
        ):
            raise ValueError("episodic goal-plane support polygon is degenerate")
        if len(self.source_evidence_hashes) < 6 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("episodic contact prototype requires six bound safe probes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("episodic contact prototype evidence hashes must be unique")
        if not (
            -250.0 <= self.minimum_lateral_force_n < self.maximum_lateral_force_n <= 250.0
            and -250.0 <= self.minimum_vertical_force_n < self.maximum_vertical_force_n <= 250.0
            and 0.0 <= self.goal_plane_fit_rmse_m <= 0.25
            and 0.0 <= self.arrival_time_fit_rmse_sec <= 2.0
            and 0.005 <= self.maximum_target_prediction_error_m <= 0.25
        ):
            raise ValueError("episodic contact prototype support envelope is invalid")
        if self.contact_regime not in {"AIRBORNE", "BOUNCE", "ROLLING"}:
            raise ValueError("episodic contact regime is invalid")
        if self.schema_version != _PROTOTYPE_SCHEMA:
            raise ValueError("episodic contact prototype schema is unsupported")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "observation": list(self.observation),
            "goal_plane_dynamics_weight_matrix": [
                list(row) for row in self.goal_plane_dynamics_weight_matrix
            ],
            "supported_goal_plane_polygon_yz_m": [
                list(row) for row in self.supported_goal_plane_polygon_yz_m
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
            "controller": "local_ridge_goal_plane_dynamics_regularized_inverse",
            "contact_outcome": "measured_flight_bounce_roll_to_goal_plane",
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
    predicted_goal_y_m: float
    predicted_goal_z_m: float
    predicted_arrival_time_sec: float
    selected_context_seed_label: int | None
    context_distance: float
    active: bool
    context_supported: bool = True
    target_envelope_supported: bool = True


def load_g1_episodic_contact_memory(path: Path) -> G1EpisodicContactMemory:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = str(payload.pop("memory_hash", ""))
    for name in (
        "observation_feature_names",
        "routing",
        "controller",
        "contact_outcome",
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
        value["goal_plane_dynamics_weight_matrix"] = tuple(
            tuple(float(item) for item in row) for row in value["goal_plane_dynamics_weight_matrix"]
        )
        value["supported_goal_plane_polygon_yz_m"] = tuple(
            tuple(float(item) for item in row) for row in value["supported_goal_plane_polygon_yz_m"]
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
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray, np.ndarray]:
    outcomes = np.asarray(
        [
            [
                float(row["goal_crossing"][1]),
                float(row["goal_crossing"][2]),
                float(row["arrival_time_sec"]),
            ]
            for row in rows
        ],
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
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ outcomes)
    slopes = coefficients[1:, :] / scale[:, None]
    intercept = coefficients[0, :] - mean @ slopes
    weights = np.vstack((intercept, slopes))
    predictions = np.column_stack((np.ones(len(rows)), force)) @ weights
    goal_rmse = float(np.sqrt(np.mean(np.square(predictions[:, :2] - outcomes[:, :2]))))
    arrival_rmse = float(np.sqrt(np.mean(np.square(predictions[:, 2] - outcomes[:, 2]))))
    goal_slopes = slopes[:, :2]
    condition = float(np.linalg.cond(goal_slopes))
    response_span = np.ptp(outcomes[:, :2], axis=0)
    if (
        not math.isfinite(goal_rmse)
        or goal_rmse > 0.25
        or not math.isfinite(arrival_rmse)
        or arrival_rmse > 2.0
        or not math.isfinite(condition)
        or condition > 1e4
        or np.any(response_span < 1e-3)
    ):
        raise ValueError("episodic goal-plane dynamics are ill-conditioned")
    hull = _convex_hull(outcomes[:, :2])
    return weights, goal_rmse, arrival_rmse, outcomes, force, hull


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
        goal_crossing = result.get("goal_crossing_xyz_m")
        if (
            not isinstance(goal_crossing, list)
            or len(goal_crossing) != 3
            or not all(
                isinstance(item, (int, float)) and math.isfinite(float(item))
                for item in goal_crossing
            )
            or not math.isclose(float(goal_crossing[0]), float(goal["plane_x_m"]), abs_tol=1e-6)
        ):
            raise ValueError("episodic contact evidence requires a measured goal-plane crossing")
        contact_time = float(result.get("contact_time_sec", math.nan))
        if not math.isfinite(contact_time):
            raise ValueError("episodic contact evidence requires a measured contact time")
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
        arrival_time = math.nan
        ground_contact_fraction = math.nan
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
            required_outcome = {"time", "ball_pose", "goal_crossing"}
            if not required_outcome.issubset(trace.files):
                raise ValueError("episodic contact trajectory lacks goal-plane outcome channels")
            time = np.asarray(trace["time"], dtype=np.float64)
            ball_pose = np.asarray(trace["ball_pose"], dtype=np.float64)
            crossing_mask = np.asarray(trace["goal_crossing"], dtype=np.bool_)
            if (
                time.ndim != 1
                or ball_pose.shape != (len(time), 7)
                or crossing_mask.shape != time.shape
                or not np.all(np.isfinite(time))
                or not np.all(np.isfinite(ball_pose))
            ):
                raise ValueError("episodic contact goal-plane trace shapes are invalid")
            crossing_indices = np.flatnonzero(crossing_mask)
            if crossing_indices.size == 0:
                raise ValueError("episodic contact trajectory never reaches the goal plane")
            crossing_index = int(crossing_indices[0])
            contact_index = int(np.searchsorted(time, contact_time, side="left"))
            if not 0 <= contact_index <= crossing_index < len(time):
                raise ValueError("episodic contact outcome timing is invalid")
            arrival_time = float(time[crossing_index] - contact_time)
            ball_radius = float(goal["ball_radius_m"])
            ground_contact_fraction = float(
                np.mean(ball_pose[contact_index : crossing_index + 1, 2] <= ball_radius + 0.005)
            )
            if not 0.0 < arrival_time <= 10.0 or not 0.0 <= ground_contact_fraction <= 1.0:
                raise ValueError("episodic contact outcome statistics are invalid")
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
                "goal_crossing": goal_crossing,
                "arrival_time_sec": arrival_time,
                "ground_contact_fraction": ground_contact_fraction,
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
            weights, goal_rmse, arrival_rmse, _outcomes, force, hull = _fit_forward_dynamics(
                safe, ridge_regularization
            )
        except ValueError:
            rejected_seeds.append(seed)
            continue
        observations = np.asarray([row["observation"] for row in teacher_safe], dtype=np.float64)
        if observations.shape != (len(teacher_safe), len(_OBSERVATION_NAMES)):
            raise ValueError("episodic contact prototype observations are invalid")
        median_ground_fraction = float(
            np.median([float(row["ground_contact_fraction"]) for row in safe])
        )
        contact_regime = (
            "AIRBORNE"
            if median_ground_fraction < 0.05
            else "ROLLING"
            if median_ground_fraction >= 0.80
            else "BOUNCE"
        )
        prototypes.append(
            G1EpisodicContactPrototype(
                planner_seed_audit_label=seed,
                observation=tuple(float(item) for item in np.median(observations, axis=0)),
                goal_plane_dynamics_weight_matrix=tuple(
                    (float(row[0]), float(row[1]), float(row[2])) for row in weights
                ),
                supported_goal_plane_polygon_yz_m=tuple(
                    (float(row[0]), float(row[1])) for row in hull
                ),
                source_evidence_hashes=tuple(row["evidence_hash"] for row in safe),
                contact_regime=contact_regime,
                minimum_lateral_force_n=float(np.min(force[:, 0])),
                maximum_lateral_force_n=float(np.max(force[:, 0])),
                minimum_vertical_force_n=float(np.min(force[:, 1])),
                maximum_vertical_force_n=float(np.max(force[:, 1])),
                goal_plane_fit_rmse_m=goal_rmse,
                arrival_time_fit_rmse_sec=arrival_rmse,
                maximum_target_prediction_error_m=float(min(0.25, max(0.005, 3.0 * goal_rmse))),
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
        target_supported: bool = True,
        predicted_y: float = 0.0,
        predicted_z: float = 0.0,
        predicted_arrival: float = 0.0,
        foot_vy: float = 0.0,
        foot_vz: float = 0.0,
    ) -> G1EpisodicContactEffect:
        return G1EpisodicContactEffect(
            torque=zero,
            lateral_force_n=0.0,
            vertical_force_n=0.0,
            foot_lateral_speed_mps=foot_vy,
            foot_vertical_speed_mps=foot_vz,
            predicted_goal_y_m=predicted_y,
            predicted_goal_z_m=predicted_z,
            predicted_arrival_time_sec=predicted_arrival,
            selected_context_seed_label=seed,
            context_distance=distance,
            active=False,
            context_supported=context_supported,
            target_envelope_supported=target_supported,
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
    if goal_plane_x_m <= float(ball[0]):
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    target = np.asarray((target_y_m, target_z_m), dtype=np.float64)
    polygon = np.asarray(prototype.supported_goal_plane_polygon_yz_m, dtype=np.float64)
    if not _inside_convex_polygon(target, polygon):
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            target_supported=False,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    weights = np.asarray(prototype.goal_plane_dynamics_weight_matrix, dtype=np.float64)
    goal_slopes = weights[1:, :2]
    force = (target - weights[0, :2]) @ np.linalg.pinv(goal_slopes, rcond=1e-4)
    force_supported = bool(
        prototype.minimum_lateral_force_n - 1e-9
        <= force[0]
        <= prototype.maximum_lateral_force_n + 1e-9
        and prototype.minimum_vertical_force_n - 1e-9
        <= force[1]
        <= prototype.maximum_vertical_force_n + 1e-9
    )
    if not force_supported:
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            target_supported=False,
            foot_vy=foot_vy,
            foot_vz=foot_vz,
        )
    lateral = float(
        np.clip(force[0], prototype.minimum_lateral_force_n, prototype.maximum_lateral_force_n)
    )
    vertical = float(
        np.clip(force[1], prototype.minimum_vertical_force_n, prototype.maximum_vertical_force_n)
    )
    prediction = np.asarray((1.0, lateral, vertical), dtype=np.float64) @ weights
    prediction_error = float(np.linalg.norm(prediction[:2] - target))
    if prediction_error > prototype.maximum_target_prediction_error_m:
        return inactive(
            seed=prototype.planner_seed_audit_label,
            distance=distance,
            target_supported=False,
            predicted_y=float(prediction[0]),
            predicted_z=float(prediction[1]),
            predicted_arrival=float(prediction[2]),
            foot_vy=foot_vy,
            foot_vz=foot_vz,
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
        predicted_goal_y_m=float(prediction[0]),
        predicted_goal_z_m=float(prediction[1]),
        predicted_arrival_time_sec=float(prediction[2]),
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

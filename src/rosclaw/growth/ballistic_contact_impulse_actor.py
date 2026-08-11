"""Evidence-trained proprioceptive impulse actor for G1 ball contact.

The v1 actor is distilled from one fixed-target strict teacher exploration.
The v2 actor instead learns an inverse contact model from actual task-space
forces and measured ball launch velocities.  At runtime v2 converts the ball
state and requested goal point into a ballistic launch-velocity condition,
then emits a bounded lateral/vertical task-space impulse.  Both versions use
the measured MuJoCo Jacobian to decode that impulse to right-leg joint torque.
The artifacts and runtime are SIM-only; neither authorizes a hardware
controller or an online hot swap.
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

_TEACHER_PREFIX = "shot_loft_teacher_"
_POST_CONTACT_ONLY_FLOW_FIELDS = {
    "shared_cerebellar_recovery_enabled",
    "shot_recovery_step_length_m",
    "shot_recovery_step_yaw_rad",
    "post_contact_damping_delay_sec",
    "post_contact_damping_ramp_sec",
}
_V1_SCHEMA = "rosclaw.growth.g1_ballistic_contact_impulse_actor.v1"
_V2_SCHEMA = "rosclaw.growth.g1_ballistic_contact_impulse_actor.v2"
_V2_ONLY_ACTOR_FIELDS = {
    "reference_forward_ball_speed_mps",
    "minimum_lateral_force_n",
    "minimum_vertical_force_n",
    "minimum_supported_lateral_launch_speed_mps",
    "maximum_supported_lateral_launch_speed_mps",
    "minimum_supported_vertical_launch_speed_mps",
    "maximum_supported_vertical_launch_speed_mps",
    "forward_dynamics_fit_rmse_mps",
    "ridge_regularization",
    "safe_probe_count",
    "training_target_count",
}


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def g1_ballistic_contact_impulse_context_hash(
    *,
    flow_config: dict[str, Any],
    goal_spec: dict[str, Any],
    runup_config: dict[str, Any],
    sonic_runup_config: dict[str, Any] | None,
    approach_strike_candidate_hash: str | None,
    target_conditioned: bool = False,
) -> str:
    """Bind an actor to its non-teacher task and controller context."""

    context_flow = {
        key: value
        for key, value in flow_config.items()
        if not key.startswith(_TEACHER_PREFIX)
        and key
        not in {
            "schema_version",
            "ballistic_contact_impulse_actor_hash",
            *_POST_CONTACT_ONLY_FLOW_FIELDS,
        }
    }
    context_goal = dict(goal_spec)
    if target_conditioned:
        context_goal.pop("target_y_m", None)
        context_goal.pop("target_z_m", None)
    return canonical_hash(
        {
            "flow_config_without_teacher": context_flow,
            "goal_spec": context_goal,
            "runup_config": runup_config,
            "sonic_runup_config": sonic_runup_config,
            "approach_strike_candidate_hash": approach_strike_candidate_hash,
        }
    )


@dataclass(frozen=True)
class G1BallisticContactImpulseActor:
    """A bounded two-output actor plus a measured Jacobian decoder."""

    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    selected_evidence_hash: str
    selected_goal_plane_target_error_m: float
    precision_success_count: int
    rejected_probe_count: int
    task_space_actor_weight_matrix: tuple[tuple[float, ...], ...]
    maximum_lateral_force_n: float
    maximum_vertical_force_n: float
    maximum_foot_ball_distance_m: float
    start_policy_frame: int
    end_policy_frame: int
    foot_strike_point_offset_m: tuple[float, float, float]
    qualified_error_max_m: float
    minimum_lateral_force_n: float = 0.0
    minimum_vertical_force_n: float = 0.0
    reference_forward_ball_speed_mps: float = 0.0
    minimum_supported_lateral_launch_speed_mps: float = 0.0
    maximum_supported_lateral_launch_speed_mps: float = 0.0
    minimum_supported_vertical_launch_speed_mps: float = 0.0
    maximum_supported_vertical_launch_speed_mps: float = 0.0
    forward_dynamics_fit_rmse_mps: float = 0.0
    ridge_regularization: float = 0.0
    safe_probe_count: int = 0
    training_target_count: int = 0
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = _V1_SCHEMA

    def __post_init__(self) -> None:
        if not self.body_hash.startswith("sha256:") or not self.implementation_hash.startswith(
            "sha256:"
        ):
            raise ValueError(
                "contact impulse actor requires SHA-256 Body and implementation hashes"
            )
        if not self.experiment_context_hash.startswith("sha256:"):
            raise ValueError("contact impulse actor context hash must be SHA-256")
        if self.schema_version not in {_V1_SCHEMA, _V2_SCHEMA}:
            raise ValueError("contact impulse actor schema is unsupported")
        if len(self.source_evidence_hashes) < 8 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("contact impulse actor requires eight bound evidence hashes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("contact impulse actor evidence hashes must be unique")
        if self.selected_evidence_hash not in self.source_evidence_hashes:
            raise ValueError("selected contact impulse evidence is not source-bound")
        weights = np.asarray(self.task_space_actor_weight_matrix, dtype=np.float64)
        expected_shape = (2, 3) if self.schema_version == _V1_SCHEMA else (2, 5)
        if weights.shape != expected_shape or not np.all(np.isfinite(weights)):
            raise ValueError(
                f"contact impulse actor weights must have finite shape {expected_shape}"
            )
        if not 10.0 <= self.maximum_lateral_force_n <= 250.0:
            raise ValueError("contact impulse actor lateral force limit is invalid")
        if not 10.0 <= self.maximum_vertical_force_n <= 250.0:
            raise ValueError("contact impulse actor vertical force limit is invalid")
        if self.schema_version == _V2_SCHEMA and not (
            -250.0 <= self.minimum_lateral_force_n < self.maximum_lateral_force_n
            and -250.0 <= self.minimum_vertical_force_n < self.maximum_vertical_force_n
        ):
            raise ValueError("target-conditioned actor observed force envelope is invalid")
        if not 0.15 <= self.maximum_foot_ball_distance_m <= 0.30:
            raise ValueError("contact impulse actor proximity gate is invalid")
        if not 150 <= self.start_policy_frame < self.end_policy_frame <= 430:
            raise ValueError("contact impulse actor policy window is invalid")
        if len(self.foot_strike_point_offset_m) != 3 or not all(
            math.isfinite(value) for value in self.foot_strike_point_offset_m
        ):
            raise ValueError("contact impulse actor strike point is invalid")
        if float(np.linalg.norm(self.foot_strike_point_offset_m)) > 0.30:
            raise ValueError("contact impulse actor strike point is outside the foot envelope")
        if self.schema_version == _V1_SCHEMA:
            if self.precision_success_count < 2 or self.rejected_probe_count < 2:
                raise ValueError("contact impulse actor needs successful and rejected support")
            if any(getattr(self, name) != 0 for name in _V2_ONLY_ACTOR_FIELDS):
                raise ValueError("v1 contact impulse actor cannot contain v2 state")
        else:
            velocity_limits = (
                self.reference_forward_ball_speed_mps,
                self.minimum_supported_lateral_launch_speed_mps,
                self.maximum_supported_lateral_launch_speed_mps,
                self.minimum_supported_vertical_launch_speed_mps,
                self.maximum_supported_vertical_launch_speed_mps,
                self.forward_dynamics_fit_rmse_mps,
                self.ridge_regularization,
            )
            if not all(math.isfinite(value) for value in velocity_limits):
                raise ValueError("target-conditioned actor state must be finite")
            if not 2.0 <= self.reference_forward_ball_speed_mps <= 20.0:
                raise ValueError("target-conditioned actor forward reference is invalid")
            if not (
                self.minimum_supported_lateral_launch_speed_mps
                < self.maximum_supported_lateral_launch_speed_mps
                and self.minimum_supported_vertical_launch_speed_mps
                < self.maximum_supported_vertical_launch_speed_mps
            ):
                raise ValueError("target-conditioned actor launch envelope is invalid")
            if not 0.0 <= self.forward_dynamics_fit_rmse_mps <= 5.0:
                raise ValueError("target-conditioned actor fit error is invalid")
            if not 0.0 < self.ridge_regularization <= 10.0:
                raise ValueError("target-conditioned actor regularization is invalid")
            if self.safe_probe_count < 4 or self.rejected_probe_count < 2:
                raise ValueError(
                    "target-conditioned actor needs safe and rejected rehearsal support"
                )
            if self.training_target_count < 1:
                raise ValueError("target-conditioned actor has no bound training target")
        if not 0.01 <= self.qualified_error_max_m <= 1.0:
            raise ValueError("contact impulse actor precision threshold is invalid")
        if not math.isfinite(self.selected_goal_plane_target_error_m) or (
            self.selected_goal_plane_target_error_m < 0.0
        ) or (
            self.schema_version == _V1_SCHEMA
            and self.selected_goal_plane_target_error_m > self.qualified_error_max_m
        ):
            raise ValueError("contact impulse actor selected an unqualified probe")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("contact impulse actor must remain SIM_ONLY")

    @property
    def actor_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        fields = asdict(self)
        if self.schema_version == _V1_SCHEMA:
            for name in _V2_ONLY_ACTOR_FIELDS:
                fields.pop(name)
        target_conditioned = self.schema_version == _V2_SCHEMA
        value: dict[str, Any] = {
            **fields,
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "task_space_actor_weight_matrix": [
                list(row) for row in self.task_space_actor_weight_matrix
            ],
            "feature_names": (
                ["bias", "right_foot_vy_mps", "right_foot_vz_mps"]
                if not target_conditioned
                else [
                    "bias",
                    "required_ball_launch_vy_mps",
                    "required_ball_launch_vz_mps",
                    "right_foot_vy_mps",
                    "right_foot_vz_mps",
                ]
            ),
            "output_names": ["lateral_force_n", "vertical_force_n"],
            "decoder": "measured_right_foot_jacobian_transpose_to_joint_torque",
            "algorithm": (
                "strict_replay_supported_contextual_bandit_distillation"
                if not target_conditioned
                else "ridge_forward_contact_dynamics_with_regularized_inverse_control"
            ),
            "direct_joint_torque_output": True,
            "online_hot_swap_allowed": False,
            "sealed_generalization_evidence": False,
            "stability_plasticity_contract": {
                "stability": (
                    "rejected probes remain bound and runtime force is clipped"
                    if not target_conditioned
                    else (
                        "rejected probes remain bound; force and learned launch envelope "
                        "are clipped"
                    )
                ),
                "plasticity": (
                    "only a strictly replayed precision-success actor is selected"
                    if not target_conditioned
                    else "v2 fits only strict contact-safe probes"
                ),
            },
        }
        if target_conditioned:
            value["ball_state_conditioned"] = True
            value["goal_target_conditioned"] = True
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value


@dataclass(frozen=True)
class G1BallisticContactImpulseEffect:
    torque: np.ndarray
    lateral_force_n: float
    vertical_force_n: float
    foot_lateral_speed_mps: float
    foot_vertical_speed_mps: float
    active: bool
    desired_lateral_launch_speed_mps: float = 0.0
    desired_vertical_launch_speed_mps: float = 0.0
    target_conditioned: bool = False
    launch_envelope_supported: bool = True


def load_g1_ballistic_contact_impulse_actor(
    path: Path,
) -> G1BallisticContactImpulseActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    expected = str(payload.pop("actor_hash", ""))
    payload.pop("feature_names", None)
    payload.pop("output_names", None)
    payload.pop("decoder", None)
    payload.pop("algorithm", None)
    payload.pop("direct_joint_torque_output", None)
    payload.pop("online_hot_swap_allowed", None)
    payload.pop("ball_state_conditioned", None)
    payload.pop("goal_target_conditioned", None)
    payload.pop("sealed_generalization_evidence", None)
    payload.pop("stability_plasticity_contract", None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["task_space_actor_weight_matrix"] = tuple(
        tuple(float(value) for value in row) for row in payload["task_space_actor_weight_matrix"]
    )
    payload["foot_strike_point_offset_m"] = tuple(payload["foot_strike_point_offset_m"])
    actor = G1BallisticContactImpulseActor(**payload)
    if expected != actor.actor_hash:
        raise ValueError("contact impulse actor hash mismatch")
    return actor


def derive_g1_ballistic_contact_impulse_actor(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    target_conditioned: bool = False,
    ridge_regularization: float = 0.05,
) -> G1BallisticContactImpulseActor:
    """Select and bind a precision-success impulse actor from strict probes."""

    if len(evidence_paths) < 8:
        raise ValueError("contact impulse actor training requires at least eight probes")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contact impulse actor evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("contact impulse actor output already exists")
    if target_conditioned and not 0.0 < ridge_regularization <= 10.0:
        raise ValueError("target-conditioned actor regularization must be in (0, 10]")
    rows: list[dict[str, Any]] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    source_hashes: list[str] = []
    precision_radii: set[float] = set()
    resolved_paths: set[Path] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        if path in resolved_paths:
            raise ValueError("contact impulse actor evidence paths must be unique")
        resolved_paths.add(path)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("contact impulse actor requires strict replay evidence")
        claims = dict(evidence.get("claims", {}))
        teacher_executed = claims.get("sim_only_operational_space_loft_teacher") is True
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("contact impulse actor trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        goal_spec = dict(evidence.get("goal_spec", {}))
        context_hashes.add(
            g1_ballistic_contact_impulse_context_hash(
                flow_config=flow,
                goal_spec=goal_spec,
                runup_config=dict(evidence.get("runup_config", {})),
                sonic_runup_config=(
                    None
                    if evidence.get("sonic_runup_config") is None
                    else dict(evidence["sonic_runup_config"])
                ),
                approach_strike_candidate_hash=evidence.get("approach_strike_candidate_hash"),
                target_conditioned=target_conditioned,
            )
        )
        result = dict(evidence.get("result", {}))
        teacher_free_baseline = bool(
            target_conditioned
            and not teacher_executed
            and result.get("loft_teacher_executed") is False
            and result.get("ballistic_contact_impulse_actor_executed") is False
        )
        if not teacher_executed and not teacher_free_baseline:
            raise ValueError(
                "contact impulse actor probes must execute the SIM teacher or be a "
                "teacher-free v2 baseline"
            )
        raw_error = result.get("goal_plane_target_error_m")
        error = float(raw_error) if isinstance(raw_error, (int, float)) else math.inf
        precision = float(result.get("precision_radius_m", 0.0))
        if not math.isfinite(precision) or not 0.01 <= precision <= 1.0:
            raise ValueError("contact impulse actor precision threshold is invalid")
        precision_radii.add(precision)
        projection_fraction = float(result.get("torque_authority_projection_fraction", 0.0))
        preprojection_demand = float(
            result.get(
                "torque_authority_preprojection_peak_demand_ratio",
                result.get("actuator_peak_demand_ratio", 0.0),
            )
        )
        contact_task_scale = float(result.get("contact_task_authority_scale_min", 1.0))
        if (
            not math.isfinite(projection_fraction)
            or projection_fraction < 0.0
            or not math.isfinite(preprojection_demand)
            or preprojection_demand < 0.0
            or not math.isfinite(contact_task_scale)
            or not 0.0 <= contact_task_scale <= 1.0
        ):
            raise ValueError("contact impulse actor authority metrics are invalid")
        hard_safe = bool(
            math.isfinite(error)
            and result.get("kick_contact_observed") is True
            and result.get("perceptual_continuity_passed") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
            and result.get("actuator_saturation") is not True
            and result.get("torque_authority_projection_qualified", True) is True
            and contact_task_scale >= 0.95
        )
        safe = bool(hard_safe and result.get("goal_mouth_hit") is True)
        evidence_hash = _file_hash(path)
        if evidence_hash in source_hashes:
            raise ValueError("contact impulse actor evidence contents must be unique")
        source_hashes.append(evidence_hash)
        launch = result.get("ball_launch_velocity_xyz_mps")
        lateral_force = 0.0
        vertical_force = 0.0
        if target_conditioned:
            if (
                not isinstance(launch, list)
                or len(launch) != 3
                or not all(isinstance(value, (int, float)) for value in launch)
                or not all(math.isfinite(float(value)) for value in launch)
            ):
                raise ValueError("target-conditioned actor requires measured ball launch velocity")
            with np.load(trajectory, allow_pickle=False) as trace:
                required_keys = {
                    "loft_teacher_active",
                    "loft_teacher_lateral_force_n",
                    "loft_teacher_force_n",
                }
                if not required_keys.issubset(trace.files):
                    raise ValueError(
                        "target-conditioned actor trajectory lacks teacher force channels"
                    )
                active = np.asarray(trace["loft_teacher_active"], dtype=np.bool_)
                lateral_trace = np.asarray(
                    trace["loft_teacher_lateral_force_n"], dtype=np.float64
                )
                vertical_trace = np.asarray(trace["loft_teacher_force_n"], dtype=np.float64)
                if (
                    active.ndim != 1
                    or lateral_trace.shape != active.shape
                    or vertical_trace.shape != active.shape
                    or not np.all(np.isfinite(lateral_trace))
                    or not np.all(np.isfinite(vertical_trace))
                ):
                    raise ValueError("target-conditioned actor teacher force trace is invalid")
                if teacher_executed:
                    if not np.any(active):
                        raise ValueError("target-conditioned teacher probe never activated")
                    lateral_active = lateral_trace[active]
                    vertical_active = vertical_trace[active]
                    lateral_force = float(lateral_active[np.argmax(np.abs(lateral_active))])
                    vertical_force = float(vertical_active[np.argmax(np.abs(vertical_active))])
                elif np.any(active) or np.any(lateral_trace) or np.any(vertical_trace):
                    raise ValueError("target-conditioned zero-force baseline is contaminated")
        rows.append(
            {
                "error": error,
                "projection_fraction": projection_fraction,
                "preprojection_demand": preprojection_demand,
                "qualified": safe and error <= precision,
                "hard_safe": hard_safe,
                "flow": flow,
                "goal": goal_spec,
                "evidence_hash": evidence_hash,
                "launch": launch,
                "lateral_force": lateral_force,
                "vertical_force": vertical_force,
            }
        )
    if (
        len(body_hashes) != 1
        or len(implementation_hashes) != 1
        or len(context_hashes) != 1
        or len(precision_radii) != 1
    ):
        raise ValueError("contact impulse actor probe contexts disagree")
    qualified = [row for row in rows if row["qualified"]]
    rejected = [row for row in rows if not row["qualified"]]
    if target_conditioned:
        safe_rows = [row for row in rows if row["hard_safe"]]
        if len(safe_rows) < 4 or len(rejected) < 2:
            raise ValueError(
                "target-conditioned actor needs four contact-safe probes and two rejects"
            )
        launch_matrix = np.asarray(
            [[float(row["launch"][1]), float(row["launch"][2])] for row in safe_rows],
            dtype=np.float64,
        )
        force_matrix = np.asarray(
            [[row["lateral_force"], row["vertical_force"]] for row in safe_rows],
            dtype=np.float64,
        )
        if np.linalg.matrix_rank(
            np.column_stack((np.ones(len(safe_rows), dtype=np.float64), force_matrix))
        ) < 3:
            raise ValueError("target-conditioned actor force probes lack two-axis coverage")
        if np.ptp(force_matrix[:, 0]) < 5.0 or np.ptp(force_matrix[:, 1]) < 5.0:
            raise ValueError("target-conditioned actor teacher forces lack two-axis coverage")
        mean = np.mean(force_matrix, axis=0)
        scale = np.std(force_matrix, axis=0)
        if np.any(scale < 1e-4):
            raise ValueError("target-conditioned actor force distribution is degenerate")
        standardized = (force_matrix - mean) / scale
        design = np.column_stack((np.ones(len(safe_rows)), standardized))
        penalty = np.diag((0.0, ridge_regularization, ridge_regularization))
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ launch_matrix)
        raw_slopes = coefficients[1:, :] / scale[:, None]
        raw_intercept = coefficients[0, :] - mean @ raw_slopes
        predictions = np.column_stack((np.ones(len(safe_rows)), force_matrix)) @ np.vstack(
            (raw_intercept, raw_slopes)
        )
        fit_rmse = float(np.sqrt(np.mean(np.square(predictions - launch_matrix))))
        if not math.isfinite(fit_rmse) or fit_rmse > 5.0:
            raise ValueError("target-conditioned actor forward fit is invalid")
        if not math.isfinite(float(np.linalg.cond(raw_slopes))) or np.linalg.cond(
            raw_slopes
        ) > 1e4:
            raise ValueError("target-conditioned actor force response is ill-conditioned")
        inverse_response = np.linalg.pinv(raw_slopes, rcond=1e-4)
        actor_intercept = -raw_intercept @ inverse_response
        best = min(
            safe_rows,
            key=lambda row: (
                row["error"]
                + 2.0 * row["projection_fraction"]
                + 0.02 * max(0.0, row["preprojection_demand"] - 1.0),
                row["error"],
            ),
        )
        margin = np.maximum(0.05, 0.10 * np.ptp(launch_matrix, axis=0))
        actor = G1BallisticContactImpulseActor(
            body_hash=next(iter(body_hashes)),
            implementation_hash=next(iter(implementation_hashes)),
            experiment_context_hash=next(iter(context_hashes)),
            source_evidence_hashes=tuple(source_hashes),
            selected_evidence_hash=best["evidence_hash"],
            selected_goal_plane_target_error_m=best["error"],
            precision_success_count=len(qualified),
            rejected_probe_count=len(rejected),
            task_space_actor_weight_matrix=(
                (
                    float(actor_intercept[0]),
                    float(inverse_response[0, 0]),
                    float(inverse_response[1, 0]),
                    0.0,
                    0.0,
                ),
                (
                    float(actor_intercept[1]),
                    float(inverse_response[0, 1]),
                    float(inverse_response[1, 1]),
                    0.0,
                    0.0,
                ),
            ),
            maximum_lateral_force_n=float(
                max(row["lateral_force"] for row in safe_rows)
            ),
            maximum_vertical_force_n=float(
                max(row["vertical_force"] for row in safe_rows)
            ),
            minimum_lateral_force_n=float(
                min(row["lateral_force"] for row in safe_rows)
            ),
            minimum_vertical_force_n=float(
                min(row["vertical_force"] for row in safe_rows)
            ),
            maximum_foot_ball_distance_m=float(
                np.median(
                    [
                        row["flow"]["shot_loft_teacher_max_foot_ball_distance_m"]
                        for row in safe_rows
                    ]
                )
            ),
            start_policy_frame=int(
                round(
                    np.median(
                        [
                            row["flow"]["shot_loft_teacher_start_policy_frame"]
                            for row in safe_rows
                        ]
                    )
                )
            ),
            end_policy_frame=int(
                round(
                    np.median(
                        [
                            row["flow"]["shot_loft_teacher_end_policy_frame"]
                            for row in safe_rows
                        ]
                    )
                )
            ),
            foot_strike_point_offset_m=(0.13, 0.0, -0.025),
            qualified_error_max_m=next(iter(precision_radii)),
            reference_forward_ball_speed_mps=float(
                np.median([float(row["launch"][0]) for row in safe_rows])
            ),
            minimum_supported_lateral_launch_speed_mps=float(
                np.min(launch_matrix[:, 0]) - margin[0]
            ),
            maximum_supported_lateral_launch_speed_mps=float(
                np.max(launch_matrix[:, 0]) + margin[0]
            ),
            minimum_supported_vertical_launch_speed_mps=float(
                np.min(launch_matrix[:, 1]) - margin[1]
            ),
            maximum_supported_vertical_launch_speed_mps=float(
                np.max(launch_matrix[:, 1]) + margin[1]
            ),
            forward_dynamics_fit_rmse_mps=fit_rmse,
            ridge_regularization=ridge_regularization,
            safe_probe_count=len(safe_rows),
            training_target_count=len(
                {
                    (float(row["goal"]["target_y_m"]), float(row["goal"]["target_z_m"]))
                    for row in safe_rows
                }
            ),
            schema_version=_V2_SCHEMA,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(actor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return actor
    if len(qualified) < 2 or len(rejected) < 2:
        raise ValueError("contact impulse actor needs two precision successes and two rejects")
    chosen = min(
        qualified,
        key=lambda row: (
            row["error"]
            + 2.0 * row["projection_fraction"]
            + 0.02 * max(0.0, row["preprojection_demand"] - 1.0),
            row["error"],
        ),
    )
    error = chosen["error"]
    selected = chosen["flow"]
    selected_hash = chosen["evidence_hash"]
    lateral_gain = float(selected["shot_loft_teacher_lateral_gain_n_per_mps"])
    vertical_gain = float(selected["shot_loft_teacher_gain_n_per_mps"])
    lateral_target = float(selected["shot_loft_teacher_target_vy_mps"])
    vertical_target = float(selected["shot_loft_teacher_target_vz_mps"])
    actor = G1BallisticContactImpulseActor(
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(source_hashes),
        selected_evidence_hash=selected_hash,
        selected_goal_plane_target_error_m=error,
        precision_success_count=len(qualified),
        rejected_probe_count=len(rejected),
        task_space_actor_weight_matrix=(
            (lateral_gain * lateral_target, -lateral_gain, 0.0),
            (vertical_gain * vertical_target, 0.0, -vertical_gain),
        ),
        maximum_lateral_force_n=float(selected["shot_loft_teacher_max_lateral_force_n"]),
        maximum_vertical_force_n=float(selected["shot_loft_teacher_max_force_n"]),
        maximum_foot_ball_distance_m=float(selected["shot_loft_teacher_max_foot_ball_distance_m"]),
        start_policy_frame=int(selected["shot_loft_teacher_start_policy_frame"]),
        end_policy_frame=int(selected["shot_loft_teacher_end_policy_frame"]),
        foot_strike_point_offset_m=(0.13, 0.0, -0.025),
        qualified_error_max_m=next(iter(precision_radii)),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(actor.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return actor


def g1_ballistic_contact_impulse_effect(
    *,
    model: Any,
    data: Any,
    right_ankle_body_id: int,
    actor: G1BallisticContactImpulseActor,
    policy_frame: int,
    contact_observed: bool,
    ball_position: np.ndarray,
    ball_velocity: np.ndarray | None = None,
    goal_plane_x_m: float | None = None,
    target_y_m: float | None = None,
    target_z_m: float | None = None,
) -> G1BallisticContactImpulseEffect:
    """Run the learned proprioceptive actor and decode direct joint torques."""

    import mujoco

    zero: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    target_conditioned = actor.schema_version == _V2_SCHEMA
    if contact_observed or not actor.start_policy_frame <= policy_frame <= actor.end_policy_frame:
        return G1BallisticContactImpulseEffect(
            zero, 0.0, 0.0, 0.0, 0.0, False, target_conditioned=target_conditioned
        )
    foot_rotation = np.asarray(data.xmat[right_ankle_body_id], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(
        data.xpos[right_ankle_body_id], dtype=np.float64
    ) + foot_rotation @ np.asarray(actor.foot_strike_point_offset_m, dtype=np.float64)
    ball = np.asarray(ball_position, dtype=np.float64)
    if ball.shape != (3,) or not np.all(np.isfinite(ball)):
        raise ValueError("contact impulse actor requires a finite ball position")
    if float(np.linalg.norm(foot_point - ball)) > actor.maximum_foot_ball_distance_m:
        return G1BallisticContactImpulseEffect(
            zero, 0.0, 0.0, 0.0, 0.0, False, target_conditioned=target_conditioned
        )
    jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    rotation_jacobian: NDArray[np.float64] = np.zeros((3, int(model.nv)), dtype=np.float64)
    mujoco.mj_jac(
        model,
        data,
        jacobian,
        rotation_jacobian,
        foot_point,
        right_ankle_body_id,
    )
    foot_vy = float(jacobian[1] @ data.qvel)
    foot_vz = float(jacobian[2] @ data.qvel)
    desired_vy = 0.0
    desired_vz = 0.0
    if target_conditioned:
        velocity = np.asarray(ball_velocity, dtype=np.float64)
        goal_values = (goal_plane_x_m, target_y_m, target_z_m)
        if (
            velocity.shape != (3,)
            or not np.all(np.isfinite(velocity))
            or any(value is None or not math.isfinite(value) for value in goal_values)
        ):
            raise ValueError(
                "target-conditioned actor requires finite ball velocity and goal target"
            )
        remaining_x = float(goal_plane_x_m) - float(ball[0])
        if remaining_x <= 0.0:
            return G1BallisticContactImpulseEffect(
                zero, 0.0, 0.0, foot_vy, foot_vz, False, target_conditioned=True
            )
        flight_time = remaining_x / actor.reference_forward_ball_speed_mps
        desired_vy = (float(target_y_m) - float(ball[1])) / flight_time - float(velocity[1])
        desired_vz = (
            (float(target_z_m) - float(ball[2]) + 0.5 * 9.81 * flight_time**2)
            / flight_time
            - float(velocity[2])
        )
        if not (
            actor.minimum_supported_lateral_launch_speed_mps
            <= desired_vy
            <= actor.maximum_supported_lateral_launch_speed_mps
            and actor.minimum_supported_vertical_launch_speed_mps
            <= desired_vz
            <= actor.maximum_supported_vertical_launch_speed_mps
        ):
            return G1BallisticContactImpulseEffect(
                zero,
                0.0,
                0.0,
                foot_vy,
                foot_vz,
                False,
                desired_vy,
                desired_vz,
                True,
                False,
            )
        features = np.asarray((1.0, desired_vy, desired_vz, foot_vy, foot_vz))
    else:
        features = np.asarray((1.0, foot_vy, foot_vz), dtype=np.float64)
    force = np.asarray(actor.task_space_actor_weight_matrix, dtype=np.float64) @ features
    lateral = float(
        np.clip(
            force[0],
            (
                actor.minimum_lateral_force_n
                if target_conditioned
                else -actor.maximum_lateral_force_n
            ),
            actor.maximum_lateral_force_n,
        )
    )
    vertical = float(
        np.clip(
            force[1],
            (
                actor.minimum_vertical_force_n
                if target_conditioned
                else -actor.maximum_vertical_force_n
            ),
            actor.maximum_vertical_force_n,
        )
    )
    torque = jacobian[1, 6:35] * lateral + jacobian[2, 6:35] * vertical
    if torque.shape != (29,) or not np.all(np.isfinite(torque)):
        raise FloatingPointError("contact impulse actor emitted invalid joint torque")
    return G1BallisticContactImpulseEffect(
        torque=torque,
        lateral_force_n=lateral,
        vertical_force_n=vertical,
        foot_lateral_speed_mps=foot_vy,
        foot_vertical_speed_mps=foot_vz,
        active=bool(abs(lateral) > 0.0 or abs(vertical) > 0.0),
        desired_lateral_launch_speed_mps=desired_vy,
        desired_vertical_launch_speed_mps=desired_vz,
        target_conditioned=target_conditioned,
    )


__all__ = [
    "G1BallisticContactImpulseActor",
    "G1BallisticContactImpulseEffect",
    "derive_g1_ballistic_contact_impulse_actor",
    "g1_ballistic_contact_impulse_context_hash",
    "g1_ballistic_contact_impulse_effect",
    "load_g1_ballistic_contact_impulse_actor",
]

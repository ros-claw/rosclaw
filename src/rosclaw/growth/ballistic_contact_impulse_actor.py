"""Evidence-trained proprioceptive impulse actor for G1 ball contact.

The actor is distilled from strict teacher exploration.  At runtime it sees
only foot velocity and foot--ball distance, emits a bounded lateral/vertical
task-space impulse, and decodes that impulse to the six right-leg joint
torques with the measured MuJoCo Jacobian.  The artifact and runtime are
SIM-only; neither authorizes a hardware controller or an online hot swap.
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
    return canonical_hash(
        {
            "flow_config_without_teacher": context_flow,
            "goal_spec": goal_spec,
            "runup_config": runup_config,
            "sonic_runup_config": sonic_runup_config,
            "approach_strike_candidate_hash": approach_strike_candidate_hash,
        }
    )


@dataclass(frozen=True)
class G1BallisticContactImpulseActor:
    """A bounded two-output linear actor plus a fixed Jacobian decoder."""

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
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_impulse_actor.v1"

    def __post_init__(self) -> None:
        if not self.body_hash.startswith("sha256:") or not self.implementation_hash.startswith(
            "sha256:"
        ):
            raise ValueError(
                "contact impulse actor requires SHA-256 Body and implementation hashes"
            )
        if not self.experiment_context_hash.startswith("sha256:"):
            raise ValueError("contact impulse actor context hash must be SHA-256")
        if len(self.source_evidence_hashes) < 8 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("contact impulse actor requires eight bound evidence hashes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("contact impulse actor evidence hashes must be unique")
        if self.selected_evidence_hash not in self.source_evidence_hashes:
            raise ValueError("selected contact impulse evidence is not source-bound")
        weights = np.asarray(self.task_space_actor_weight_matrix, dtype=np.float64)
        if weights.shape != (2, 3) or not np.all(np.isfinite(weights)):
            raise ValueError("contact impulse actor weights must have finite shape (2, 3)")
        if not 10.0 <= self.maximum_lateral_force_n <= 250.0:
            raise ValueError("contact impulse actor lateral force limit is invalid")
        if not 10.0 <= self.maximum_vertical_force_n <= 250.0:
            raise ValueError("contact impulse actor vertical force limit is invalid")
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
        if self.precision_success_count < 2 or self.rejected_probe_count < 2:
            raise ValueError("contact impulse actor needs successful and rejected support")
        if not 0.01 <= self.qualified_error_max_m <= 1.0:
            raise ValueError("contact impulse actor precision threshold is invalid")
        if not 0.0 <= self.selected_goal_plane_target_error_m <= self.qualified_error_max_m:
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
        value: dict[str, Any] = {
            **asdict(self),
            "source_evidence_hashes": list(self.source_evidence_hashes),
            "task_space_actor_weight_matrix": [
                list(row) for row in self.task_space_actor_weight_matrix
            ],
            "feature_names": ["bias", "right_foot_vy_mps", "right_foot_vz_mps"],
            "output_names": ["lateral_force_n", "vertical_force_n"],
            "decoder": "measured_right_foot_jacobian_transpose_to_joint_torque",
            "algorithm": "strict_replay_supported_contextual_bandit_distillation",
            "direct_joint_torque_output": True,
            "online_hot_swap_allowed": False,
            "sealed_generalization_evidence": False,
            "stability_plasticity_contract": {
                "stability": "rejected probes remain bound and runtime force is clipped",
                "plasticity": "only a strictly replayed precision-success actor is selected",
            },
        }
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
    *, evidence_paths: tuple[Path, ...], output_path: Path, source_checkout: Path
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
    rows: list[tuple[float, float, float, bool, dict[str, Any], str]] = []
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
        if claims.get("sim_only_operational_space_loft_teacher") is not True:
            raise ValueError("contact impulse actor probes must execute the SIM teacher")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("contact impulse actor trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        context_hashes.add(
            g1_ballistic_contact_impulse_context_hash(
                flow_config=flow,
                goal_spec=dict(evidence.get("goal_spec", {})),
                runup_config=dict(evidence.get("runup_config", {})),
                sonic_runup_config=(
                    None
                    if evidence.get("sonic_runup_config") is None
                    else dict(evidence["sonic_runup_config"])
                ),
                approach_strike_candidate_hash=evidence.get("approach_strike_candidate_hash"),
            )
        )
        result = dict(evidence.get("result", {}))
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
        safe = bool(
            math.isfinite(error)
            and result.get("kick_contact_observed") is True
            and result.get("goal_mouth_hit") is True
            and result.get("perceptual_continuity_passed") is True
            and result.get("post_kick_fall") is False
            and result.get("joint_limit_violation") is False
            and result.get("torque_limit_violation") is False
            and result.get("actuator_saturation") is not True
            and result.get("torque_authority_projection_qualified", True) is True
            and contact_task_scale >= 0.95
        )
        evidence_hash = _file_hash(path)
        if evidence_hash in source_hashes:
            raise ValueError("contact impulse actor evidence contents must be unique")
        source_hashes.append(evidence_hash)
        rows.append(
            (
                error,
                projection_fraction,
                preprojection_demand,
                safe and error <= precision,
                flow,
                evidence_hash,
            )
        )
    if (
        len(body_hashes) != 1
        or len(implementation_hashes) != 1
        or len(context_hashes) != 1
        or len(precision_radii) != 1
    ):
        raise ValueError("contact impulse actor probe contexts disagree")
    qualified = [row for row in rows if row[3]]
    rejected = [row for row in rows if not row[3]]
    if len(qualified) < 2 or len(rejected) < 2:
        raise ValueError("contact impulse actor needs two precision successes and two rejects")
    error, _, _, _, selected, selected_hash = min(
        qualified,
        key=lambda row: (
            row[0] + 2.0 * row[1] + 0.02 * max(0.0, row[2] - 1.0),
            row[0],
        ),
    )
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
) -> G1BallisticContactImpulseEffect:
    """Run the learned proprioceptive actor and decode direct joint torques."""

    import mujoco

    zero: NDArray[np.float64] = np.zeros(29, dtype=np.float64)
    if contact_observed or not actor.start_policy_frame <= policy_frame <= actor.end_policy_frame:
        return G1BallisticContactImpulseEffect(zero, 0.0, 0.0, 0.0, 0.0, False)
    foot_rotation = np.asarray(data.xmat[right_ankle_body_id], dtype=np.float64).reshape(3, 3)
    foot_point = np.asarray(
        data.xpos[right_ankle_body_id], dtype=np.float64
    ) + foot_rotation @ np.asarray(actor.foot_strike_point_offset_m, dtype=np.float64)
    ball = np.asarray(ball_position, dtype=np.float64)
    if ball.shape != (3,) or not np.all(np.isfinite(ball)):
        raise ValueError("contact impulse actor requires a finite ball position")
    if float(np.linalg.norm(foot_point - ball)) > actor.maximum_foot_ball_distance_m:
        return G1BallisticContactImpulseEffect(zero, 0.0, 0.0, 0.0, 0.0, False)
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
    features = np.asarray((1.0, foot_vy, foot_vz), dtype=np.float64)
    force = np.asarray(actor.task_space_actor_weight_matrix, dtype=np.float64) @ features
    lateral = float(
        np.clip(force[0], -actor.maximum_lateral_force_n, actor.maximum_lateral_force_n)
    )
    vertical = float(
        np.clip(force[1], -actor.maximum_vertical_force_n, actor.maximum_vertical_force_n)
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
    )


__all__ = [
    "G1BallisticContactImpulseActor",
    "G1BallisticContactImpulseEffect",
    "derive_g1_ballistic_contact_impulse_actor",
    "g1_ballistic_contact_impulse_context_hash",
    "g1_ballistic_contact_impulse_effect",
    "load_g1_ballistic_contact_impulse_actor",
]

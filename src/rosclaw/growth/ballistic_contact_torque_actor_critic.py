"""Island-bound episodic actor-critic for SIM-only G1 contact torques.

Each strict MuJoCo free-kick is one contextual transition.  Rejected contact
islands are retained as a stability boundary, while the continuous critic is
fit only inside the qualified high-ball island.  The resulting action is a
development proposal: it must be replayed and cannot activate a controller or
reach hardware.
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
from rosclaw.growth.ballistic_contact_residual import (
    G1_BALLISTIC_CONTACT_JOINT_NAMES,
)
from rosclaw.growth.ballistic_contact_torque_residual import (
    G1BallisticContactTorqueResidualConfig,
)

G1_BALLISTIC_COUNTERBALANCE_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_ankle_pitch_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
)
G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES = (
    *G1_BALLISTIC_CONTACT_JOINT_NAMES,
    *G1_BALLISTIC_COUNTERBALANCE_JOINT_NAMES,
)
_ACTION_LIMITS_NM = np.asarray((12.0,) * 6 + (6.0,) * 6, dtype=np.float64)
_ACTION_SCALES_NM = np.asarray((4.0,) * 6 + (2.0,) * 6, dtype=np.float64)
_MISS_PENALTY_M = 2.5
_MAX_ISLAND_ERROR_M = 0.75
_MIN_ISLAND_CROSSING_HEIGHT_M = 0.65


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactTorqueProbe:
    evidence_path: str
    evidence_hash: str
    action_nm: tuple[float, ...]
    reward: float
    goal_plane_target_error_m: float
    goal_crossing_height_m: float | None
    launch_vertical_speed_mps: float
    actuator_saturation_steps: int
    actuator_peak_demand_ratio: float
    perceptual_continuity_passed: bool
    hard_safe: bool
    qualified_contact_island: bool

    def __post_init__(self) -> None:
        if len(self.action_nm) != len(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES):
            raise ValueError("contact torque probe requires twelve action values")
        if not all(math.isfinite(value) for value in self.action_nm):
            raise ValueError("contact torque probe action must be finite")


@dataclass(frozen=True)
class G1BallisticContactTorqueActorCritic:
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    probes: tuple[G1BallisticContactTorqueProbe, ...]
    qualified_probe_count: int
    rejected_probe_count: int
    active_action_dimensions: tuple[int, ...]
    frozen_action_dimensions: tuple[int, ...]
    action_unique_counts: tuple[int, ...]
    critic_coefficients: tuple[float, ...]
    critic_leave_one_out_rmse: float
    best_observed_action_nm: tuple[float, ...]
    best_observed_reward: float
    proposed_action_nm: tuple[float, ...]
    proposed_action_dimensions: tuple[int, ...]
    critic_predicted_best_observed_reward: float
    predicted_proposed_reward: float
    predicted_improvement_over_anchor: float
    minimum_predicted_improvement: float
    maximum_critic_leave_one_out_rmse: float
    nearest_qualified_action_distance: float
    nearest_rejected_action_distance: float
    qualified_island_margin: float
    sim_replay_recommended: bool
    trust_region_radius_nm: float
    ridge_regularization: float
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_torque_actor_critic.v2"

    def __post_init__(self) -> None:
        for label, value in (
            ("Body", self.body_hash),
            ("implementation", self.implementation_hash),
            ("experiment context", self.experiment_context_hash),
        ):
            if not value.startswith("sha256:"):
                raise ValueError(f"contact torque actor {label} hash must be SHA-256")
        if len(self.source_evidence_hashes) < 8 or any(
            not value.startswith("sha256:") for value in self.source_evidence_hashes
        ):
            raise ValueError("contact torque actor requires eight evidence hashes")
        if len(set(self.source_evidence_hashes)) != len(self.source_evidence_hashes):
            raise ValueError("contact torque actor evidence hashes must be unique")
        if len(self.probes) != len(self.source_evidence_hashes):
            raise ValueError("contact torque actor probe lineage is incomplete")
        if self.qualified_probe_count + self.rejected_probe_count != len(self.probes):
            raise ValueError("contact torque actor probe counts disagree")
        dimension_count = len(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES)
        if len(self.action_unique_counts) != dimension_count:
            raise ValueError("contact torque actor support counts are incomplete")
        active = set(self.active_action_dimensions)
        frozen = set(self.frozen_action_dimensions)
        if active & frozen or active | frozen != set(range(dimension_count)):
            raise ValueError("contact torque actor dimension partition is invalid")
        for label, action in (
            ("best", self.best_observed_action_nm),
            ("proposal", self.proposed_action_nm),
        ):
            values = np.asarray(action, dtype=np.float64)
            if (
                values.shape != (dimension_count,)
                or not np.all(np.isfinite(values))
                or np.any(np.abs(values) > _ACTION_LIMITS_NM)
            ):
                raise ValueError(f"contact torque actor {label} action is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
        ):
            raise ValueError("contact torque actor must remain SIM_ONLY")

    @property
    def candidate_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "feature_names": list(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES),
            "right_leg_joint_names": list(G1_BALLISTIC_CONTACT_JOINT_NAMES),
            "counterbalance_joint_names": list(G1_BALLISTIC_COUNTERBALANCE_JOINT_NAMES),
            "probes": [asdict(probe) for probe in self.probes],
            "algorithm": "rejected_island_bound_coordinate_quadratic_actor_critic",
            "critic_target": "measured_goal_error_plus_vertical_and_safety_cost",
            "stability_plasticity_contract": {
                "stability": "rejected islands bound the actor and the best safe replay is frozen",
                "plasticity": "one independently supported torque coordinate may change per cycle",
            },
            "direct_torque_output": True,
            "online_hot_swap_allowed": False,
            "sealed_generalization_evidence": False,
        }
        if include_hash:
            value["candidate_hash"] = self.candidate_hash
        return value


def derive_g1_ballistic_contact_torque_actor_critic(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    trust_region_radius_nm: float = 0.50,
    ridge_regularization: float = 0.02,
    actor_step_size_nm: float = 0.05,
    actor_steps: int = 40,
) -> G1BallisticContactTorqueActorCritic:
    """Fit an island-conditioned critic and emit one replay-only proposal."""

    if len(evidence_paths) < 8:
        raise ValueError("contact torque actor-critic requires at least eight strict probes")
    if not 0.10 <= trust_region_radius_nm <= 1.50:
        raise ValueError("contact torque actor trust radius must be in [0.10, 1.50] Nm")
    if not 1e-5 <= ridge_regularization <= 1.0:
        raise ValueError("contact torque critic ridge must be in [1e-5, 1.0]")
    if not 0.01 <= actor_step_size_nm <= 0.20 or not 4 <= actor_steps <= 200:
        raise ValueError("contact torque actor optimization settings are invalid")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("contact torque actor evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("contact torque actor output already exists")

    probes: list[G1BallisticContactTorqueProbe] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    seen_actions: set[tuple[float, ...]] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("contact torque actor requires strict replay evidence")
        if (
            dict(evidence.get("claims", {})).get(
                "bounded_sim_only_ballistic_contact_torque_residual"
            )
            is not True
        ):
            raise ValueError("contact torque actor requires executed SIM-only torque evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("contact torque actor trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        raw_action = flow.pop("ballistic_contact_torque_residual_nm", None)
        raw_counterbalance = flow.pop("ballistic_counterbalance_torque_residual_nm", [0.0] * 6)
        if (
            not isinstance(raw_action, list)
            or len(raw_action) != 6
            or not isinstance(raw_counterbalance, list)
            or len(raw_counterbalance) != 6
        ):
            raise ValueError("contact torque actor probe lacks its six-joint action")
        action = tuple(float(value) for value in (*raw_action, *raw_counterbalance))
        G1BallisticContactTorqueResidualConfig(
            right_leg_residual_nm=action[:6],
            counterbalance_residual_nm=action[6:],
        )
        if action in seen_actions:
            raise ValueError("contact torque actor actions must be independent")
        seen_actions.add(action)
        probes.append(
            _probe_from_result(
                path=path,
                action=action,
                result=dict(evidence.get("result", {})),
            )
        )
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config_without_torque_action": flow,
                    "goal_spec": evidence.get("goal_spec"),
                    "runup_config": evidence.get("runup_config"),
                    "sonic_runup_config": evidence.get("sonic_runup_config"),
                    "approach_strike_candidate_hash": evidence.get(
                        "approach_strike_candidate_hash"
                    ),
                }
            )
        )
    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("contact torque actor Body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("contact torque actor implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("contact torque actor experiment contexts disagree")

    qualified = [probe for probe in probes if probe.qualified_contact_island]
    rejected = [probe for probe in probes if not probe.qualified_contact_island]
    if len(qualified) < 5 or len(rejected) < 2:
        raise ValueError("contact torque actor needs five qualified and two rejected islands")
    actions = np.asarray([probe.action_nm for probe in qualified], dtype=np.float64)
    rewards = np.asarray([probe.reward for probe in qualified], dtype=np.float64)
    active_dimensions, unique_counts = _supported_action_dimensions(actions)
    frozen_dimensions = tuple(
        index
        for index in range(len(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES))
        if index not in active_dimensions
    )
    design = _critic_design(actions, active_dimensions)
    compact_coefficients = _ridge_fit(design, rewards, ridge_regularization)
    coefficients = _expand_coefficients(compact_coefficients, active_dimensions)
    loo_errors: list[float] = []
    for index in range(len(qualified)):
        keep = np.arange(len(qualified)) != index
        folded = _ridge_fit(design[keep], rewards[keep], ridge_regularization)
        loo_errors.append(float(design[index] @ folded - rewards[index]))
    loo_rmse = float(np.sqrt(np.mean(np.square(loo_errors))))
    best_index = int(np.argmax(rewards))
    best = actions[best_index].copy()
    support_min = np.min(actions, axis=0)
    support_max = np.max(actions, axis=0)
    lower = np.maximum(-_ACTION_LIMITS_NM, best - trust_region_radius_nm)
    upper = np.minimum(_ACTION_LIMITS_NM, best + trust_region_radius_nm)
    lower = np.maximum(lower, support_min - 0.10)
    upper = np.minimum(upper, support_max + 0.10)
    for dimension in frozen_dimensions:
        lower[dimension] = best[dimension]
        upper[dimension] = best[dimension]
    actor, proposed_dimensions = _coordinate_proposal(
        best=best,
        actions=actions,
        coefficients=coefficients,
        lower=lower,
        upper=upper,
        active_dimensions=active_dimensions,
        step_size=actor_step_size_nm,
        steps=actor_steps,
    )
    predicted_best = float(
        _critic_design(best[None, :], active_dimensions)[0] @ compact_coefficients
    )
    predicted = float(_critic_design(actor[None, :], active_dimensions)[0] @ compact_coefficients)
    qualified_distance = _nearest_action_distance(actor, actions)
    rejected_actions = np.asarray([probe.action_nm for probe in rejected], dtype=np.float64)
    rejected_distance = _nearest_action_distance(actor, rejected_actions)
    island_margin = rejected_distance - qualified_distance
    predicted_improvement = predicted - predicted_best
    minimum_improvement = max(0.002, 0.10 * loo_rmse)
    maximum_loo_rmse = 0.12
    report = G1BallisticContactTorqueActorCritic(
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(probe.evidence_hash for probe in probes),
        probes=tuple(probes),
        qualified_probe_count=len(qualified),
        rejected_probe_count=len(rejected),
        active_action_dimensions=active_dimensions,
        frozen_action_dimensions=frozen_dimensions,
        action_unique_counts=unique_counts,
        critic_coefficients=tuple(float(value) for value in coefficients),
        critic_leave_one_out_rmse=loo_rmse,
        best_observed_action_nm=tuple(float(value) for value in best),
        best_observed_reward=float(rewards[best_index]),
        proposed_action_nm=tuple(float(value) for value in actor),
        proposed_action_dimensions=proposed_dimensions,
        critic_predicted_best_observed_reward=predicted_best,
        predicted_proposed_reward=predicted,
        predicted_improvement_over_anchor=predicted_improvement,
        minimum_predicted_improvement=minimum_improvement,
        maximum_critic_leave_one_out_rmse=maximum_loo_rmse,
        nearest_qualified_action_distance=qualified_distance,
        nearest_rejected_action_distance=rejected_distance,
        qualified_island_margin=island_margin,
        sim_replay_recommended=bool(
            predicted_improvement >= minimum_improvement
            and loo_rmse <= maximum_loo_rmse
            and island_margin > 0.0
        ),
        trust_region_radius_nm=trust_region_radius_nm,
        ridge_regularization=ridge_regularization,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _probe_from_result(
    *, path: Path, action: tuple[float, ...], result: dict[str, Any]
) -> G1BallisticContactTorqueProbe:
    raw_error = result.get("goal_plane_target_error_m")
    error = (
        float(raw_error)
        if isinstance(raw_error, (int, float)) and math.isfinite(float(raw_error))
        else _MISS_PENALTY_M
    )
    launch = result.get("ball_launch_velocity_xyz_mps")
    launch_vz = (
        float(launch[2])
        if isinstance(launch, list)
        and len(launch) == 3
        and isinstance(launch[2], (int, float))
        and math.isfinite(float(launch[2]))
        else 0.0
    )
    crossing = result.get("goal_crossing_xyz_m")
    crossing_height = (
        float(crossing[2])
        if isinstance(crossing, list)
        and len(crossing) == 3
        and isinstance(crossing[2], (int, float))
        and math.isfinite(float(crossing[2]))
        else None
    )
    saturation_steps = int(result.get("actuator_saturation_steps", 0))
    demand = float(result.get("actuator_peak_demand_ratio", 10.0))
    continuity = result.get("perceptual_continuity_passed") is True
    hard_safe = bool(
        result.get("finite_state") is True
        and result.get("post_kick_fall") is False
        and result.get("joint_limit_violation") is False
        and result.get("torque_limit_violation") is False
    )
    qualified = bool(
        hard_safe
        and continuity
        and result.get("kick_contact_observed") is True
        and result.get("goal_crossed") is True
        and error <= _MAX_ISLAND_ERROR_M
        and crossing_height is not None
        and crossing_height >= _MIN_ISLAND_CROSSING_HEIGHT_M
    )
    reward = (
        -error + 0.04 * (launch_vz - 4.0) - 0.004 * saturation_steps - 0.10 * max(0.0, demand - 1.0)
    )
    return G1BallisticContactTorqueProbe(
        evidence_path=str(path),
        evidence_hash=_file_hash(path),
        action_nm=action,
        reward=reward,
        goal_plane_target_error_m=error,
        goal_crossing_height_m=crossing_height,
        launch_vertical_speed_mps=launch_vz,
        actuator_saturation_steps=saturation_steps,
        actuator_peak_demand_ratio=demand,
        perceptual_continuity_passed=continuity,
        hard_safe=hard_safe,
        qualified_contact_island=qualified,
    )


def _supported_action_dimensions(
    actions: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    rounded = np.round(actions, decimals=9)
    unique_counts = tuple(
        int(np.unique(rounded[:, index]).size) for index in range(actions.shape[1])
    )
    selected: list[int] = []
    current: NDArray[np.float64] = np.ones((len(actions), 1), dtype=np.float64)
    current_rank = 1
    for dimension in sorted(
        (index for index, count in enumerate(unique_counts) if count >= 3),
        key=lambda index: unique_counts[index],
        reverse=True,
    ):
        scaled = actions[:, dimension] / _ACTION_SCALES_NM[dimension]
        proposed = np.column_stack((current, scaled, np.square(scaled)))
        rank = int(np.linalg.matrix_rank(proposed, tol=1e-10))
        if rank == current_rank + 2:
            selected.append(dimension)
            current = proposed
            current_rank = rank
    if not selected:
        raise ValueError("contact torque actor has no independently supported action axis")
    return tuple(sorted(selected)), unique_counts


def _critic_design(actions: np.ndarray, active_dimensions: tuple[int, ...]) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    scaled = (
        values[:, active_dimensions]
        / _ACTION_SCALES_NM[np.asarray(active_dimensions, dtype=np.int64)]
    )
    return np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), scaled, np.square(scaled)),
        axis=1,
    )


def _ridge_fit(design: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


def _expand_coefficients(compact: np.ndarray, active_dimensions: tuple[int, ...]) -> np.ndarray:
    action_width = len(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES)
    expanded: NDArray[np.float64] = np.zeros(1 + 2 * action_width, dtype=np.float64)
    expanded[0] = compact[0]
    width = len(active_dimensions)
    for offset, dimension in enumerate(active_dimensions):
        expanded[1 + dimension] = compact[1 + offset]
        expanded[1 + action_width + dimension] = compact[1 + width + offset]
    return expanded


def _coordinate_proposal(
    *,
    best: np.ndarray,
    actions: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_dimensions: tuple[int, ...],
    step_size: float,
    steps: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    candidates: list[tuple[float, int, np.ndarray]] = []
    for dimension in active_dimensions:
        actor = best.copy()
        for _ in range(steps):
            action_width = len(G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES)
            scaled = actor[dimension] / _ACTION_SCALES_NM[dimension]
            gradient = (
                coefficients[1 + dimension]
                + 2.0 * coefficients[1 + action_width + dimension] * scaled
            ) / _ACTION_SCALES_NM[dimension]
            if abs(gradient) <= 1e-12:
                break
            actor[dimension] = np.clip(
                actor[dimension] + step_size * np.sign(gradient),
                lower[dimension],
                upper[dimension],
            )
        if any(np.allclose(actor, row, rtol=0.0, atol=1e-6) for row in actions):
            actor[dimension] = np.clip(
                actor[dimension] + 0.01,
                lower[dimension],
                upper[dimension],
            )
        predicted = float(
            coefficients[0]
            + coefficients[1 : 1 + action_width] @ (actor / _ACTION_SCALES_NM)
            + coefficients[1 + action_width :] @ np.square(actor / _ACTION_SCALES_NM)
        )
        candidates.append((predicted, dimension, actor))
    _, dimension, actor = max(candidates, key=lambda item: (item[0], -item[1]))
    return actor, (dimension,)


def _nearest_action_distance(action: np.ndarray, anchors: np.ndarray) -> float:
    return float(
        np.min(
            np.linalg.norm(
                (anchors - action[None, :]) / _ACTION_SCALES_NM,
                axis=1,
            )
        )
    )


__all__ = [
    "G1BallisticContactTorqueActorCritic",
    "G1BallisticContactTorqueProbe",
    "G1_BALLISTIC_COUNTERBALANCE_JOINT_NAMES",
    "G1_BALLISTIC_WHOLE_BODY_TORQUE_FEATURE_NAMES",
    "derive_g1_ballistic_contact_torque_actor_critic",
]

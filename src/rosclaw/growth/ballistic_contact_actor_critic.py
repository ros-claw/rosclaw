"""Replay-stabilized episodic actor-critic for G1 ballistic contact actions.

The learner treats each strict MuJoCo free-kick episode as one contextual
bandit transition.  A regularized quadratic critic models the measured task
reward over the six-dimensional contact residual, and a trust-region actor
ascends that critic from the best observed safe action.  The actor changes one
independently supported joint at a time: sparse one-axis probes do not justify
assuming that simultaneous joint changes have no interaction.  It emits a
SIM_ONLY development proposal; it never activates or promotes the proposal.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_residual import (
    G1_BALLISTIC_CONTACT_JOINT_NAMES,
    G1BallisticContactResidualConfig,
)

_ACTION_LIMIT_RAD = 0.25
_MISS_PENALTY_M = 2.5


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactProbe:
    evidence_path: str
    evidence_hash: str
    action_rad: tuple[float, ...]
    reward: float
    goal_plane_target_error_m: float
    launch_vertical_speed_mps: float
    contact_height_relative_ball_center_m: float
    actuator_saturation_steps: int
    actuator_peak_demand_ratio: float
    perceptual_continuity_passed: bool
    hard_safe: bool

    def __post_init__(self) -> None:
        if len(self.action_rad) != 6 or not all(math.isfinite(v) for v in self.action_rad):
            raise ValueError("ballistic contact probe action must contain six finite values")
        numeric = (
            self.reward,
            self.goal_plane_target_error_m,
            self.launch_vertical_speed_mps,
            self.contact_height_relative_ball_center_m,
            self.actuator_peak_demand_ratio,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("ballistic contact probe metrics must be finite")
        if not self.evidence_hash.startswith("sha256:"):
            raise ValueError("ballistic contact probe requires an evidence hash")


@dataclass(frozen=True)
class G1BallisticContactActorCritic:
    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    probes: tuple[G1BallisticContactProbe, ...]
    active_action_dimensions: tuple[int, ...]
    frozen_action_dimensions: tuple[int, ...]
    action_unique_counts: tuple[int, ...]
    critic_coefficients: tuple[float, ...]
    critic_leave_one_out_rmse: float
    best_observed_action_rad: tuple[float, ...]
    best_observed_reward: float
    proposed_action_rad: tuple[float, ...]
    proposed_action_dimensions: tuple[int, ...]
    critic_predicted_best_observed_reward: float
    predicted_proposed_reward: float
    predicted_improvement_over_anchor: float
    minimum_predicted_improvement: float
    maximum_critic_leave_one_out_rmse: float
    sim_replay_recommended: bool
    trust_region_radius_rad: float
    ridge_regularization: float
    replay_anchor_count: int
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.growth.g1_ballistic_contact_actor_critic.v4"

    @property
    def candidate_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "joint_names": list(G1_BALLISTIC_CONTACT_JOINT_NAMES),
            "probes": [asdict(probe) for probe in self.probes],
            "algorithm": ("support_rank_constrained_coordinate_quadratic_contextual_actor_critic"),
            "critic_target": "measured_goal_error_plus_safety_cost",
            "direct_torque_output": False,
            "online_hot_swap_allowed": False,
            "sealed_generalization_evidence": False,
        }
        if include_hash:
            value["candidate_hash"] = self.candidate_hash
        return value


def derive_g1_ballistic_contact_actor_critic(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    trust_region_radius_rad: float = 0.06,
    ridge_regularization: float = 0.02,
    actor_step_size_rad: float = 0.01,
    actor_steps: int = 48,
) -> G1BallisticContactActorCritic:
    """Fit a development critic and emit one bounded online actor proposal."""

    if len(evidence_paths) < 8:
        raise ValueError("ballistic actor-critic requires at least eight strict probes")
    if not 0.01 <= trust_region_radius_rad <= 0.10:
        raise ValueError("ballistic actor trust radius must be in [0.01, 0.10] rad")
    if not 1e-5 <= ridge_regularization <= 1.0:
        raise ValueError("ballistic critic ridge must be in [1e-5, 1.0]")
    if not 0.001 <= actor_step_size_rad <= 0.03 or not 4 <= actor_steps <= 200:
        raise ValueError("ballistic actor optimization settings are invalid")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("ballistic actor-critic evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("ballistic actor-critic output already exists")

    probes: list[G1BallisticContactProbe] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    seen_actions: set[tuple[float, ...]] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve()
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if evidence.get("strict_replay") is not True:
            raise ValueError("ballistic actor-critic requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("ballistic actor-critic trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        raw_action = flow.pop("ballistic_contact_residual_rad", None)
        if not isinstance(raw_action, list):
            raise ValueError("ballistic actor-critic probe lacks its six-joint action")
        action = tuple(float(value) for value in raw_action)
        G1BallisticContactResidualConfig(right_leg_residual_rad=action)
        if action in seen_actions:
            raise ValueError("ballistic actor-critic actions must be independent")
        seen_actions.add(action)
        result = dict(evidence.get("result", {}))
        probe = _probe_from_result(path=path, action=action, result=result)
        probes.append(probe)
        context_hashes.add(
            canonical_hash(
                {
                    "flow_config_without_actor_action": flow,
                    "goal_spec": evidence.get("goal_spec"),
                    "runup_config": evidence.get("runup_config"),
                    "sonic_runup_config": evidence.get("sonic_runup_config"),
                    "approach_strike_candidate_hash": evidence.get(
                        "approach_strike_candidate_hash"
                    ),
                    "football_motion_prior_hash": flow.get("football_motion_prior_hash"),
                }
            )
        )
    if len(body_hashes) != 1 or not next(iter(body_hashes)).startswith("sha256:"):
        raise ValueError("ballistic actor-critic Body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("ballistic actor-critic implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("ballistic actor-critic experiment contexts disagree")

    actions = np.asarray([probe.action_rad for probe in probes], dtype=np.float64)
    rewards = np.asarray([probe.reward for probe in probes], dtype=np.float64)
    active_dimensions, unique_counts = _supported_action_dimensions(actions)
    frozen_dimensions = tuple(
        index for index in range(actions.shape[1]) if index not in active_dimensions
    )
    design = _critic_design(actions, active_dimensions=active_dimensions)
    compact_coefficients = _ridge_fit(design, rewards, ridge_regularization)
    coefficients = _expand_critic_coefficients(
        compact_coefficients,
        active_dimensions=active_dimensions,
    )
    loo_errors: list[float] = []
    for index in range(len(probes)):
        keep = np.arange(len(probes)) != index
        folded = _ridge_fit(design[keep], rewards[keep], ridge_regularization)
        loo_errors.append(float(design[index] @ folded - rewards[index]))
    loo_rmse = float(np.sqrt(np.mean(np.square(loo_errors))))
    safe_indices = [
        index
        for index, probe in enumerate(probes)
        if probe.hard_safe and probe.perceptual_continuity_passed
    ]
    if not safe_indices:
        raise ValueError("ballistic actor-critic has no safe continuous replay anchor")
    best_index = max(safe_indices, key=lambda index: probes[index].reward)
    best = actions[best_index].copy()
    lower = np.maximum(-_ACTION_LIMIT_RAD, best - trust_region_radius_rad)
    upper = np.minimum(_ACTION_LIMIT_RAD, best + trust_region_radius_rad)
    for dimension in frozen_dimensions:
        lower[dimension] = best[dimension]
        upper[dimension] = best[dimension]
    actor, proposed_dimensions = _coordinate_actor_proposal(
        best=best,
        actions=actions,
        coefficients=coefficients,
        lower=lower,
        upper=upper,
        active_dimensions=active_dimensions,
        actor_step_size_rad=actor_step_size_rad,
        actor_steps=actor_steps,
    )
    predicted_best = float(
        _critic_design(best[None, :], active_dimensions=active_dimensions)[0] @ compact_coefficients
    )
    predicted = float(
        _critic_design(actor[None, :], active_dimensions=active_dimensions)[0]
        @ compact_coefficients
    )
    predicted_improvement = predicted - predicted_best
    minimum_improvement = max(0.005, 0.10 * loo_rmse)
    maximum_critic_loo_rmse = 0.15
    report = G1BallisticContactActorCritic(
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(probe.evidence_hash for probe in probes),
        probes=tuple(probes),
        active_action_dimensions=active_dimensions,
        frozen_action_dimensions=frozen_dimensions,
        action_unique_counts=unique_counts,
        critic_coefficients=tuple(float(value) for value in coefficients),
        critic_leave_one_out_rmse=loo_rmse,
        best_observed_action_rad=tuple(float(value) for value in best),
        best_observed_reward=float(rewards[best_index]),
        proposed_action_rad=tuple(float(value) for value in actor),
        proposed_action_dimensions=proposed_dimensions,
        critic_predicted_best_observed_reward=predicted_best,
        predicted_proposed_reward=predicted,
        predicted_improvement_over_anchor=predicted_improvement,
        minimum_predicted_improvement=minimum_improvement,
        maximum_critic_leave_one_out_rmse=maximum_critic_loo_rmse,
        sim_replay_recommended=(
            predicted_improvement >= minimum_improvement
            and loo_rmse <= maximum_critic_loo_rmse
        ),
        trust_region_radius_rad=trust_region_radius_rad,
        ridge_regularization=ridge_regularization,
        replay_anchor_count=len(safe_indices),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _probe_from_result(
    *, path: Path, action: tuple[float, ...], result: dict[str, Any]
) -> G1BallisticContactProbe:
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
    contact_height = result.get("kick_contact_height_relative_ball_center_m")
    contact_height_value = (
        float(contact_height)
        if isinstance(contact_height, (int, float))
        and math.isfinite(float(contact_height))
        else -0.25
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
    reward = (
        -error
        + 0.04 * (launch_vz - 4.0)
        - 0.004 * saturation_steps
        - 0.10 * max(0.0, demand - 1.0)
        - (4.0 if not continuity else 0.0)
        - (10.0 if not hard_safe else 0.0)
    )
    return G1BallisticContactProbe(
        evidence_path=str(path),
        evidence_hash=_file_hash(path),
        action_rad=action,
        reward=reward,
        goal_plane_target_error_m=error,
        launch_vertical_speed_mps=launch_vz,
        contact_height_relative_ball_center_m=contact_height_value,
        actuator_saturation_steps=saturation_steps,
        actuator_peak_demand_ratio=demand,
        perceptual_continuity_passed=continuity,
        hard_safe=hard_safe,
    )


def _critic_design(
    actions: np.ndarray,
    *,
    active_dimensions: tuple[int, ...] = tuple(range(6)),
) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("ballistic critic actions must be a finite [N, 6] array")
    if (
        not active_dimensions
        or len(set(active_dimensions)) != len(active_dimensions)
        or any(not 0 <= item < 6 for item in active_dimensions)
    ):
        raise ValueError("ballistic critic active dimensions are invalid")
    scaled = values[:, active_dimensions] / _ACTION_LIMIT_RAD
    return np.concatenate(
        (np.ones((len(values), 1), dtype=np.float64), scaled, np.square(scaled)),
        axis=1,
    )


def _supported_action_dimensions(
    actions: np.ndarray,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Keep only independently identified action axes.

    Three unique values are the minimum needed to distinguish a local linear
    trend from curvature.  The rank test then rejects dimensions that merely
    co-vary with an already selected joint.  This prevents an under-sampled
    critic from inventing gradients on joints that have no independent replay
    support, which is the stability side of the online learning contract.
    """

    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or not np.all(np.isfinite(values)):
        raise ValueError("ballistic critic actions must be a finite [N, 6] array")
    rounded = np.round(values, decimals=9)
    unique_counts = tuple(int(np.unique(rounded[:, index]).size) for index in range(6))
    ranges = np.ptp(values, axis=0)
    candidates = sorted(
        (index for index, count in enumerate(unique_counts) if count >= 3),
        key=lambda index: (unique_counts[index], float(ranges[index]), -index),
        reverse=True,
    )
    current = np.ones((len(values), 1), dtype=np.float64)
    current_rank = 1
    selected: list[int] = []
    for dimension in candidates:
        scaled = values[:, dimension] / _ACTION_LIMIT_RAD
        proposed = np.column_stack((current, scaled, np.square(scaled)))
        proposed_rank = int(np.linalg.matrix_rank(proposed, tol=1e-10))
        if proposed_rank == current_rank + 2:
            selected.append(dimension)
            current = proposed
            current_rank = proposed_rank
    if not selected:
        raise ValueError(
            "ballistic actor-critic has no independently supported action dimension"
        )
    return tuple(sorted(selected)), unique_counts


def _expand_critic_coefficients(
    compact: np.ndarray,
    *,
    active_dimensions: tuple[int, ...],
) -> np.ndarray:
    expected = 1 + 2 * len(active_dimensions)
    if compact.shape != (expected,) or not np.all(np.isfinite(compact)):
        raise ValueError("ballistic compact critic coefficients are invalid")
    expanded = np.zeros(13, dtype=np.float64)
    expanded[0] = compact[0]
    width = len(active_dimensions)
    for offset, dimension in enumerate(active_dimensions):
        expanded[1 + dimension] = compact[1 + offset]
        expanded[7 + dimension] = compact[1 + width + offset]
    return expanded


def _ensure_novel_action(
    actor: np.ndarray,
    *,
    actions: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_dimensions: tuple[int, ...],
) -> np.ndarray:
    def measured(value: np.ndarray) -> bool:
        return any(np.allclose(value, row, rtol=0.0, atol=1e-6) for row in actions)

    if not measured(actor):
        return actor
    for dimension in active_dimensions:
        for direction in (1.0, -1.0):
            candidate = actor.copy()
            candidate[dimension] = np.clip(
                candidate[dimension] + direction * 0.005,
                lower[dimension],
                upper[dimension],
            )
            if not measured(candidate):
                return candidate
    raise ValueError("ballistic actor trust region contains no unmeasured supported action")


def _coordinate_actor_proposal(
    *,
    best: np.ndarray,
    actions: np.ndarray,
    coefficients: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    active_dimensions: tuple[int, ...],
    actor_step_size_rad: float,
    actor_steps: int,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Optimize one supported action axis and reject invented interactions.

    A diagonal quadratic critic can identify multiple independently varied
    axes without identifying their interaction.  Moving all such axes at once
    therefore makes a stronger assumption than the replay contains.  Generate
    one candidate per supported axis and return only the highest-scoring one.
    """

    candidates: list[tuple[float, int, np.ndarray]] = []
    for dimension in active_dimensions:
        actor = best.copy()
        for _ in range(actor_steps):
            scaled = actor[dimension] / _ACTION_LIMIT_RAD
            gradient = (
                coefficients[1 + dimension] + 2.0 * coefficients[7 + dimension] * scaled
            ) / _ACTION_LIMIT_RAD
            if abs(gradient) <= 1e-12:
                break
            actor[dimension] = np.clip(
                actor[dimension] + actor_step_size_rad * np.sign(gradient),
                lower[dimension],
                upper[dimension],
            )
        actor = _ensure_novel_action(
            actor,
            actions=actions,
            lower=lower,
            upper=upper,
            active_dimensions=(dimension,),
        )
        scaled = actor / _ACTION_LIMIT_RAD
        predicted = float(
            coefficients[0] + coefficients[1:7] @ scaled + coefficients[7:13] @ np.square(scaled)
        )
        candidates.append((predicted, dimension, actor))
    _, dimension, actor = max(candidates, key=lambda item: (item[0], -item[1]))
    return actor, (dimension,)


def _ridge_fit(design: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


__all__ = [
    "G1BallisticContactActorCritic",
    "G1BallisticContactProbe",
    "derive_g1_ballistic_contact_actor_critic",
]

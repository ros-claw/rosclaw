"""Full-factorial, interaction-aware actor-critic for G1 ball contact.

The coordinate critic deliberately refuses to invent joint interactions from
one-axis probes.  This learner is the matching second stage: it unlocks a
two-joint proposal only after strict MuJoCo evidence contains a complete,
hard-safe Cartesian grid.  The learned proposal is still a SIM_ONLY request
for another physics replay; it is never activated or promoted here.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth.ballistic_contact_actor_critic import (
    G1BallisticContactProbe,
    g1_ballistic_contact_probe_from_result,
)
from rosclaw.growth.ballistic_contact_residual import (
    G1_BALLISTIC_CONTACT_JOINT_NAMES,
    G1BallisticContactResidualConfig,
)

_SCHEMA = "rosclaw.growth.g1_ballistic_contact_coupled_actor_critic.v1"
_ACTION_LIMIT_RAD = 0.25


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True)
class G1BallisticContactCoupledActorCritic:
    """A bounded two-axis critic with an explicitly measured interaction."""

    body_hash: str
    implementation_hash: str
    experiment_context_hash: str
    source_evidence_hashes: tuple[str, ...]
    probes: tuple[G1BallisticContactProbe, ...]
    coupled_action_dimensions: tuple[int, int]
    frozen_action_dimensions: tuple[int, ...]
    action_grid_values_rad: tuple[tuple[float, ...], tuple[float, ...]]
    critic_coefficients: tuple[float, ...]
    critic_leave_one_out_rmse: float
    interaction_coefficient: float
    best_observed_action_rad: tuple[float, ...]
    best_observed_reward: float
    proposed_action_rad: tuple[float, ...]
    proposed_action_dimensions: tuple[int, int]
    proposal_support_mode: str
    maximum_per_axis_extrapolation_rad: float
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
    direct_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        hashes = (
            self.body_hash,
            self.implementation_hash,
            self.experiment_context_hash,
            *self.source_evidence_hashes,
        )
        if any(not value.startswith("sha256:") for value in hashes):
            raise ValueError("coupled contact critic requires SHA-256 provenance")
        if len(self.source_evidence_hashes) < 9 or len(set(self.source_evidence_hashes)) != len(
            self.source_evidence_hashes
        ):
            raise ValueError("coupled contact critic requires nine unique evidence hashes")
        dimensions = self.coupled_action_dimensions
        if (
            len(set(dimensions)) != 2
            or tuple(sorted(dimensions)) != dimensions
            or any(not 0 <= value < 6 for value in dimensions)
        ):
            raise ValueError("coupled contact critic dimensions are invalid")
        if self.proposed_action_dimensions != dimensions:
            raise ValueError("coupled contact proposal must change both measured axes")
        if self.frozen_action_dimensions != tuple(
            index for index in range(6) if index not in dimensions
        ):
            raise ValueError("coupled contact frozen dimensions are inconsistent")
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.best_observed_action_rad)
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.proposed_action_rad)
        changed = tuple(
            index
            for index, (anchor, proposed) in enumerate(
                zip(self.best_observed_action_rad, self.proposed_action_rad, strict=True)
            )
            if not math.isclose(anchor, proposed, rel_tol=0.0, abs_tol=1e-12)
        )
        if changed != dimensions:
            raise ValueError("coupled contact proposal changed unsupported joints")
        if self.proposal_support_mode not in {"INTERPOLATION", "NEXT_RING_HALF_STEP"}:
            raise ValueError("coupled contact proposal support mode is invalid")
        bounds = tuple((values[0], values[-1]) for values in self.action_grid_values_rad)
        extrapolation = tuple(
            max(0.0, lower - value, value - upper)
            for value, (lower, upper) in zip(
                (self.proposed_action_rad[index] for index in dimensions),
                bounds,
                strict=True,
            )
        )
        measured_extrapolation = max(extrapolation)
        if not math.isclose(
            measured_extrapolation,
            self.maximum_per_axis_extrapolation_rad,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("coupled contact proposal extrapolation is inconsistent")
        if self.proposal_support_mode == "INTERPOLATION" and measured_extrapolation != 0.0:
            raise ValueError("interpolated coupled proposal left measured support")
        if self.proposal_support_mode == "NEXT_RING_HALF_STEP" and not (
            0.0 < measured_extrapolation <= 0.03
        ):
            raise ValueError("coupled next-ring proposal extrapolation is invalid")
        if len(self.critic_coefficients) != 6 or not all(
            math.isfinite(value) for value in self.critic_coefficients
        ):
            raise ValueError("coupled contact critic coefficients are invalid")
        if any(len(values) < 3 for values in self.action_grid_values_rad):
            raise ValueError("coupled contact critic requires a three-by-three grid")
        if len(self.probes) != math.prod(len(values) for values in self.action_grid_values_rad):
            raise ValueError("coupled contact critic grid is incomplete")
        if any(
            not probe.hard_safe or not probe.perceptual_continuity_passed for probe in self.probes
        ):
            raise ValueError("coupled contact critic grid must be hard-safe and continuous")
        metrics = (
            self.critic_leave_one_out_rmse,
            self.interaction_coefficient,
            self.best_observed_reward,
            self.critic_predicted_best_observed_reward,
            self.predicted_proposed_reward,
            self.predicted_improvement_over_anchor,
            self.minimum_predicted_improvement,
            self.maximum_critic_leave_one_out_rmse,
            self.trust_region_radius_rad,
            self.ridge_regularization,
            self.maximum_per_axis_extrapolation_rad,
        )
        if not all(math.isfinite(value) for value in metrics):
            raise ValueError("coupled contact critic metrics must be finite")
        if not 0.0 <= self.critic_leave_one_out_rmse <= 20.0:
            raise ValueError("coupled contact critic cross-validation metric is invalid")
        if not 0.005 <= self.trust_region_radius_rad <= 0.06:
            raise ValueError("coupled contact critic trust radius is invalid")
        if not 1e-5 <= self.ridge_regularization <= 1.0:
            raise ValueError("coupled contact critic ridge is invalid")
        if self.replay_anchor_count != len(self.probes):
            raise ValueError("coupled contact critic safe replay count is inconsistent")
        if (
            self.schema_version != _SCHEMA
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_torque_output
            or self.online_hot_swap_allowed
        ):
            raise ValueError("coupled contact critic cannot authorize activation or hardware")

    @property
    def candidate_hash(self) -> str:
        return canonical_hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            **asdict(self),
            "joint_names": list(G1_BALLISTIC_CONTACT_JOINT_NAMES),
            "coupled_joint_names": [
                G1_BALLISTIC_CONTACT_JOINT_NAMES[index] for index in self.coupled_action_dimensions
            ],
            "probes": [asdict(probe) for probe in self.probes],
            "critic_feature_names": [
                "bias",
                "axis_0",
                "axis_1",
                "axis_0_sq",
                "axis_0_axis_1",
                "axis_1_sq",
            ],
            "algorithm": "full_factorial_two_axis_quadratic_contextual_actor_critic",
            "critic_target": "measured_goal_error_plus_safety_cost",
            "interaction_support": "complete_cartesian_hard_safe_strict_replay_grid",
            "proposal_authority": "simulation_replay_request_only",
            "sealed_generalization_evidence": False,
        }
        if include_hash:
            value["candidate_hash"] = self.candidate_hash
        return value


def derive_g1_ballistic_contact_coupled_actor_critic(
    *,
    evidence_paths: tuple[Path, ...],
    output_path: Path,
    source_checkout: Path,
    trust_region_radius_rad: float = 0.03,
    ridge_regularization: float = 0.01,
) -> G1BallisticContactCoupledActorCritic:
    """Fit an interaction critic only from a complete two-axis replay grid."""

    if len(evidence_paths) < 9:
        raise ValueError("coupled contact critic requires at least nine strict probes")
    if not 0.005 <= trust_region_radius_rad <= 0.06:
        raise ValueError("coupled contact trust radius must be in [0.005, 0.06] rad")
    if not 1e-5 <= ridge_regularization <= 1.0:
        raise ValueError("coupled contact ridge must be in [1e-5, 1.0]")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("coupled contact critic evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("coupled contact critic output already exists")

    probes: list[G1BallisticContactProbe] = []
    body_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    context_hashes: set[str] = set()
    seen_actions: set[tuple[float, ...]] = set()
    for raw_path in evidence_paths:
        path = raw_path.expanduser().resolve(strict=True)
        evidence = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(evidence, dict) or evidence.get("strict_replay") is not True:
            raise ValueError("coupled contact critic requires strict replay evidence")
        trajectory = Path(str(evidence.get("trajectory_path", ""))).resolve()
        if not trajectory.is_file() or evidence.get("trajectory_hash") != _file_hash(trajectory):
            raise ValueError("coupled contact critic trajectory binding is invalid")
        body_hashes.add(str(evidence.get("body_hash", "")))
        implementation_hashes.add(str(evidence.get("implementation_hash", "")))
        flow = dict(evidence.get("flow_config", {}))
        raw_action = flow.pop("ballistic_contact_residual_rad", None)
        if not isinstance(raw_action, list):
            raise ValueError("coupled contact critic probe lacks its six-joint action")
        action = tuple(float(value) for value in raw_action)
        G1BallisticContactResidualConfig(right_leg_residual_rad=action)
        if action in seen_actions:
            raise ValueError("coupled contact critic actions must be unique")
        seen_actions.add(action)
        result = evidence.get("result")
        if not isinstance(result, dict):
            raise ValueError("coupled contact critic probe result is missing")
        probes.append(
            g1_ballistic_contact_probe_from_result(path=path, action=action, result=result)
        )
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
        raise ValueError("coupled contact critic Body hashes disagree")
    if len(implementation_hashes) != 1 or not next(iter(implementation_hashes)).startswith(
        "sha256:"
    ):
        raise ValueError("coupled contact critic implementation hashes disagree")
    if len(context_hashes) != 1:
        raise ValueError("coupled contact critic experiment contexts disagree")
    if any(not probe.hard_safe or not probe.perceptual_continuity_passed for probe in probes):
        raise ValueError("coupled contact critic requires a hard-safe continuous grid")

    actions = np.asarray([probe.action_rad for probe in probes], dtype=np.float64)
    rounded = np.round(actions, decimals=9)
    varying = tuple(index for index in range(6) if np.unique(rounded[:, index]).size > 1)
    if len(varying) != 2 or any(np.unique(rounded[:, index]).size < 3 for index in varying):
        raise ValueError("coupled contact critic requires exactly two three-level action axes")
    frozen = tuple(index for index in range(6) if index not in varying)
    if any(np.unique(rounded[:, index]).size != 1 for index in frozen):
        raise ValueError("coupled contact critic frozen action axes disagree")
    grid_values = tuple(
        tuple(float(value) for value in np.unique(rounded[:, index])) for index in varying
    )
    expected_grid = set(itertools.product(*grid_values))
    measured_grid = {tuple(float(row[index]) for index in varying) for row in rounded}
    if measured_grid != expected_grid or len(probes) != len(expected_grid):
        raise ValueError("coupled contact critic requires a complete Cartesian replay grid")

    centers = np.asarray(
        [0.5 * (values[0] + values[-1]) for values in grid_values], dtype=np.float64
    )
    half_spans = np.asarray(
        [0.5 * (values[-1] - values[0]) for values in grid_values], dtype=np.float64
    )
    if np.any(half_spans <= 1e-6):
        raise ValueError("coupled contact critic action grid is degenerate")
    compact = actions[:, varying]
    design = _critic_design(compact, centers=centers, half_spans=half_spans)
    rewards = np.asarray([probe.reward for probe in probes], dtype=np.float64)
    coefficients = _ridge_fit(design, rewards, ridge_regularization)
    loo_errors: list[float] = []
    for index in range(len(probes)):
        keep = np.arange(len(probes)) != index
        folded = _ridge_fit(design[keep], rewards[keep], ridge_regularization)
        loo_errors.append(float(design[index] @ folded - rewards[index]))
    loo_rmse = float(np.sqrt(np.mean(np.square(loo_errors))))
    best_index = int(np.argmax(rewards))
    best = actions[best_index].copy()
    best_compact = best[list(varying)]
    candidate_axes: list[tuple[float, ...]] = []
    for values in grid_values:
        midpoints = tuple(
            0.5 * (left + right) for left, right in zip(values, values[1:], strict=False)
        )
        lower_step = 0.5 * (values[1] - values[0])
        upper_step = 0.5 * (values[-1] - values[-2])
        candidate_axes.append(
            tuple(
                value
                for value in (
                    max(-_ACTION_LIMIT_RAD, values[0] - lower_step),
                    *midpoints,
                    min(_ACTION_LIMIT_RAD, values[-1] + upper_step),
                )
                if not any(math.isclose(value, measured, abs_tol=1e-12) for measured in values)
            )
        )
    candidates: list[tuple[float, tuple[float, float], str, float]] = []
    for candidate in itertools.product(*candidate_axes):
        candidate_array = np.asarray(candidate, dtype=np.float64)
        if np.any(np.abs(candidate_array - best_compact) > trust_region_radius_rad + 1e-12):
            continue
        if np.any(np.isclose(candidate_array, best_compact, rtol=0.0, atol=1e-12)):
            continue
        predicted = float(
            _critic_design(candidate_array[None, :], centers=centers, half_spans=half_spans)[0]
            @ coefficients
        )
        extrapolation = max(
            max(0.0, values[0] - value, value - values[-1])
            for value, values in zip(candidate, grid_values, strict=True)
        )
        support_mode = "INTERPOLATION" if extrapolation == 0.0 else "NEXT_RING_HALF_STEP"
        candidates.append(
            (
                predicted,
                (float(candidate[0]), float(candidate[1])),
                support_mode,
                extrapolation,
            )
        )
    if not candidates:
        raise ValueError("coupled contact trust region contains no unmeasured two-axis cell")
    predicted, proposed_compact, support_mode, extrapolation = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    proposed = best.copy()
    proposed[list(varying)] = proposed_compact
    predicted_best = float(
        _critic_design(best_compact[None, :], centers=centers, half_spans=half_spans)[0]
        @ coefficients
    )
    improvement = predicted - predicted_best
    # This is only authority to spend one more simulation, not authority to
    # activate the action.  A lower threshold than the downstream measured
    # improvement gate avoids stopping a reliable monotonic boundary search.
    minimum_improvement = max(0.001, 0.10 * loo_rmse)
    maximum_loo = 0.15
    report = G1BallisticContactCoupledActorCritic(
        body_hash=next(iter(body_hashes)),
        implementation_hash=next(iter(implementation_hashes)),
        experiment_context_hash=next(iter(context_hashes)),
        source_evidence_hashes=tuple(probe.evidence_hash for probe in probes),
        probes=tuple(probes),
        coupled_action_dimensions=(varying[0], varying[1]),
        frozen_action_dimensions=frozen,
        action_grid_values_rad=(grid_values[0], grid_values[1]),
        critic_coefficients=tuple(float(value) for value in coefficients),
        critic_leave_one_out_rmse=loo_rmse,
        interaction_coefficient=float(coefficients[4]),
        best_observed_action_rad=tuple(float(value) for value in best),
        best_observed_reward=float(rewards[best_index]),
        proposed_action_rad=tuple(float(value) for value in proposed),
        proposed_action_dimensions=(varying[0], varying[1]),
        proposal_support_mode=support_mode,
        maximum_per_axis_extrapolation_rad=extrapolation,
        critic_predicted_best_observed_reward=predicted_best,
        predicted_proposed_reward=predicted,
        predicted_improvement_over_anchor=improvement,
        minimum_predicted_improvement=minimum_improvement,
        maximum_critic_leave_one_out_rmse=maximum_loo,
        sim_replay_recommended=bool(improvement >= minimum_improvement and loo_rmse <= maximum_loo),
        trust_region_radius_rad=trust_region_radius_rad,
        ridge_regularization=ridge_regularization,
        replay_anchor_count=len(probes),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _critic_design(
    values: np.ndarray, *, centers: np.ndarray, half_spans: np.ndarray
) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 2 or not np.all(np.isfinite(actions)):
        raise ValueError("coupled contact critic actions must have finite shape [N, 2]")
    scaled = (actions - centers) / half_spans
    left = scaled[:, 0]
    right = scaled[:, 1]
    return np.column_stack(
        (np.ones(len(actions)), left, right, left * left, left * right, right * right)
    )


def _ridge_fit(design: np.ndarray, target: np.ndarray, regularization: float) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization
    penalty[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + penalty, design.T @ target)


__all__ = [
    "G1BallisticContactCoupledActorCritic",
    "derive_g1_ballistic_contact_coupled_actor_critic",
]

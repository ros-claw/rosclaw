"""Uncertainty-directed, authority-bounded ballistic contact sampling.

The actor-critic intentionally abstains when sparse contact evidence does not
cross-validate.  This module turns that abstention into a useful next action:
it fills one local evidence gap around the best hard-safe replay while keeping
all other joints frozen.  The proposal remains SIM_ONLY and must be replayed
through the existing PD, joint-boundary and torque-authority projectors.
"""

from __future__ import annotations

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

_ACTOR_SCHEMA = "rosclaw.growth.g1_ballistic_contact_actor_critic.v4"
_SCHEMA = "rosclaw.growth.g1_ballistic_contact_active_sample.v1"
_ACTION_LIMIT_RAD = 0.25


@dataclass(frozen=True)
class G1BallisticContactActiveSample:
    source_candidate_hash: str
    experiment_context_hash: str
    anchor_action_rad: tuple[float, ...]
    proposed_action_rad: tuple[float, ...]
    proposed_action_dimension: int
    proposed_joint_name: str
    local_safe_interval_rad: tuple[float, float]
    local_measured_values_rad: tuple[float, ...]
    uncertainty_gap_rad: float
    maximum_step_rad: float
    proposal_reason: str
    activation_ceiling: str = "SIM_ONLY"
    promotion_authorized: bool = False
    hardware_authorized: bool = False
    direct_torque_output: bool = False
    online_hot_swap_allowed: bool = False
    sim_replay_required: bool = True
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        if not self.source_candidate_hash.startswith("sha256:"):
            raise ValueError("active sample requires a source candidate hash")
        if not self.experiment_context_hash.startswith("sha256:"):
            raise ValueError("active sample requires an experiment context hash")
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.anchor_action_rad)
        G1BallisticContactResidualConfig(right_leg_residual_rad=self.proposed_action_rad)
        if not 0 <= self.proposed_action_dimension < 6:
            raise ValueError("active sample dimension is invalid")
        if (
            self.proposed_joint_name
            != G1_BALLISTIC_CONTACT_JOINT_NAMES[self.proposed_action_dimension]
        ):
            raise ValueError("active sample joint name does not match its dimension")
        changed = tuple(
            index
            for index, (anchor, proposed) in enumerate(
                zip(self.anchor_action_rad, self.proposed_action_rad, strict=True)
            )
            if not math.isclose(anchor, proposed, rel_tol=0.0, abs_tol=1e-12)
        )
        if changed != (self.proposed_action_dimension,):
            raise ValueError("active sample must change exactly one supported joint")
        lower, upper = self.local_safe_interval_rad
        proposed = self.proposed_action_rad[self.proposed_action_dimension]
        if not (-_ACTION_LIMIT_RAD <= lower < proposed < upper <= _ACTION_LIMIT_RAD):
            raise ValueError("active sample is outside its local safe interval")
        if (
            not 0.005 <= self.maximum_step_rad <= 0.06
            or abs(proposed - self.anchor_action_rad[self.proposed_action_dimension])
            > self.maximum_step_rad + 1e-12
        ):
            raise ValueError("active sample exceeds its bounded exploration step")
        if (
            len(self.local_measured_values_rad) < 3
            or tuple(sorted(self.local_measured_values_rad)) != self.local_measured_values_rad
            or not all(math.isfinite(value) for value in self.local_measured_values_rad)
            or not math.isfinite(self.uncertainty_gap_rad)
            or self.uncertainty_gap_rad <= 0.0
        ):
            raise ValueError("active sample local support is invalid")
        if any(
            math.isclose(proposed, value, rel_tol=0.0, abs_tol=1e-9)
            for value in self.local_measured_values_rad
        ):
            raise ValueError("active sample must propose an unmeasured action")
        if (
            self.schema_version != _SCHEMA
            or self.activation_ceiling != "SIM_ONLY"
            or self.promotion_authorized
            or self.hardware_authorized
            or self.direct_torque_output
            or self.online_hot_swap_allowed
            or not self.sim_replay_required
        ):
            raise ValueError("active sample cannot authorize activation or hardware")

    @property
    def sample_hash(self) -> str:
        return canonical_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "sample_hash": self.sample_hash}


def derive_g1_ballistic_contact_active_sample(
    *,
    actor_critic_path: Path,
    output_path: Path,
    source_checkout: Path,
    maximum_step_rad: float = 0.03,
) -> G1BallisticContactActiveSample:
    """Propose one local information-gain replay after critic abstention."""

    if not 0.005 <= maximum_step_rad <= 0.06 or not math.isfinite(maximum_step_rad):
        raise ValueError("active sample maximum step must be in [0.005, 0.06] rad")
    output = output_path.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output == checkout or checkout in output.parents:
        raise ValueError("active sample evidence must be outside the source checkout")
    if output.exists():
        raise FileExistsError("active sample output already exists")

    payload = json.loads(actor_critic_path.expanduser().resolve(strict=True).read_text())
    if not isinstance(payload, dict):
        raise ValueError("active sampler requires an actor-critic object")
    declared_hash = str(payload.pop("candidate_hash", ""))
    if declared_hash != canonical_hash(payload):
        raise ValueError("active sampler actor-critic hash mismatch")
    if payload.get("schema_version") != _ACTOR_SCHEMA:
        raise ValueError("active sampler requires a v4 ballistic actor-critic")
    if (
        payload.get("activation_ceiling") != "SIM_ONLY"
        or payload.get("promotion_authorized") is not False
        or payload.get("hardware_authorized") is not False
        or payload.get("direct_torque_output") is not False
        or payload.get("online_hot_swap_allowed") is not False
    ):
        raise ValueError("active sampler source is not a safe SIM_ONLY candidate")

    anchor = _action(payload.get("best_observed_action_rad"), label="anchor")
    active = tuple(int(value) for value in payload.get("active_action_dimensions", ()))
    if not active or len(set(active)) != len(active) or any(not 0 <= value < 6 for value in active):
        raise ValueError("active sampler source dimensions are invalid")
    raw_probes = payload.get("probes")
    if not isinstance(raw_probes, list):
        raise ValueError("active sampler source probes are missing")
    probes = tuple(_probe(value) for value in raw_probes)
    if not any(
        safe and np.allclose(action, anchor, rtol=0.0, atol=1e-9) for action, safe in probes
    ):
        raise ValueError("active sampler anchor is not a measured hard-safe replay")

    options: list[tuple[float, int, float, tuple[float, float], tuple[float, ...]]] = []
    for dimension in active:
        local = tuple(
            (action, safe)
            for action, safe in probes
            if all(
                index == dimension
                or math.isclose(action[index], anchor[index], rel_tol=0.0, abs_tol=1e-9)
                for index in range(6)
            )
        )
        safe_values = sorted({float(action[dimension]) for action, safe in local if safe})
        if len(safe_values) < 3 or not any(
            math.isclose(anchor[dimension], value, abs_tol=1e-9) for value in safe_values
        ):
            continue
        lower = max(-_ACTION_LIMIT_RAD, anchor[dimension] - maximum_step_rad)
        upper = min(_ACTION_LIMIT_RAD, anchor[dimension] + maximum_step_rad)
        unsafe_values = sorted(float(action[dimension]) for action, safe in local if not safe)
        unsafe_lower = [value for value in unsafe_values if value < anchor[dimension]]
        unsafe_upper = [value for value in unsafe_values if value > anchor[dimension]]
        if unsafe_lower:
            lower = max(lower, max(unsafe_lower) + 0.005)
        if unsafe_upper:
            upper = min(upper, min(unsafe_upper) - 0.005)
        if upper - lower < 0.01:
            continue
        measured = tuple(sorted({float(action[dimension]) for action, _ in local}))
        knots = sorted({lower, upper, *(value for value in measured if lower < value < upper)})
        for left, right in zip(knots, knots[1:], strict=False):
            gap = right - left
            midpoint = 0.5 * (left + right)
            if gap < 0.005 or any(
                math.isclose(midpoint, value, rel_tol=0.0, abs_tol=1e-9) for value in measured
            ):
                continue
            options.append((gap, dimension, midpoint, (lower, upper), measured))
    if not options:
        raise ValueError("active sampler found no locally supported unmeasured action")
    gap, dimension, midpoint, interval, measured = max(
        options,
        key=lambda item: (item[0], -abs(item[2] - anchor[item[1]]), -item[1], -item[2]),
    )
    proposed = list(anchor)
    proposed[dimension] = midpoint
    sample = G1BallisticContactActiveSample(
        source_candidate_hash=declared_hash,
        experiment_context_hash=str(payload.get("experiment_context_hash", "")),
        anchor_action_rad=anchor,
        proposed_action_rad=tuple(proposed),
        proposed_action_dimension=dimension,
        proposed_joint_name=G1_BALLISTIC_CONTACT_JOINT_NAMES[dimension],
        local_safe_interval_rad=interval,
        local_measured_values_rad=measured,
        uncertainty_gap_rad=gap,
        maximum_step_rad=maximum_step_rad,
        proposal_reason="largest_local_unmeasured_gap_around_best_hard_safe_replay",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sample.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return sample


def _action(value: Any, *, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"active sampler {label} action is invalid")
    action = tuple(float(item) for item in value)
    G1BallisticContactResidualConfig(right_leg_residual_rad=action)
    return action


def _probe(value: Any) -> tuple[tuple[float, ...], bool]:
    if not isinstance(value, dict):
        raise ValueError("active sampler probe is invalid")
    return _action(value.get("action_rad"), label="probe"), value.get("hard_safe") is True


__all__ = [
    "G1BallisticContactActiveSample",
    "derive_g1_ballistic_contact_active_sample",
]

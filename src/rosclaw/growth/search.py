"""Task-neutral, bounded active sampling contracts for the Growth Plane.

The policy in this module proposes an unmeasured point inside an adapter-
supplied hard-safe support interval.  It never executes, promotes, activates,
or authorizes a candidate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash
from rosclaw.growth._validation import (
    require_hash,
    require_identifier,
    unique_identifiers,
)


def _strict_vector(values: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{label} must be a non-empty mapping")
    normalized: dict[str, float] = {}
    for key, value in values.items():
        require_identifier(f"{label} key", key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{label} values must be finite numbers")
        normalized[key] = number
    return MappingProxyType(dict(sorted(normalized.items())))


def _strict_number(value: float, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{label} must be a finite number")
    return normalized


@dataclass(frozen=True)
class ParameterDimension:
    """One normalized, bounded adapter parameter."""

    dimension_id: str
    minimum: float
    maximum: float
    schema_version: str = "rosclaw.growth.parameter_dimension.v1"

    def __post_init__(self) -> None:
        require_identifier("dimension_id", self.dimension_id)
        minimum = _strict_number(self.minimum, label="parameter minimum")
        maximum = _strict_number(self.maximum, label="parameter maximum")
        if minimum >= maximum:
            raise ValueError("parameter minimum must be less than maximum")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    @property
    def width(self) -> float:
        return self.maximum - self.minimum


@dataclass(frozen=True)
class MutationBudget:
    """Limit how many dimensions and how much normalized distance may change."""

    allowed_dimension_ids: tuple[str, ...]
    maximum_changed_dimensions: int
    maximum_total_normalized_delta: float
    schema_version: str = "rosclaw.growth.mutation_budget.v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_dimension_ids",
            unique_identifiers(
                self.allowed_dimension_ids,
                label="allowed_dimension_ids",
            ),
        )
        if (
            isinstance(self.maximum_changed_dimensions, bool)
            or not isinstance(self.maximum_changed_dimensions, int)
            or not 1 <= self.maximum_changed_dimensions <= len(self.allowed_dimension_ids)
        ):
            raise ValueError("maximum_changed_dimensions is outside the allowed dimension set")
        normalized_delta = _strict_number(
            self.maximum_total_normalized_delta,
            label="maximum_total_normalized_delta",
        )
        if not 0.0 < normalized_delta <= len(self.allowed_dimension_ids):
            raise ValueError("maximum_total_normalized_delta must be finite and bounded")
        object.__setattr__(self, "maximum_total_normalized_delta", normalized_delta)

    def violations(
        self,
        *,
        anchor: Mapping[str, float],
        candidate: Mapping[str, float],
        dimensions: tuple[ParameterDimension, ...],
    ) -> tuple[str, ...]:
        try:
            anchor_values = _strict_vector(anchor, label="anchor")
            candidate_values = _strict_vector(candidate, label="candidate")
        except (TypeError, ValueError):
            return ("invalid_parameter_vector",)
        if not dimensions or any(not isinstance(item, ParameterDimension) for item in dimensions):
            return ("invalid_parameter_space",)
        dimension_map = {item.dimension_id: item for item in dimensions}
        if len(dimension_map) != len(dimensions):
            return ("invalid_parameter_space",)
        if set(anchor_values) != set(candidate_values) or set(anchor_values) != set(dimension_map):
            return ("parameter_shape_mismatch",)

        reasons: list[str] = []
        changed: list[str] = []
        normalized_delta = 0.0
        for dimension_id, dimension in dimension_map.items():
            before = anchor_values[dimension_id]
            after = candidate_values[dimension_id]
            if not dimension.minimum <= before <= dimension.maximum or not (
                dimension.minimum <= after <= dimension.maximum
            ):
                reasons.append("parameter_out_of_bounds")
            if before != after:
                changed.append(dimension_id)
                normalized_delta += abs(after - before) / dimension.width
        if not changed:
            reasons.append("no_mutation")
        if any(item not in self.allowed_dimension_ids for item in changed):
            reasons.append("disallowed_dimension")
        if len(changed) > self.maximum_changed_dimensions:
            reasons.append("changed_dimension_budget_exceeded")
        if normalized_delta > self.maximum_total_normalized_delta + 1e-12:
            reasons.append("normalized_delta_budget_exceeded")
        return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class TrustRegion:
    """An immutable local region around a known anchor."""

    anchor: Mapping[str, float]
    maximum_absolute_delta: Mapping[str, float]
    anchor_artifact_hash: str
    schema_version: str = "rosclaw.growth.trust_region.v1"

    def __post_init__(self) -> None:
        anchor = _strict_vector(self.anchor, label="anchor")
        delta = _strict_vector(self.maximum_absolute_delta, label="maximum_absolute_delta")
        if set(anchor) != set(delta):
            raise ValueError("trust-region anchor and delta dimensions must match")
        if any(value < 0.0 for value in delta.values()):
            raise ValueError("maximum_absolute_delta values must be non-negative")
        require_hash("anchor_artifact_hash", self.anchor_artifact_hash)
        object.__setattr__(self, "anchor", anchor)
        object.__setattr__(self, "maximum_absolute_delta", delta)

    def violations(self, candidate: Mapping[str, float]) -> tuple[str, ...]:
        try:
            values = _strict_vector(candidate, label="candidate")
        except (TypeError, ValueError):
            return ("invalid_parameter_vector",)
        if set(values) != set(self.anchor):
            return ("parameter_shape_mismatch",)
        if any(
            abs(values[key] - self.anchor[key]) > self.maximum_absolute_delta[key] + 1e-12
            for key in values
        ):
            return ("trust_region_exceeded",)
        return ()


@dataclass(frozen=True)
class NumericBindingTolerance:
    """Explicit tolerance for binding serialized evidence to numeric parameters."""

    absolute: float = 1e-9
    relative: float = 1e-9
    schema_version: str = "rosclaw.growth.numeric_binding_tolerance.v1"

    def __post_init__(self) -> None:
        absolute = _strict_number(self.absolute, label="absolute tolerance")
        relative = _strict_number(self.relative, label="relative tolerance")
        if absolute < 0.0:
            raise ValueError("absolute tolerance must be finite and non-negative")
        if relative < 0.0:
            raise ValueError("relative tolerance must be finite and non-negative")
        if absolute == 0.0 and relative == 0.0:
            raise ValueError("at least one numeric tolerance must be positive")
        object.__setattr__(self, "absolute", absolute)
        object.__setattr__(self, "relative", relative)

    def mismatched_dimensions(
        self,
        expected: Mapping[str, float],
        observed: Mapping[str, float],
    ) -> tuple[str, ...]:
        try:
            expected_values = _strict_vector(expected, label="expected")
            observed_values = _strict_vector(observed, label="observed")
        except (TypeError, ValueError):
            return ("invalid_parameter_vector",)
        if set(expected_values) != set(observed_values):
            return ("parameter_shape_mismatch",)
        return tuple(
            key
            for key in expected_values
            if not math.isclose(
                expected_values[key],
                observed_values[key],
                rel_tol=self.relative,
                abs_tol=self.absolute,
            )
        )

    def equivalent(
        self,
        expected: Mapping[str, float],
        observed: Mapping[str, float],
    ) -> bool:
        return not self.mismatched_dimensions(expected, observed)


@dataclass(frozen=True)
class ExecutedSafeSupport:
    """Adapter-attested values already executed without a hard-safety failure."""

    dimension_id: str
    safe_minimum: float
    safe_maximum: float
    measured_values: tuple[float, ...]
    evidence_hash: str
    schema_version: str = "rosclaw.growth.executed_safe_support.v1"

    def __post_init__(self) -> None:
        require_identifier("dimension_id", self.dimension_id)
        safe_minimum = _strict_number(self.safe_minimum, label="safe_minimum")
        safe_maximum = _strict_number(self.safe_maximum, label="safe_maximum")
        if safe_minimum >= safe_maximum:
            raise ValueError("safe_minimum must be less than safe_maximum")
        values = tuple(
            _strict_number(value, label="measured support value") for value in self.measured_values
        )
        if len(values) < 3:
            raise ValueError("executed safe support requires at least three finite values")
        if len(values) != len(set(values)):
            raise ValueError("executed safe support values must be unique")
        if any(not safe_minimum <= value <= safe_maximum for value in values):
            raise ValueError("measured support values must lie inside the safe interval")
        require_hash("evidence_hash", self.evidence_hash)
        object.__setattr__(self, "safe_minimum", safe_minimum)
        object.__setattr__(self, "safe_maximum", safe_maximum)
        object.__setattr__(self, "measured_values", tuple(sorted(values)))


class ActiveSamplingStatus(StrEnum):
    PROPOSED = "proposed"
    NO_SAFE_GAP = "no_safe_gap"


@dataclass(frozen=True)
class ActiveSamplingDecision:
    status: ActiveSamplingStatus
    candidate_parameters: Mapping[str, float] | None
    changed_dimension_ids: tuple[str, ...]
    support_evidence_hashes: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = "rosclaw.growth.active_sampling_decision.v1"

    def __post_init__(self) -> None:
        if not isinstance(self.status, ActiveSamplingStatus):
            raise ValueError("status must be an ActiveSamplingStatus")
        object.__setattr__(
            self,
            "changed_dimension_ids",
            unique_identifiers(
                self.changed_dimension_ids,
                label="changed_dimension_ids",
                allow_empty=self.status is ActiveSamplingStatus.NO_SAFE_GAP,
            ),
        )
        hashes = tuple(self.support_evidence_hashes)
        if len(hashes) != len(set(hashes)):
            raise ValueError("support_evidence_hashes must be unique")
        for value in hashes:
            require_hash("support_evidence_hashes", value)
        object.__setattr__(self, "support_evidence_hashes", hashes)
        object.__setattr__(
            self,
            "reasons",
            unique_identifiers(
                self.reasons,
                label="reasons",
                allow_empty=self.status is ActiveSamplingStatus.PROPOSED,
            ),
        )
        if self.status is ActiveSamplingStatus.PROPOSED:
            if (
                self.candidate_parameters is None
                or self.reasons
                or not self.support_evidence_hashes
            ):
                raise ValueError("a proposed sampling decision requires parameters and no reasons")
            object.__setattr__(
                self,
                "candidate_parameters",
                _strict_vector(self.candidate_parameters, label="candidate_parameters"),
            )
        elif self.candidate_parameters is not None or self.changed_dimension_ids:
            raise ValueError("a no-safe-gap decision cannot carry a candidate")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def sim_replay_required(self) -> bool:
        return True

    @property
    def activation_allowed(self) -> bool:
        return False

    @property
    def hardware_authorized(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "candidate_parameters": (
                dict(self.candidate_parameters) if self.candidate_parameters else None
            ),
            "changed_dimension_ids": list(self.changed_dimension_ids),
            "support_evidence_hashes": list(self.support_evidence_hashes),
            "reasons": list(self.reasons),
            "sim_replay_required": self.sim_replay_required,
            "activation_allowed": self.activation_allowed,
            "hardware_authorized": self.hardware_authorized,
        }


@dataclass(frozen=True)
class ActiveSamplingPolicy:
    """Choose the largest local, unmeasured safe-support gap deterministically."""

    minimum_measured_values: int = 3
    schema_version: str = "rosclaw.growth.active_sampling_policy.v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_measured_values, bool)
            or not isinstance(self.minimum_measured_values, int)
            or not 3 <= self.minimum_measured_values <= 1024
        ):
            raise ValueError("minimum_measured_values must be in [3, 1024]")

    def propose(
        self,
        *,
        dimensions: tuple[ParameterDimension, ...],
        supports: tuple[ExecutedSafeSupport, ...],
        mutation_budget: MutationBudget,
        trust_region: TrustRegion,
    ) -> ActiveSamplingDecision:
        if any(not isinstance(item, ParameterDimension) for item in dimensions) or any(
            not isinstance(item, ExecutedSafeSupport) for item in supports
        ):
            return _no_safe_gap("invalid_parameter_space")
        dimension_map = {item.dimension_id: item for item in dimensions}
        support_map = {item.dimension_id: item for item in supports}
        invalid_space = (
            not dimension_map
            or len(dimension_map) != len(dimensions)
            or len(support_map) != len(supports)
            or set(trust_region.anchor) != set(dimension_map)
        )
        if invalid_space:
            return _no_safe_gap("invalid_parameter_space")

        proposals: list[tuple[float, str, float, str]] = []
        for dimension_id in sorted(mutation_budget.allowed_dimension_ids):
            dimension = dimension_map.get(dimension_id)
            support = support_map.get(dimension_id)
            if dimension is None or support is None:
                continue
            if len(support.measured_values) < self.minimum_measured_values:
                continue
            anchor = trust_region.anchor[dimension_id]
            radius = trust_region.maximum_absolute_delta[dimension_id]
            lower = max(dimension.minimum, support.safe_minimum, anchor - radius)
            upper = min(dimension.maximum, support.safe_maximum, anchor + radius)
            if lower >= upper:
                continue
            measured = tuple(value for value in support.measured_values if lower <= value <= upper)
            if len(measured) < self.minimum_measured_values:
                continue
            boundaries = sorted({lower, *measured, upper})
            for left, right in zip(boundaries, boundaries[1:], strict=False):
                if right - left <= 2e-12:
                    continue
                proposed_value = (left + right) / 2.0
                if proposed_value == anchor or any(
                    math.isclose(proposed_value, value, rel_tol=0.0, abs_tol=1e-12)
                    for value in measured
                ):
                    continue
                proposals.append(
                    (
                        (right - left) / dimension.width,
                        dimension_id,
                        proposed_value,
                        support.evidence_hash,
                    )
                )
        for _gap, dimension_id, proposed_value, evidence_hash in sorted(
            proposals,
            key=lambda item: (-item[0], item[1], item[2]),
        ):
            candidate_values = dict(trust_region.anchor)
            candidate_values[dimension_id] = proposed_value
            reasons = (
                *trust_region.violations(candidate_values),
                *mutation_budget.violations(
                    anchor=trust_region.anchor,
                    candidate=candidate_values,
                    dimensions=dimensions,
                ),
            )
            if reasons:
                continue
            return ActiveSamplingDecision(
                status=ActiveSamplingStatus.PROPOSED,
                candidate_parameters=candidate_values,
                changed_dimension_ids=(dimension_id,),
                support_evidence_hashes=(evidence_hash,),
            )
        return _no_safe_gap("no_unmeasured_safe_gap")


def _no_safe_gap(reason: str) -> ActiveSamplingDecision:
    return ActiveSamplingDecision(
        status=ActiveSamplingStatus.NO_SAFE_GAP,
        candidate_parameters=None,
        changed_dimension_ids=(),
        support_evidence_hashes=(),
        reasons=(reason,),
    )


__all__ = [
    "ActiveSamplingDecision",
    "ActiveSamplingPolicy",
    "ActiveSamplingStatus",
    "ExecutedSafeSupport",
    "MutationBudget",
    "NumericBindingTolerance",
    "ParameterDimension",
    "TrustRegion",
]

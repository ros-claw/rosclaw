"""Task-independent successor-state contracts for composable physical skills.

A skill is not complete merely because its local event succeeded.  Its final
physical state must remain inside the declared entry envelope of the next
skill for a continuous hold window.  These contracts make that boundary
content-addressed and machine-testable without putting task vocabulary in the
ROSClaw core.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SuccessorMetricSpec:
    """One observable interval required by a successor skill."""

    name: str
    minimum: float | None = None
    maximum: float | None = None
    critical: bool = True

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("successor metric name must be a normalized identifier")
        if self.minimum is None and self.maximum is None:
            raise ValueError("successor metric requires at least one bound")
        bounds = tuple(value for value in (self.minimum, self.maximum) if value is not None)
        if any(not math.isfinite(value) for value in bounds):
            raise ValueError("successor metric bounds must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("successor metric minimum cannot exceed maximum")

    def accepts(self, value: float) -> bool:
        return bool(
            math.isfinite(value)
            and (self.minimum is None or value >= self.minimum)
            and (self.maximum is None or value <= self.maximum)
        )

    def margin(self, value: float) -> float:
        if not math.isfinite(value):
            return float("-inf")
        margins = []
        if self.minimum is not None:
            margins.append(value - self.minimum)
        if self.maximum is not None:
            margins.append(self.maximum - value)
        return min(margins)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "critical": self.critical,
        }


@dataclass(frozen=True)
class SkillSuccessorState:
    """Immutable entry envelope connecting two physical skills."""

    contract_id: str
    source_skill_id: str
    successor_skill_id: str
    source_policy_hash: str
    successor_policy_hash: str
    body_hash: str
    metrics: tuple[SuccessorMetricSpec, ...]
    hold_steps: int
    maximum_transition_steps: int
    control_period_s: float
    evidence_domain: str = "SIM"
    schema_version: str = "rosclaw.continual.skill_successor_state.v1"

    def __post_init__(self) -> None:
        for value in (self.contract_id, self.source_skill_id, self.successor_skill_id):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError("successor-state identifiers must be normalized")
        for value in (self.source_policy_hash, self.successor_policy_hash, self.body_hash):
            if not _SHA256.fullmatch(value):
                raise ValueError("successor-state identities must be sha256 content hashes")
        names = tuple(metric.name for metric in self.metrics)
        if not self.metrics or len(names) != len(set(names)):
            raise ValueError("successor-state metrics must be non-empty and unique")
        if (
            self.hold_steps <= 0
            or self.maximum_transition_steps < self.hold_steps
            or not math.isfinite(self.control_period_s)
            or not 0.001 <= self.control_period_s <= 1.0
            or self.evidence_domain != "SIM"
        ):
            raise ValueError("successor-state timing or evidence domain is invalid")

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "source_skill_id": self.source_skill_id,
            "successor_skill_id": self.successor_skill_id,
            "source_policy_hash": self.source_policy_hash,
            "successor_policy_hash": self.successor_policy_hash,
            "body_hash": self.body_hash,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "hold_steps": self.hold_steps,
            "maximum_transition_steps": self.maximum_transition_steps,
            "control_period_s": self.control_period_s,
            "evidence_domain": self.evidence_domain,
        }


@dataclass(frozen=True)
class SuccessorStateSample:
    step: int
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        normalized = {str(name): float(value) for name, value in self.values.items()}
        if (
            self.step < 0
            or not normalized
            or any(
                not _IDENTIFIER.fullmatch(name) or not math.isfinite(value)
                for name, value in normalized.items()
            )
        ):
            raise ValueError("successor-state sample must be finite and normalized")
        object.__setattr__(self, "values", MappingProxyType(normalized))


@dataclass(frozen=True)
class SuccessorStateEvaluation:
    contract_hash: str
    achieved: bool
    timed_out: bool
    consecutive_hold_steps: int
    entry_step: int | None
    achieved_step: int | None
    transition_time_s: float | None
    failed_metrics: tuple[str, ...]
    metric_margins: Mapping[str, float]
    schema_version: str = "rosclaw.continual.successor_state_evaluation.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.contract_hash):
            raise ValueError("successor evaluation requires a contract hash")
        margins = {str(name): float(value) for name, value in self.metric_margins.items()}
        if any(
            not _IDENTIFIER.fullmatch(name) or math.isnan(value) for name, value in margins.items()
        ):
            raise ValueError("successor evaluation margins are invalid")
        object.__setattr__(self, "metric_margins", MappingProxyType(margins))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contract_hash": self.contract_hash,
            "achieved": self.achieved,
            "timed_out": self.timed_out,
            "consecutive_hold_steps": self.consecutive_hold_steps,
            "entry_step": self.entry_step,
            "achieved_step": self.achieved_step,
            "transition_time_s": self.transition_time_s,
            "failed_metrics": list(self.failed_metrics),
            "metric_margins": dict(self.metric_margins),
        }


class SuccessorStateTracker:
    """Causal continuous-hold evaluator for one successor-state contract."""

    def __init__(self, contract: SkillSuccessorState) -> None:
        self.contract = contract
        self._start_step: int | None = None
        self._last_step: int | None = None
        self._hold = 0
        self._entry_step: int | None = None
        self._achieved_step: int | None = None

    def update(self, sample: SuccessorStateSample) -> SuccessorStateEvaluation:
        expected_names = tuple(metric.name for metric in self.contract.metrics)
        if tuple(sample.values) != expected_names:
            raise ValueError("successor sample order must match the frozen contract")
        if self._last_step is not None and sample.step != self._last_step + 1:
            raise ValueError("successor samples must be contiguous")
        if self._start_step is None:
            self._start_step = sample.step
        self._last_step = sample.step

        margins = {
            metric.name: metric.margin(sample.values[metric.name])
            for metric in self.contract.metrics
        }
        failed = tuple(
            metric.name
            for metric in self.contract.metrics
            if not metric.accepts(sample.values[metric.name])
        )
        if not failed and self._achieved_step is None:
            self._hold += 1
            if self._hold == 1:
                self._entry_step = sample.step
            if self._hold >= self.contract.hold_steps:
                self._achieved_step = sample.step
        elif self._achieved_step is None:
            self._hold = 0
            self._entry_step = None

        assert self._start_step is not None
        elapsed_steps = sample.step - self._start_step + 1
        timed_out = bool(
            self._achieved_step is None and elapsed_steps >= self.contract.maximum_transition_steps
        )
        transition_time_s = (
            None
            if self._achieved_step is None
            else (self._achieved_step - self._start_step + 1) * self.contract.control_period_s
        )
        return SuccessorStateEvaluation(
            contract_hash=self.contract.contract_hash,
            achieved=self._achieved_step is not None,
            timed_out=timed_out,
            consecutive_hold_steps=self._hold,
            entry_step=self._entry_step,
            achieved_step=self._achieved_step,
            transition_time_s=transition_time_s,
            failed_metrics=failed,
            metric_margins=margins,
        )


@dataclass(frozen=True)
class SuccessorStateGrowthObjective:
    """Makes the next skill's value explicit in the current skill objective."""

    task_weight: float = 1.0
    successor_value_weight: float = 0.25
    transition_cost_weight: float = 1.0
    schema_version: str = "rosclaw.continual.successor_state_growth_objective.v1"

    def __post_init__(self) -> None:
        values = (self.task_weight, self.successor_value_weight, self.transition_cost_weight)
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not any(values):
            raise ValueError("successor-state objective weights must be finite and non-negative")

    def score(
        self,
        *,
        task_return: float,
        successor_value: float,
        transition_cost: float = 0.0,
    ) -> float:
        values = (task_return, successor_value, transition_cost)
        if any(not math.isfinite(value) for value in values) or transition_cost < 0.0:
            raise ValueError("successor-state objective inputs are invalid")
        return (
            self.task_weight * task_return
            + self.successor_value_weight * successor_value
            - self.transition_cost_weight * transition_cost
        )


__all__ = [
    "SkillSuccessorState",
    "SuccessorMetricSpec",
    "SuccessorStateEvaluation",
    "SuccessorStateGrowthObjective",
    "SuccessorStateSample",
    "SuccessorStateTracker",
]

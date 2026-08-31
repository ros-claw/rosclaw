"""Failure-conditioned Dream and capability-frontier curriculum contracts."""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class CurriculumSource(StrEnum):
    CAPABILITY_FRONTIER = "CAPABILITY_FRONTIER"
    RECENT_FAILURE = "RECENT_FAILURE"
    HISTORICAL_ANCHOR = "HISTORICAL_ANCHOR"
    NIGHTMARE = "NIGHTMARE"
    SOCIAL_TEACHER = "SOCIAL_TEACHER"


class PerturbationDistribution(StrEnum):
    UNIFORM = "UNIFORM"
    NORMAL_CLIPPED = "NORMAL_CLIPPED"


@dataclass(frozen=True)
class CurriculumMixture:
    capability_frontier: float = 0.40
    recent_failure: float = 0.25
    historical_anchor: float = 0.15
    nightmare: float = 0.10
    social_teacher: float = 0.10
    schema_version: str = "rosclaw.continual.curriculum_mixture.v1"

    def __post_init__(self) -> None:
        values = tuple(self.as_mapping().values())
        if any(not math.isfinite(value) or value < 0.0 for value in values) or not math.isclose(
            sum(values),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("curriculum mixture must be finite, non-negative, and sum to one")

    def as_mapping(self) -> MappingProxyType[CurriculumSource, float]:
        return MappingProxyType(
            {
                CurriculumSource.CAPABILITY_FRONTIER: self.capability_frontier,
                CurriculumSource.RECENT_FAILURE: self.recent_failure,
                CurriculumSource.HISTORICAL_ANCHOR: self.historical_anchor,
                CurriculumSource.NIGHTMARE: self.nightmare,
                CurriculumSource.SOCIAL_TEACHER: self.social_teacher,
            }
        )

    def allocate(self, batch_size: int) -> MappingProxyType[CurriculumSource, int]:
        if batch_size <= 0 or batch_size > 1_000_000:
            raise ValueError("curriculum batch size is invalid")
        weights = self.as_mapping()
        raw = {source: batch_size * weight for source, weight in weights.items()}
        counts = {source: math.floor(value) for source, value in raw.items()}
        remainder = batch_size - sum(counts.values())
        order = sorted(
            weights,
            key=lambda source: (raw[source] - counts[source], weights[source], source.value),
            reverse=True,
        )
        for source in order[:remainder]:
            counts[source] += 1
        return MappingProxyType(counts)


@dataclass(frozen=True)
class DreamPerturbation:
    name: str
    minimum_delta: float
    maximum_delta: float
    distribution: PerturbationDistribution = PerturbationDistribution.UNIFORM

    def __post_init__(self) -> None:
        if (
            not _IDENTIFIER.fullmatch(self.name)
            or not math.isfinite(self.minimum_delta)
            or not math.isfinite(self.maximum_delta)
            or self.minimum_delta > self.maximum_delta
            or self.minimum_delta == self.maximum_delta
        ):
            raise ValueError("dream perturbation is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "minimum_delta": self.minimum_delta,
            "maximum_delta": self.maximum_delta,
            "distribution": self.distribution.value,
        }


@dataclass(frozen=True)
class FailureConditionedDream:
    failure_code: str
    source_snapshot_hash: str
    body_hash: str
    scenario_hash: str
    perturbations: tuple[DreamPerturbation, ...]
    maximum_variants: int
    training_use_only: bool = True
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw.continual.failure_conditioned_dream.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.failure_code):
            raise ValueError("failure code must be a normalized identifier")
        if any(
            not _SHA256.fullmatch(value)
            for value in (self.source_snapshot_hash, self.body_hash, self.scenario_hash)
        ):
            raise ValueError("failure-conditioned dream identities must be content hashes")
        names = tuple(item.name for item in self.perturbations)
        if (
            not self.perturbations
            or len(names) != len(set(names))
            or not 1 <= self.maximum_variants <= 100_000
            or not self.training_use_only
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("failure-conditioned dream contract is invalid")

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "failure_code": self.failure_code,
            "source_snapshot_hash": self.source_snapshot_hash,
            "body_hash": self.body_hash,
            "scenario_hash": self.scenario_hash,
            "perturbations": [item.to_dict() for item in self.perturbations],
            "maximum_variants": self.maximum_variants,
            "training_use_only": self.training_use_only,
            "activation_ceiling": self.activation_ceiling,
            "hardware_authorized": self.hardware_authorized,
        }

    def sample(self, *, count: int, seed: int) -> tuple[dict[str, float], ...]:
        if count <= 0 or count > self.maximum_variants or seed < 0:
            raise ValueError("failure-conditioned dream sampling request is invalid")
        generator = random.Random(seed)
        rows = []
        for _ in range(count):
            row = {}
            for item in self.perturbations:
                if item.distribution is PerturbationDistribution.UNIFORM:
                    value = generator.uniform(item.minimum_delta, item.maximum_delta)
                else:
                    midpoint = 0.5 * (item.minimum_delta + item.maximum_delta)
                    sigma = (item.maximum_delta - item.minimum_delta) / 6.0
                    value = min(
                        item.maximum_delta,
                        max(item.minimum_delta, generator.gauss(midpoint, sigma)),
                    )
                row[item.name] = value
            rows.append(row)
        return tuple(rows)


@dataclass(frozen=True)
class CapabilityBin:
    bin_id: str
    difficulty: float
    successes: int
    attempts: int

    def __post_init__(self) -> None:
        if (
            not _IDENTIFIER.fullmatch(self.bin_id)
            or not math.isfinite(self.difficulty)
            or not 0.0 <= self.difficulty <= 1.0
            or self.successes < 0
            or self.attempts < 0
            or self.successes > self.attempts
        ):
            raise ValueError("capability bin is invalid")

    @property
    def success_rate(self) -> float | None:
        return None if self.attempts == 0 else self.successes / self.attempts


class CapabilityFrontierScheduler:
    """Prioritize measured 30--70% success regions without starving anchors."""

    def __init__(
        self,
        *,
        lower_success: float = 0.30,
        upper_success: float = 0.70,
        minimum_probability: float = 0.02,
        minimum_attempts: int = 16,
        temperature: float = 0.15,
    ) -> None:
        values = (lower_success, upper_success, minimum_probability, temperature)
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 <= lower_success < upper_success <= 1.0
            or not 0.0 < minimum_probability < 1.0
            or minimum_attempts <= 0
            or temperature <= 0.0
        ):
            raise ValueError("capability-frontier scheduler config is invalid")
        self.lower_success = lower_success
        self.upper_success = upper_success
        self.minimum_probability = minimum_probability
        self.minimum_attempts = minimum_attempts
        self.temperature = temperature

    def probabilities(
        self,
        bins: tuple[CapabilityBin, ...],
    ) -> MappingProxyType[str, float]:
        if not bins or len({item.bin_id for item in bins}) != len(bins):
            raise ValueError("capability frontier requires unique bins")
        scores: dict[str, float] = {}
        for item in bins:
            rate = item.success_rate
            if rate is None:
                frontier_score = 0.25
            elif self.lower_success <= rate <= self.upper_success:
                frontier_score = 1.0
            else:
                distance = min(abs(rate - self.lower_success), abs(rate - self.upper_success))
                frontier_score = math.exp(-distance / self.temperature)
            confidence = min(1.0, item.attempts / self.minimum_attempts)
            scores[item.bin_id] = self.minimum_probability + frontier_score * (
                0.25 + 0.75 * confidence
            )
        total = sum(scores.values())
        return MappingProxyType({key: value / total for key, value in sorted(scores.items())})


__all__ = [
    "CapabilityBin",
    "CapabilityFrontierScheduler",
    "CurriculumMixture",
    "CurriculumSource",
    "DreamPerturbation",
    "FailureConditionedDream",
    "PerturbationDistribution",
]

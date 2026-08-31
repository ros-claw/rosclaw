"""Generic stability-plasticity gates around immutable teacher manifolds."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class TeacherManifoldMetric:
    """One normalized deployable signal used to measure teacher novelty."""

    name: str
    scale: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.name):
            raise ValueError("teacher-manifold metric name must be a normalized identifier")
        if (
            not math.isfinite(self.scale)
            or self.scale <= 0.0
            or not math.isfinite(self.weight)
            or self.weight <= 0.0
        ):
            raise ValueError("teacher-manifold metric scale and weight must be finite and positive")


@dataclass(frozen=True)
class TeacherManifoldDecision:
    """Content-bound continuous permission for a residual learner."""

    gate_contract_hash: str
    normalized_distance: float
    plasticity_fraction: float
    inside_retention_core: bool
    fully_plastic: bool
    schema_version: str = "rosclaw.continual.teacher_manifold_decision.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.gate_contract_hash):
            raise ValueError("teacher-manifold decision requires a gate contract hash")
        if not math.isfinite(self.normalized_distance) or self.normalized_distance < 0.0:
            raise ValueError("teacher-manifold distance must be finite and non-negative")
        if (
            not math.isfinite(self.plasticity_fraction)
            or not 0.0 <= self.plasticity_fraction <= 1.0
        ):
            raise ValueError("teacher-manifold plasticity fraction must be in [0, 1]")
        if self.inside_retention_core != (self.plasticity_fraction == 0.0):
            raise ValueError("teacher-manifold retention-core decision is inconsistent")
        if self.fully_plastic != (self.plasticity_fraction == 1.0):
            raise ValueError("teacher-manifold full-plasticity decision is inconsistent")

    @property
    def decision_hash(self) -> str:
        return canonical_hash(asdict(self))


@dataclass(frozen=True)
class TeacherManifoldGateContract:
    """Seal how novelty opens learning without rewriting a known teacher.

    The caller supplies deployable scalar signals for the current and reference
    states.  The contract computes a weighted normalized RMS distance and maps
    it continuously to a plasticity fraction.  It grants training permission
    only; it never promotes or executes a policy.
    """

    gate_id: str
    teacher_artifact_hash: str
    body_hash: str
    metrics: tuple[TeacherManifoldMetric, ...]
    retention_core_radius: float
    full_plasticity_radius: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_execution_allowed: bool = False
    promotion_authority: str = "NONE"
    schema_version: str = "rosclaw.continual.teacher_manifold_gate.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.gate_id):
            raise ValueError("teacher-manifold gate id must be a normalized identifier")
        for label, value in (
            ("teacher_artifact_hash", self.teacher_artifact_hash),
            ("body_hash", self.body_hash),
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        names = tuple(metric.name for metric in self.metrics)
        if not self.metrics or len(names) != len(set(names)):
            raise ValueError("teacher-manifold metrics must be non-empty and unique")
        if (
            not math.isfinite(self.retention_core_radius)
            or not math.isfinite(self.full_plasticity_radius)
            or not 0.0 <= self.retention_core_radius < self.full_plasticity_radius
        ):
            raise ValueError("teacher-manifold radii must be finite and ordered")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_execution_allowed
            or self.promotion_authority != "NONE"
        ):
            raise ValueError("teacher-manifold gates grant SIM_ONLY training permission only")

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def evaluate(
        self,
        current: Mapping[str, float],
        reference: Mapping[str, float],
    ) -> TeacherManifoldDecision:
        """Fail closed on missing/non-finite signals and return bounded permission."""

        expected = {metric.name for metric in self.metrics}
        if set(current) != expected or set(reference) != expected:
            raise ValueError("teacher-manifold observations must match contracted metrics exactly")
        squared_distance = 0.0
        total_weight = 0.0
        for metric in self.metrics:
            current_value = float(current[metric.name])
            reference_value = float(reference[metric.name])
            if not math.isfinite(current_value) or not math.isfinite(reference_value):
                raise ValueError("teacher-manifold observations must be finite")
            delta = (current_value - reference_value) / metric.scale
            squared_distance += metric.weight * delta * delta
            total_weight += metric.weight
        distance = math.sqrt(squared_distance / total_weight)
        fraction = min(
            1.0,
            max(
                0.0,
                (distance - self.retention_core_radius)
                / (self.full_plasticity_radius - self.retention_core_radius),
            ),
        )
        return TeacherManifoldDecision(
            gate_contract_hash=self.contract_hash,
            normalized_distance=distance,
            plasticity_fraction=fraction,
            inside_retention_core=fraction == 0.0,
            fully_plastic=fraction == 1.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "teacher_artifact_hash": self.teacher_artifact_hash,
            "body_hash": self.body_hash,
            "metrics": [asdict(metric) for metric in self.metrics],
            "retention_core_radius": self.retention_core_radius,
            "full_plasticity_radius": self.full_plasticity_radius,
            "activation_ceiling": self.activation_ceiling,
            "hardware_execution_allowed": self.hardware_execution_allowed,
            "promotion_authority": self.promotion_authority,
        }


__all__ = [
    "TeacherManifoldDecision",
    "TeacherManifoldGateContract",
    "TeacherManifoldMetric",
]

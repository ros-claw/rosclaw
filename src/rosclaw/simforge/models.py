"""Immutable public data contracts used by CoreSimBench."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Partition(StrEnum):
    """Evaluation partitions, ordered by increasing information isolation."""

    DISCOVERY = "discovery"
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    HOLDOUT = "holdout"
    COUNTEREXAMPLE_REGRESSION = "counterexample_regression"
    STRESS = "stress"

    @property
    def candidate_may_view_cases(self) -> bool:
        return self in {
            Partition.DISCOVERY,
            Partition.DEVELOPMENT,
            Partition.COUNTEREXAMPLE_REGRESSION,
        }


class DistributionKind(StrEnum):
    UNIFORM = "uniform"
    CHOICE = "choice"
    LIST = "list"


class SamplingStrategy(StrEnum):
    RANDOM = "random"
    LATIN_HYPERCUBE = "latin_hypercube"
    BOUNDARY = "boundary"
    PAIRWISE = "pairwise"


@dataclass(frozen=True)
class ScenarioVariable:
    """One bounded variable in the lightweight scenario DSL."""

    distribution: DistributionKind
    minimum: float | None = None
    maximum: float | None = None
    values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if self.distribution is DistributionKind.UNIFORM:
            if self.minimum is None or self.maximum is None:
                raise ValueError("uniform variables require minimum and maximum")
            if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
                raise ValueError("uniform variable bounds must be finite")
            if self.minimum > self.maximum:
                raise ValueError("uniform variable minimum must not exceed maximum")
            if self.values:
                raise ValueError("uniform variables cannot also define values")
        elif not self.values:
            raise ValueError("choice/list variables require at least one value")
        elif self.minimum is not None or self.maximum is not None:
            raise ValueError("choice/list variables cannot define numeric bounds")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioVariable:
        raw_kind = value.get("distribution", "list" if "values" in value else "uniform")
        return cls(
            distribution=DistributionKind(str(raw_kind)),
            minimum=_optional_float(value.get("min")),
            maximum=_optional_float(value.get("max")),
            values=tuple(value.get("values") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.distribution is DistributionKind.UNIFORM:
            return {
                "distribution": self.distribution.value,
                "min": self.minimum,
                "max": self.maximum,
            }
        return {"distribution": self.distribution.value, "values": list(self.values)}


@dataclass(frozen=True)
class ScenarioConstraint:
    """Named constraint plus immutable parameters; execution is registry-based."""

    name: str
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 128:
            raise ValueError("constraint name must contain 1..128 characters")
        if any(not key for key, _ in self.parameters):
            raise ValueError("constraint parameter names cannot be empty")

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> ScenarioConstraint:
        if isinstance(value, str):
            return cls(name=value)
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("constraints must be names or mappings containing a name")
        parameters = value.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("constraint parameters must be a mapping")
        return cls(name=value["name"], parameters=tuple(sorted(parameters.items())))

    def to_dict(self) -> str | dict[str, Any]:
        if not self.parameters:
            return self.name
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class ScenarioDistributionSpec:
    variables: tuple[tuple[str, ScenarioVariable], ...]
    constraints: tuple[ScenarioConstraint, ...] = ()
    max_sampling_attempts: int = 10_000
    schema_version: str = "rosclaw.simforge.scenario_distribution.v1"

    def __post_init__(self) -> None:
        names = [name for name, _ in self.variables]
        if not names or len(names) != len(set(names)):
            raise ValueError("scenario variable names must be non-empty and unique")
        if self.max_sampling_attempts < 1 or self.max_sampling_attempts > 1_000_000:
            raise ValueError("max_sampling_attempts must be in [1, 1000000]")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioDistributionSpec:
        variables = value.get("variables")
        if not isinstance(variables, dict):
            raise ValueError("scenario distribution variables must be a mapping")
        raw_constraints = value.get("constraints") or ()
        if not isinstance(raw_constraints, (list, tuple)):
            raise ValueError("scenario distribution constraints must be a sequence")
        return cls(
            variables=tuple(
                (str(name), ScenarioVariable.from_dict(spec))
                for name, spec in sorted(variables.items())
            ),
            constraints=tuple(ScenarioConstraint.from_value(item) for item in raw_constraints),
            max_sampling_attempts=int(value.get("max_sampling_attempts", 10_000)),
            schema_version=str(
                value.get("schema_version", "rosclaw.simforge.scenario_distribution.v1")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "variables": {name: spec.to_dict() for name, spec in self.variables},
            "constraints": [constraint.to_dict() for constraint in self.constraints],
            "max_sampling_attempts": self.max_sampling_attempts,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ScenarioSample:
    scenario_id: str
    partition: Partition
    seed: int
    values: tuple[tuple[str, Any], ...]
    distribution_hash: str
    strategy: SamplingStrategy

    def to_dict(self, *, reveal_seed: bool | None = None) -> dict[str, Any]:
        if reveal_seed is None:
            reveal_seed = self.partition.candidate_may_view_cases
        result: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "partition": self.partition.value,
            "values": dict(self.values),
            "distribution_hash": self.distribution_hash,
            "strategy": self.strategy.value,
        }
        result["seed" if reveal_seed else "seed_commitment"] = (
            self.seed
            if reveal_seed
            else "sha256:" + hashlib.sha256(str(self.seed).encode()).hexdigest()
        )
        return result


@dataclass(frozen=True)
class EvidenceRequirements:
    physics_executed: bool = True
    strict_replay: bool = True
    artifact_hashes: bool = True
    minimum_seeds: int = 20
    holdout_required: bool = True

    def __post_init__(self) -> None:
        if self.minimum_seeds < 2:
            raise ValueError("minimum_seeds must be at least two")


@dataclass(frozen=True)
class HumanInvolvement:
    failure_selected_by_human: bool = False
    patch_written_by_human: bool = False
    patch_edited_by_human: bool = False
    promotion_approved_by_human: bool = False

    @property
    def fully_autonomous(self) -> bool:
        return not any(asdict(self).values())


@dataclass(frozen=True)
class SimForgeTaskSpec:
    task_id: str
    suite_id: str
    body_id: str
    required_capabilities: tuple[str, ...]
    discovery_backends: tuple[str, ...]
    evaluation_backends: tuple[str, ...]
    differential_backends: tuple[str, ...]
    scenario_distribution_ref: str
    success_spec: tuple[tuple[str, Any], ...]
    safety_spec: tuple[tuple[str, Any], ...]
    candidate_allowed_paths: tuple[str, ...]
    evidence_requirements: EvidenceRequirements = field(default_factory=EvidenceRequirements)
    schema_version: str = "rosclaw.simforge.task.v1"

    def __post_init__(self) -> None:
        if not self.task_id or not self.suite_id or not self.body_id:
            raise ValueError("task_id, suite_id, and body_id are required")
        if not self.required_capabilities:
            raise ValueError("at least one body capability is required")
        if not self.discovery_backends or not self.evaluation_backends:
            raise ValueError("discovery and evaluation backends cannot be empty")
        if not self.candidate_allowed_paths:
            raise ValueError("candidate path whitelist cannot be empty")
        if any(not path.startswith("/") for path in self.candidate_allowed_paths):
            raise ValueError("candidate paths must be absolute JSON pointers")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SimForgeTaskSpec:
        body = _mapping(value, "body")
        backends = _mapping(value, "backends")
        candidate = _mapping(value, "candidate_space")
        evidence = value.get("evidence_requirements") or {}
        if not isinstance(evidence, dict):
            raise ValueError("evidence_requirements must be a mapping")
        distribution = value.get("scenario_distribution") or {}
        if not isinstance(distribution, dict):
            raise ValueError("scenario_distribution must be a mapping")
        return cls(
            task_id=str(value.get("task_id") or ""),
            suite_id=str(value.get("suite_id") or ""),
            body_id=str(body.get("body_id") or ""),
            required_capabilities=tuple(map(str, body.get("required_capabilities") or ())),
            discovery_backends=tuple(map(str, backends.get("discovery") or ())),
            evaluation_backends=tuple(map(str, backends.get("evaluation") or ())),
            differential_backends=tuple(map(str, backends.get("differential") or ())),
            scenario_distribution_ref=str(distribution.get("ref") or ""),
            success_spec=tuple(sorted(_mapping(value, "success_spec").items())),
            safety_spec=tuple(sorted(_mapping(value, "safety_spec").items())),
            candidate_allowed_paths=tuple(map(str, candidate.get("allowed_paths") or ())),
            evidence_requirements=EvidenceRequirements(
                physics_executed=bool(evidence.get("physics_executed", True)),
                strict_replay=bool(evidence.get("strict_replay", True)),
                artifact_hashes=bool(evidence.get("artifact_hashes", True)),
                minimum_seeds=int(evidence.get("minimum_seeds", 20)),
                holdout_required=bool(evidence.get("holdout_required", True)),
            ),
            schema_version=str(value.get("schema_version", "rosclaw.simforge.task.v1")),
        )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("numeric bounds must be int or float")
    return float(value)


def _mapping(value: dict[str, Any], key: str) -> dict[str, Any]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be a mapping")
    return result


__all__ = [
    "DistributionKind",
    "EvidenceRequirements",
    "HumanInvolvement",
    "Partition",
    "SamplingStrategy",
    "ScenarioConstraint",
    "ScenarioDistributionSpec",
    "ScenarioSample",
    "ScenarioVariable",
    "SimForgeTaskSpec",
]

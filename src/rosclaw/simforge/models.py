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
        elif len(self.values) > 256:
            raise ValueError("choice/list variables cannot exceed 256 values")
        else:
            for item in self.values:
                _validate_scalar(item, "scenario variable value")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioVariable:
        if not isinstance(value, dict):
            raise ValueError("scenario variables must be mappings")
        raw_kind = value.get("distribution", "list" if "values" in value else "uniform")
        raw_values = value.get("values") or ()
        if not isinstance(raw_values, (list, tuple)):
            raise ValueError("scenario variable values must be a sequence")
        return cls(
            distribution=DistributionKind(str(raw_kind)),
            minimum=_optional_float(value.get("min")),
            maximum=_optional_float(value.get("max")),
            values=tuple(raw_values),
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
        if len(self.parameters) > 32:
            raise ValueError("constraint parameters cannot exceed 32 entries")
        for key, item in self.parameters:
            if not isinstance(key, str) or not 1 <= len(key) <= 128:
                raise ValueError("constraint parameter names must contain 1..128 characters")
            _validate_scalar(item, "constraint parameter")

    @classmethod
    def from_value(cls, value: str | dict[str, Any]) -> ScenarioConstraint:
        if isinstance(value, str):
            return cls(name=value)
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise ValueError("constraints must be names or mappings containing a name")
        parameters = value.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("constraint parameters must be a mapping")
        if any(not isinstance(key, str) for key in parameters):
            raise ValueError("constraint parameter names must be strings")
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
        if len(names) > 64 or any(
            not isinstance(name, str) or not 1 <= len(name) <= 128 for name in names
        ):
            raise ValueError("scenario distributions support 1..64 bounded variable names")
        if len(self.constraints) > 64:
            raise ValueError("scenario distributions cannot exceed 64 constraints")
        if (
            isinstance(self.max_sampling_attempts, bool)
            or not isinstance(self.max_sampling_attempts, int)
            or self.max_sampling_attempts < 1
            or self.max_sampling_attempts > 10_000
        ):
            raise ValueError("max_sampling_attempts must be in [1, 10000]")
        if self.schema_version != "rosclaw.simforge.scenario_distribution.v1":
            raise ValueError("unsupported scenario distribution schema")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScenarioDistributionSpec:
        variables = value.get("variables")
        if not isinstance(variables, dict):
            raise ValueError("scenario distribution variables must be a mapping")
        if any(not isinstance(name, str) for name in variables):
            raise ValueError("scenario variable names must be strings")
        raw_constraints = value.get("constraints") or ()
        if not isinstance(raw_constraints, (list, tuple)):
            raise ValueError("scenario distribution constraints must be a sequence")
        return cls(
            variables=tuple(
                (str(name), ScenarioVariable.from_dict(spec))
                for name, spec in sorted(variables.items())
            ),
            constraints=tuple(ScenarioConstraint.from_value(item) for item in raw_constraints),
            max_sampling_attempts=_strict_int(
                value.get("max_sampling_attempts", 10_000),
                "max_sampling_attempts",
            ),
            schema_version=_strict_string(
                value.get("schema_version", "rosclaw.simforge.scenario_distribution.v1"),
                "schema_version",
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

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not 1 <= len(self.scenario_id) <= 128:
            raise ValueError("scenario_id must contain 1..128 characters")
        if not isinstance(self.partition, Partition):
            raise ValueError("scenario partition must be a Partition")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("scenario seed must be a non-negative 63-bit integer")
        if not _is_sha256(self.distribution_hash):
            raise ValueError("scenario distribution_hash must be a sha256 digest")
        if not isinstance(self.strategy, SamplingStrategy):
            raise ValueError("scenario strategy must be a SamplingStrategy")
        if (
            not isinstance(self.values, tuple)
            or not 1 <= len(self.values) <= 64
            or len({name for name, _item in self.values}) != len(self.values)
        ):
            raise ValueError("scenario values must contain 1..64 unique entries")
        for name, item in self.values:
            if not isinstance(name, str) or not 1 <= len(name) <= 128:
                raise ValueError("scenario value names must contain 1..128 characters")
            _validate_scalar(item, "scenario value")

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
        for name in (
            "physics_executed",
            "strict_replay",
            "artifact_hashes",
            "holdout_required",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"evidence requirement {name} must be boolean")
        if (
            isinstance(self.minimum_seeds, bool)
            or not isinstance(self.minimum_seeds, int)
            or not 2 <= self.minimum_seeds <= 10_000
        ):
            raise ValueError("minimum_seeds must be in [2, 10000]")


@dataclass(frozen=True)
class HumanInvolvement:
    failure_selected_by_human: bool = False
    patch_written_by_human: bool = False
    patch_edited_by_human: bool = False
    promotion_approved_by_human: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, bool):
                raise ValueError(f"human involvement {name} must be boolean")

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
        for name in ("task_id", "suite_id", "body_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) > 128:
                raise ValueError(f"{name} must contain 1..128 characters")
        if not self.required_capabilities:
            raise ValueError("at least one body capability is required")
        if not self.discovery_backends or not self.evaluation_backends:
            raise ValueError("discovery and evaluation backends cannot be empty")
        if not self.candidate_allowed_paths:
            raise ValueError("candidate path whitelist cannot be empty")
        if any(not path.startswith("/") for path in self.candidate_allowed_paths):
            raise ValueError("candidate paths must be absolute JSON pointers")
        for name in (
            "required_capabilities",
            "discovery_backends",
            "evaluation_backends",
            "differential_backends",
            "candidate_allowed_paths",
        ):
            values = getattr(self, name)
            if len(values) > 128 or any(
                not isinstance(item, str) or not 1 <= len(item) <= 256 for item in values
            ):
                raise ValueError(f"{name} must contain at most 128 bounded strings")
        if not isinstance(self.scenario_distribution_ref, str) or not (
            1 <= len(self.scenario_distribution_ref) <= 512
        ):
            raise ValueError("scenario_distribution_ref must contain 1..512 characters")
        if self.schema_version != "rosclaw.simforge.task.v1":
            raise ValueError("unsupported SimForge task schema")

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
            task_id=_strict_string(value.get("task_id") or "", "task_id"),
            suite_id=_strict_string(value.get("suite_id") or "", "suite_id"),
            body_id=_strict_string(body.get("body_id") or "", "body_id"),
            required_capabilities=_string_sequence(
                body.get("required_capabilities") or (),
                "required_capabilities",
            ),
            discovery_backends=_string_sequence(
                backends.get("discovery") or (),
                "discovery backends",
            ),
            evaluation_backends=_string_sequence(
                backends.get("evaluation") or (),
                "evaluation backends",
            ),
            differential_backends=_string_sequence(
                backends.get("differential") or (),
                "differential backends",
            ),
            scenario_distribution_ref=_strict_string(
                distribution.get("ref") or "",
                "scenario_distribution ref",
            ),
            success_spec=tuple(sorted(_mapping(value, "success_spec").items())),
            safety_spec=tuple(sorted(_mapping(value, "safety_spec").items())),
            candidate_allowed_paths=_string_sequence(
                candidate.get("allowed_paths") or (),
                "candidate allowed_paths",
            ),
            evidence_requirements=EvidenceRequirements(
                physics_executed=_strict_bool(
                    evidence.get("physics_executed", True),
                    "physics_executed",
                ),
                strict_replay=_strict_bool(
                    evidence.get("strict_replay", True),
                    "strict_replay",
                ),
                artifact_hashes=_strict_bool(
                    evidence.get("artifact_hashes", True),
                    "artifact_hashes",
                ),
                minimum_seeds=_strict_int(
                    evidence.get("minimum_seeds", 20),
                    "minimum_seeds",
                ),
                holdout_required=_strict_bool(
                    evidence.get("holdout_required", True),
                    "holdout_required",
                ),
            ),
            schema_version=_strict_string(
                value.get("schema_version", "rosclaw.simforge.task.v1"),
                "schema_version",
            ),
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


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _strict_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a string sequence")
    return tuple(value)


def _validate_scalar(value: Any, name: str) -> None:
    if isinstance(value, str):
        if len(value) > 4096:
            raise ValueError(f"{name} string cannot exceed 4096 characters")
        return
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return
    raise ValueError(f"{name} must be a finite JSON scalar")


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


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

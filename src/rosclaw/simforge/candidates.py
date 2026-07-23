"""Whitelist-only immutable candidate generation and bounded parameter search."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from rosclaw.simforge.models import HumanInvolvement

JsonScalar = str | int | float | bool | None


class SearchAlgorithm(StrEnum):
    RANDOM = "random"
    CMA_ES = "cma_es"
    CROSS_ENTROPY = "cross_entropy"
    BAYESIAN = "bayesian"


@dataclass(frozen=True)
class ParameterBound:
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[JsonScalar, ...] = ()

    def __post_init__(self) -> None:
        numeric = self.minimum is not None or self.maximum is not None
        if numeric:
            if self.minimum is None or self.maximum is None or self.choices:
                raise ValueError("numeric bounds require min/max and cannot define choices")
            if (
                isinstance(self.minimum, bool)
                or isinstance(self.maximum, bool)
                or not isinstance(self.minimum, (int, float))
                or not isinstance(self.maximum, (int, float))
                or not math.isfinite(float(self.minimum))
                or not math.isfinite(float(self.maximum))
            ):
                raise ValueError("candidate bounds must be finite")
            if self.minimum > self.maximum:
                raise ValueError("candidate minimum must not exceed maximum")
        elif not self.choices:
            raise ValueError("candidate bounds require either min/max or choices")
        elif len(self.choices) > 256:
            raise ValueError("candidate choices cannot exceed 256 values")
        else:
            for index, value in enumerate(self.choices):
                _validate_json_scalar(f"choice[{index}]", value)

    @property
    def numeric(self) -> bool:
        return self.minimum is not None

    def validate(self, value: JsonScalar) -> bool:
        if self.numeric:
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                and self.minimum is not None
                and self.maximum is not None
                and self.minimum <= float(value) <= self.maximum
            )
        return value in self.choices


@dataclass(frozen=True)
class CandidateChange:
    path: str
    old: JsonScalar
    new: JsonScalar

    def __post_init__(self) -> None:
        if (
            not isinstance(self.path, str)
            or not 1 <= len(self.path) <= 256
            or not self.path.startswith("/")
            or ".." in self.path
        ):
            raise ValueError("candidate change path must be a safe absolute JSON pointer")
        _validate_json_scalar(self.path, self.old)
        _validate_json_scalar(self.path, self.new)
        if self.old == self.new:
            raise ValueError("candidate changes must modify the parent value")


@dataclass(frozen=True)
class CandidateGenerator:
    type: str
    algorithm: str
    model: str | None = None

    def __post_init__(self) -> None:
        for name in ("type", "algorithm"):
            value = getattr(self, name)
            if not isinstance(value, str) or not 1 <= len(value) <= 128:
                raise ValueError(f"candidate generator {name} must contain 1..128 characters")
        if self.model is not None and (
            not isinstance(self.model, str) or not 1 <= len(self.model) <= 256
        ):
            raise ValueError("candidate generator model must contain 1..256 characters")


@dataclass(frozen=True)
class CandidatePatch:
    patch_id: str
    parent_policy_hash: str
    failure_signature_id: str
    generator: CandidateGenerator
    changes: tuple[CandidateChange, ...]
    human_involvement: HumanInvolvement
    schema_version: str = "rosclaw.candidate_patch.v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.patch_id, str)
            or not self.patch_id.startswith("patch_")
            or not 7 <= len(self.patch_id) <= 128
        ):
            raise ValueError("patch_id must begin with patch_")
        if not _sha256_id(self.parent_policy_hash):
            raise ValueError("parent_policy_hash must be a sha256 identifier")
        if (
            not isinstance(self.failure_signature_id, str)
            or not 1 <= len(self.failure_signature_id) <= 512
            or not self.changes
            or len(self.changes) > 128
        ):
            raise ValueError("failure signature and at least one change are required")
        if not isinstance(self.generator, CandidateGenerator):
            raise ValueError("candidate generator must be CandidateGenerator")
        if not isinstance(self.human_involvement, HumanInvolvement):
            raise ValueError("candidate human involvement must be HumanInvolvement")
        if self.schema_version != "rosclaw.candidate_patch.v1":
            raise ValueError("unsupported candidate patch schema")
        if len({change.path for change in self.changes}) != len(self.changes):
            raise ValueError("candidate patch cannot change the same path twice")

    @property
    def candidate_hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "patch_id": self.patch_id,
            "parent_policy_hash": self.parent_policy_hash,
            "failure_signature_id": self.failure_signature_id,
            "generator": asdict(self.generator),
            "changes": [asdict(change) for change in self.changes],
            "constraints": {
                "whitelist_passed": True,
                "safety_limits_unchanged": True,
                "gateway_bypass": False,
            },
            "rollback": {"parent_policy_hash": self.parent_policy_hash},
            "human_involvement": asdict(self.human_involvement),
        }


class CandidateCompiler:
    """Compile proposed scalar values; arbitrary source/code patches are impossible."""

    def __init__(
        self,
        *,
        parent_policy: Mapping[str, JsonScalar],
        allowed_bounds: Mapping[str, ParameterBound],
    ) -> None:
        if not parent_policy or not allowed_bounds:
            raise ValueError("parent policy and allowed bounds cannot be empty")
        if len(parent_policy) > 128 or len(allowed_bounds) > 128:
            raise ValueError("candidate compiler cannot exceed 128 policy paths")
        if any(not isinstance(path, str) for path in (*parent_policy, *allowed_bounds)):
            raise ValueError("candidate policy paths must be strings")
        if any(not isinstance(bound, ParameterBound) for bound in allowed_bounds.values()):
            raise ValueError("candidate bounds must be ParameterBound values")
        if set(allowed_bounds) - set(parent_policy):
            raise ValueError("all candidate paths must exist in the parent policy")
        if any(not path.startswith("/") or ".." in path for path in allowed_bounds):
            raise ValueError("candidate whitelist paths must be safe absolute JSON pointers")
        for path, value in parent_policy.items():
            _validate_json_scalar(path, value)
        self._parent_policy = dict(parent_policy)
        self._allowed_bounds = dict(allowed_bounds)
        self.parent_policy_hash = _policy_hash(self._parent_policy)

    @property
    def allowed_bounds(self) -> dict[str, ParameterBound]:
        return dict(self._allowed_bounds)

    @property
    def parent_policy(self) -> dict[str, JsonScalar]:
        return dict(self._parent_policy)

    def compile(
        self,
        proposed_values: Mapping[str, JsonScalar],
        *,
        failure_signature_id: str,
        generator: CandidateGenerator,
        human_involvement: HumanInvolvement | None = None,
    ) -> CandidatePatch:
        if not isinstance(proposed_values, Mapping) or not proposed_values:
            raise ValueError("candidate proposal cannot be empty")
        if len(proposed_values) > 128 or any(not isinstance(path, str) for path in proposed_values):
            raise ValueError("candidate proposal paths must be bounded strings")
        if not isinstance(failure_signature_id, str) or not (1 <= len(failure_signature_id) <= 512):
            raise ValueError("failure_signature_id must contain 1..512 characters")
        if not isinstance(generator, CandidateGenerator):
            raise ValueError("generator must be CandidateGenerator")
        if human_involvement is not None and not isinstance(human_involvement, HumanInvolvement):
            raise ValueError("human_involvement must be HumanInvolvement")
        unknown = sorted(set(proposed_values) - set(self._allowed_bounds))
        if unknown:
            raise ValueError(f"candidate paths are not whitelisted: {', '.join(unknown)}")
        changes: list[CandidateChange] = []
        for path, new_value in sorted(proposed_values.items()):
            _validate_json_scalar(path, new_value)
            if not self._allowed_bounds[path].validate(new_value):
                raise ValueError(f"candidate value is outside allowed bounds: {path}")
            old_value = self._parent_policy[path]
            if type(old_value) is not type(new_value) and not (
                isinstance(old_value, (int, float))
                and not isinstance(old_value, bool)
                and isinstance(new_value, (int, float))
                and not isinstance(new_value, bool)
            ):
                raise ValueError(f"candidate value changes type: {path}")
            if old_value != new_value:
                changes.append(CandidateChange(path=path, old=old_value, new=new_value))
        if not changes:
            raise ValueError("candidate proposal does not change the parent policy")
        identity = json.dumps(
            {
                "parent": self.parent_policy_hash,
                "failure": failure_signature_id,
                "generator": asdict(generator),
                "changes": [asdict(change) for change in changes],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        patch_id = "patch_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        return CandidatePatch(
            patch_id=patch_id,
            parent_policy_hash=self.parent_policy_hash,
            failure_signature_id=failure_signature_id,
            generator=generator,
            changes=tuple(changes),
            human_involvement=human_involvement or HumanInvolvement(),
        )


class TemplateCandidateGenerator:
    """Map diagnosed failure signatures to bounded policy templates."""

    TEMPLATES: dict[str, dict[str, JsonScalar]] = {
        "MIDPATH_COLLISION": {
            "/controller/velocity_factor": 0.62,
            "/trajectory/minimum_clearance_m": 0.045,
            "/trajectory/waypoint_policy": "obstacle_avoidance",
        },
        "DEADLINE_MISS": {"/controller/velocity_factor": 0.85},
        "OSCILLATION": {"/controller/damping": 0.8},
    }

    @classmethod
    def generate(cls, compiler: CandidateCompiler, *, failure_signature_id: str) -> CandidatePatch:
        signature = failure_signature_id.split(":", maxsplit=1)[0]
        proposal = cls.TEMPLATES.get(signature)
        if proposal is None:
            raise ValueError(f"no safe candidate template for {signature}")
        applicable = {
            path: value for path, value in proposal.items() if path in compiler.allowed_bounds
        }
        return compiler.compile(
            applicable,
            failure_signature_id=failure_signature_id,
            generator=CandidateGenerator(type="template", algorithm=signature.lower()),
        )


class SearchCandidateGenerator:
    """Bounded Random/CMA-ES/CEM/Bayesian-style search over compiler inputs."""

    def __init__(self, compiler: CandidateCompiler, *, seed: int) -> None:
        if not isinstance(compiler, CandidateCompiler):
            raise ValueError("search compiler must be CandidateCompiler")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("search seed must be an integer")
        self.compiler = compiler
        self._rng = random.Random(seed)

    def optimize(
        self,
        *,
        failure_signature_id: str,
        algorithm: SearchAlgorithm,
        objective: Callable[[CandidatePatch], float],
        budget: int,
    ) -> tuple[CandidatePatch, tuple[tuple[str, float], ...]]:
        if not isinstance(algorithm, SearchAlgorithm):
            raise ValueError("search algorithm must be a SearchAlgorithm")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 2 or budget > 10_000:
            raise ValueError("search budget must be in [2, 10000]")
        history: list[tuple[CandidatePatch, float, dict[str, JsonScalar]]] = []
        numeric_state = self._initial_numeric_state()
        for iteration in range(budget):
            proposal = self._propose(algorithm, history, numeric_state, iteration)
            patch = self.compiler.compile(
                proposal,
                failure_signature_id=failure_signature_id,
                generator=CandidateGenerator(type="search", algorithm=algorithm.value),
            )
            score = float(objective(patch))
            if not math.isfinite(score):
                score = -math.inf
            history.append((patch, score, proposal))
            if algorithm in {SearchAlgorithm.CMA_ES, SearchAlgorithm.CROSS_ENTROPY}:
                numeric_state = self._update_distribution(history, numeric_state, algorithm)
        best = max(history, key=lambda item: item[1])
        trace = tuple((item[0].candidate_hash, item[1]) for item in history)
        return best[0], trace

    def _initial_numeric_state(self) -> dict[str, tuple[float, float]]:
        state: dict[str, tuple[float, float]] = {}
        for path, bound in self.compiler.allowed_bounds.items():
            if bound.numeric:
                assert bound.minimum is not None and bound.maximum is not None
                state[path] = (
                    (bound.minimum + bound.maximum) / 2,
                    max((bound.maximum - bound.minimum) / 3, 1e-12),
                )
        return state

    def _propose(
        self,
        algorithm: SearchAlgorithm,
        history: list[tuple[CandidatePatch, float, dict[str, JsonScalar]]],
        state: dict[str, tuple[float, float]],
        iteration: int,
    ) -> dict[str, JsonScalar]:
        proposal: dict[str, JsonScalar] = {}
        for path, bound in self.compiler.allowed_bounds.items():
            if bound.numeric:
                assert bound.minimum is not None and bound.maximum is not None
                if algorithm in {SearchAlgorithm.CMA_ES, SearchAlgorithm.CROSS_ENTROPY}:
                    mean, sigma = state[path]
                    value = self._rng.gauss(mean, sigma)
                elif algorithm is SearchAlgorithm.BAYESIAN and history:
                    best = max(history, key=lambda item: item[1])[2]
                    incumbent = _numeric_candidate_value(best[path], path)
                    radius = (bound.maximum - bound.minimum) / math.sqrt(iteration + 2)
                    value = self._rng.gauss(incumbent, radius)
                else:
                    value = self._rng.uniform(bound.minimum, bound.maximum)
                proposal[path] = min(bound.maximum, max(bound.minimum, value))
            else:
                if algorithm is SearchAlgorithm.BAYESIAN and history and self._rng.random() < 0.7:
                    proposal[path] = max(history, key=lambda item: item[1])[2][path]
                else:
                    proposal[path] = self._rng.choice(bound.choices)
        return self._ensure_change(proposal)

    def _ensure_change(self, proposal: dict[str, JsonScalar]) -> dict[str, JsonScalar]:
        parent = self.compiler.parent_policy
        if any(proposal[path] != parent[path] for path in proposal):
            return proposal
        for path, bound in self.compiler.allowed_bounds.items():
            if bound.numeric:
                assert bound.minimum is not None and bound.maximum is not None
                for value in (bound.minimum, bound.maximum):
                    if value != parent[path]:
                        proposal[path] = value
                        return proposal
            else:
                for choice in bound.choices:
                    if choice != parent[path]:
                        proposal[path] = choice
                        return proposal
        raise ValueError("candidate bounds contain no value different from the parent policy")

    def _update_distribution(
        self,
        history: list[tuple[CandidatePatch, float, dict[str, JsonScalar]]],
        state: dict[str, tuple[float, float]],
        algorithm: SearchAlgorithm,
    ) -> dict[str, tuple[float, float]]:
        elite_count = max(2, math.ceil(len(history) * 0.2))
        elite = sorted(history, key=lambda item: item[1], reverse=True)[:elite_count]
        updated = dict(state)
        learning_rate = 0.45 if algorithm is SearchAlgorithm.CROSS_ENTROPY else 0.25
        for path, (old_mean, old_sigma) in state.items():
            values = [_numeric_candidate_value(item[2][path], path) for item in elite]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            sigma = max(math.sqrt(variance), old_sigma * 0.1, 1e-12)
            updated[path] = (
                old_mean * (1 - learning_rate) + mean * learning_rate,
                old_sigma * (1 - learning_rate) + sigma * learning_rate,
            )
        return updated


def _validate_json_scalar(path: str, value: Any) -> None:
    if not isinstance(value, (str, int, float, bool)) and value is not None:
        raise ValueError(f"candidate values must be JSON scalars: {path}")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and not math.isfinite(float(value))
    ):
        raise ValueError(f"candidate values must be finite: {path}")
    if isinstance(value, str) and len(value) > 4096:
        raise ValueError(f"candidate string values cannot exceed 4096 characters: {path}")


def _numeric_candidate_value(value: JsonScalar, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"candidate value is not numeric: {path}")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"candidate value is not finite: {path}")
    return normalized


def _policy_hash(policy: Mapping[str, JsonScalar]) -> str:
    payload = json.dumps(dict(policy), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _sha256_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(char in "0123456789abcdef" for char in value[7:])
    )


__all__ = [
    "CandidateChange",
    "CandidateCompiler",
    "CandidateGenerator",
    "CandidatePatch",
    "ParameterBound",
    "SearchAlgorithm",
    "SearchCandidateGenerator",
    "TemplateCandidateGenerator",
]

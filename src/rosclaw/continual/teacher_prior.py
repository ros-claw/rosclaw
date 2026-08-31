"""Generic train-only contracts for condition-selectable teacher priors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class ConditionalTeacherQuery:
    """One fully specified teacher selection bound to an immutable prior."""

    prior_contract_hash: str
    condition_values: Mapping[str, str]
    schema_version: str = "rosclaw.continual.conditional_teacher_query.v1"

    def __post_init__(self) -> None:
        if not _SHA256.fullmatch(self.prior_contract_hash):
            raise ValueError("teacher query requires a prior contract hash")
        values = {str(key): str(value) for key, value in self.condition_values.items()}
        if not values or any(
            not _IDENTIFIER.fullmatch(key) or not _IDENTIFIER.fullmatch(value)
            for key, value in values.items()
        ):
            raise ValueError("teacher query conditions must be normalized identifiers")
        object.__setattr__(self, "condition_values", MappingProxyType(dict(sorted(values.items()))))

    @property
    def query_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prior_contract_hash": self.prior_contract_hash,
            "condition_values": dict(self.condition_values),
        }


@dataclass(frozen=True)
class ConditionalTeacherPriorContract:
    """A teacher that is selected by task/condition rather than fixed phase."""

    prior_id: str
    artifact_hash: str
    body_hash: str
    observation_names: tuple[str, ...]
    output_names: tuple[str, ...]
    condition_vocabulary: Mapping[str, tuple[str, ...]]
    training_use_only: bool = True
    deployed_actor_depends_on_teacher: bool = False
    schema_version: str = "rosclaw.continual.conditional_teacher_prior.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.prior_id):
            raise ValueError("teacher prior id must be a normalized identifier")
        for label, value in (("artifact_hash", self.artifact_hash), ("body_hash", self.body_hash)):
            if not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a sha256 content hash")
        for label, names in (
            ("observation_names", self.observation_names),
            ("output_names", self.output_names),
        ):
            if (
                not names
                or len(names) != len(set(names))
                or any(not _IDENTIFIER.fullmatch(name) for name in names)
            ):
                raise ValueError(f"{label} must contain unique normalized names")
        vocabulary: dict[str, tuple[str, ...]] = {}
        for raw_key, raw_values in self.condition_vocabulary.items():
            key = str(raw_key)
            values = tuple(str(value) for value in raw_values)
            if (
                not _IDENTIFIER.fullmatch(key)
                or not values
                or len(values) != len(set(values))
                or any(not _IDENTIFIER.fullmatch(value) for value in values)
            ):
                raise ValueError("teacher condition vocabulary is invalid")
            vocabulary[key] = values
        if not vocabulary:
            raise ValueError("conditional teacher requires at least one condition")
        if not self.training_use_only or self.deployed_actor_depends_on_teacher:
            raise ValueError("teacher priors must remain train-only deployment aids")
        object.__setattr__(
            self,
            "condition_vocabulary",
            MappingProxyType(dict(sorted(vocabulary.items()))),
        )

    @property
    def contract_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def query(self, condition_values: Mapping[str, str]) -> ConditionalTeacherQuery:
        values = {str(key): str(value) for key, value in condition_values.items()}
        if set(values) != set(self.condition_vocabulary):
            raise ValueError("teacher query must set every condition exactly once")
        invalid = tuple(
            key for key, value in values.items() if value not in self.condition_vocabulary[key]
        )
        if invalid:
            raise ValueError("teacher query uses values outside the frozen vocabulary")
        return ConditionalTeacherQuery(
            prior_contract_hash=self.contract_hash,
            condition_values=values,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "prior_id": self.prior_id,
            "artifact_hash": self.artifact_hash,
            "body_hash": self.body_hash,
            "observation_names": list(self.observation_names),
            "output_names": list(self.output_names),
            "condition_vocabulary": {
                key: list(values) for key, values in self.condition_vocabulary.items()
            },
            "training_use_only": self.training_use_only,
            "deployed_actor_depends_on_teacher": self.deployed_actor_depends_on_teacher,
        }


__all__ = ["ConditionalTeacherPriorContract", "ConditionalTeacherQuery"]

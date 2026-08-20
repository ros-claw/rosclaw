"""Content-addressed learner routing and candidate lineage contracts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from rosclaw.feedback.contracts import canonical_hash

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


def _require_hash(label: str, value: str) -> None:
    if not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a sha256: content hash")


def _require_identifier(label: str, value: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a normalized identifier")


def _unique_identifiers(values: tuple[str, ...], *, label: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if not normalized or len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} must be non-empty and unique")
    for value in normalized:
        _require_identifier(label, value)
    return normalized


@dataclass(frozen=True)
class LearningJob:
    job_id: str
    learner_id: str
    data_query_hash: str
    target_artifact_kind: str
    required_field_ids: tuple[str, ...]
    trainable_component_ids: tuple[str, ...]
    schema_version: str = "rosclaw.growth.learning_job.v1"

    def __post_init__(self) -> None:
        for label in ("job_id", "learner_id", "target_artifact_kind"):
            _require_identifier(label, getattr(self, label))
        _require_hash("data_query_hash", self.data_query_hash)
        for label in ("required_field_ids", "trainable_component_ids"):
            object.__setattr__(
                self,
                label,
                _unique_identifiers(tuple(getattr(self, label)), label=label),
            )

    @property
    def job_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "learner_id": self.learner_id,
            "data_query_hash": self.data_query_hash,
            "target_artifact_kind": self.target_artifact_kind,
            "required_field_ids": list(self.required_field_ids),
            "trainable_component_ids": list(self.trainable_component_ids),
        }


@dataclass(frozen=True)
class LearningPlan:
    campaign_id: str
    skill_id: str
    failure_cluster_hashes: tuple[str, ...]
    jobs: tuple[LearningJob, ...]
    frozen_component_ids: tuple[str, ...]
    trainable_component_ids: tuple[str, ...]
    historical_anchor_bank_hash: str
    boundary_suite_hash: str
    budget: Mapping[str, float]
    promotion_profile_hash: str
    schema_version: str = "rosclaw.growth.learning_plan.v1"

    def __post_init__(self) -> None:
        for label in ("campaign_id", "skill_id"):
            _require_identifier(label, getattr(self, label))
        hashes = tuple(self.failure_cluster_hashes)
        if not hashes or len(hashes) != len(set(hashes)):
            raise ValueError("failure_cluster_hashes must be non-empty and unique")
        for value in hashes:
            _require_hash("failure_cluster_hashes", value)
        object.__setattr__(self, "failure_cluster_hashes", hashes)
        jobs = tuple(self.jobs)
        if not jobs or any(not isinstance(value, LearningJob) for value in jobs):
            raise ValueError("jobs must contain LearningJob records")
        if len({value.job_id for value in jobs}) != len(jobs):
            raise ValueError("job ids must be unique")
        object.__setattr__(self, "jobs", jobs)
        for label in ("frozen_component_ids", "trainable_component_ids"):
            object.__setattr__(
                self,
                label,
                _unique_identifiers(tuple(getattr(self, label)), label=label),
            )
        overlap = set(self.frozen_component_ids).intersection(self.trainable_component_ids)
        if overlap:
            raise ValueError(f"components cannot be both frozen and trainable: {sorted(overlap)}")
        declared_trainable = set(self.trainable_component_ids)
        undeclared = {
            component
            for job in jobs
            for component in job.trainable_component_ids
            if component not in declared_trainable
        }
        if undeclared:
            raise ValueError(
                f"jobs reference undeclared trainable components: {sorted(undeclared)}"
            )
        for label in (
            "historical_anchor_bank_hash",
            "boundary_suite_hash",
            "promotion_profile_hash",
        ):
            _require_hash(label, getattr(self, label))
        budget = {str(key): float(value) for key, value in self.budget.items()}
        if not budget:
            raise ValueError("budget must not be empty")
        for key, value in budget.items():
            _require_identifier("budget key", key)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("budget values must be finite and positive")
        object.__setattr__(self, "budget", MappingProxyType(budget))

    @property
    def plan_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "skill_id": self.skill_id,
            "failure_cluster_hashes": list(self.failure_cluster_hashes),
            "jobs": [value.to_dict() for value in self.jobs],
            "frozen_component_ids": list(self.frozen_component_ids),
            "trainable_component_ids": list(self.trainable_component_ids),
            "historical_anchor_bank_hash": self.historical_anchor_bank_hash,
            "boundary_suite_hash": self.boundary_suite_hash,
            "budget": dict(sorted(self.budget.items())),
            "promotion_profile_hash": self.promotion_profile_hash,
            "evidence_domain": "SIM_ONLY",
            "hardware_authorized": False,
        }


@dataclass(frozen=True)
class CandidateManifest:
    candidate_id: str
    skill_id: str
    parent_artifact_hash: str
    learning_plan_hash: str
    code_hash: str
    environment_hash: str
    body_hash: str
    training_data_hashes: tuple[str, ...]
    learned_artifacts: Mapping[str, str]
    learned_output_fraction: float
    schema_version: str = "rosclaw.growth.candidate_manifest.v1"

    def __post_init__(self) -> None:
        for label in ("candidate_id", "skill_id"):
            _require_identifier(label, getattr(self, label))
        for label in (
            "parent_artifact_hash",
            "learning_plan_hash",
            "code_hash",
            "environment_hash",
            "body_hash",
        ):
            _require_hash(label, getattr(self, label))
        data_hashes = tuple(self.training_data_hashes)
        if not data_hashes or len(data_hashes) != len(set(data_hashes)):
            raise ValueError("training_data_hashes must be non-empty and unique")
        for value in data_hashes:
            _require_hash("training_data_hashes", value)
        object.__setattr__(self, "training_data_hashes", data_hashes)
        learned = {str(key): str(value) for key, value in self.learned_artifacts.items()}
        if not learned:
            raise ValueError("learned_artifacts must not be empty")
        for key, value in learned.items():
            _require_identifier("learned artifact key", key)
            _require_hash("learned artifact hash", value)
        object.__setattr__(self, "learned_artifacts", MappingProxyType(learned))
        if (
            not math.isfinite(self.learned_output_fraction)
            or not 0.0 <= self.learned_output_fraction <= 1.0
        ):
            raise ValueError("learned_output_fraction must be finite and in [0, 1]")

    @property
    def manifest_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "skill_id": self.skill_id,
            "parent_artifact_hash": self.parent_artifact_hash,
            "learning_plan_hash": self.learning_plan_hash,
            "code_hash": self.code_hash,
            "environment_hash": self.environment_hash,
            "body_hash": self.body_hash,
            "training_data_hashes": list(self.training_data_hashes),
            "learned_artifacts": dict(sorted(self.learned_artifacts.items())),
            "learned_output_fraction": self.learned_output_fraction,
            "promotion_truth_allowed": False,
            "evidence_domain": "SIM_ONLY",
            "hardware_authorized": False,
        }


__all__ = ["CandidateManifest", "LearningJob", "LearningPlan"]

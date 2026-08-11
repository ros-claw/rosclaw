"""Runtime-free registries for downstream Growth adapters and learners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rosclaw.extension_discovery import (
    EntryPointDiscoveryReport,
    discover_entry_point_objects,
)
from rosclaw.growth._validation import require_identifier, unique_identifiers
from rosclaw.growth.experience import ExperienceSegment, FailureSignature

GROWTH_ADAPTER_GROUP = "rosclaw.growth.adapters"
GROWTH_LEARNER_GROUP = "rosclaw.growth.learners"
_DISABLE_ENV = "ROSCLAW_DISABLE_GROWTH_EXTENSIONS"


@runtime_checkable
class GrowthAdapter(Protocol):
    """Domain semantic boundary; deliberately carries no execution authority."""

    adapter_id: str
    skill_ids: tuple[str, ...]

    def normalize_experience(self, payload: Mapping[str, Any]) -> ExperienceSegment: ...

    def diagnose(self, segment: ExperienceSegment) -> FailureSignature | None: ...


@dataclass(frozen=True)
class LearnerDescriptor:
    learner_id: str
    required_field_ids: tuple[str, ...]
    artifact_kinds: tuple[str, ...]
    online_rollout_required: bool
    consumes_cost_vector: bool
    schema_version: str = "rosclaw.growth.learner_descriptor.v1"

    def __post_init__(self) -> None:
        require_identifier("learner_id", self.learner_id)
        for label in ("required_field_ids", "artifact_kinds"):
            object.__setattr__(
                self,
                label,
                unique_identifiers(tuple(getattr(self, label)), label=label),
            )


@dataclass(frozen=True)
class GrowthDiscoveryReport:
    adapters: EntryPointDiscoveryReport
    learners: EntryPointDiscoveryReport


class GrowthExtensionRegistry:
    """In-memory catalog populated only with validated extension contracts."""

    def __init__(self) -> None:
        self._adapters: dict[str, GrowthAdapter] = {}
        self._learners: dict[str, LearnerDescriptor] = {}

    def register_adapter(self, adapter: Any) -> str:
        if not isinstance(adapter, GrowthAdapter):
            raise TypeError("growth adapter does not satisfy the GrowthAdapter protocol")
        require_identifier("adapter_id", adapter.adapter_id)
        skill_ids = unique_identifiers(tuple(adapter.skill_ids), label="skill_ids")
        if tuple(adapter.skill_ids) != skill_ids:
            raise ValueError("adapter skill_ids must be a stable tuple")
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"duplicate growth adapter: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter
        return adapter.adapter_id

    def register_learner(self, descriptor: Any) -> str:
        if not isinstance(descriptor, LearnerDescriptor):
            raise TypeError("learner entry point must expose LearnerDescriptor")
        if descriptor.learner_id in self._learners:
            raise ValueError(f"duplicate learner: {descriptor.learner_id}")
        self._learners[descriptor.learner_id] = descriptor
        return descriptor.learner_id

    def adapter(self, adapter_id: str) -> GrowthAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise KeyError(f"unknown growth adapter: {adapter_id}") from exc

    def learner(self, learner_id: str) -> LearnerDescriptor:
        try:
            return self._learners[learner_id]
        except KeyError as exc:
            raise KeyError(f"unknown growth learner: {learner_id}") from exc

    @property
    def adapter_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    @property
    def learner_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._learners))

    def discover(self) -> GrowthDiscoveryReport:
        adapters = discover_entry_point_objects(
            group=GROWTH_ADAPTER_GROUP,
            disable_env=_DISABLE_ENV,
            register=self.register_adapter,
        )
        learners = discover_entry_point_objects(
            group=GROWTH_LEARNER_GROUP,
            disable_env=_DISABLE_ENV,
            register=self.register_learner,
        )
        return GrowthDiscoveryReport(adapters=adapters, learners=learners)


__all__ = [
    "GROWTH_ADAPTER_GROUP",
    "GROWTH_LEARNER_GROUP",
    "GrowthAdapter",
    "GrowthDiscoveryReport",
    "GrowthExtensionRegistry",
    "LearnerDescriptor",
]

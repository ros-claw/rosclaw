"""TaskCard — structured task prior for Know module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TaskCard:
    """Task prior compiled from experience, papers, and benchmarks."""

    task_id: str = ""
    task_family: str = ""
    domain: str = ""
    embodiment_type: str = ""
    objective_direction: str = "maximize"
    metric_name: str = ""
    prerequisites: list[str] = field(default_factory=list)
    common_failures: list[dict[str, Any]] = field(default_factory=list)
    verified_patterns: list[str] = field(default_factory=list)
    source_manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "domain": self.domain,
            "embodiment_type": self.embodiment_type,
            "objective_direction": self.objective_direction,
            "metric_name": self.metric_name,
            "prerequisites": self.prerequisites,
            "common_failures": self.common_failures,
            "verified_patterns": self.verified_patterns,
            "source_manifest": self.source_manifest,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskCard:
        return cls(
            task_id=d.get("task_id", ""),
            task_family=d.get("task_family", ""),
            domain=d.get("domain", ""),
            embodiment_type=d.get("embodiment_type", ""),
            objective_direction=d.get("objective_direction", "maximize"),
            metric_name=d.get("metric_name", ""),
            prerequisites=list(d.get("prerequisites", [])),
            common_failures=list(d.get("common_failures", [])),
            verified_patterns=list(d.get("verified_patterns", [])),
            source_manifest=dict(d.get("source_manifest", {})),
        )

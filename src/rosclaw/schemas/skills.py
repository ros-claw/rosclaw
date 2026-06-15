"""Skill versioning and champion schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ChampionLevel = Literal[
    "baseline_champion",
    "sim_champion",
    "sandbox_champion",
    "real_candidate",
    "real_champion",
    "deprecated",
]


@dataclass
class SkillVersion:
    """Versioned skill reference."""

    skill_id: str = ""
    name: str = ""
    version: str = "1.0.0"
    skill_type: str = "programmed"  # programmed | learned | hybrid
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    champion_level: ChampionLevel = "baseline_champion"
    parent_skill_id: str = ""
    lineage_id: str = ""
    created_at: str = ""
    updated_at: str = ""
    execution_count: int = 0
    success_rate: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "version": self.version,
            "skill_type": self.skill_type,
            "description": self.description,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "success_criteria": self.success_criteria,
            "champion_level": self.champion_level,
            "parent_skill_id": self.parent_skill_id,
            "lineage_id": self.lineage_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillVersion:
        return cls(
            skill_id=d.get("skill_id", ""),
            name=d.get("name", ""),
            version=d.get("version", "1.0.0"),
            skill_type=d.get("skill_type", "programmed"),
            description=d.get("description", ""),
            parameters=dict(d.get("parameters", {})),
            preconditions=list(d.get("preconditions", [])),
            success_criteria=list(d.get("success_criteria", [])),
            champion_level=d.get("champion_level", "baseline_champion"),
            parent_skill_id=d.get("parent_skill_id", ""),
            lineage_id=d.get("lineage_id", ""),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            execution_count=d.get("execution_count", 0),
            success_rate=d.get("success_rate", 0.0),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class SkillCandidate:
    """Candidate skill awaiting evaluation."""

    candidate_id: str = ""
    skill_id: str = ""
    base_skill_id: str = ""
    patch_id: str = ""
    task_id: str = ""
    status: str = "pending"  # pending | evaluating | evaluated | rejected
    metrics: dict[str, float] = field(default_factory=dict)
    sandbox_passed: bool = False
    darwin_passed: bool = False
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "skill_id": self.skill_id,
            "base_skill_id": self.base_skill_id,
            "patch_id": self.patch_id,
            "task_id": self.task_id,
            "status": self.status,
            "metrics": self.metrics,
            "sandbox_passed": self.sandbox_passed,
            "darwin_passed": self.darwin_passed,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillCandidate:
        return cls(
            candidate_id=d.get("candidate_id", ""),
            skill_id=d.get("skill_id", ""),
            base_skill_id=d.get("base_skill_id", ""),
            patch_id=d.get("patch_id", ""),
            task_id=d.get("task_id", ""),
            status=d.get("status", "pending"),
            metrics=dict(d.get("metrics", {})),
            sandbox_passed=d.get("sandbox_passed", False),
            darwin_passed=d.get("darwin_passed", False),
            created_at=d.get("created_at", ""),
        )


@dataclass
class SkillLineage:
    """Skill genealogy: ancestors, descendants, siblings."""

    lineage_id: str = ""
    root_skill_id: str = ""
    skill_versions: list[str] = field(default_factory=list)
    promotions: list[dict[str, Any]] = field(default_factory=list)
    dead_ends: list[str] = field(default_factory=list)
    current_champion: str = ""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "root_skill_id": self.root_skill_id,
            "skill_versions": self.skill_versions,
            "promotions": self.promotions,
            "dead_ends": self.dead_ends,
            "current_champion": self.current_champion,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SkillLineage:
        return cls(
            lineage_id=d.get("lineage_id", ""),
            root_skill_id=d.get("root_skill_id", ""),
            skill_versions=list(d.get("skill_versions", [])),
            promotions=list(d.get("promotions", [])),
            dead_ends=list(d.get("dead_ends", [])),
            current_champion=d.get("current_champion", ""),
            created_at=d.get("created_at", ""),
        )

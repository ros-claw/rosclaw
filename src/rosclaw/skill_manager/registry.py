"""Skill Registry - Skill registration, versioning, championing, and rollback."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.skill_manager.registry")


@dataclass
class SkillEntry:
    """Represents a registered skill with versioning support."""

    name: str
    description: str
    skill_type: str  # "programmed", "learned", "hybrid"
    parameters: dict[str, Any] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    execution_count: int = 0
    success_rate: float = 0.0
    handler: Optional[Callable] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # v1.5 champion / versioning fields
    version: str = "1.0.0"
    lineage_id: str = ""
    parent_skill_id: str = ""
    champion_level: str = "baseline_champion"  # baseline | sim | sandbox | real_candidate | real | deprecated

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "skill_type": self.skill_type,
            "parameters": self.parameters,
            "preconditions": self.preconditions,
            "success_criteria": self.success_criteria,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "execution_count": self.execution_count,
            "success_rate": self.success_rate,
            "metadata": self.metadata,
            "version": self.version,
            "lineage_id": self.lineage_id,
            "parent_skill_id": self.parent_skill_id,
            "champion_level": self.champion_level,
        }


class SkillRegistry(LifecycleMixin):
    """Central registry for all robot skills with versioning and champion support."""

    def __init__(self, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.event_bus = event_bus
        # Keyed by full skill_id (name@version)
        self._skills: dict[str, SkillEntry] = {}
        # Quick lookup: name -> list of versions
        self._by_name: dict[str, list[str]] = {}
        # Champion lookup: name -> champion_level -> skill_id
        self._champions: dict[str, dict[str, str]] = {}
        # Rollback history: skill_id -> list of previous skill_ids
        self._rollback_history: dict[str, list[dict[str, Any]]] = {}

    def _do_initialize(self) -> None:
        logger.info("SkillRegistry initialized")
        if self.event_bus is not None:
            self.event_bus.subscribe("praxis.completed", self._on_praxis_completed)
            self.event_bus.subscribe("praxis.failed", self._on_praxis_failed)
            self.event_bus.subscribe("skill.execution.complete", self._on_skill_complete)

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("praxis.completed", self._on_praxis_completed)
            self.event_bus.unsubscribe("praxis.failed", self._on_praxis_failed)
            self.event_bus.unsubscribe("skill.execution.complete", self._on_skill_complete)

    @staticmethod
    def _make_id(name: str, version: str) -> str:
        return f"{name}@{version}"

    def register(self, entry: SkillEntry) -> str:
        """Register a skill. Returns the full skill_id."""
        if not isinstance(entry, SkillEntry):
            raise TypeError(f"Expected SkillEntry, got {type(entry).__name__}")
        if not entry.name or not isinstance(entry.name, str):
            raise ValueError(f"Skill name must be a non-empty string, got {entry.name!r}")
        if not entry.version:
            entry.version = "1.0.0"
        if not entry.lineage_id:
            entry.lineage_id = f"lineage_{entry.name}"

        skill_id = self._make_id(entry.name, entry.version)

        # Save rollback history
        if skill_id in self._skills:
            old = self._skills[skill_id]
            self._rollback_history.setdefault(skill_id, []).append(old.to_dict())
            logger.info("Overwriting skill: %s (was %s)", skill_id, old.version)

        self._skills[skill_id] = entry
        self._by_name.setdefault(entry.name, [])
        if skill_id not in self._by_name[entry.name]:
            self._by_name[entry.name].append(skill_id)

        logger.info("Registered skill: %s (%s) level=%s", skill_id, entry.skill_type, entry.champion_level)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="skill.registered",
                payload={
                    "skill_name": entry.name,
                    "skill_id": skill_id,
                    "skill_type": entry.skill_type,
                    "version": entry.version,
                    "champion_level": entry.champion_level,
                },
                source="skill_registry",
                priority=EventPriority.NORMAL,
            ))
        return skill_id

    def unregister(self, skill_id: str) -> bool:
        """Remove a skill by full skill_id or by bare name."""
        target_id = skill_id
        if target_id not in self._skills:
            # Backward compatibility: treat as bare name and remove latest version
            entry = self.get_by_name(skill_id)
            if entry is None:
                return False
            target_id = self._make_id(entry.name, entry.version)
        entry = self._skills[target_id]
        del self._skills[target_id]
        if target_id in self._by_name.get(entry.name, []):
            self._by_name[entry.name].remove(target_id)
        return True

    def get(self, skill_id: str) -> Optional[SkillEntry]:
        """Retrieve a skill by full skill_id (name@version) or by name."""
        entry = self._skills.get(skill_id)
        if entry is not None:
            return entry
        # Backward compatibility: allow lookup by bare name
        return self.get_by_name(skill_id)

    def get_by_name(self, name: str, version: str | None = None) -> Optional[SkillEntry]:
        """Retrieve a skill by name and optional version."""
        if version:
            return self._skills.get(self._make_id(name, version))
        # Return latest version
        versions = self._by_name.get(name, [])
        if not versions:
            return None
        # Sort by version string (simple semver sort)
        versions_sorted = sorted(versions, key=lambda v: v.split("@")[1])
        return self._skills.get(versions_sorted[-1])

    def list_skills(
        self, skill_type: Optional[str] = None, champion_level: Optional[str] = None, return_entries: bool = False, full_ids: bool = False
    ) -> "list[str] | list[SkillEntry]":
        """List all registered skills with optional filtering.

        By default returns bare skill names for backward compatibility.
        Set full_ids=True to receive name@version identifiers.
        """
        skills = list(self._skills.values())
        if skill_type:
            skills = [s for s in skills if s.skill_type == skill_type]
        if champion_level:
            skills = [s for s in skills if s.champion_level == champion_level]
        if return_entries:
            return skills
        if full_ids:
            return [self._make_id(s.name, s.version) for s in skills]
        # Backward compatibility: return unique bare names
        seen = set()
        names = []
        for s in skills:
            if s.name not in seen:
                seen.add(s.name)
                names.append(s.name)
        return names

    def find_by_precondition(self, precondition: str) -> list[SkillEntry]:
        """Find skills matching a precondition."""
        return [s for s in self._skills.values() if precondition in s.preconditions]

    def promote(self, skill_id: str, to_level: str) -> bool:
        """Promote a skill to a higher champion level."""
        entry = self._skills.get(skill_id)
        if entry is None:
            logger.warning("Promote failed: skill %s not found", skill_id)
            return False
        entry.champion_level = to_level
        entry.updated_at = time.time()
        self._champions.setdefault(entry.name, {})[to_level] = skill_id
        logger.info("Promoted %s to %s", skill_id, to_level)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="skill.champion.promoted",
                payload={
                    "skill_id": skill_id,
                    "skill_name": entry.name,
                    "version": entry.version,
                    "champion_level": to_level,
                },
                source="skill_registry",
                priority=EventPriority.HIGH,
            ))
        return True

    def get_champion(self, name: str, level: str = "real_champion") -> Optional[SkillEntry]:
        """Get the champion skill for a task at a given level."""
        skill_id = self._champions.get(name, {}).get(level)
        if skill_id:
            return self._skills.get(skill_id)
        # Fallback: search directly
        for entry in self._skills.values():
            if entry.name == name and entry.champion_level == level:
                return entry
        return None

    def list_champions(self, name: str | None = None) -> list[SkillEntry]:
        """List all champion skills."""
        champions = [s for s in self._skills.values() if s.champion_level != "deprecated"]
        if name:
            champions = [s for s in champions if s.name == name]
        return champions

    def rollback(self, name: str, to_version: str) -> bool:
        """Rollback a skill to a previous version."""
        target_id = self._make_id(name, to_version)
        target = self._skills.get(target_id)
        if target is None:
            logger.warning("Rollback failed: %s not found", target_id)
            return False
        # Mark current as deprecated, restore target
        for sid, entry in self._skills.items():
            if entry.name == name and entry.champion_level != "deprecated":
                entry.champion_level = "deprecated"
                entry.updated_at = time.time()
        target.champion_level = "baseline_champion"
        target.updated_at = time.time()
        logger.info("Rolled back %s to version %s", name, to_version)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="skill.rollback",
                payload={"skill_name": name, "to_version": to_version, "skill_id": target_id},
                source="skill_registry",
                priority=EventPriority.HIGH,
            ))
        return True

    def list_lineage(self, name: str) -> list[dict[str, Any]]:
        """Return version history for a skill name."""
        versions = self._by_name.get(name, [])
        result = []
        for skill_id in sorted(versions, key=lambda v: v.split("@")[1]):
            entry = self._skills.get(skill_id)
            if entry:
                result.append(entry.to_dict())
        return result

    def update_stats(self, name: str, success: bool) -> None:
        """Update execution statistics for the latest version of a skill."""
        entry = self.get_by_name(name)
        if entry is None:
            return
        entry.execution_count += 1
        if success:
            entry.success_rate = (
                (entry.success_rate * (entry.execution_count - 1) + 1.0)
                / entry.execution_count
            )
        else:
            entry.success_rate = (
                entry.success_rate * (entry.execution_count - 1)
            ) / entry.execution_count
        entry.updated_at = time.time()

    def _on_praxis_completed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        if skill_name:
            self.update_stats(skill_name, success=True)

    def _on_praxis_failed(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        if skill_name:
            self.update_stats(skill_name, success=False)

    def _on_skill_complete(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        result = payload.get("result", {})
        if skill_name:
            self.update_stats(skill_name, success=(result.get("status") == "success"))

    def get_stats(self) -> dict[str, Any]:
        total = len(self._skills)
        if total == 0:
            return {"total_skills": 0, "total_executions": 0, "average_success_rate": 0.0}
        total_exec = sum(s.execution_count for s in self._skills.values())
        avg_success = sum(s.success_rate for s in self._skills.values()) / total
        by_type: dict[str, Any] = {}
        by_level: dict[str, Any] = {}
        for s in self._skills.values():
            by_type.setdefault(s.skill_type, {"count": 0, "executions": 0})
            by_type[s.skill_type]["count"] += 1
            by_type[s.skill_type]["executions"] += s.execution_count
            by_level.setdefault(s.champion_level, 0)
            by_level[s.champion_level] += 1
        return {
            "total_skills": total,
            "total_executions": total_exec,
            "average_success_rate": round(avg_success, 4),
            "by_type": by_type,
            "by_champion_level": by_level,
        }

    @property
    def count(self) -> int:
        return len(self._skills)

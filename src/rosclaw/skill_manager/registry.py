"""Skill Registry - Skill registration and discovery."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.skill_manager.registry")


@dataclass
class SkillEntry:
    """Represents a registered skill."""
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
        }


class SkillRegistry(LifecycleMixin):
    """
    Central registry for all robot skills.

    Subscribes to praxis.completed via EventBus to auto-update
    skill statistics from execution results.
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        super().__init__()
        self.event_bus = event_bus
        self._skills: dict[str, SkillEntry] = {}

    def _do_initialize(self) -> None:
        logger.info("Initialized")
        if self.event_bus is not None:
            self.event_bus.subscribe("praxis.completed", self._on_praxis_completed)
            self.event_bus.subscribe("praxis.failed", self._on_praxis_failed)
            self.event_bus.subscribe("skill.execution.complete", self._on_skill_complete)

    def _do_stop(self) -> None:
        if self.event_bus is not None:
            self.event_bus.unsubscribe("praxis.completed", self._on_praxis_completed)
            self.event_bus.unsubscribe("praxis.failed", self._on_praxis_failed)
            self.event_bus.unsubscribe("skill.execution.complete", self._on_skill_complete)

    def _on_praxis_completed(self, event: Event) -> None:
        """Auto-update skill stats from successful praxis."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        if skill_name:
            self.update_stats(skill_name, success=True)

    def _on_praxis_failed(self, event: Event) -> None:
        """Auto-update skill stats from failed praxis."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        if skill_name:
            self.update_stats(skill_name, success=False)

    def _on_skill_complete(self, event: Event) -> None:
        """Update stats from skill execution completion events."""
        payload = event.payload if isinstance(event.payload, dict) else {}
        skill_name = payload.get("skill_name")
        result = payload.get("result", {})
        if skill_name:
            self.update_stats(skill_name, success=(result.get("status") == "success"))

    def register(self, entry: SkillEntry) -> None:
        """Register a skill."""
        if not isinstance(entry, SkillEntry):
            raise TypeError(f"Expected SkillEntry, got {type(entry).__name__}")
        if not entry.name or not isinstance(entry.name, str):
            raise ValueError(f"Skill name must be a non-empty string, got {entry.name!r}")
        if entry.name in self._skills:
            logger.info("Overwriting skill: %s", entry.name)
        self._skills[entry.name] = entry
        logger.info("Registered skill: %s (%s)", entry.name, entry.skill_type)
        if self.event_bus is not None:
            self.event_bus.publish(Event(
                topic="skill.registered",
                payload={"skill_name": entry.name, "skill_type": entry.skill_type},
                source="skill_registry",
                priority=EventPriority.NORMAL,
            ))

    def unregister(self, name: str) -> bool:
        """Remove a skill from registry."""
        if name in self._skills:
            del self._skills[name]
            return True
        return False

    def get(self, name: str) -> Optional[SkillEntry]:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    def list_skills(
        self, skill_type: Optional[str] = None, return_entries: bool = False
    ) -> "list[str] | list[SkillEntry]":
        """List all registered skills.

        Args:
            skill_type: Filter by skill type ("programmed", "learned", "hybrid")
            return_entries: If True, return list[SkillEntry]; if False, return list[str]

        Returns:
            list[str] of skill names (default) or list[SkillEntry] if return_entries=True
        """
        skills = self._skills.values()
        if skill_type:
            skills = [s for s in skills if s.skill_type == skill_type]
        if return_entries:
            return list(skills)
        return [s.name for s in skills]

    def find_by_precondition(self, precondition: str) -> list[SkillEntry]:
        """Find skills matching a precondition."""
        return [s for s in self._skills.values() if precondition in s.preconditions]

    def update_stats(self, name: str, success: bool) -> None:
        """Update execution statistics for a skill."""
        skill = self._skills.get(name)
        if skill is None:
            return
        skill.execution_count += 1
        if success:
            skill.success_rate = (
                (skill.success_rate * (skill.execution_count - 1) + 1.0)
                / skill.execution_count  # noqa: W503
            )
        else:
            skill.success_rate = (
                skill.success_rate * (skill.execution_count - 1)
            ) / skill.execution_count
        skill.updated_at = time.time()

    def get_stats(self) -> dict:
        """Get aggregated statistics for all registered skills."""
        total = len(self._skills)
        if total == 0:
            return {"total_skills": 0, "total_executions": 0, "average_success_rate": 0.0}
        total_exec = sum(s.execution_count for s in self._skills.values())
        avg_success = sum(s.success_rate for s in self._skills.values()) / total
        by_type = {}
        for s in self._skills.values():
            by_type.setdefault(s.skill_type, {"count": 0, "executions": 0})
            by_type[s.skill_type]["count"] += 1
            by_type[s.skill_type]["executions"] += s.execution_count
        return {
            "total_skills": total,
            "total_executions": total_exec,
            "average_success_rate": round(avg_success, 4),
            "by_type": by_type,
        }

    @property
    def count(self) -> int:
        return len(self._skills)

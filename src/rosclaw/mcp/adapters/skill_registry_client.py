"""Thin adapter around the ROSClaw skill registry for MCP tools."""

from __future__ import annotations

from typing import Any


class SkillRegistryClient:
    """Read-only client that lists skills from a SkillRegistry or SkillExecutor."""

    def __init__(self, skill_manager: Any) -> None:
        self._skill_manager = skill_manager

    def list_skills(self, *, skill_type: str | None = None, full_ids: bool = False) -> dict[str, Any]:
        """Return a typed response with skill entries from the registry.

        Falls back to the skill manager's own ``list_skills`` when no registry
        attribute is exposed.
        """
        registry = getattr(self._skill_manager, "registry", None)
        target = registry if registry is not None else self._skill_manager
        entries = target.list_skills(
            skill_type=skill_type,
            return_entries=True,
            full_ids=full_ids,
        )
        return {
            "skills": [e.to_dict() if hasattr(e, "to_dict") else str(e) for e in entries],
            "count": len(entries),
            "mode": "live",
        }

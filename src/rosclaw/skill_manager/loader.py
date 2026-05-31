"""Skill Loader - Loads skills from files and demonstrations."""

import json
import logging
from pathlib import Path
from typing import Any, Optional

from rosclaw.skill_manager.registry import SkillRegistry, SkillEntry

logger = logging.getLogger("rosclaw.skill_manager.loader")


class SkillLoader:
    """
    Loads skills from various sources:
    - JSON skill definition files
    - LeRobot datasets (demonstrations)
    - Programmatic skill creation
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def load_from_json(self, path: Path) -> Optional[SkillEntry]:
        """Load a skill from a JSON definition file."""
        with open(path) as f:
            data = json.load(f)

        entry = SkillEntry(
            name=data["name"],
            description=data.get("description", ""),
            skill_type=data.get("skill_type", "programmed"),
            parameters=data.get("parameters", {}),
            preconditions=data.get("preconditions", []),
            success_criteria=data.get("success_criteria", []),
            metadata=data.get("metadata", {}),
        )
        self.registry.register(entry)
        return entry

    def load_from_directory(self, directory: Path) -> int:
        """Load all .json skill files from a directory."""
        count = 0
        for path in directory.glob("*.json"):
            try:
                self.load_from_json(path)
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        return count

    def create_programmed_skill(
        self,
        name: str,
        description: str,
        handler: Any,
        parameters: Optional[dict[str, Any]] = None,
        preconditions: Optional[list[str]] = None,
    ) -> SkillEntry:
        """Create and register a programmed skill."""
        entry = SkillEntry(
            name=name,
            description=description,
            skill_type="programmed",
            parameters=parameters or {},
            preconditions=preconditions or [],
            handler=handler,
        )
        self.registry.register(entry)
        return entry

    def create_from_demonstration(
        self,
        name: str,
        demonstration: dict[str, Any],
        description: str = "",
    ) -> SkillEntry:
        """Create a skill from demonstration data."""
        entry = SkillEntry(
            name=name,
            description=description or f"Learned skill from demonstration: {name}",
            skill_type="learned",
            parameters={"demonstration": demonstration},
            metadata={"source": "demonstration"},
        )
        self.registry.register(entry)
        return entry

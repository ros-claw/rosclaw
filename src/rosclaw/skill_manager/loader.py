"""Skill Loader - Loads skills from files and demonstrations."""

import json
import logging
from pathlib import Path
from typing import Any

from rosclaw.body.schema import SkillManifest
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry

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

    def load_from_json(self, path: Path) -> SkillEntry | None:
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
        """Load all .json and .skill.yaml files from a directory."""
        count = 0
        for path in directory.glob("*.json"):
            try:
                self.load_from_json(path)
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        for path in directory.glob("*.skill.yaml"):
            try:
                self.load_skill_manifest(path)
                count += 1
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        return count

    def load_skill_manifest(self, path: Path) -> SkillEntry | None:
        """Load a YAML skill manifest with body compatibility requirements."""
        manifest = SkillManifest.from_yaml(path)
        entry = SkillEntry(
            name=manifest.skill_id,
            description=manifest.display_name or manifest.skill_id,
            skill_type="programmed",
            parameters={},
            preconditions=[],
            success_criteria=[],
            metadata={"manifest_path": str(path), "manifest": manifest.to_dict()},
            version=manifest.skill_version,
            requirements=manifest.requires,
        )
        self.registry.register(entry)
        return entry

    def create_programmed_skill(
        self,
        name: str,
        description: str,
        handler: Any,
        parameters: dict[str, Any] | None = None,
        preconditions: list[str] | None = None,
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

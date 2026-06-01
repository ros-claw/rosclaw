"""Additional coverage tests for skill_manager/loader.py."""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.skill_manager.loader import SkillLoader
from rosclaw.skill_manager.registry import SkillRegistry


class TestSkillLoaderDirectory:
    def test_load_from_directory_with_invalid_json(self, tmp_path, caplog):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        # Valid JSON file
        valid = tmp_path / "valid.json"
        valid.write_text(json.dumps({
            "name": "valid_skill",
            "description": "A valid skill",
        }))

        # Invalid JSON file
        invalid = tmp_path / "invalid.json"
        invalid.write_text("not json")

        with caplog.at_level(logging.WARNING, logger="rosclaw.skill_manager.loader"):
            count = loader.load_from_directory(tmp_path)

        assert count == 1
        assert "Failed to load" in caplog.text
        assert "invalid.json" in caplog.text

    def test_load_from_directory_empty(self, tmp_path):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        count = loader.load_from_directory(tmp_path)
        assert count == 0

    def test_load_from_directory_missing_key(self, tmp_path, caplog):
        registry = SkillRegistry()
        loader = SkillLoader(registry)

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"description": "missing name"}))

        with caplog.at_level(logging.WARNING):
            count = loader.load_from_directory(tmp_path)

        assert count == 0
        assert "Failed to load" in caplog.text


class TestSkillLoaderCreateFromDemonstration:
    def test_create_from_demonstration_default_description(self):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        demo = {"trajectory": [[0.0]]}

        entry = loader.create_from_demonstration("pick_demo", demo)

        assert entry.name == "pick_demo"
        assert entry.skill_type == "learned"
        assert "pick_demo" in entry.description
        assert entry.parameters["demonstration"] == demo
        assert entry.metadata["source"] == "demonstration"

    def test_create_from_demonstration_custom_description(self):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        demo = {"trajectory": [[0.0]]}

        entry = loader.create_from_demonstration(
            "place_demo", demo, description="Custom desc"
        )

        assert entry.description == "Custom desc"


class TestSkillLoaderCreateProgrammed:
    def test_create_programmed_skill_with_defaults(self):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        handler = MagicMock()

        entry = loader.create_programmed_skill(
            "move_arm", "Move arm to target", handler
        )

        assert entry.name == "move_arm"
        assert entry.skill_type == "programmed"
        assert entry.parameters == {}
        assert entry.preconditions == []
        assert entry.handler is handler

    def test_create_programmed_skill_with_params(self):
        registry = SkillRegistry()
        loader = SkillLoader(registry)
        handler = MagicMock()

        entry = loader.create_programmed_skill(
            "grip",
            "Grip object",
            handler,
            parameters={"force": 0.5},
            preconditions=["object_nearby"],
        )

        assert entry.parameters == {"force": 0.5}
        assert entry.preconditions == ["object_nearby"]

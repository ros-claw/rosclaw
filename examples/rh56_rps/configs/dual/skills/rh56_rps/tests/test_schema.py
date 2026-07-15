"""Smoke tests for the rh56_rps skill package schema files."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml


class TestSkillSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).parent.parent

    def _load(self, name: str) -> dict:
        with open(self.root / name, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def test_skill_yaml_has_required_keys(self) -> None:
        data = self._load("skill.yaml")
        self.assertEqual(data.get("schema_version"), "rosclaw.skill.v1")
        self.assertIn("metadata", data)
        self.assertIn("identity", data)
        self.assertIn("execution", data)

    def test_behavior_tree_is_valid_xml(self) -> None:
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.root / "behavior_tree.xml")
        tags = {elem.tag for elem in tree.getroot().iter()}
        self.assertIn("SandboxValidate", tags)
        self.assertIn("VerifyOutcome", tags)
        self.assertIn("Fallback", tags)


if __name__ == "__main__":
    unittest.main()

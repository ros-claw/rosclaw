"""Placeholder schema tests for {name}."""

from pathlib import Path

from rosclaw.skill.models import SkillPackage
from rosclaw.skill.validators import validate_package


def test_skill_package_valid():
    pkg = SkillPackage.load(Path(__file__).parent.parent)
    report = validate_package(pkg)
    assert report.ok, report.errors

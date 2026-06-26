"""Tests for the keyword-based BodyQueryEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.query import BodyQueryEngine
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def query_engine(tmp_path: Path, monkeypatch) -> BodyQueryEngine:
    """Create a linked body and return a BodyQueryEngine for it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService()
    service.create_or_init(robot="unitree-g1", name="g1-query", mode="single")
    resolver = BodyResolver()
    effective = resolver.get_effective_body(recompile_if_stale=False)
    body_yaml = resolver.get_current_body_yaml()
    calibration = resolver.get_calibration()
    maintenance = resolver.get_maintenance_events()
    return BodyQueryEngine(effective, body_yaml, calibration, maintenance)


def test_identity(query_engine: BodyQueryEngine):
    result = query_engine.answer("What robot body is this?")
    assert "g1-query" in result.answer or "unitree-g1" in result.answer
    assert result.evidence.get("robot_model") == "unitree-g1"


def test_bypass_sandbox_refused(query_engine: BodyQueryEngine):
    result = query_engine.answer("Can I bypass sandbox validation?")
    assert result.answer.startswith("No")
    assert result.evidence["policy"] == "physical_execution_requires_sandbox"
    assert "Refused" in result.actionable_policy[0]


def test_walk_enabled(query_engine: BodyQueryEngine):
    result = query_engine.answer("Can it walk?")
    assert "walk" in result.answer.lower()
    assert "capabilities" in result.evidence


def test_vision(query_engine: BodyQueryEngine):
    result = query_engine.answer("Can it see?")
    assert "visual" in result.answer.lower() or "camera" in result.answer.lower()


def test_calibration(query_engine: BodyQueryEngine):
    result = query_engine.answer("What is the calibration status?")
    assert "calibration" in result.answer.lower()
    assert "calibration_status" in result.evidence


def test_no_faults(query_engine: BodyQueryEngine):
    result = query_engine.answer("Are there any faults?")
    assert "no open faults" in result.answer.lower()


def test_capabilities_list(query_engine: BodyQueryEngine):
    result = query_engine.answer("What capabilities does it have?")
    assert "enabled capabilities" in result.answer.lower()
    assert "capabilities" in result.evidence


def test_safety_status(query_engine: BodyQueryEngine):
    result = query_engine.answer("Is it safe?")
    assert "safety" in result.answer.lower()
    assert "safety_status" in result.evidence


def test_fallback(query_engine: BodyQueryEngine):
    result = query_engine.answer("What is the meaning of life?")
    assert "don't have a specific answer" in result.answer.lower()
    assert "capabilities" in result.evidence

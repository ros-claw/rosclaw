"""Tests for generated summary files under refs/generated/."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def linked_body(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService(workspace=tmp_path)
    service.create_or_init(robot="unitree-g1", name="test-001", mode="single")
    return tmp_path


def test_generated_summaries_exist_under_refs_generated(linked_body: Path):
    generated_dir = linked_body / "body" / "refs" / "generated"
    assert (generated_dir / "body.summary.json").exists()
    assert (generated_dir / "embodiment.agent.json").exists()
    assert (generated_dir / "safety.summary.json").exists()


def test_body_summary_matches_effective_hash(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    effective = resolver.get_effective_body()
    summary_path = linked_body / "body" / "refs" / "generated" / "body.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["effective_body_hash"] == effective.effective_body_hash
    assert summary["robot_instance_id"] == effective.body_instance_id


def test_agent_summary_has_safety_policy(linked_body: Path):
    agent_path = linked_body / "body" / "refs" / "generated" / "embodiment.agent.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    assert agent["agent_policy"]["physical_execution_requires_sandbox"] is True
    assert agent["agent_policy"]["direct_real_robot_execution_allowed"] is False


def test_safety_summary_has_status_and_limits(linked_body: Path):
    safety_path = linked_body / "body" / "refs" / "generated" / "safety.summary.json"
    safety = json.loads(safety_path.read_text(encoding="utf-8"))
    assert "safety_status" in safety
    assert "global_limits" in safety
    assert "open_faults" in safety


def test_generated_summaries_regenerated_on_render(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    safety_path = linked_body / "body" / "refs" / "generated" / "safety.summary.json"
    safety_path.write_text("{}", encoding="utf-8")
    resolver.refresh_all_artifacts(reason="test summary regeneration")
    regenerated = json.loads(safety_path.read_text(encoding="utf-8"))
    assert regenerated.get("schema", "").startswith("rosclaw.generated.")

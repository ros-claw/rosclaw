"""Tests for Agent View renderer and required EMBODIMENT.md sections."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rosclaw.body.agent_view import BodyAgentView, BodyAgentViewRenderer
from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService


@pytest.fixture
def linked_body(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService(workspace=tmp_path)
    service.create_or_init(robot="unitree-g1", name="test-001", mode="single")
    return tmp_path


def test_embodiment_has_all_required_sections(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    md_text = resolver.embodiment_md_path.read_text(encoding="utf-8")
    for section in BodyAgentViewRenderer.required_sections():
        assert f"## {section}" in md_text, f"Missing section: {section}"


def test_embodiment_states_it_is_not_permission_bypass(linked_body: Path):
    md_text = (linked_body / "body" / "EMBODIMENT.md").read_text(encoding="utf-8")
    assert "not a permission bypass" in md_text
    assert "sandbox" in md_text.lower()


def test_render_preserves_human_notes(linked_body: Path):
    embodiment_path = linked_body / "body" / "EMBODIMENT.md"
    original = embodiment_path.read_text(encoding="utf-8")
    notes = "\n\nThis is a human note that must survive re-rendering.\n\n"
    embodiment_path.write_text(
        original + "\n\n<!-- HUMAN-NOTES-START -->\n" + notes + "<!-- HUMAN-NOTES-END -->\n",
        encoding="utf-8",
    )

    renderer = BodyAgentViewRenderer(workspace=linked_body)
    renderer.render_all(reason="test human notes preservation")

    rendered = embodiment_path.read_text(encoding="utf-8")
    assert "This is a human note that must survive re-rendering." in rendered


def test_render_records_render_event(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    before = len(MaintenanceLog(resolver.maintenance_log_path).read_events(type_filter="render"))
    renderer = BodyAgentViewRenderer(workspace=linked_body)
    renderer.render_all(reason="test render event")
    after = len(MaintenanceLog(resolver.maintenance_log_path).read_events(type_filter="render"))
    assert after == before + 1


def test_show_agent_summary_text(linked_body: Path):
    view = BodyAgentView(workspace=linked_body)
    summary = view.get_agent_summary()
    assert "sandbox validation" in summary
    assert "direct_real_robot_execution_allowed" not in summary  # short text summary


def test_state_json_is_valid_and_safe(linked_body: Path):
    view = BodyAgentView(workspace=linked_body)
    state_json = view.get_state_json()
    state = json.loads(state_json)
    assert state["body_instance_id"] == "test-001"
    assert state["agent_policy"]["physical_execution_requires_sandbox"] is True
    assert state["agent_policy"]["direct_real_robot_execution_allowed"] is False


def test_body_query_engine_refuses_bypass(linked_body: Path):
    from rosclaw.body.query import BodyQueryEngine

    resolver = BodyResolver(workspace=linked_body)
    engine = BodyQueryEngine(
        effective=resolver.get_effective_body(),
        body_yaml=resolver.get_current_body_yaml(),
        calibration=resolver.get_calibration(),
        maintenance=resolver.get_maintenance_events(),
    )
    result = engine.answer("Can this robot bypass sandbox validation?")
    assert "sandbox" in result.answer.lower()
    assert "not allowed" in result.answer.lower() or "mandatory" in result.answer.lower()

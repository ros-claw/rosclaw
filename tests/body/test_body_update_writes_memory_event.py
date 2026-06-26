"""Test that body updates can trigger memory event writes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.memory.body_events import BodyMemoryEventWriter


@pytest.fixture
def linked_workspace(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    BodyInstanceService(workspace=tmp_path).create_or_init(robot="unitree-g1", name="g1-update", mode="single")
    return tmp_path


def test_body_fault_update_writes_memory_event(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body_before = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.return_value = "record-fault"
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    # Simulate a body fault being added.
    from rosclaw.body.notes import MaintenanceLog

    MaintenanceLog(resolver.maintenance_log_path).write_fault_event(
        body_instance_id=body_before.body_instance_id,
        component="right_arm_actuator_group",
        severity="high",
        summary="right arm overheating",
        fault_id="fault-001",
    )

    body_after = resolver.get_effective_body(recompile_if_stale=True)
    assert body_after.effective_body_hash != body_before.effective_body_hash

    diff = resolver.differ.diff_effective_bodies(body_before, body_after)

    result = writer.write_body_change(
        body_instance_id=body_after.body_instance_id,
        old_hash=body_before.effective_body_hash,
        new_hash=body_after.effective_body_hash,
        diff=diff,
        affected_skills=["dual_arm_lift"],
    )

    assert result["recorded"] is True
    assert result["record_id"] == "record-fault"
    event = result["event"]
    assert event["old_effective_body_hash"] == body_before.effective_body_hash
    assert event["new_effective_body_hash"] == body_after.effective_body_hash
    assert "dual_arm_lift" in event["affected_skills"]


def test_body_update_does_not_block_when_memory_fails(linked_workspace: Path):
    resolver = BodyResolver(workspace=linked_workspace)
    body_before = resolver.resolve("rosclaw://body/current/effective")

    mock_client = MagicMock()
    mock_client.insert.side_effect = RuntimeError("memory unavailable")
    writer = BodyMemoryEventWriter(memory_client=mock_client)

    from rosclaw.body.notes import MaintenanceLog

    MaintenanceLog(resolver.maintenance_log_path).write_render_event(
        body_instance_id=body_before.body_instance_id,
        reason="non-blocking test",
    )
    body_after = resolver.get_effective_body(recompile_if_stale=True)
    diff = resolver.differ.diff_effective_bodies(body_before, body_after)

    result = writer.write_body_change(
        body_instance_id=body_after.body_instance_id,
        old_hash=body_before.effective_body_hash,
        new_hash=body_after.effective_body_hash,
        diff=diff,
        affected_skills=[],
    )

    assert result["recorded"] is False
    assert "memory unavailable" in result["reason"]

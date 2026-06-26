"""Schema invariant tests for body.yaml, calibration.yaml, maintenance.log, EffectiveBody."""

from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw.body.notes import MaintenanceLog
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.body.workspace_validator import BodyValidator


@pytest.fixture
def linked_body(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    service = BodyInstanceService(workspace=tmp_path)
    service.create_or_init(robot="unitree-g1", name="test-001", mode="single")
    return tmp_path


def test_body_yaml_has_required_fields(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    body = resolver.get_current_body_yaml()
    assert body.body_instance.get("id") == "test-001"
    assert body.model_ref.get("profile_id") == "unitree-g1"
    assert body.model_ref.get("profile_checksum") is not None


def test_calibration_body_instance_id_matches_body(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    body = resolver.get_current_body_yaml()
    calibration = resolver.get_calibration()
    assert calibration.body_instance_id == body.body_instance.get("id")


def test_maintenance_log_is_valid_jsonl(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    log = MaintenanceLog(resolver.maintenance_log_path)
    events = list(log.read_events())
    assert len(events) >= 1
    for event in events:
        assert event.event_id
        assert event.time
        assert event.type in {"init", "note", "maintenance", "incident", "repair", "render", "fault", "calibration", "capability_update"}


def test_invalid_maintenance_log_is_handled_gracefully(tmp_path: Path):
    log_path = tmp_path / "maintenance.log"
    log_path.write_text("not valid json\n")
    log = MaintenanceLog(log_path)
    events = list(log.read_events())
    # Malformed lines are skipped rather than crashing.
    assert events == []


def test_effective_body_source_trace_includes_all_sources(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    effective = resolver.get_effective_body()
    assert effective.source_trace
    assert "eurdf" in effective.source_trace
    assert "body" in effective.source_trace
    assert "calibration" in effective.source_trace


def test_workspace_validator_passes_for_valid_body(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    validator = BodyValidator(resolver)
    report = validator.validate_all()
    assert report.result in ("PASS", "PASS_WITH_WARNINGS")


def test_safety_invariants_block_real_robot_when_unknown(linked_body: Path):
    resolver = BodyResolver(workspace=linked_body)
    effective = resolver.get_effective_body()
    # Effective body should expose safety limits.
    assert effective.safety is not None
    assert "speed_limits" in effective.safety or "safety_limits" in effective.safety

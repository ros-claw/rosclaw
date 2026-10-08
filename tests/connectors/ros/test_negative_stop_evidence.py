"""Independent closed actuator/pose source contracts; no physical dispatch."""

import importlib.util
import json
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog
from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
BINDING = {
    "run_id": "run",
    "body_snapshot_hash": "body",
    "attachment_hash": "brush",
    "producer_id": "actor",
}


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "negative_stop_test", ROOT / "negative_stop_evidence.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path, fault=None):
    base = datetime(2026, 10, 8, tzinfo=UTC)
    samples = [
        {
            "source": "independent_gazebo_ground_truth_subscription",
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "time_sec": 1 + i / 10,
            "captured_at": (base + timedelta(seconds=1 + i / 10)).isoformat(),
        }
        for i in range(31)
    ]
    path = tmp_path / "brush.jsonl"
    audit = CoverageAuditLog(
        path,
        context={**BINDING, "source": "simulator_owned_actuator", "evidence_domain": "SIMULATION"},
        capacity=256,
    )
    for i in range(101):
        stamp = i / 20
        enabled = fault == "on" and 40 <= i <= 80
        kind = "TRANSITION" if fault == "on" and i in (40, 81) else "WATERMARK"
        binding = (
            {**BINDING, "producer_id": "foreign"} if fault == "binding" and i == 50 else BINDING
        )
        event = BrushStateEvent(
            **binding,
            sequence=i + (1 if fault == "sequence" and i == 50 else 0),
            sim_time_sec=stamp,
            kind=kind,
            enabled=enabled,
            captured_at=(base + timedelta(seconds=stamp)).isoformat(),
            complete=not (fault == "incomplete" and i == 50),
        )
        payload = {
            "event": asdict(event),
            "artifact_hash": event.artifact_hash(),
            "lease_remaining_sec": 1.0 if fault == "lease" and i == 50 else -1.0,
            "lease_updates": 3,
        }
        if fault == "event_hash" and i == 50:
            payload["artifact_hash"] = "wrong"
        if fault == "gap" and 40 <= i <= 50:
            continue
        if fault == "unbracketed" and i > 79:
            continue
        audit.emit("brush_state_event", payload, sim_time=stamp)
    summary = audit.close()
    assert summary["complete"]
    if fault == "open":
        summary["writer_stopped"] = False
    if fault == "drop":
        summary["dropped_events"] = 1
    if fault == "closure":
        summary["events_written"] += 1
    Path(str(path) + ".summary.json").write_text(json.dumps(summary))
    if fault == "partial":
        with path.open("ab") as stream:
            stream.write(b'{"partial":')
    if fault == "chain":
        path.write_bytes(
            path.read_bytes().replace(
                b'"source": "simulator_owned_actuator"', b'"source": "altered"', 1
            )
        )
    if fault == "moving":
        samples[15]["x"] = 0.02
    if fault == "stale_pose":
        samples[-1]["time_sec"] = samples[-2]["time_sec"]
    return path, {"samples": samples}


def test_independent_stop_and_closed_actor_are_corresponding_not_task_acceptance(reader, tmp_path):
    path, stop = fixture(tmp_path)
    result = reader.closed_brush_stop(path, BINDING, stop)
    assert result["brush_off"] and result["lease_released"]
    assert result["physical_acceptance"] == "NOT_VERIFIED"
    assert result["stop_window_actuator_events"] >= 20
    assert result["independent_stop"]["sample_count"] == 31


@pytest.mark.parametrize(
    "fault",
    [
        "on",
        "lease",
        "binding",
        "sequence",
        "incomplete",
        "event_hash",
        "gap",
        "unbracketed",
        "open",
        "drop",
        "closure",
        "partial",
        "chain",
        "moving",
        "stale_pose",
    ],
)
def test_negative_terminal_cannot_substitute_for_missing_or_unsafe_sources(reader, tmp_path, fault):
    path, stop = fixture(tmp_path, fault)
    with pytest.raises((ValueError, TypeError)):
        reader.closed_brush_stop(path, BINDING, stop)

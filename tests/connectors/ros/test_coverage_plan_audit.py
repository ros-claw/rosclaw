"""Passive diagnostics must preserve trace integrity and verifier authority."""

import json
import math

import pytest

from rosclaw.connectors.ros.diagnosis.coverage_audit import (
    CoverageAuditLog,
    plan_projection,
    read_audit,
    trajectory_metrics,
)
from rosclaw.connectors.ros.verification.coverage import CoverageVerifier


def test_append_chain_freezes_input_and_refuses_overwrite(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = CoverageAuditLog(path)
    payload = {"poses": [{"x": 1}]}
    writer.emit("path", payload)
    payload["poses"][0]["x"] = 2
    writer.emit("path", payload)
    assert writer.close()["complete"]
    rows = read_audit(path)
    assert [r["payload"]["poses"][0]["x"] for r in rows] == [1, 2]
    with pytest.raises(FileExistsError):
        CoverageAuditLog(path)
    rows[0]["payload"]["poses"][0]["x"] = 3
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="integrity"):
        read_audit(path)


def test_loss_invalid_payload_and_unfinished_tail_are_not_complete(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        CoverageAuditLog(tmp_path / "unbounded.jsonl", capacity=0)
    writer = CoverageAuditLog(tmp_path / "events.jsonl", max_event_bytes=400)
    writer.emit("path", {"large": "x" * 500})
    writer.emit("feedback", {"x": float("nan")})
    result = writer.close()
    assert not result["complete"]
    assert result["dropped_events"] == 2
    path = tmp_path / "incomplete.jsonl"
    path.write_text('{"sequence": 1}')
    with pytest.raises(ValueError, match="unfinished"):
        read_audit(path)


def test_motion_classes_account_for_all_time_and_wrap_yaw():
    trace = [
        {"x": 0, "y": 0, "yaw": math.pi - 0.01, "time_sec": 0},
        {"x": 1, "y": 0, "yaw": -math.pi + 0.01, "time_sec": 1},
        {"x": 1, "y": 0, "yaw": -math.pi + 1, "time_sec": 2},
        {"x": 1, "y": 0, "yaw": -math.pi + 1, "time_sec": 3},
    ]
    result = trajectory_metrics(trace)
    assert result["observed_distance_m"] == 1
    assert result["accumulated_rotation_rad"] == pytest.approx(1.01)
    assert result["driving_sec"] == result["stationary_turn_sec"] == result["stop_wait_sec"] == 1
    with pytest.raises(ValueError, match="increase"):
        trajectory_metrics([trace[0], trace[0]])


def test_prediction_is_separate_and_requires_orientation_and_frame():
    grid = {
        "width": 4,
        "height": 4,
        "resolution": 1,
        "accessible_cells": list(range(16)),
        "frame_id": "map",
        "cleaning_polygon": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
    }
    measured = CoverageVerifier(**grid)
    prediction = plan_projection(grid, [{"x": 0.5, "y": 0.5, "yaw": 0}], frame_id="map")
    assert prediction["predicted_coverage_ratio"] == 1 / 16
    assert measured.result()["coverage_ratio"] == 0
    with pytest.raises(KeyError):
        plan_projection(grid, [{"x": 0.5, "y": 0.5}], frame_id="map")
    with pytest.raises(ValueError, match="frame mismatch"):
        plan_projection(grid, [{"x": 0.5, "y": 0.5, "yaw": 0}], frame_id="odom")


def test_continuous_plan_projection_does_not_omit_between_waypoints():
    grid = {
        "width": 4,
        "height": 1,
        "resolution": 1,
        "accessible_cells": list(range(4)),
        "cleaning_polygon": [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
    }
    result = plan_projection(
        grid, [{"x": 0.5, "y": 0.5, "yaw": 0}, {"x": 3.5, "y": 0.5, "yaw": 0}], frame_id="map"
    )
    assert result["predicted_cells"] == [0, 1, 2, 3]


def test_audit_flush_failure_does_not_change_result_or_leak_execution_lock(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.mission.executor import RosCoverageSimulationExecutor
    from rosclaw.kernel import ActionState, ExecutionMode

    executor = RosCoverageSimulationExecutor(
        owner="daemon_test",
        client=None,
        control=None,
        witness=SimpleNamespace(latest=(0, {"time_sec": 1}), samples=[]),
        output=tmp_path,
        body_id="fixture",
        body_snapshot_hash="body",
        grid={},
    )
    expected = executor._result(ActionState.COMPLETED)
    monkeypatch.setattr(executor, "_localize", lambda _action: expected)
    original_close = CoverageAuditLog.close

    def failing_summary(self, **kwargs):
        original_close(self, **kwargs)
        raise OSError("injected diagnostic flush failure")

    monkeypatch.setattr(CoverageAuditLog, "close", failing_summary)
    result = executor(
        SimpleNamespace(
            action_id="test-localize",
            execution_mode=ExecutionMode.SIMULATION,
            body_id="fixture",
            body_snapshot_hash="body",
            capability_id="localization.set_initial_pose",
        )
    )
    assert result is expected
    assert executor.execution_lock.acquire(blocking=False)
    executor.execution_lock.release()
    assert executor.audit is None

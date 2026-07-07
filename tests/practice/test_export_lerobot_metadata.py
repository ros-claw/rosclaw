"""Tests for LeRobot exporter metadata and parquet content."""

from __future__ import annotations

import json
import tempfile

import pytest

from rosclaw.practice.config import PracticeConfig, SourceConfig
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.exporters.lerobot_exporter import LeRobotExporter
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import (
    FailureEventPayload,
    PhysicalFeedbackPayload,
    PracticeEventEnvelope,
)
from rosclaw.runtime.bus import RuntimeBus

pytest.importorskip("pyarrow")


def _run_session_with_extras(tmp: str) -> str:
    bus = RuntimeBus()
    recorder = PracticeRecorder(bus, data_root=tmp, publish_to_event_bus=False)
    recorder.initialize()
    recorder.start()
    cfg = PracticeConfig(
        robot_id="test_bot",
        task_name="ok_contact",
        data_root=tmp,
        sources=SourceConfig(agent=True, runtime=True),
        mock=True,
        publish_to_event_bus=False,
    )
    coord = PracticeCoordinator(cfg, runtime_bus=bus, recorder=recorder)
    coord.initialize()
    coord.start()
    practice_id = coord._session.practice_id

    coord.emit_event(
        PracticeEventEnvelope(
            practice_id=practice_id,
            robot_id="test_bot",
            body_id="body_rh56_left",
            source="runtime",
            event_type="physical_feedback_event",
            payload=PhysicalFeedbackPayload(
                frame_id="f0",
                body_id="body_rh56_left",
                timestamp=0.0,
                target={"thumb": 1.0, "index": 2.0},
                actual={"thumb": 1.1, "index": 2.1},
                force_net={"thumb": 100.0, "index": 110.0},
                primary_event="desired_contact",
            ).model_dump(),
        )
    )
    coord.emit_event(
        PracticeEventEnvelope(
            practice_id=practice_id,
            robot_id="test_bot",
            source="runtime",
            event_type="failure_event",
            payload=FailureEventPayload(
                failure_id="fail_1",
                failure_type="over_contact",
                severity="high",
                source="sandbox",
                description="too much force",
            ).model_dump(),
        )
    )

    coord.stop()
    recorder.stop()
    return practice_id


def test_lerobot_parquet_content():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session_with_extras(tmp)
        exporter = LeRobotExporter(tmp)
        out = exporter.export(practice_id)

        import pyarrow.parquet as pq

        state_table = pq.read_table(out / "data" / "observation.state.parquet")
        action_table = pq.read_table(out / "data" / "action.parquet")

        assert state_table.num_rows == 1
        assert action_table.num_rows == 1
        assert "observation.state" in state_table.column_names
        assert "action" in action_table.column_names
        assert state_table.column("episode_index")[0].as_py() == 0


def test_lerobot_rosclaw_extra_contains_non_feedback_events():
    with tempfile.TemporaryDirectory() as tmp:
        practice_id = _run_session_with_extras(tmp)
        exporter = LeRobotExporter(tmp)
        out = exporter.export(practice_id)

        extras_path = out / "rosclaw_extra.jsonl"
        with open(extras_path, encoding="utf-8") as f:
            extras = [json.loads(line) for line in f if line.strip()]

        failure_extras = [e for e in extras if e.get("event_type") == "failure_event"]
        assert len(failure_extras) == 1
        assert failure_extras[0]["payload"]["failure_type"] == "over_contact"

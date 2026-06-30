"""Tests for the Runtime Kernel v2 PracticeRecorder consumer."""

from __future__ import annotations

import json

from rosclaw.core.event_bus import EventBus
from rosclaw.practice.coordinator import PracticeCoordinator
from rosclaw.practice.recorder import PracticeRecorder
from rosclaw.practice.schemas import PracticeEventEnvelope
from rosclaw.runtime import RuntimeBus, RuntimeEvent


def test_recorder_starts_session_on_practice_start(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    recorder = PracticeRecorder(bus, data_root=str(tmp_path))
    recorder.initialize()
    recorder.start()

    bus.publish(
        RuntimeEvent(
            type="practice.start",
            source="practice_coordinator",
            robot="realsense-d405",
            body_id="d405_lab_01",
            payload={
                "practice_id": "prac_test_001",
                "robot_id": "realsense-d405",
                "task_id": "inspect_tabletop",
                "task_name": "Inspect tabletop",
                "skill_id": "realsense_capture_rgbd",
                "sources": {"camera": True, "provider": True, "runtime": True},
            },
            metadata={"trace_id": "prac_test_001"},
        )
    )

    assert recorder.session is not None
    assert recorder.session.practice_id == "prac_test_001"
    assert recorder.session.robot_id == "realsense-d405"

    session_dir = tmp_path / "sessions" / "prac_test_001"
    assert session_dir.exists()
    assert (session_dir / "manifest.yaml").exists()

    recorder.stop()


def test_recorder_writes_events_jsonl_and_catalog(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    recorder = PracticeRecorder(bus, data_root=str(tmp_path))
    recorder.initialize()
    recorder.start()

    bus.publish(
        RuntimeEvent(
            type="practice.start",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={"practice_id": "prac_test_002", "robot_id": "realsense-d405"},
            metadata={"trace_id": "prac_test_002"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="camera.rgbd_frame",
            source="realsense_camera",
            robot="realsense-d405",
            payload={"camera_id": "d405", "width": 640, "height": 480},
            metadata={"trace_id": "prac_test_002"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="provider.result",
            source="cosmos_reasoner",
            robot="realsense-d405",
            payload={"provider_id": "cosmos", "status": "success"},
            metadata={"trace_id": "prac_test_002"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="practice.stop",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={
                "practice_id": "prac_test_002",
                "outcome": "SUCCESS",
                "event_count": 2,
            },
            metadata={"trace_id": "prac_test_002"},
        )
    )

    recorder.stop()

    session_dir = tmp_path / "sessions" / "prac_test_002"
    events_path = session_dir / "raw" / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    event_types = [json.loads(line)["event_type"] for line in lines]
    assert "camera.rgbd_frame" in event_types
    assert "provider.result" in event_types

    timeline_path = session_dir / "timeline.jsonl"
    assert timeline_path.exists()
    assert len(timeline_path.read_text(encoding="utf-8").strip().splitlines()) == 2

    episode_path = session_dir / "episode.json"
    assert episode_path.exists()
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    assert episode["outcome"] == "SUCCESS"
    assert episode["practice_id"] == "prac_test_002"


def test_recorder_publishes_legacy_practice_event(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    captured = []
    event_bus.subscribe("practice.event", lambda e: captured.append(e))

    recorder = PracticeRecorder(bus, data_root=str(tmp_path), event_bus=event_bus)
    recorder.initialize()
    recorder.start()

    bus.publish(
        RuntimeEvent(
            type="practice.start",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={"practice_id": "prac_test_003", "robot_id": "realsense-d405"},
            metadata={"trace_id": "prac_test_003"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="camera.rgbd_frame",
            source="realsense_camera",
            robot="realsense-d405",
            payload={"camera_id": "d405"},
            metadata={"trace_id": "prac_test_003"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="practice.stop",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={"practice_id": "prac_test_003", "outcome": "SUCCESS"},
            metadata={"trace_id": "prac_test_003"},
        )
    )

    recorder.stop()

    # The camera event should have been republished as a legacy practice.event.
    assert any(e.topic == "practice.event" for e in captured)


def test_recorder_reuses_embedded_envelope(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    recorder = PracticeRecorder(bus, data_root=str(tmp_path))
    recorder.initialize()
    recorder.start()

    from rosclaw.practice.schemas import PracticeEventEnvelope

    envelope = PracticeEventEnvelope(
        practice_id="prac_test_004",
        robot_id="realsense-d405",
        source="camera",
        event_type="camera.rgbd_frame",
        payload={"camera_id": "d405"},
        tags=["rgbd"],
    )

    bus.publish(
        RuntimeEvent(
            type="practice.start",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={"practice_id": "prac_test_004", "robot_id": "realsense-d405"},
            metadata={"trace_id": "prac_test_004"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="camera.rgbd_frame",
            source="realsense_camera",
            robot="realsense-d405",
            payload=envelope.model_dump(mode="json"),
            metadata={"trace_id": "prac_test_004"},
        )
    )
    bus.publish(
        RuntimeEvent(
            type="practice.stop",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={"practice_id": "prac_test_004", "outcome": "SUCCESS"},
            metadata={"trace_id": "prac_test_004"},
        )
    )

    recorder.stop()

    session_dir = tmp_path / "sessions" / "prac_test_004"
    lines = (session_dir / "raw" / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["practice_id"] == "prac_test_004"
    assert stored["tags"] == ["rgbd"]


def test_coordinator_delegates_recording_to_recorder(tmp_path):
    """PracticeCoordinator with a recorder uses RuntimeBus instead of file writes."""
    from rosclaw.practice.config import PracticeConfig

    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    recorder = PracticeRecorder(bus, data_root=str(tmp_path))
    config = PracticeConfig(
        robot_id="realsense-d405",
        robot_type="realsense-d405",
        task_id="inspect",
        task_name="Inspect",
        skill_id="realsense_capture_rgbd",
        data_root=str(tmp_path),
        publish_to_event_bus=False,
    )
    coordinator = PracticeCoordinator(config=config, runtime_bus=bus, recorder=recorder)

    recorder.initialize()
    recorder.start()
    coordinator.initialize()
    coordinator.start()

    coordinator.emit_event(
        PracticeEventEnvelope(
            practice_id=coordinator.session.practice_id,
            robot_id="realsense-d405",
            source="camera",
            event_type="camera.rgbd_frame",
            payload={"camera_id": "d405", "width": 640},
        )
    )

    coordinator.stop()
    recorder.stop()

    practice_id = coordinator.session.practice_id
    session_dir = tmp_path / "sessions" / practice_id
    events_path = session_dir / "raw" / "events.jsonl"
    assert events_path.exists()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    event_types = [json.loads(line)["event_type"] for line in lines]
    assert "camera.rgbd_frame" in event_types

    assert (session_dir / "episode.json").exists()
    assert coordinator.summary is not None
    assert coordinator.summary.outcome == "SUCCESS"

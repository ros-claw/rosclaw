"""Tests for the Runtime Kernel v2 MemoryConsumer."""

from __future__ import annotations

from rosclaw.core.event_bus import EventBus
from rosclaw.memory.consumer import MemoryConsumer
from rosclaw.memory.seekdb_client import SeekDBMemoryClient
from rosclaw.runtime import RuntimeBus, RuntimeEvent


def test_memory_consumer_stores_camera_artifact(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    client = SeekDBMemoryClient()
    client.connect()
    consumer = MemoryConsumer(bus, robot_id="realsense-d405", seekdb_client=client)
    consumer.initialize()
    consumer.start()

    bus.publish(
        RuntimeEvent(
            type="camera.rgbd_frame",
            source="realsense_camera",
            robot="realsense-d405",
            payload={
                "camera_id": "d405",
                "rgb_ref": "artifact://rgb_frame/2026-06-30/ep1/frame.jpg",
                "depth_ref": "artifact://depth_frame/2026-06-30/ep1/frame.png",
            },
            metadata={"trace_id": "ep1"},
        )
    )

    consumer.stop()

    artifacts = client.query("artifacts", filters={"episode_id": "ep1"})
    assert len(artifacts) == 2
    uris = {a["uri"] for a in artifacts}
    assert "artifact://rgb_frame/2026-06-30/ep1/frame.jpg" in uris


def test_memory_consumer_stores_provider_result(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    client = SeekDBMemoryClient()
    client.connect()
    consumer = MemoryConsumer(bus, robot_id="realsense-d405", seekdb_client=client)
    consumer.initialize()
    consumer.start()

    bus.publish(
        RuntimeEvent(
            type="provider.result",
            source="cosmos_reasoner",
            robot="realsense-d405",
            payload={"provider_id": "cosmos", "status": "success", "answer": "cup"},
            metadata={"trace_id": "ep2"},
        )
    )

    consumer.stop()

    events = client.query("praxis_events", filters={"episode_id": "ep2"})
    assert len(events) == 1
    assert events[0]["event_type"] == "provider.result"


def test_memory_consumer_stores_practice_stop_experience(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    client = SeekDBMemoryClient()
    client.connect()
    consumer = MemoryConsumer(bus, robot_id="realsense-d405", seekdb_client=client)
    consumer.initialize()
    consumer.start()

    bus.publish(
        RuntimeEvent(
            type="practice.stop",
            source="practice_coordinator",
            robot="realsense-d405",
            payload={
                "practice_id": "ep3",
                "robot_id": "realsense-d405",
                "outcome": "SUCCESS",
                "duration_ms": 1500.0,
                "task": {"skill_id": "realsense_capture_rgbd"},
            },
            metadata={"trace_id": "ep3"},
        )
    )

    consumer.stop()

    experiences = client.query("experience_graph", filters={"id": "ep3"})
    assert len(experiences) == 1
    assert experiences[0]["event_type"] == "practice_episode"
    assert experiences[0]["outcome"] == "success"


def test_memory_consumer_indexes_artifact_uri(tmp_path):
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    client = SeekDBMemoryClient()
    client.connect()
    consumer = MemoryConsumer(bus, robot_id="realsense-d405", seekdb_client=client)
    consumer.initialize()
    consumer.start()

    bus.publish(
        RuntimeEvent(
            type="camera.rgbd_frame",
            source="realsense_camera",
            robot="realsense-d405",
            payload={
                "camera_id": "d435i",
                "rgb_ref": "file:///data/rosclaw/practice/sessions/ep4/frames/rgb/frame_0001.jpg",
            },
            metadata={"trace_id": "ep4"},
        )
    )

    consumer.stop()

    # Query by URI substring is not supported by exact-filter SeekDBMemoryClient,
    # but the schema index on uri exists and exact URI lookup works.
    artifacts = client.query(
        "artifacts",
        filters={"uri": "file:///data/rosclaw/practice/sessions/ep4/frames/rgb/frame_0001.jpg"},
    )
    assert len(artifacts) == 1
    assert artifacts[0]["episode_id"] == "ep4"

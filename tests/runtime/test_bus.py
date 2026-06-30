"""Tests for Runtime Kernel v2 foundational components."""

from __future__ import annotations

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.runtime import (
    RuntimeBus,
    RuntimeComponent,
    RuntimeConsumer,
    RuntimeEvent,
    RuntimeKernelService,
    RuntimeProducer,
    RuntimeReplay,
    SchemaValidationError,
)
from rosclaw.runtime.registry import RuntimeComponentRegistry


class DummyProducer(RuntimeProducer):
    def __init__(self, bus):
        super().__init__("dummy_producer", bus)
        self.events = []

    def _do_start(self):
        self.publish_event("camera.rgbd_frame", {"width": 640})

    def _do_stop(self):
        pass


class DummyConsumer(RuntimeConsumer):
    def __init__(self, bus):
        super().__init__("dummy_consumer", bus)
        self.events = []

    def on_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)


class TypedPayload(RuntimeEvent):
    """Used as a schema validator."""

    score: float


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

def test_runtime_event_topic():
    ev = RuntimeEvent(type="camera.rgbd_frame", source="cam")
    assert ev.topic == "rosclaw.camera.rgbd_frame"


def test_runtime_event_roundtrip():
    ev = RuntimeEvent(
        type="skill.complete",
        source="executor",
        robot="realsense-d405",
        body_id="d405_lab_01",
        payload={"skill_id": "realsense_capture_rgbd"},
        metadata={"trace_id": "abc"},
    )
    payload = ev.to_event_bus_payload()
    restored = RuntimeEvent.from_event_bus_payload(payload, topic=ev.topic)
    assert restored.type == ev.type
    assert restored.robot == ev.robot
    assert restored.body_id == ev.body_id
    assert restored.payload["skill_id"] == "realsense_capture_rgbd"


# ---------------------------------------------------------------------------
# RuntimeBus
# ---------------------------------------------------------------------------

def test_runtime_bus_publish_subscribe():
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    received = []
    bus.subscribe("camera.rgbd_frame", lambda ev: received.append(ev))
    bus.publish(RuntimeEvent(type="camera.rgbd_frame", source="cam", payload={"w": 1}))
    assert len(received) == 1
    assert received[0].source == "cam"


def test_runtime_bus_prefix_subscription():
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    received = []
    bus.subscribe_prefix("camera.*", lambda ev: received.append(ev.type))
    bus.publish(RuntimeEvent(type="camera.rgbd_frame", source="cam"))
    bus.publish(RuntimeEvent(type="camera.depth", source="cam"))
    bus.publish(RuntimeEvent(type="skill.complete", source="executor"))
    assert sorted(received) == ["camera.depth", "camera.rgbd_frame"]


def test_runtime_bus_schema_validation():
    from pydantic import BaseModel

    class ScorePayload(BaseModel):
        score: float

    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    bus.register_schema("provider.result", ScorePayload)

    bus.publish(RuntimeEvent(type="provider.result", payload={"score": 0.9}))

    with pytest.raises(SchemaValidationError):
        bus.publish(RuntimeEvent(type="provider.result", payload={"score": "bad"}))


def test_runtime_bus_history():
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    bus.publish(RuntimeEvent(type="camera.rgbd_frame", source="cam"))
    hist = bus.get_history("camera.rgbd_frame")
    assert len(hist) == 1


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

def test_registry_lifecycle():
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    registry = RuntimeComponentRegistry()
    consumer = DummyConsumer(bus)
    producer = DummyProducer(bus)
    # Consumer first so it is subscribed before the producer publishes.
    registry.register("consumer", consumer)
    registry.register("producer", producer)

    registry.initialize_all()
    registry.start_all()
    # Producer published one event; consumer should have received it.
    assert any(e.type == "camera.rgbd_frame" for e in consumer.events)
    registry.stop_all()


# ---------------------------------------------------------------------------
# Service lifecycle
# ---------------------------------------------------------------------------

def test_kernel_service_lifecycle(tmp_path):
    service = RuntimeKernelService(home=str(tmp_path))
    service.initialize()
    service.start()
    service.bus.publish(RuntimeEvent(type="runtime.start", source="test"))
    service.stop()


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def test_replay_from_history():
    event_bus = EventBus(normalize_topics=False)
    bus = RuntimeBus(event_bus=event_bus)
    replay = RuntimeReplay(bus)
    bus.publish(RuntimeEvent(type="skill.invoke", payload={"skill_id": "s1"}, metadata={"trace_id": "ep1"}))
    bus.publish(RuntimeEvent(type="skill.complete", payload={"skill_id": "s1"}, metadata={"trace_id": "ep1"}))
    episode = replay.replay_episode("ep1")
    assert len(episode) == 2
    skills = replay.replay_skill("s1", episode_id="ep1")
    assert len(skills) == 2

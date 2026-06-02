"""EventBus coverage tests — fills gaps not covered by test_event_bus_extended.py."""

import asyncio
import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority


# --- Event.derive ---


def test_event_derive_inherits_trace_id():
    evt = Event(topic="a", payload=1, trace_id="trace_123")
    derived = evt.derive(topic="b", payload=2)
    assert derived.trace_id == "trace_123"
    assert derived.topic == "b"
    assert derived.payload == 2


def test_event_derive_overrides_fields():
    evt = Event(topic="a", payload=1, source="src1", priority=EventPriority.LOW)
    derived = evt.derive(
        topic="b",
        payload=2,
        source="src2",
        priority=EventPriority.HIGH,
        metadata={"extra": True},
    )
    assert derived.topic == "b"
    assert derived.payload == 2
    assert derived.source == "src2"
    assert derived.priority == EventPriority.HIGH
    assert derived.metadata == {"extra": True}
    assert derived.trace_id == evt.trace_id


# --- _topic_matches wildcard patterns ---


def test_topic_matches_hash_wildcard():
    bus = EventBus()
    received = []
    bus.subscribe("#", lambda e: received.append(e.payload))
    bus.publish(Event(topic="anything.at.all", payload="matched"))
    assert received == ["matched"]


def test_topic_matches_question_wildcard():
    bus = EventBus()
    received = []
    bus.subscribe("robot.?oint", lambda e: received.append(e.payload))
    bus.publish(Event(topic="robot.joint", payload="yes"))
    bus.publish(Event(topic="robot.moint", payload="also"))
    bus.publish(Event(topic="robot.joints", payload="no"))
    assert received == ["yes", "also"]


def test_topic_matches_star_wildcard():
    bus = EventBus()
    received = []
    bus.subscribe("robot.*.state", lambda e: received.append(e.topic))
    bus.publish(Event(topic="robot.joint.state", payload=1))
    bus.publish(Event(topic="robot.gripper.state", payload=2))
    bus.publish(Event(topic="robot.joint.velocity", payload=3))
    assert received == ["robot.joint.state", "robot.gripper.state"]


# --- publish trace_id injection ---


def test_publish_injects_trace_id_from_request_id():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(
        topic="test",
        payload={"request_id": "req_abc"},
    ))
    assert received == ["req_abc"]


def test_publish_injects_trace_id_from_correlation_id():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(
        topic="test",
        payload={"correlation_id": "corr_xyz"},
    ))
    assert received == ["corr_xyz"]


def test_publish_injects_trace_id_from_episode_id():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(
        topic="test",
        payload={"episode_id": "ep_99"},
    ))
    assert received == ["ep_99"]


def test_publish_generates_fallback_trace_id():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(topic="test", payload={}))
    assert len(received) == 1
    assert received[0].startswith("trace_")


def test_publish_trace_id_non_dict_payload():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(topic="test", payload=[1, 2, 3]))
    assert len(received) == 1
    assert received[0].startswith("trace_")


def test_publish_preserves_existing_trace_id():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.trace_id))
    bus.publish(Event(topic="test", payload={"request_id": "new"}, trace_id="old"))
    assert received == ["old"]


# --- get_history topic filter ---


def test_get_history_filtered_by_topic():
    bus = EventBus()
    bus.publish(Event(topic="a", payload=1))
    bus.publish(Event(topic="b", payload=2))
    bus.publish(Event(topic="a", payload=3))
    history = bus.get_history(topic="a")
    assert len(history) == 2
    assert history[0].payload == 1
    assert history[1].payload == 3


def test_get_history_filtered_by_topic_and_limit():
    bus = EventBus()
    for i in range(5):
        bus.publish(Event(topic="a", payload=i))
    history = bus.get_history(topic="a", limit=2)
    assert len(history) == 2
    assert history[-1].payload == 4


# --- EventBus without topic normalization ---


def test_event_bus_no_normalization():
    bus = EventBus(normalize_topics=False)
    received = []
    bus.subscribe("agent.command", lambda e: received.append(e.topic))
    bus.publish(Event(topic="agent.command", payload=1))
    assert received == ["agent.command"]


# --- unsubscribe callback not found (silent) ---


def test_unsubscribe_callback_not_found_is_silent():
    bus = EventBus()
    def handler(e):  # noqa: E306
        pass
    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)
    bus.unsubscribe("test", handler)  # second time: not found, should be silent
    assert bus.subscriber_count("test") == 0


# --- sync subscriber exception logging ---


def test_sync_subscriber_exception_logged(caplog):
    import logging
    bus = EventBus()

    def bad_handler(e):
        raise ValueError("boom")

    bus.subscribe("test", bad_handler)
    with caplog.at_level(logging.WARNING, logger="rosclaw.core.event_bus"):
        bus.publish(Event(topic="test", payload="x"))
    assert "Error in sync subscriber" in caplog.text


# --- async subscriber exception logging ---


@pytest.mark.asyncio
async def test_async_subscriber_exception_logged(caplog):
    bus = EventBus()

    async def bad_handler(e):
        raise RuntimeError("async boom")

    bus.subscribe_async("test", bad_handler)
    bus.publish(Event(topic="test", payload="x"))
    await asyncio.sleep(0.1)
    assert "Error in async subscriber" in caplog.text


# --- EventPriority ordering ---


def test_event_priority_values():
    assert EventPriority.CRITICAL.value == 0
    assert EventPriority.HIGH.value == 1
    assert EventPriority.NORMAL.value == 2
    assert EventPriority.LOW.value == 3
    assert EventPriority.BACKGROUND.value == 4

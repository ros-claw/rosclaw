"""Extended EventBus tests — covers all missing branches from EVENT_BUS_TEST_PLAN.md.

Target: 59% -> 95%+ coverage on event_bus.py
"""

import asyncio

import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority


# --- Input Validation (Lines 85, 87, 89, 98-106) ---


def test_subscribe_rejects_non_callable():
    bus = EventBus()
    with pytest.raises(TypeError, match="Handler must be callable"):
        bus.subscribe("topic", "not_a_function")


def test_subscribe_rejects_non_string_topic():
    bus = EventBus()
    with pytest.raises(TypeError, match="Topic must be str"):
        bus.subscribe(123, lambda e: None)


def test_subscribe_rejects_empty_topic():
    bus = EventBus()
    with pytest.raises(ValueError, match="Topic cannot be empty"):
        bus.subscribe("", lambda e: None)


def test_subscribe_async_rejects_non_callable():
    bus = EventBus()
    with pytest.raises(TypeError, match="Handler must be callable"):
        bus.subscribe_async("topic", "not_a_function")


def test_subscribe_async_rejects_non_string_topic():
    bus = EventBus()
    with pytest.raises(TypeError, match="Topic must be str"):
        bus.subscribe_async(123, lambda e: None)


def test_subscribe_async_rejects_empty_topic():
    bus = EventBus()
    with pytest.raises(ValueError, match="Topic cannot be empty"):
        bus.subscribe_async("", lambda e: None)


def test_subscribe_async_creates_new_topic():
    bus = EventBus()

    async def handler(e):
        pass

    bus.subscribe_async("new.topic", handler)
    assert "new.topic" in bus.topics


# --- Unsubscribe Async (Line 113) ---


def test_unsubscribe_async_callback():
    bus = EventBus()

    async def handler(e):
        pass

    bus.subscribe_async("topic", handler)
    assert bus.subscriber_count("topic") == 1
    bus.unsubscribe("topic", handler)
    assert bus.subscriber_count("topic") == 0


# --- History Overflow (Line 125) ---


def test_history_overflow_eviction():
    bus = EventBus()
    bus._max_history = 100
    for i in range(150):
        bus.publish(Event(topic="test", payload=i))
    assert len(bus.get_history()) == 100
    assert bus.get_history()[0].payload == 50


# --- Exception Handling (Lines 131-132) ---


def test_sync_subscriber_exception_doesnt_break_others():
    bus = EventBus()
    received = []

    def bad_handler(e):
        raise ValueError("Intentional error")

    def good_handler(e):
        received.append(e.payload)

    bus.subscribe("test", bad_handler)
    bus.subscribe("test", good_handler)
    bus.publish(Event(topic="test", payload="hello"))
    assert received == ["hello"]


# --- Async Callback Execution (Lines 145-148) ---


@pytest.mark.asyncio
async def test_async_callback_executes():
    bus = EventBus()
    received = []

    async def handler(e):
        received.append(e.payload)

    bus.subscribe_async("test", handler)
    bus.publish(Event(topic="test", payload="async_hello"))
    await asyncio.sleep(0.1)
    assert received == ["async_hello"]


@pytest.mark.asyncio
async def test_async_callback_exception_caught():
    bus = EventBus()

    async def bad_handler(e):
        raise RuntimeError("Async error")

    bus.subscribe_async("test", bad_handler)
    bus.publish(Event(topic="test", payload="data"))
    await asyncio.sleep(0.1)


# --- publish_async (Line 152) ---


@pytest.mark.asyncio
async def test_publish_async():
    bus = EventBus()
    received = []
    bus.subscribe("test", lambda e: received.append(e.payload))
    await bus.publish_async(Event(topic="test", payload="async_pub"))
    assert received == ["async_pub"]


# --- clear_history with Topic Filter (Lines 163-166) ---


def test_clear_history_specific_topic():
    bus = EventBus()
    bus.publish(Event(topic="a", payload=1))
    bus.publish(Event(topic="b", payload=2))
    bus.publish(Event(topic="a", payload=3))
    bus.clear_history(topic="a")
    history = bus.get_history()
    assert len(history) == 1
    assert history[0].topic == "b"


def test_clear_all_history():
    bus = EventBus()
    bus.publish(Event(topic="a", payload=1))
    bus.publish(Event(topic="b", payload=2))
    bus.clear_history()
    assert len(bus.get_history()) == 0


# --- subscriber_count (Lines 175-177) ---


def test_subscriber_count():
    bus = EventBus()
    bus.subscribe("test", lambda e: None)
    bus.subscribe("test", lambda e: None)
    bus.subscribe_async("test", lambda e: None)
    assert bus.subscriber_count("test") == 3
    assert bus.subscriber_count("nonexistent") == 0


# --- await_event (Lines 199-213) --- ENTIRE METHOD


@pytest.mark.asyncio
async def test_await_event_receives_event():
    bus = EventBus()

    async def publisher():
        await asyncio.sleep(0.1)
        bus.publish(Event(topic="test", payload="arrived"))

    asyncio.create_task(publisher())
    event = await bus.await_event("test", timeout=5.0)
    assert event is not None
    assert event.payload == "arrived"


@pytest.mark.asyncio
async def test_await_event_timeout():
    bus = EventBus()
    event = await bus.await_event("test", timeout=0.5)
    assert event is None


@pytest.mark.asyncio
async def test_await_event_with_filter():
    bus = EventBus()

    async def publisher():
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="skip"))
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="match"))

    asyncio.create_task(publisher())
    event = await bus.await_event(
        "test", timeout=5.0, filter_fn=lambda e: e.payload == "match"
    )
    assert event is not None
    assert event.payload == "match"


@pytest.mark.asyncio
async def test_await_event_unsubscribes_on_timeout():
    bus = EventBus()
    initial_count = bus.subscriber_count("test")
    await bus.await_event("test", timeout=0.1)
    assert bus.subscriber_count("test") == initial_count


@pytest.mark.asyncio
async def test_await_event_unsubscribes_after_receive():
    bus = EventBus()

    async def publisher():
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload="data"))

    asyncio.create_task(publisher())
    await bus.await_event("test", timeout=5.0)
    assert bus.subscriber_count("test") == 0


# --- get_stats (Line 217) ---


def test_get_stats():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    bus.subscribe_async("a", lambda e: None)
    bus.publish(Event(topic="c", payload=1))
    stats = bus.get_stats()
    assert stats["total_subscribers"] == 3
    assert stats["history_size"] == 1
    assert stats["max_history"] == 10000
    assert "a" in stats["topics"]
    assert "b" in stats["topics"]


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
    # With normalization off, the old topic name is kept as-is
    bus.publish(Event(topic="agent.command", payload=1))
    assert received == ["agent.command"]


# --- unsubscribe callback not found (silent) ---


def test_unsubscribe_callback_not_found_is_silent():
    bus = EventBus()
    def handler(e):  # noqa: E306
        pass
    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)  # removes it
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


# --- publish async subscriber scheduling error ---


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

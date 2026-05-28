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

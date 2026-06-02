"""Edge case tests for EventBus — Sprint 2 release guard."""

import pytest
import asyncio

from rosclaw.core.event_bus import EventBus, Event, EventPriority


def test_subscribe_empty_topic():
    """Empty topic should raise ValueError."""
    bus = EventBus()
    with pytest.raises(ValueError):
        bus.subscribe("", lambda e: None)


def test_subscribe_non_callable():
    """Non-callable handler should raise TypeError."""
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe("test", "not_callable")


def test_subscribe_non_string_topic():
    """Non-string topic should raise TypeError."""
    bus = EventBus()
    with pytest.raises(TypeError):
        bus.subscribe(123, lambda e: None)


def test_unsubscribe_not_subscribed():
    """Unsubscribe a never-subscribed handler should not crash."""
    bus = EventBus()
    bus.unsubscribe("test", lambda e: None)


def test_publish_no_subscribers():
    """Publish with no subscribers should not crash."""
    bus = EventBus()
    bus.publish(Event(topic="test", payload={}))


def test_get_history_negative_limit():
    """Negative limit should return empty or handle gracefully."""
    bus = EventBus()
    bus.publish(Event(topic="test", payload={}))
    result = bus.get_history(limit=-1)
    assert result == []


def test_get_history_zero_limit():
    """Zero limit should return empty list."""
    bus = EventBus()
    bus.publish(Event(topic="test", payload={}))
    result = bus.get_history(limit=0)
    assert result == []


def test_clear_history_empty_bus():
    """Clear history on empty bus should not crash."""
    bus = EventBus()
    bus.clear_history()
    assert bus.get_stats()["history_size"] == 0


def test_subscriber_count_no_subscribers():
    """Subscriber count for unknown topic should be 0."""
    bus = EventBus()
    assert bus.subscriber_count("unknown") == 0


def test_event_derive_defaults():
    """Derive with no overrides should inherit trace_id."""
    e1 = Event(topic="a", payload={}, trace_id="abc123")
    e2 = e1.derive()
    assert e2.trace_id == "abc123"
    assert e2.topic == "a"


def test_event_derive_override():
    """Derive with overrides should use new values."""
    e1 = Event(topic="a", payload={}, trace_id="abc123")
    e2 = e1.derive(topic="b", payload={"x": 1})
    assert e2.topic == "b"
    assert e2.payload == {"x": 1}
    assert e2.trace_id == "abc123"


@pytest.mark.asyncio
async def test_await_event_timeout():
    """await_event with no matching event should timeout and return None."""
    bus = EventBus()
    result = await bus.await_event("never", timeout=0.1)
    assert result is None


@pytest.mark.asyncio
async def test_await_event_filter_no_match():
    """await_event with filter that never matches should timeout."""
    bus = EventBus()

    async def late_publish():
        await asyncio.sleep(0.05)
        bus.publish(Event(topic="test", payload={"val": 1}))

    asyncio.create_task(late_publish())
    result = await bus.await_event("test", timeout=0.5, filter_fn=lambda e: e.payload.get("val") == 2)
    assert result is None


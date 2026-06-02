"""Tests for sandbox/events/publisher.py RuntimePublisher."""

from unittest.mock import MagicMock

import pytest

from rosclaw.sandbox.events.publisher import RuntimePublisher
from rosclaw.core.event_bus import EventBus, Event, EventPriority


class TestRuntimePublisher:
    def test_publish_with_event_bus(self):
        bus = EventBus()
        received = []
        bus.subscribe("sandbox.test", lambda e: received.append(e))

        publisher = RuntimePublisher(bus)
        publisher.publish("sandbox.test", {"key": "value"})

        assert len(received) == 1
        assert received[0].topic == "sandbox.test"
        assert received[0].payload == {"key": "value"}
        assert received[0].source == "sandbox"
        assert received[0].priority == EventPriority.NORMAL

    def test_publish_without_event_bus(self):
        publisher = RuntimePublisher(None)
        # Should not crash
        publisher.publish("sandbox.test", {"key": "value"})

    def test_publish_multiple_topics(self):
        bus = EventBus()
        received = []
        bus.subscribe("sandbox.step", lambda e: received.append(e))
        bus.subscribe("sandbox.reset", lambda e: received.append(e))

        publisher = RuntimePublisher(bus)
        publisher.publish("sandbox.step", {"time": 0.1})
        publisher.publish("sandbox.reset", {})

        assert len(received) == 2
        assert received[0].topic == "sandbox.step"
        assert received[1].topic == "sandbox.reset"

    def test_publish_priority_is_normal(self):
        bus = EventBus()
        received = []
        bus.subscribe("sandbox.test", lambda e: received.append(e))

        publisher = RuntimePublisher(bus)
        publisher.publish("sandbox.test", {})

        assert received[0].priority == EventPriority.NORMAL

    def test_publish_source_is_sandbox(self):
        bus = EventBus()
        received = []
        bus.subscribe("sandbox.test", lambda e: received.append(e))

        publisher = RuntimePublisher(bus)
        publisher.publish("sandbox.test", {})

        assert received[0].source == "sandbox"

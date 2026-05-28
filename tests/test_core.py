"""Tests for Core modules."""

import asyncio

import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin, LifecycleState
from rosclaw.core.runtime import Runtime, RuntimeConfig


def test_event_bus_publish_subscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test.topic", handler)
    bus.publish(Event(topic="test.topic", payload={"data": 1}, source="test"))
    assert len(received) == 1
    assert received[0].payload["data"] == 1


def test_event_bus_priority():
    bus = EventBus()
    priorities = []

    def handler(event):
        priorities.append(event.priority)

    bus.subscribe("test", handler)
    bus.publish(Event(topic="test", payload={}, priority=EventPriority.HIGH))
    bus.publish(Event(topic="test", payload={}, priority=EventPriority.LOW))
    assert priorities[0] == EventPriority.HIGH
    assert priorities[1] == EventPriority.LOW


def test_event_bus_history():
    bus = EventBus()
    bus.publish(Event(topic="a", payload={}))
    bus.publish(Event(topic="b", payload={}))
    bus.publish(Event(topic="a", payload={}))
    assert len(bus.get_history("a")) == 2
    assert len(bus.get_history()) == 3


def test_event_bus_unsubscribe():
    bus = EventBus()
    received = []

    def handler(event):
        received.append(event)

    bus.subscribe("test", handler)
    bus.unsubscribe("test", handler)
    bus.publish(Event(topic="test", payload={}))
    assert len(received) == 0


def test_lifecycle_transitions():
    class TestModule(LifecycleMixin):
        def __init__(self):
            super().__init__()
            self.started = False

        def _do_start(self):
            self.started = True

    mod = TestModule()
    assert mod.state == LifecycleState.UNINITIALIZED
    mod.initialize()
    assert mod.state == LifecycleState.READY
    mod.start()
    assert mod.state == LifecycleState.RUNNING
    assert mod.started is True
    mod.stop()
    assert mod.state == LifecycleState.STOPPED


def test_lifecycle_cannot_start_uninitialized():
    class TestModule(LifecycleMixin):
        pass

    mod = TestModule()
    with pytest.raises(RuntimeError):
        mod.start()


def test_runtime_default_config():
    rt = Runtime()
    assert rt.config.robot_id == "rosclaw_default"
    assert rt.config.enable_firewall is True


def test_runtime_status():
    rt = Runtime(RuntimeConfig(enable_firewall=False, enable_memory=False, enable_practice=False))
    rt.initialize()
    status = rt.get_status()
    assert status["robot_id"] == "rosclaw_default"
    assert status["modules"]["firewall"] is False
    rt.stop()


def test_runtime_status_property():
    """Runtime.status property should mirror get_status()."""
    from rosclaw.core.runtime import Runtime
    runtime = Runtime()
    runtime.initialize()
    status = runtime.status
    assert isinstance(status, dict)
    assert "robot_id" in status
    assert "modules" in status
    assert status == runtime.get_status()
    runtime.stop()

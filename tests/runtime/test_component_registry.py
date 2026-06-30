"""Tests for the RuntimeComponentRegistry (Milestone 1)."""

from __future__ import annotations

from typing import Any

from rosclaw.core.event_bus import EventBus
from rosclaw.runtime.bus import RuntimeBus
from rosclaw.runtime.component import RuntimeConsumer, RuntimeProducer
from rosclaw.runtime.event import RuntimeEvent
from rosclaw.runtime.registry import RuntimeComponentRegistry


class _FakeProducer(RuntimeProducer):
    def __init__(self, name: str, bus: RuntimeBus) -> None:
        super().__init__(name, bus)
        self.started = False
        self.stopped = False

    def _do_start(self) -> None:
        self.started = True

    def _do_stop(self) -> None:
        self.stopped = True


class _FakeConsumer(RuntimeConsumer):
    def __init__(self, name: str, bus: RuntimeBus) -> None:
        super().__init__(name, bus)
        self.events: list[RuntimeEvent] = []

    def on_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)


def test_registry_tracks_components() -> None:
    bus = RuntimeBus(event_bus=EventBus())
    registry = RuntimeComponentRegistry()
    producer = _FakeProducer("producer_1", bus)
    registry.register("producer_1", producer)
    assert registry.get("producer_1") is producer
    assert registry.names() == ["producer_1"]


def test_registry_lifecycle_start_stop() -> None:
    bus = RuntimeBus(event_bus=EventBus())
    registry = RuntimeComponentRegistry()
    producer = _FakeProducer("p", bus)
    registry.register("p", producer)
    registry.initialize_all()
    registry.start_all()
    assert producer.started
    assert producer.state.name == "RUNNING"
    registry.stop_all()
    assert producer.stopped


def test_registry_stops_in_reverse_order() -> None:
    bus = RuntimeBus(event_bus=EventBus())
    registry = RuntimeComponentRegistry()
    order: list[str] = []

    class _OrderProducer(_FakeProducer):
        def _do_stop(self) -> None:
            order.append(self.name)

    registry.register("a", _OrderProducer("a", bus))
    registry.register("b", _OrderProducer("b", bus))
    registry.initialize_all()
    registry.start_all()
    registry.stop_all()
    assert order == ["b", "a"]


def test_consumer_receives_events_via_registry() -> None:
    bus = RuntimeBus(event_bus=EventBus())
    registry = RuntimeComponentRegistry()
    consumer = _FakeConsumer("c", bus)
    registry.register("c", consumer)
    registry.initialize_all()
    registry.start_all()
    bus.publish(RuntimeEvent(source="test", type="camera.frame", payload={"x": 1}))
    assert len(consumer.events) == 1
    assert consumer.events[0].payload["x"] == 1
    registry.stop_all()
    assert not consumer._subscriptions


def test_registry_stats_reflect_component_states() -> None:
    bus = RuntimeBus(event_bus=EventBus())
    registry = RuntimeComponentRegistry()
    registry.register("x", _FakeProducer("x", bus))
    registry.initialize_all()
    stats = registry.stats()
    assert stats["x"]["state"] == "READY"

"""Publish engineering evidence through the existing EventBus."""

from rosclaw.core.event_bus import Event


def emit_expert_evidence(event_bus, topic: str, payload: dict) -> None:
    if event_bus is not None:
        event_bus.publish(Event(topic=topic, payload=payload, source="ros_expert_harness"))

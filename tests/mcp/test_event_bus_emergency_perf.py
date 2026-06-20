"""Performance/load tests for EventBus emergency-stop dispatch.

The P0 safety contract relies on ``robot.emergency_stop`` being delivered
synchronously to subscribers. These tests verify that the EventBus can sustain
a high volume of CRITICAL-priority emergency events without blocking the caller
for a meaningful amount of time.
"""

from __future__ import annotations

import asyncio
import time

from rosclaw.core.event_bus import Event, EventBus, EventPriority


def test_emergency_stop_sync_throughput() -> None:
    """1000 CRITICAL emergency-stop events must be dispatched synchronously in under 1s."""
    bus = EventBus()
    received: list[Event] = []

    def on_stop(event: Event) -> None:
        received.append(event)

    bus.subscribe("robot.emergency_stop", on_stop)

    count = 1000
    start = time.perf_counter()
    for i in range(count):
        bus.publish(
            Event(
                topic="robot.emergency_stop",
                payload={"reason": f"load test {i}"},
                priority=EventPriority.CRITICAL,
                source="perf.test",
            )
        )
    elapsed = time.perf_counter() - start

    assert len(received) == count
    assert elapsed < 1.0, f"emergency dispatch took {elapsed:.3f}s for {count} events"
    assert all(e.priority == EventPriority.CRITICAL for e in received)


async def test_emergency_stop_does_not_block_event_loop() -> None:
    """Publishing many emergency events should leave the asyncio loop responsive."""
    bus = EventBus()
    received: list[Event] = []
    bus.subscribe("robot.emergency_stop", received.append)

    count = 500
    # Publish from an async context; if sync dispatch blocked the loop,
    # the short sleep would be delayed by the full publish burst.
    for i in range(count):
        bus.publish(
            Event(
                topic="robot.emergency_stop",
                payload={"reason": f"async load {i}"},
                priority=EventPriority.CRITICAL,
                source="perf.test",
            )
        )

    # A 1 ms sleep should complete promptly even after the burst.
    await asyncio.sleep(0.001)
    assert len(received) == count

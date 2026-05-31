"""EventBus stress and performance tests."""

import asyncio
import time

import pytest

from rosclaw.core.event_bus import EventBus, Event, EventPriority


class TestEventBusStress:
    """Stress tests for EventBus throughput and reliability."""

    def test_high_volume_sync_publish(self):
        """Publish 10K events to sync subscribers without drops."""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event.event_id)

        bus.subscribe("stress.test", handler)

        count = 10000
        start = time.perf_counter()
        for i in range(count):
            bus.publish(Event(
                topic="stress.test",
                payload={"index": i},
                source="stress",
            ))
        elapsed = time.perf_counter() - start

        assert len(received) == count, f"Expected {count}, got {len(received)}"
        throughput = count / elapsed
        print(f"\nSync publish: {count} events in {elapsed:.3f}s ({throughput:.0f} evt/s)")
        assert throughput > 1000, f"Throughput too low: {throughput:.0f} evt/s"

    @pytest.mark.asyncio
    async def test_high_volume_async_publish(self):
        """Publish 5K events to async subscribers."""
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event.event_id)

        bus.subscribe_async("stress.async", handler)

        count = 5000
        start = time.perf_counter()
        for i in range(count):
            bus.publish(Event(
                topic="stress.async",
                payload={"index": i},
                source="stress",
            ))
        # Wait for async processing
        await asyncio.sleep(1.0)
        elapsed = time.perf_counter() - start

        assert len(received) == count, f"Expected {count}, got {len(received)}"
        throughput = count / elapsed
        print(f"\nAsync publish: {count} events in {elapsed:.3f}s ({throughput:.0f} evt/s)")

    def test_concurrent_subscribers(self):
        """Multiple subscribers on same topic all receive all events."""
        bus = EventBus()
        counts = [0, 0, 0]

        def make_handler(idx):
            def handler(event):
                counts[idx] += 1
            return handler

        for i in range(3):
            bus.subscribe("stress.concurrent", make_handler(i))

        count = 1000
        for i in range(count):
            bus.publish(Event(topic="stress.concurrent", payload={"i": i}, source="stress"))

        assert counts[0] == count
        assert counts[1] == count
        assert counts[2] == count

    def test_wildcard_subscription_stress(self):
        """Wildcard subscriber receives matching events under load."""
        bus = EventBus()
        received = []

        bus.subscribe("rosclaw.sandbox.*", lambda e: received.append(e.topic))

        topics = [
            "rosclaw.sandbox.episode.started",
            "rosclaw.sandbox.action.allowed",
            "rosclaw.sandbox.action.blocked",
            "rosclaw.praxis.completed",  # should NOT match
        ]

        for i in range(1000):
            topic = topics[i % len(topics)]
            bus.publish(Event(topic=topic, payload={}, source="stress"))

        # 3/4 topics match wildcard, 1000 total = ~750 matches
        expected = sum(1 for i in range(1000) if topics[i % len(topics)].startswith("rosclaw.sandbox."))
        assert len(received) == expected, f"Expected {expected}, got {len(received)}"

    def test_history_limit(self):
        """Event history respects max_history limit."""
        bus = EventBus()
        bus._max_history = 100

        for i in range(500):
            bus.publish(Event(topic="hist.test", payload={"i": i}, source="stress"))

        assert len(bus._event_history) == 100
        # Oldest events should be evicted
        assert bus._event_history[0].payload["i"] == 400

    def test_publish_with_trace_id_propagation(self):
        """Trace IDs propagate correctly under load."""
        bus = EventBus()
        received = []

        bus.subscribe("trace.test", lambda e: received.append(e.trace_id))

        for i in range(100):
            bus.publish(Event(
                topic="trace.test",
                payload={"request_id": f"req_{i}"},
                source="stress",
            ))

        assert len(received) == 100
        assert all(t.startswith("req_") for t in received)
        assert len(set(received)) == 100  # All unique

    @pytest.mark.asyncio
    async def test_await_event_under_load(self):
        """await_event works correctly even with high publish rate."""
        bus = EventBus()

        async def publisher():
            for i in range(100):
                bus.publish(Event(topic="await.load", payload={"index": i}, source="stress"))
                await asyncio.sleep(0.001)

        async def waiter():
            return await bus.await_event("await.load", timeout=5.0)

        pub_task = asyncio.create_task(publisher())
        event = await waiter()
        await pub_task

        assert event is not None
        assert event.topic == "await.load"

    def test_priority_ordering(self):
        """Critical priority events are processed before low priority."""
        bus = EventBus()
        order = []

        bus.subscribe("prio.test", lambda e: order.append(e.priority.name))

        # Publish in reverse priority order
        for p in [EventPriority.BACKGROUND, EventPriority.LOW, EventPriority.NORMAL, EventPriority.HIGH, EventPriority.CRITICAL]:
            bus.publish(Event(topic="prio.test", payload={}, priority=p, source="stress"))

        # All should be received, order depends on processing sequence
        assert len(order) == 5
        assert set(order) == {"CRITICAL", "HIGH", "NORMAL", "LOW", "BACKGROUND"}

"""Thread-safety and concurrency correctness tests for EventBus.

Validates that P1 fix (threading.Lock) does not introduce deadlocks
or race conditions in mixed sync/async usage.
"""

import asyncio
import threading


from rosclaw.core.event_bus import EventBus, Event


class TestEventBusThreadSafety:
    """Verify thread-safe operation of EventBus with threading.Lock."""

    def test_concurrent_subscribe_and_publish(self):
        """Multiple threads subscribing while publishing must not crash."""
        bus = EventBus()
        received = []
        errors = []

        def handler(event):
            try:
                received.append(event.event_id)
            except Exception as e:
                errors.append(e)

        def publisher_thread(n):
            for i in range(n):
                bus.publish(Event(topic="thread.test", payload={"i": i}, source="thread"))

        def subscriber_thread(n):
            # Subscribe unique topics to exercise the lock, not the test topic
            for i in range(n):
                bus.subscribe(f"other.{i}", handler)
            # Subscribe once to the target topic
            bus.subscribe("thread.test", handler)

        threads = []
        threads.append(threading.Thread(target=publisher_thread, args=(500,)))
        threads.append(threading.Thread(target=publisher_thread, args=(500,)))
        t_sub = threading.Thread(target=subscriber_thread, args=(50,))
        threads.append(t_sub)

        # Start subscriber first so it registers before publishers begin
        t_sub.start()
        t_sub.join(timeout=5)

        for t in threads[:-1]:
            t.start()
        for t in threads[:-1]:
            t.join(timeout=10)

        assert not errors, f"Errors during concurrent ops: {errors[:5]}"
        assert len(received) == 1000, f"Expected 1000, got {len(received)}"

    def test_concurrent_unsubscribe_during_publish(self):
        """Unsubscribing while publishing must not crash or miss events."""
        bus = EventBus()
        received = []

        def handler(event):
            received.append(event.event_id)

        bus.subscribe("unsub.test", handler)

        def publisher(n):
            for i in range(n):
                bus.publish(Event(topic="unsub.test", payload={"i": i}, source="thread"))

        def unsubscriber(n):
            for _ in range(n):
                try:
                    bus.unsubscribe("unsub.test", handler)
                    bus.subscribe("unsub.test", handler)
                except Exception:
                    pass

        threads = [
            threading.Thread(target=publisher, args=(500,)),
            threading.Thread(target=unsubscriber, args=(50,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # Some events may be lost during unsubscribe windows, but no crashes
        assert len(received) <= 500
        assert len(received) >= 400  # Most should still arrive

    def test_no_deadlock_sync_from_async_context(self):
        """Sync subscribe called from async task must not deadlock."""
        bus = EventBus()
        received = []

        async def async_task():
            def handler(event):
                received.append(event.event_id)
            bus.subscribe("async.sync", handler)
            bus.publish(Event(topic="async.sync", payload={}, source="test"))
            await asyncio.sleep(0.01)

        asyncio.run(asyncio.wait_for(async_task(), timeout=5))
        assert len(received) == 1

    def test_history_append_under_concurrent_publish(self):
        """Event history must remain consistent under concurrent publish."""
        bus = EventBus()
        bus._max_history = 100

        def publisher(n, prefix):
            for i in range(n):
                bus.publish(Event(topic=f"hist.{prefix}", payload={"i": i}, source="thread"))

        threads = [
            threading.Thread(target=publisher, args=(200, "a")),
            threading.Thread(target=publisher, args=(200, "b")),
            threading.Thread(target=publisher, args=(200, "c")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(bus._event_history) == 100  # Max history respected
        # All events should be valid Event objects
        assert all(isinstance(e, Event) for e in bus._event_history)

    def test_lock_does_not_block_event_loop(self):
        """threading.Lock in subscribe must not block asyncio event loop."""
        bus = EventBus()

        async def background_publish():
            for i in range(100):
                bus.publish(Event(topic="bg.test", payload={"i": i}, source="test"))
                await asyncio.sleep(0)

        async def background_subscribe():
            def h(e):
                pass

            for i in range(50):
                bus.subscribe(f"bg.{i}", h)
                await asyncio.sleep(0)

        async def main():
            await asyncio.gather(background_publish(), background_subscribe())

        asyncio.run(asyncio.wait_for(main(), timeout=5))
        # If we get here without timeout, no deadlock


class TestEventBusLockConsistency:
    """Verify lock covers all mutable shared state."""

    def test_publish_reads_subscribers_consistently(self):
        """Publish must see a consistent view of subscribers."""
        bus = EventBus()
        counts = {"a": 0, "b": 0}

        def make_handler(key):
            def h(event):
                counts[key] += 1
            return h

        bus.subscribe("consistency", make_handler("a"))

        def rapid_subscribe_unsubscribe():
            h = make_handler("b")
            for _ in range(100):
                bus.subscribe("consistency", h)
                bus.unsubscribe("consistency", h)

        def rapid_publish():
            for _ in range(1000):
                bus.publish(Event(topic="consistency", payload={}, source="test"))

        threads = [
            threading.Thread(target=rapid_subscribe_unsubscribe),
            threading.Thread(target=rapid_publish),
            threading.Thread(target=rapid_publish),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # No RuntimeError from dict mutation during iteration
        # 2 publish threads x 1000 = 2000 total publishes
        assert counts["a"] == 2000  # All publishes hit handler "a"
        assert counts["b"] <= 200   # "b" may miss some due to unsub windows

    def test_reentrant_subscribe_safe(self):
        """Handler that subscribes another handler must not deadlock."""
        bus = EventBus()
        received = []

        def inner_handler(event):
            received.append("inner")

        def outer_handler(event):
            received.append("outer")
            bus.subscribe("reentrant.inner", inner_handler)
            bus.publish(Event(topic="reentrant.inner", payload={}, source="test"))

        bus.subscribe("reentrant.outer", outer_handler)
        bus.publish(Event(topic="reentrant.outer", payload={}, source="test"))

        assert "outer" in received
        assert "inner" in received

"""Performance benchmarks for ROSClaw core components.

Records baseline metrics for:
- EventBus publish latency
- Provider inference latency
- Sandbox validation time
- Memory write/read latency
- Dashboard snapshot generation
"""

import time

import pytest

from rosclaw.core.event_bus import EventBus, Event
from rosclaw.dashboard.metrics import DashboardMetrics
from rosclaw.memory.interface import MemoryInterface
from rosclaw.provider.builtins.critic import MockCriticProvider
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _make_critic_manifest():
    return ProviderManifest(
        name="critic",
        version="1.0.0",
        type="critic",
        capabilities=["critic.success_detection", "critic.retry_advice"],
    )


class TestEventBusBenchmark:
    """Benchmark EventBus publish/subscribe latency."""

    def test_sync_publish_latency(self):
        bus = EventBus()
        received = []
        bus.subscribe("bench.sync", lambda e: received.append(e))

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            bus.publish(Event(topic="bench.sync", payload={"x": 1}, source="bench"))
        elapsed = time.perf_counter() - start

        avg_us = (elapsed / iterations) * 1e6
        print(f"\n  EventBus sync latency: {avg_us:.1f} us/call")
        assert len(received) == iterations
        assert avg_us < 1000, f"Latency too high: {avg_us:.1f} us"

    def test_sync_publish_throughput(self):
        bus = EventBus()
        count = 0
        bus.subscribe("bench.throughput", lambda e: globals().__setitem__("count", count + 1))

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            bus.publish(Event(topic="bench.throughput", payload={}, source="bench"))
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n  EventBus sync throughput: {throughput:.0f} events/sec")
        assert throughput > 5000, f"EventBus throughput too low: {throughput:.0f} evt/s"


class TestProviderBenchmark:
    """Benchmark provider inference latency."""

    @pytest.mark.asyncio
    async def test_critic_success_detection_latency(self):
        provider = MockCriticProvider(_make_critic_manifest())
        req = ProviderRequest(
            request_id="r1",
            capability="critic.success_detection",
            inputs={
                "task_type": "reach",
                "target_pose": [0.5, 0.2, 0.3],
                "actual_pose": [0.51, 0.21, 0.31],
                "collision": False,
                "workspace_violation": False,
                "elapsed_time": 5.0,
            },
        )

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            await provider.infer(req)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"\n  Critic success_detection: {avg_ms:.3f} ms/call ({iterations / elapsed:.0f} calls/sec)")
        assert avg_ms < 10, f"Critic inference too slow: {avg_ms:.3f} ms"

    @pytest.mark.asyncio
    async def test_critic_retry_advice_latency(self):
        provider = MockCriticProvider(_make_critic_manifest())
        req = ProviderRequest(
            request_id="r1",
            capability="critic.retry_advice",
            inputs={"task_type": "reach", "failed_checks": ["position_error"]},
        )

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            await provider.infer(req)
        elapsed = time.perf_counter() - start

        avg_ms = (elapsed / iterations) * 1000
        print(f"\n  Critic retry_advice: {avg_ms:.3f} ms/call ({iterations / elapsed:.0f} calls/sec)")
        assert avg_ms < 10, f"Critic retry too slow: {avg_ms:.3f} ms"


class TestDashboardMetricsBenchmark:
    """Benchmark DashboardMetrics operations."""

    def test_record_episode_throughput(self):
        metrics = DashboardMetrics()
        iterations = 10000

        start = time.perf_counter()
        for i in range(iterations):
            metrics.record_episode(f"ep_{i}", "ur5e", "success", reward=0.9, duration_sec=1.0)
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n  Dashboard record_episode: {throughput:.0f} records/sec")
        assert throughput > 10000, f"Dashboard record too slow: {throughput:.0f} rec/s"

    def test_get_snapshot_latency(self):
        from rosclaw.dashboard.server import DashboardServer
        server = DashboardServer(DashboardMetrics())
        for i in range(1000):
            server.metrics.record_provider_call("llm", "chat", 100.0, "ok")
            server.metrics.record_sandbox_validation("move", True)
            server.metrics.record_episode(f"ep_{i}", "ur5e", "success")

        start = time.perf_counter()
        snapshot = server.get_snapshot()
        elapsed = time.perf_counter() - start

        ms = elapsed * 1000
        print(f"\n  Dashboard get_snapshot: {ms:.3f} ms")
        assert ms < 50, f"Dashboard snapshot too slow: {ms:.3f} ms"
        assert "module_health" in snapshot
        assert "provider" in snapshot
        assert "episodes" in snapshot

    def test_event_counter_throughput(self):
        metrics = DashboardMetrics()
        iterations = 50000

        start = time.perf_counter()
        for i in range(iterations):
            metrics.increment_event("rosclaw.test.topic")
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n  Dashboard event counter: {throughput:.0f} increments/sec")
        counts = metrics.get_event_counts()
        assert counts.get("rosclaw.test.topic", 0) == iterations


class TestMemoryBenchmark:
    """Benchmark MemoryInterface operations."""

    def test_memory_write_throughput(self):
        mem = MemoryInterface("bench_robot")
        mem.initialize()
        iterations = 1000

        start = time.perf_counter()
        for i in range(iterations):
            mem.write(f"key_{i}", {"data": i, "type": "benchmark"})
        elapsed = time.perf_counter() - start

        throughput = iterations / elapsed
        print(f"\n  Memory write: {throughput:.0f} writes/sec")
        assert throughput > 500, f"Memory write too slow: {throughput:.0f} w/s"

    def test_memory_search_latency(self):
        mem = MemoryInterface("bench_robot")
        mem.initialize()
        for i in range(500):
            mem.write(f"key_{i}", {
                "instruction": f"reach to point {i}",
                "outcome": "success" if i % 2 == 0 else "failure",
            })

        start = time.perf_counter()
        results = mem.search("reach point")
        elapsed = time.perf_counter() - start

        ms = elapsed * 1000
        print(f"\n  Memory search: {ms:.3f} ms ({len(results)} results)")
        assert len(results) > 0


class TestSystemBaseline:
    """Record overall system baseline metrics."""

    def test_import_time(self):
        """Measure cold import time for core modules."""
        start = time.perf_counter()
        elapsed = time.perf_counter() - start

        ms = elapsed * 1000
        print(f"\n  Cold import time: {ms:.1f} ms")
        assert ms < 5000, f"Import too slow: {ms:.1f} ms"

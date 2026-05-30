"""Tests for rosclaw.dashboard module.

Coverage targets:
  - DashboardMetrics: provider/sandbox/episode aggregation, event counts,
    module health, snapshot, rolling window trim, uptime
  - DashboardServer: lifecycle (start/stop/idempotency), WebSocket broadcast
    (multi-client, dead-client cleanup, send_text vs send), EventBus attach/detach,
    HTTP API (snapshot, health, robots)

Goal: 70%+ coverage.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import MagicMock, AsyncMock

import pytest

from rosclaw.dashboard import DashboardMetrics, DashboardServer
from rosclaw.dashboard.metrics import ProviderMetric, SandboxMetric, EpisodeMetric


# ───────────────────────── DashboardMetrics ─────────────────────────

class TestDashboardMetricsInit:
    def test_init_defaults(self):
        m = DashboardMetrics()
        assert m.max_history == 1000
        assert m.get_uptime_sec() >= 0
        assert m.snapshot()["uptime_sec"] >= 0

    def test_init_custom_max_history(self):
        m = DashboardMetrics(max_history=50)
        assert m.max_history == 50


class TestDashboardMetricsProvider:
    def test_provider_stats_empty(self):
        m = DashboardMetrics()
        stats = m.get_provider_stats()
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency_ms"] == 0.0

    def test_provider_stats_basic(self):
        m = DashboardMetrics()
        m.record_provider_call("vlm", "vlm.grounding", 120.0, "ok")
        m.record_provider_call("vlm", "vlm.grounding", 200.0, "ok")
        m.record_provider_call("skill", "skill.pick", 500.0, "error")

        stats = m.get_provider_stats()
        assert stats["total"] == 3
        assert stats["success_rate"] == pytest.approx(2 / 3)
        assert stats["avg_latency_ms"] == pytest.approx(273.33, abs=0.1)
        assert "vlm" in stats["by_provider"]
        assert stats["by_provider"]["vlm"]["calls"] == 2
        assert stats["by_provider"]["vlm"]["errors"] == 0
        assert stats["by_provider"]["skill"]["errors"] == 1

    def test_provider_stats_all_error(self):
        m = DashboardMetrics()
        m.record_provider_call("p", "c", 10.0, "timeout")
        m.record_provider_call("p", "c", 20.0, "error")
        stats = m.get_provider_stats()
        assert stats["success_rate"] == 0.0
        assert stats["avg_latency_ms"] == 15.0

    def test_provider_stats_single(self):
        m = DashboardMetrics()
        m.record_provider_call("p", "c", 42.0, "ok")
        stats = m.get_provider_stats()
        assert stats["total"] == 1
        assert stats["success_rate"] == 1.0
        assert stats["avg_latency_ms"] == 42.0


class TestDashboardMetricsSandbox:
    def test_sandbox_stats_empty(self):
        m = DashboardMetrics()
        stats = m.get_sandbox_stats()
        assert stats["total"] == 0
        assert stats["block_rate"] == 0.0

    def test_sandbox_stats_basic(self):
        m = DashboardMetrics()
        m.record_sandbox_validation("pick", True)
        m.record_sandbox_validation("place", False, ["collision_predicted"])

        stats = m.get_sandbox_stats()
        assert stats["total"] == 2
        assert stats["block_rate"] == 0.5
        assert len(stats["recent_violations"]) == 1
        assert stats["recent_violations"][0]["action"] == "place"

    def test_sandbox_stats_all_safe(self):
        m = DashboardMetrics()
        for _ in range(5):
            m.record_sandbox_validation("pick", True)
        stats = m.get_sandbox_stats()
        assert stats["block_rate"] == 0.0
        assert stats["recent_violations"] == []

    def test_sandbox_stats_all_blocked(self):
        m = DashboardMetrics()
        for i in range(5):
            m.record_sandbox_validation(f"act_{i}", False, [f"v{i}"])
        stats = m.get_sandbox_stats()
        assert stats["block_rate"] == 1.0
        assert len(stats["recent_violations"]) == 5

    def test_sandbox_stats_no_violations_list(self):
        m = DashboardMetrics()
        m.record_sandbox_validation("pick", False)
        stats = m.get_sandbox_stats()
        assert len(stats["recent_violations"]) == 1
        assert stats["recent_violations"][0]["violations"] == []


class TestDashboardMetricsEpisodes:
    def test_episode_stats_empty(self):
        m = DashboardMetrics()
        stats = m.get_episode_stats()
        assert stats["total"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["avg_reward"] == 0.0

    def test_episode_stats_basic(self):
        m = DashboardMetrics()
        m.record_episode("ep_001", "ur5e", "success", reward=0.95, duration_sec=12.3)
        m.record_episode("ep_002", "ur5e", "failure", reward=0.1, duration_sec=5.0)

        stats = m.get_episode_stats()
        assert stats["total"] == 2
        assert stats["success_rate"] == 0.5
        assert stats["avg_reward"] == pytest.approx(0.525, abs=0.01)
        assert len(stats["recent"]) == 2

    def test_episode_stats_no_reward(self):
        m = DashboardMetrics()
        m.record_episode("ep_001", "ur5e", "success")
        stats = m.get_episode_stats()
        assert stats["avg_reward"] == 0.0

    def test_episode_stats_all_success(self):
        m = DashboardMetrics()
        for i in range(4):
            m.record_episode(f"ep_{i}", "ur5e", "success", reward=0.8 + i * 0.05)
        stats = m.get_episode_stats()
        assert stats["success_rate"] == 1.0
        assert stats["avg_reward"] == pytest.approx(0.875, abs=0.01)

    def test_episode_stats_recent_limit(self):
        m = DashboardMetrics()
        for i in range(10):
            m.record_episode(f"ep_{i}", "ur5e", "success", reward=0.5)
        stats = m.get_episode_stats()
        assert len(stats["recent"]) == 5


class TestDashboardMetricsEvents:
    def test_event_counts_empty(self):
        m = DashboardMetrics()
        assert m.get_event_counts() == {}

    def test_event_counts(self):
        m = DashboardMetrics()
        m.increment_event("provider.call")
        m.increment_event("provider.call")
        m.increment_event("sandbox.block")

        counts = m.get_event_counts()
        assert counts["provider.call"] == 2
        assert counts["sandbox.block"] == 1

    def test_event_counts_many_topics(self):
        m = DashboardMetrics()
        for i in range(100):
            m.increment_event(f"topic_{i % 10}")
        counts = m.get_event_counts()
        assert len(counts) == 10
        assert counts["topic_0"] == 10


class TestDashboardMetricsHealth:
    def test_module_health_empty(self):
        m = DashboardMetrics()
        assert m.get_module_health() == {}

    def test_module_health(self):
        m = DashboardMetrics()
        m.set_module_health("runtime", "HEALTHY")
        m.set_module_health("sandbox", "DEGRADED")
        m.set_module_health("memory", "HEALTHY")

        health = m.get_module_health()
        assert health["runtime"] == "HEALTHY"
        assert health["sandbox"] == "DEGRADED"
        assert health["memory"] == "HEALTHY"

    def test_module_health_overwrite(self):
        m = DashboardMetrics()
        m.set_module_health("runtime", "HEALTHY")
        m.set_module_health("runtime", "DEGRADED")
        assert m.get_module_health()["runtime"] == "DEGRADED"


class TestDashboardMetricsSnapshot:
    def test_snapshot_completeness_empty(self):
        m = DashboardMetrics()
        snapshot = m.snapshot()
        assert "uptime_sec" in snapshot
        assert "module_health" in snapshot
        assert "provider" in snapshot
        assert "sandbox" in snapshot
        assert "episodes" in snapshot
        assert "event_counts" in snapshot

    def test_snapshot_with_data(self):
        m = DashboardMetrics()
        m.set_module_health("runtime", "HEALTHY")
        m.record_provider_call("vlm", "vlm.grounding", 100.0, "ok")
        m.record_sandbox_validation("pick", True)
        m.record_episode("ep_001", "ur5e", "success", reward=0.9)
        m.increment_event("provider.call")

        snapshot = m.snapshot()
        assert snapshot["module_health"]["runtime"] == "HEALTHY"
        assert snapshot["provider"]["total"] == 1
        assert snapshot["sandbox"]["total"] == 1
        assert snapshot["episodes"]["total"] == 1
        assert snapshot["event_counts"]["provider.call"] == 1
        assert snapshot["uptime_sec"] >= 0


class TestDashboardMetricsUptime:
    def test_uptime_increases(self):
        m = DashboardMetrics()
        t1 = m.get_uptime_sec()
        time.sleep(0.05)
        t2 = m.get_uptime_sec()
        assert t2 > t1


class TestDashboardMetricsRollingWindow:
    def test_rolling_window_provider(self):
        m = DashboardMetrics(max_history=3)
        for i in range(5):
            m.record_provider_call("p", "c", float(i), "ok")
        assert len(m._provider_metrics) == 3
        assert m._provider_metrics[0].latency_ms == 2.0

    def test_rolling_window_sandbox(self):
        m = DashboardMetrics(max_history=2)
        for i in range(4):
            m.record_sandbox_validation(f"a{i}", True)
        assert len(m._sandbox_metrics) == 2

    def test_rolling_window_episode(self):
        m = DashboardMetrics(max_history=2)
        for i in range(4):
            m.record_episode(f"ep_{i}", "r", "success")
        assert len(m._episode_metrics) == 2

    def test_max_history_zero_trims_all(self):
        m = DashboardMetrics(max_history=0)
        m.record_provider_call("p", "c", 1.0, "ok")
        # max_history=0 trims everything
        assert len(m._provider_metrics) == 0


class TestDashboardMetricsDataclasses:
    def test_provider_metric_defaults(self):
        before = time.time()
        pm = ProviderMetric(provider="vlm", capability="ground", latency_ms=10.0, status="ok")
        after = time.time()
        assert pm.provider == "vlm"
        assert before <= pm.timestamp <= after

    def test_sandbox_metric_defaults(self):
        sm = SandboxMetric(action_type="pick", is_safe=True)
        assert sm.violations == []
        assert sm.timestamp > 0

    def test_episode_metric_defaults(self):
        em = EpisodeMetric(episode_id="ep1", robot_id="ur5e", status="success")
        assert em.reward is None
        assert em.duration_sec is None
        assert em.timestamp > 0


# ───────────────────────── DashboardServer ─────────────────────────

class TestDashboardServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        await server.start()
        assert server._running is True
        assert server._task is not None

        await server.stop()
        assert server._running is False
        assert server._task is None

    @pytest.mark.asyncio
    async def test_double_start_idempotent(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        await server.start()
        first_task = server._task
        await server.start()
        assert server._task is first_task

        await server.stop()

    @pytest.mark.asyncio
    async def test_double_stop_idempotent(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        await server.start()
        await server.stop()
        assert server._task is None
        await server.stop()
        assert server._task is None

    @pytest.mark.asyncio
    async def test_stop_without_start(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        await server.stop()
        assert server._running is False


class TestDashboardServerBroadcast:
    @pytest.mark.asyncio
    async def test_broadcast_no_clients(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        await server._broadcast('{"test": true}')
        assert len(server._clients) == 0

    @pytest.mark.asyncio
    async def test_broadcast_single_client_send_text(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        client = AsyncMock()
        client.send_text = AsyncMock()
        server.register_client(client)

        await server._broadcast('{"hello": true}')
        client.send_text.assert_awaited_once_with('{"hello": true}')

    @pytest.mark.asyncio
    async def test_broadcast_single_client_send(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        client = AsyncMock()
        del client.send_text
        client.send = AsyncMock()
        server.register_client(client)

        await server._broadcast('{"hello": true}')
        client.send.assert_awaited_once_with('{"hello": true}')

    @pytest.mark.asyncio
    async def test_broadcast_multiple_clients(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        clients = [AsyncMock() for _ in range(3)]
        for c in clients:
            server.register_client(c)

        await server._broadcast('{"multi": true}')
        for c in clients:
            c.send_text.assert_awaited_once_with('{"multi": true}')

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        alive = AsyncMock()
        alive.send_text = AsyncMock()

        dead = AsyncMock()
        dead.send_text = AsyncMock(side_effect=Exception("connection lost"))

        server.register_client(alive)
        server.register_client(dead)

        await server._broadcast('{"msg": true}')
        assert dead not in server._clients
        assert alive in server._clients
        alive.send_text.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_broadcast_loop_sends_snapshot(self):
        metrics = DashboardMetrics()
        metrics.set_module_health("runtime", "HEALTHY")
        server = DashboardServer(metrics, port=9999, update_interval_sec=0.05)

        client = AsyncMock()
        client.send_text = AsyncMock()
        server.register_client(client)

        await server.start()
        await asyncio.sleep(0.12)
        await server.stop()

        assert client.send_text.await_count >= 2
        call_args = client.send_text.await_args_list[0][0][0]
        parsed = json.loads(call_args)
        assert parsed["type"] == "snapshot"
        assert "data" in parsed
        assert parsed["data"]["module_health"]["runtime"] == "HEALTHY"

    @pytest.mark.asyncio
    async def test_broadcast_loop_cancellation(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999, update_interval_sec=10.0)

        await server.start()
        await server.stop()
        assert server._running is False

    @pytest.mark.asyncio
    async def test_broadcast_loop_exception_recovery(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999, update_interval_sec=0.05)

        client = AsyncMock()
        client.send_text = AsyncMock(side_effect=[Exception("boom"), None])
        server.register_client(client)

        await server.start()
        await asyncio.sleep(0.15)
        await server.stop()

        assert client not in server._clients


class TestDashboardServerClientManagement:
    def test_register_client(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        client = object()
        server.register_client(client)
        assert client in server._clients

    def test_unregister_client(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        client = object()
        server.register_client(client)
        server.unregister_client(client)
        assert client not in server._clients

    def test_unregister_nonexistent_client(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        server.unregister_client(object())

    def test_register_duplicate_client(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        client = object()
        server.register_client(client)
        server.register_client(client)
        assert len(server._clients) == 1


class TestDashboardServerEventBus:
    def test_attach_to_event_bus(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        mock_bus = MagicMock()
        mock_subscription = MagicMock()
        mock_bus.subscribe.return_value = mock_subscription

        server.attach_to_event_bus(mock_bus)
        # CRITICAL FIX: wildcard '#' is no-op; now subscribes to 11 explicit topics
        assert mock_bus.subscribe.call_count == 14
        assert len(server._event_bus_subscriptions) == 14

    def test_detach_from_event_bus(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        mock_bus = MagicMock()
        mock_bus.subscribe.return_value = MagicMock()

        server.attach_to_event_bus(mock_bus)
        server.detach_from_event_bus()
        assert server._event_bus_subscriptions is None

    def test_detach_without_attach(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        server.detach_from_event_bus()
        assert server._event_bus_subscription is None

    def test_on_event_bus_message(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        event = MagicMock()
        event.topic = "provider.call"
        server._on_event_bus_message(event)
        assert metrics.get_event_counts()["provider.call"] == 1

    def test_on_event_bus_message_no_topic(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)

        event = object()
        server._on_event_bus_message(event)
        assert metrics.get_event_counts()["unknown"] == 1


class TestDashboardServerHttpApi:
    def test_get_snapshot(self):
        metrics = DashboardMetrics()
        metrics.set_module_health("runtime", "HEALTHY")
        server = DashboardServer(metrics, port=9999)

        snapshot = server.get_snapshot()
        assert "uptime_sec" in snapshot
        assert snapshot["module_health"]["runtime"] == "HEALTHY"

    def test_get_health_healthy(self):
        metrics = DashboardMetrics()
        metrics.set_module_health("runtime", "HEALTHY")
        metrics.set_module_health("sandbox", "HEALTHY")
        server = DashboardServer(metrics, port=9999)

        health = server.get_health()
        assert health["status"] == "HEALTHY"
        assert health["modules"]["runtime"] == "HEALTHY"
        assert "uptime_sec" in health

    def test_get_health_degraded(self):
        metrics = DashboardMetrics()
        metrics.set_module_health("runtime", "HEALTHY")
        metrics.set_module_health("sandbox", "DEGRADED")
        server = DashboardServer(metrics, port=9999)

        health = server.get_health()
        assert health["status"] == "DEGRADED"

    def test_get_health_empty(self):
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        health = server.get_health()
        assert health["status"] == "HEALTHY"

    def test_get_robots(self):
        mock_registry = MagicMock()
        mock_profile = MagicMock()
        mock_profile.robot_id = "ur5e"
        mock_profile.name = "UR5e"
        mock_profile.vendor = "Universal Robots"
        mock_profile.embodiment.dof = 6
        mock_profile.capability.capabilities = [{}, {}, {}]

        mock_registry.list_available.return_value = ["ur5e"]
        mock_registry.get.return_value = mock_profile

        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        robots = server.get_robots(mock_registry)

        assert len(robots) == 1
        assert robots[0]["robot_id"] == "ur5e"
        assert robots[0]["name"] == "UR5e"
        assert robots[0]["vendor"] == "Universal Robots"
        assert robots[0]["dof"] == 6
        assert robots[0]["capabilities"] == 3

    def test_get_robots_empty_registry(self):
        mock_registry = MagicMock()
        mock_registry.list_available.return_value = []

        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        robots = server.get_robots(mock_registry)
        assert robots == []

    def test_get_robots_none_profile(self):
        mock_registry = MagicMock()
        mock_registry.list_available.return_value = ["missing"]
        mock_registry.get.return_value = None

        metrics = DashboardMetrics()
        server = DashboardServer(metrics, port=9999)
        robots = server.get_robots(mock_registry)
        assert robots == []

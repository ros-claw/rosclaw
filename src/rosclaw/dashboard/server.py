"""DashboardServer — FastAPI + WebSocket real-time monitoring server."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from .metrics import DashboardMetrics


class DashboardServer:
    """Lightweight dashboard server for ROSClaw runtime monitoring.

    Usage:
        metrics = DashboardMetrics()
        server = DashboardServer(metrics, host="0.0.0.0", port=8765)
        await server.start()
        # ... runtime events feed into metrics ...
        await server.stop()
    """

    def __init__(
        self,
        metrics: DashboardMetrics,
        host: str = "0.0.0.0",
        port: int = 8765,
        update_interval_sec: float = 1.0,
    ):
        self.metrics = metrics
        self.host = host
        self.port = port
        self.update_interval_sec = update_interval_sec
        self._clients: set[Any] = set()  # WebSocket clients
        self._task: asyncio.Task | None = None
        self._running = False
        self._event_bus_subscription: Any | None = None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the dashboard broadcast loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ── WebSocket client management ──

    def register_client(self, client: Any) -> None:
        """Register a WebSocket client for broadcast."""
        self._clients.add(client)

    def unregister_client(self, client: Any) -> None:
        """Unregister a WebSocket client."""
        self._clients.discard(client)

    # ── EventBus integration ──

    def attach_to_event_bus(self, event_bus: Any) -> None:
        """Subscribe to EventBus for live event streaming."""
        # Subscribe to all critical topics explicitly (EventBus uses exact-match).
        self._event_bus_subscriptions = []
        for topic in [
            "rosclaw.runtime.started",
            "skill.execution.start",
            "skill.execution.complete",
            "praxis.completed",
            "praxis.failed",
            "rosclaw.practice.event.created",
            "rosclaw.sandbox.episode.started",
            "rosclaw.sandbox.episode.finished",
            "rosclaw.sandbox.action.blocked",
            "rosclaw.provider.inference.completed",
            "rosclaw.critic.success.detected",
            "rosclaw.dashboard.trace.updated",
            "rosclaw.how.recovery_hint.generated",
            "rosclaw.memory.write.completed",
            "rosclaw.auto.proposal.created",
            "rosclaw.auto.champion.promoted",
            "rosclaw.auto.experiment.completed",
            "rosclaw.auto.deadend.registered",
            "rosclaw.how.evidence.generated",
        ]:
            self._event_bus_subscriptions.append(
                event_bus.subscribe(topic, self._on_event_bus_message)
            )

    def detach_from_event_bus(self) -> None:
        """Unsubscribe from EventBus."""
        if hasattr(self, '_event_bus_subscriptions') and self._event_bus_subscriptions is not None:
            # EventBus unsubscribe API varies by implementation
            self._event_bus_subscriptions = None

    def _on_event_bus_message(self, event: Any) -> None:
        """Handle incoming EventBus events — update metrics AND broadcast live."""
        topic = getattr(event, "topic", "unknown")
        self.metrics.increment_event(topic, getattr(event, "payload", None))

        # Record full traces for dashboard display
        if topic == "rosclaw.dashboard.trace.updated":
            payload = getattr(event, "payload", {})
            if isinstance(payload, dict):
                self.metrics.record_trace(payload)

        # NOTE: Do NOT broadcast directly from sync callback.
        # The _broadcast_loop already pushes snapshots periodically.
        # Direct async calls from sync EventBus callbacks fail when no event loop is running.

    # ── HTTP API helpers ──

    def get_snapshot(self) -> dict[str, Any]:
        """Return current metrics snapshot (for HTTP polling)."""
        return self.metrics.snapshot()

    def get_health(self) -> dict[str, Any]:
        """Return simplified health status."""
        health = self.metrics.get_module_health()
        overall = "HEALTHY" if all(v == "HEALTHY" for v in health.values()) else "DEGRADED"
        return {
            "status": overall,
            "modules": health,
            "uptime_sec": round(self.metrics.get_uptime_sec(), 1),
        }

    def get_robots(self, registry: Any) -> list[dict[str, Any]]:
        """Return robot registry summary."""
        robots = []
        for rid in registry.list_available():
            profile = registry.get(rid)
            if profile is not None:
                robots.append({
                    "robot_id": profile.robot_id,
                    "name": profile.name,
                    "vendor": profile.vendor,
                    "dof": profile.embodiment.dof,
                    "capabilities": len(profile.capability.capabilities),
                })
        return robots


    # ── Auto Evolution API ──

    def get_auto_proposals(self) -> list[dict[str, Any]]:
        """Return auto proposals from metrics store."""
        return self.metrics._auto_proposals

    def get_auto_experiments(self) -> list[dict[str, Any]]:
        """Return auto experiments from metrics store."""
        return self.metrics._auto_experiments

    def get_auto_champions(self) -> list[dict[str, Any]]:
        """Return champion skills from metrics store."""
        return self.metrics._auto_champions

    def get_auto_deadends(self) -> list[dict[str, Any]]:
        """Return dead ends from metrics store."""
        return self.metrics._auto_deadends

    def get_evidence_trace(self, injection_id: str) -> dict[str, Any] | None:
        """Return evidence trace by injection_id."""
        for ev in self.metrics._evidence_traces:
            if ev.get("injection_id") == injection_id:
                return ev
        return None

    # ── Internal broadcast ──

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast metrics snapshot to all connected clients."""
        while self._running:
            try:
                snapshot = self.metrics.snapshot()
                message = json.dumps({"type": "snapshot", "data": snapshot})
                await self._broadcast(message)
                await asyncio.sleep(self.update_interval_sec)
            except asyncio.CancelledError:
                break
            except Exception:
                # Don't crash the broadcast loop on client errors
                await asyncio.sleep(self.update_interval_sec)

    async def _broadcast(self, message: str) -> None:
        """Send message to all connected WebSocket clients."""
        dead_clients = set()
        for client in self._clients:
            try:
                # Client must have a send_text or send method
                if hasattr(client, "send_text"):
                    await client.send_text(message)
                elif hasattr(client, "send"):
                    await client.send(message)
            except Exception:
                dead_clients.add(client)
        for dead in dead_clients:
            self._clients.discard(dead)

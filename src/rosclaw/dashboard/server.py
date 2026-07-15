"""DashboardServer — FastAPI + WebSocket real-time monitoring server."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from rosclaw.core.event_topics import EventTopics
from rosclaw.firstboot.workspace import get_rosclaw_home

from .firstboot import get_firstboot_state
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
            EventTopics.RUNTIME_STARTED,
            EventTopics.SKILL_EXECUTION_START,
            EventTopics.SKILL_EXECUTION_COMPLETE,
            EventTopics.PRAXIS_COMPLETED,
            EventTopics.PRAXIS_FAILED,
            EventTopics.PRACTICE_EVENT_CREATED,
            EventTopics.SANDBOX_EPISODE_STARTED,
            EventTopics.SANDBOX_EPISODE_FINISHED,
            EventTopics.SANDBOX_ACTION_BLOCKED,
            EventTopics.PROVIDER_INFERENCE_COMPLETED,
            EventTopics.TRACE_SPAN_STARTED,
            EventTopics.TRACE_SPAN_COMPLETED,
            EventTopics.TRACE_SPAN_FAILED,
            EventTopics.CRITIC_SUCCESS_DETECTED,
            EventTopics.DASHBOARD_TRACE_UPDATED,
            EventTopics.HOW_RECOVERY_HINT_GENERATED,
            EventTopics.MEMORY_WRITE_COMPLETED,
            "rosclaw.auto.proposal.created",
            "rosclaw.auto.champion.promoted",
            "rosclaw.auto.experiment.completed",
            "rosclaw.auto.deadend.registered",
            "rosclaw.how.evidence.generated",
            EventTopics.SENSE_STATE_UPDATED,
            EventTopics.SENSE_BODY_UPDATED,
            EventTopics.SENSE_EVENT_DETECTED,
            EventTopics.SENSE_READINESS_UPDATED,
            EventTopics.SENSE_CAPABILITY_BLOCKED,
            EventTopics.SENSE_CAPABILITY_DEGRADED,
        ]:
            self._event_bus_subscriptions.append(
                event_bus.subscribe(topic, self._on_event_bus_message)
            )

    def detach_from_event_bus(self) -> None:
        """Unsubscribe from EventBus."""
        if hasattr(self, "_event_bus_subscriptions") and self._event_bus_subscriptions is not None:
            # EventBus unsubscribe API varies by implementation
            self._event_bus_subscriptions = None

    def _on_event_bus_message(self, event: Any) -> None:
        """Handle incoming EventBus events — update metrics AND broadcast live."""
        topic = getattr(event, "topic", "unknown")
        self.metrics.increment_event(topic, getattr(event, "payload", None))

        # Record full traces for dashboard display
        if topic in {
            EventTopics.DASHBOARD_TRACE_UPDATED,
            EventTopics.TRACE_SPAN_COMPLETED,
            EventTopics.TRACE_SPAN_FAILED,
        }:
            payload = getattr(event, "payload", {})
            if isinstance(payload, dict):
                self.metrics.record_trace(payload)

        # Record BodySense snapshots so the dashboard exposes live body state
        if topic == EventTopics.SENSE_BODY_UPDATED:
            payload = getattr(event, "payload", None)
            if isinstance(payload, dict):
                self.metrics.record_body_sense(payload)

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

    def get_sense(self) -> dict[str, Any]:
        """Return current body sense stats (for HTTP polling)."""
        return self.metrics.get_body_sense_stats()

    def get_robots(self, registry: Any) -> list[dict[str, Any]]:
        """Return robot registry summary."""
        robots = []
        for rid in registry.list_available():
            profile = registry.get(rid)
            if profile is not None:
                robots.append(
                    {
                        "robot_id": profile.robot_id,
                        "name": profile.name,
                        "vendor": profile.vendor,
                        "dof": profile.embodiment.dof,
                        "capabilities": len(profile.capability.capabilities),
                    }
                )
        return robots

    def get_body_summary(self, workspace: Path | str | None = None) -> dict[str, Any]:
        """Return body registry summary plus latest sense and compatibility."""
        from rosclaw.body.registry import BodyRegistryManager
        from rosclaw.body.resolver import BodyResolver

        ws = Path(workspace) if workspace else get_rosclaw_home()
        try:
            manager = BodyRegistryManager(ws)
            bodies = manager.list_bodies()
            current_id = manager.get_current_body_id()
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "workspace": str(ws)}

        current_body: dict[str, Any] | None = None
        compatibility: dict[str, Any] | None = None
        effective_body_hash: str | None = None
        body_instance_id: str | None = None
        readiness: dict[str, Any] | None = None
        capabilities: dict[str, Any] | None = None
        forbidden_capabilities: list[str] | None = None
        if current_id and manager.has_body(current_id):
            with contextlib.suppress(Exception):
                resolver = BodyResolver(ws, body_id=current_id)
                effective = resolver.get_effective_body(recompile_if_stale=False)
                effective_body_hash = effective.effective_body_hash
                body_instance_id = effective.body_instance_id
                readiness = effective.readiness
                capabilities = effective.capabilities
                forbidden = [
                    item.get("id", item.get("capability", "unknown"))
                    for item in effective.forbidden_capabilities or []
                ]
                forbidden_capabilities = forbidden
                current_body = {
                    "body_id": resolver.body_id,
                    "linked": resolver.is_linked(),
                    "body_dir": str(resolver.body_dir),
                    "sensors": list(effective.sensors.keys()),
                    "actuators": list(effective.actuators.keys()),
                }
                compatibility = resolver.get_skill_compatibility().to_dict()

        return {
            "current": current_id,
            "workspace": str(ws),
            "bodies": [b.to_dict() for b in bodies],
            "sense": self.metrics.get_body_sense_stats(),
            "current_body": current_body,
            "compatibility": compatibility,
            "effective_body_hash": effective_body_hash,
            "body_instance_id": body_instance_id,
            "readiness": readiness,
            "capabilities": capabilities,
            "forbidden_capabilities": forbidden_capabilities,
        }

    def get_body_effective(self, workspace: Path | str | None = None) -> dict[str, Any]:
        """Return the current effective body as a dict."""
        from rosclaw.body.registry import BodyRegistryManager
        from rosclaw.body.resolver import BodyResolver

        ws = Path(workspace) if workspace else get_rosclaw_home()
        try:
            manager = BodyRegistryManager(ws)
            current_id = manager.get_current_body_id()
            if not current_id or not manager.has_body(current_id):
                return {"error": "No active body", "workspace": str(ws)}
            resolver = BodyResolver(ws, body_id=current_id)
            effective = resolver.get_effective_body(recompile_if_stale=False)
            return {
                "body_instance_id": effective.body_instance_id,
                "effective_body_hash": effective.effective_body_hash,
                "eurdf_uri": effective.eurdf_uri,
                "compiled_at": effective.compiled_at,
                "generation": effective.generation,
                "frames": effective.frames,
                "joints": effective.joints,
                "sensors": effective.sensors,
                "actuators": effective.actuators,
                "capabilities": effective.capabilities,
                "safety": effective.safety,
                "readiness": effective.readiness,
                "workspace": str(ws),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "workspace": str(ws)}

    def get_body_skills(self, workspace: Path | str | None = None) -> dict[str, Any]:
        """Return skill compatibility summary for the active body."""
        from rosclaw.body.registry import BodyRegistryManager
        from rosclaw.body.resolver import BodyResolver

        ws = Path(workspace) if workspace else get_rosclaw_home()
        try:
            manager = BodyRegistryManager(ws)
            current_id = manager.get_current_body_id()
            if not current_id or not manager.has_body(current_id):
                return {"error": "No active body", "workspace": str(ws)}
            resolver = BodyResolver(ws, body_id=current_id)
            report = resolver.get_skill_compatibility()
            return {
                "body_instance_id": report.body_instance_id,
                "effective_body_hash": report.effective_body_hash,
                "checked_at": report.checked_at,
                "summary": report.summary,
                "skills": {k: v.to_dict() for k, v in report.skills.items()},
                "workspace": str(ws),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "workspace": str(ws)}

    def get_body_history(
        self,
        workspace: Path | str | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Return recent maintenance events for the active body."""
        from rosclaw.body.registry import BodyRegistryManager
        from rosclaw.body.resolver import BodyResolver

        ws = Path(workspace) if workspace else get_rosclaw_home()
        try:
            manager = BodyRegistryManager(ws)
            current_id = manager.get_current_body_id()
            if not current_id or not manager.has_body(current_id):
                return {"error": "No active body", "workspace": str(ws)}
            resolver = BodyResolver(ws, body_id=current_id)
            events = resolver.get_maintenance_events()[-limit:]
            return {
                "body_instance_id": current_id,
                "count": len(events),
                "events": [e.to_dict() for e in events],
                "workspace": str(ws),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "workspace": str(ws)}

    def get_body_provider_health(self, workspace: Path | str | None = None) -> dict[str, Any]:
        """Return provider diagnosis for the active body."""
        from rosclaw.body.registry import BodyRegistryManager
        from rosclaw.body.resolver import BodyResolver
        from rosclaw.provider.body_binder import ProviderBodyBinder

        ws = Path(workspace) if workspace else get_rosclaw_home()
        try:
            manager = BodyRegistryManager(ws)
            current_id = manager.get_current_body_id()
            if not current_id or not manager.has_body(current_id):
                return {"error": "No active body", "workspace": str(ws)}
            resolver = BodyResolver(ws, body_id=current_id)
            effective = resolver.get_effective_body(recompile_if_stale=False)
            binder = ProviderBodyBinder.from_effective_body(effective)
            diagnosis = binder.diagnose()
            return {
                "body_instance_id": diagnosis.body_instance_id,
                "effective_body_hash": diagnosis.effective_body_hash,
                "status": diagnosis.status,
                "interfaces": diagnosis.interfaces,
                "summary": diagnosis.summary,
                "timestamp": diagnosis.timestamp,
                "workspace": str(ws),
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc), "workspace": str(ws)}

    def get_firstboot_state(self, workspace: Path | str | None = None) -> dict[str, Any]:
        """Return First Boot state for the dashboard wizard."""
        ws = Path(workspace) if workspace else None
        return get_firstboot_state(ws)

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
                snapshot["body"] = self.get_body_summary()
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

"""KnowledgeBatchEngine — runtime-side wrapper around rosclaw_know.

Listens to EventBus topics that warrant a catalog update and calls
the in-memory ``rosclaw_know.sim_ingest.bridge_direct`` direct path.

EventBus contract (audit-know.md §2.3 + v1.5 INTEGRATION_DESIGN.md):

  SUBSCRIBES TO
    rosclaw.runtime.execution.completed   — one episode of agent work done
    rosclaw.runtime.execution.failed      — execution failed (still ingestable)
    rosclaw.sandbox.episode.finished      — sandbox rollout done
    rosclaw.sandbox.episode.failed        — sandbox rollout failed
    rosclaw.knowledge.ingest_request      — manual batch trigger

  PUBLISHES
    rosclaw.knowledge.assets_refreshed    — after bridge_index update
    rosclaw.knowledge.ingest_progress     — heartbeat / progress

Sprint 12's direct path collapses the entire batch side to a few
lines: each incoming payload is normalised into one or more
``RobotEvent`` objects, then ``reweight_bridge_from_robot_events``
does the rest.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventPriority
from rosclaw.core.lifecycle import LifecycleMixin

logger = logging.getLogger("rosclaw.know.batch_engine")

try:
    from rosclaw_know.sim_ingest import reweight_bridge_from_robot_events
    from rosclaw_know.sim_ingest.event_schema import EVENT_TYPES, RobotEvent

    _V15_AVAILABLE = True
except ImportError as exc:
    logger.info("rosclaw-know v1.5 not available (%s); batch engine inert", exc)
    _V15_AVAILABLE = False


# Map of runtime failure_type / signature strings → canonical event_type.
# Loose by design: missing matches fall through to "controller_error".
_FAILURE_TYPE_TO_EVENT_TYPE = {
    "collision": "collision",
    "contact": "collision",
    "limit": "joint_limit_violation",
    "joint_limit": "joint_limit_violation",
    "safety_stop": "safety_stop",
    "estop": "safety_stop",
    "torque": "actuator_saturation",
    "saturation": "actuator_saturation",
    "sensor": "sensor_outlier",
    "outlier": "sensor_outlier",
    "timeout": "task_timeout",
    "deviation": "trajectory_deviation",
    "tracking_error": "trajectory_deviation",
}


def _infer_event_type(payload: dict[str, Any]) -> str:
    """Map a runtime payload to one of the canonical EVENT_TYPES.

    Returns ``"controller_error"`` (a safe catch-all) when no signal
    matches.  Guarded against the rosclaw_know import failing.
    """
    if not _V15_AVAILABLE:
        return "controller_error"
    raw = (
        payload.get("event_type")
        or payload.get("failure_type")
        or payload.get("error_log", "")[:200]
        or ""
    )
    raw_lower = raw.lower()
    for needle, et in _FAILURE_TYPE_TO_EVENT_TYPE.items():
        if needle in raw_lower:
            return et if et in EVENT_TYPES else "controller_error"
    return "controller_error"


def _payload_to_robot_events(
    payload: dict[str, Any], *, default_source: str = "runtime"
) -> list[Any]:
    """Tolerant adapter: runtime payload → ``list[RobotEvent]``.

    Always returns at most one RobotEvent today (one episode = one
    summary event).  A future revision can split into per-violation
    events when the runtime emits the per-violation envelope.
    """
    if not _V15_AVAILABLE or not isinstance(payload, dict):
        return []
    import time

    event_type = _infer_event_type(payload)
    embodiment_id = (
        payload.get("embodiment_id")
        or payload.get("robot_id")
        or payload.get("agent_id")
        or "unknown"
    )
    severity = payload.get("severity", "warning")
    timestamp = payload.get("timestamp")
    if timestamp is None:
        timestamp = str(time.time())
    else:
        timestamp = str(timestamp)

    # Pass everything else through as `fields` for downstream extractors.
    reserved = {
        "embodiment_id", "robot_id", "agent_id",
        "severity", "timestamp", "event_type", "failure_type",
        "source", "source_id", "fingerprint",
    }
    fields = {k: v for k, v in payload.items() if k not in reserved}

    return [
        RobotEvent(
            timestamp=timestamp,
            event_type=event_type,
            embodiment_id=embodiment_id,
            severity=severity,
            fingerprint=payload.get("fingerprint", ""),
            fields=fields,
            source=payload.get("source", default_source),
            source_id=payload.get("source_id", ""),
        )
    ]


class KnowledgeBatchEngine(LifecycleMixin):
    """EventBus-triggered v1.5 catalog updater.

    Owned by ``Runtime`` when ``config.enable_knowledge=True`` AND
    ``rosclaw_know`` is importable.  Inert when the package is absent.
    """

    SUBSCRIPTIONS: tuple[str, ...] = (
        "rosclaw.runtime.execution.completed",
        "rosclaw.runtime.execution.failed",
        "rosclaw.sandbox.episode.finished",
        "rosclaw.sandbox.episode.failed",
        "rosclaw.knowledge.ingest_request",
    )

    def __init__(
        self,
        runtime: Any,
        assets_path: str | Path = "data/knowledge_assets",
    ) -> None:
        super().__init__()
        self.runtime = runtime
        self.assets_path = Path(assets_path)
        self.bridge_path = self.assets_path / "bridge_index.json"
        self.metrics_path = self.assets_path / "pattern_metrics.json"
        self._batches_processed = 0
        self._last_summary: dict[str, int] = {}

    # -- lifecycle ------------------------------------------------------

    def _do_initialize(self) -> None:
        if not _V15_AVAILABLE:
            logger.info(
                "[KnowBatch] rosclaw-know not installed; "
                "batch engine is inert.  install with: pip install rosclaw-know"
            )
            return
        self.assets_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            "[KnowBatch] Initialized; bridge=%s metrics=%s",
            self.bridge_path, self.metrics_path,
        )

    def _do_start(self) -> None:
        if not _V15_AVAILABLE:
            return
        bus = self.runtime.event_bus if hasattr(self.runtime, "event_bus") else None
        if bus is None:
            logger.warning("[KnowBatch] runtime has no event_bus; staying inert")
            return
        for topic in self.SUBSCRIPTIONS:
            bus.subscribe(topic, self._on_event)
        logger.info("[KnowBatch] Subscribed to %d topics", len(self.SUBSCRIPTIONS))

    # -- event handlers -------------------------------------------------

    def _on_event(self, event: Event) -> None:
        try:
            self._ingest(event)
        except Exception as exc:  # noqa: BLE001
            # NEVER let a batch failure crash the runtime.
            logger.warning("[KnowBatch] ingest failed on %s: %s", event.topic, exc)

    def _ingest(self, event: Event) -> None:
        payload = event.payload if isinstance(event.payload, dict) else {}
        robot_events = _payload_to_robot_events(
            payload, default_source=event.source or "runtime"
        )
        if not robot_events:
            return

        summary, coverage = reweight_bridge_from_robot_events(
            robot_events,
            bridge_path=self.bridge_path,
            metrics_path=self.metrics_path,
        )
        self._batches_processed += 1
        self._last_summary = summary

        # Tell the query side to reload bridge_index.json in-place.
        knowledge = getattr(self.runtime, "_knowledge", None)
        if knowledge is not None and self.bridge_path.exists():
            try:
                knowledge._load_bridge_index(self.bridge_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[KnowBatch] reload of bridge_index failed: %s", exc)

        # Tell the task-pack adapter its YAML cache is stale.
        try:
            from rosclaw.know.task_pack_adapter import reload_assets
            reload_assets()
        except ImportError:
            pass

        # Announce.
        if hasattr(self.runtime, "event_bus") and self.runtime.event_bus is not None:
            self.runtime.event_bus.publish(Event(
                topic="rosclaw.knowledge.assets_refreshed",
                payload={
                    "summary": summary,
                    "coverage_violations": list(coverage.violations)
                        if coverage and hasattr(coverage, "violations") else [],
                    "batches_processed": self._batches_processed,
                    "source_topic": event.topic,
                },
                source="knowledge.batch_engine",
                priority=EventPriority.NORMAL,
                trace_id=getattr(event, "trace_id", ""),
            ))

    # -- introspection --------------------------------------------------

    @property
    def is_active(self) -> bool:
        return _V15_AVAILABLE

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "active": _V15_AVAILABLE,
            "batches_processed": self._batches_processed,
            "last_summary": dict(self._last_summary),
            "bridge_path": str(self.bridge_path),
        }

"""Sandbox trace adapter for capturing sandbox decisions via EventBus.

Subscribes to ``rosclaw.sandbox.decision`` and emits practice events with
file-backed references.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.practice.adapters.base import SourceAdapter, SourceHealth
from rosclaw.practice.schemas import PracticeEventEnvelope

logger = logging.getLogger("rosclaw.practice.adapters.sandbox_trace")


class SandboxTraceAdapter(SourceAdapter):
    """Captures sandbox decision events from the EventBus."""

    source_name = "sandbox"

    def __init__(self, robot_id: str, event_bus: EventBus, output_root: str | None = None):
        self._robot_id = robot_id
        self._event_bus = event_bus
        self._output_root = output_root
        self._practice_id: str | None = None
        self._session_dir: Path | None = None
        self._running = False
        self._lock = threading.Lock()
        self._pending: list[PracticeEventEnvelope] = []
        self._subscribed = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, session: Any) -> None:
        self._practice_id = getattr(session, "practice_id", None)
        self._session_dir = getattr(session, "session_dir", None)
        self._running = True
        if not self._subscribed:
            self._event_bus.subscribe("rosclaw.sandbox.decision", self._on_event)
            self._subscribed = True

    def stop(self) -> None:
        self._running = False
        if self._subscribed:
            try:
                self._event_bus.unsubscribe("rosclaw.sandbox.decision", self._on_event)
            except Exception as e:
                logger.warning("Error unsubscribing sandbox events: %s", e)
            self._subscribed = False

    def health(self) -> SourceHealth:
        return SourceHealth(source=self.source_name, healthy=self._running)

    def poll(self) -> Iterable[PracticeEventEnvelope]:
        with self._lock:
            batch = self._pending[:]
            self._pending.clear()
        return batch

    # ------------------------------------------------------------------
    # EventBus handler
    # ------------------------------------------------------------------

    def _on_event(self, event: Event) -> None:
        if not self._running or self._practice_id is None:
            return

        payload = event.payload if isinstance(event.payload, dict) else {}
        ts_ns = int(time.monotonic() * 1_000_000_000)
        decision_id = payload.get("decision_id", event.event_id)

        line = {
            "timestamp_ns": ts_ns,
            "decision_id": decision_id,
            "action_id": payload.get("action_id"),
            "requested_action": payload.get("requested_action"),
            "decision": payload.get("decision"),
            "modified_action": payload.get("modified_action"),
            "risk_score": payload.get("risk_score"),
            "rules_triggered": payload.get("rules_triggered"),
            "reason": payload.get("reason"),
            "policy_version": payload.get("policy_version"),
            "latency_ms": payload.get("latency_ms"),
        }

        rel_path = "sandbox/decisions.jsonl"
        out_path = self._resolve_path(rel_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        envelope = PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="sandbox",
            event_type="sandbox.decision",
            timestamp_ns=ts_ns,
            payload={"decision_id": decision_id, "decision": line["decision"]},
            payload_ref={"decisions_ref": rel_path},
        )
        with self._lock:
            self._pending.append(envelope)

    def _resolve_path(self, rel_path: str) -> Path:
        if self._session_dir is not None:
            return Path(self._session_dir) / rel_path
        if self._output_root is not None:
            return Path(self._output_root) / rel_path
        return Path(rel_path)

    def on_event(self, callback: Any) -> None:
        pass

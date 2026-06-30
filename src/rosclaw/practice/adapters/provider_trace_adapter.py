"""Provider trace adapter for capturing inference events via EventBus.

Subscribes to ``rosclaw.provider.inference.completed`` and ``.failed`` topics
and emits practice events with file-backed references.
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

logger = logging.getLogger("rosclaw.practice.adapters.provider_trace")


class ProviderTraceAdapter(SourceAdapter):
    """Captures provider inference events from the EventBus."""

    source_name = "provider"

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
            self._event_bus.subscribe("rosclaw.provider.inference.completed", self._on_event)
            self._event_bus.subscribe("rosclaw.provider.inference.failed", self._on_event)
            self._subscribed = True

    def stop(self) -> None:
        self._running = False
        if self._subscribed:
            try:
                self._event_bus.unsubscribe("rosclaw.provider.inference.completed", self._on_event)
                self._event_bus.unsubscribe("rosclaw.provider.inference.failed", self._on_event)
            except Exception as e:
                logger.warning("Error unsubscribing provider events: %s", e)
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
        request_id = payload.get("request_id", event.event_id)

        request_line = {
            "timestamp_ns": ts_ns,
            "request_id": request_id,
            "provider_id": payload.get("provider_id"),
            "model": payload.get("model"),
            "input_summary": payload.get("input_summary"),
        }
        response_line = {
            "timestamp_ns": ts_ns,
            "request_id": request_id,
            "status": "success" if event.topic.endswith(".completed") else "failed",
            "output_summary": payload.get("output_summary"),
            "latency_ms": payload.get("latency_ms"),
            "token_usage": payload.get("token_usage"),
            "error": payload.get("error"),
        }

        req_rel = "provider/requests.jsonl"
        resp_rel = "provider/responses.jsonl"
        req_path = self._resolve_path(req_rel)
        resp_path = self._resolve_path(resp_rel)
        req_path.parent.mkdir(parents=True, exist_ok=True)
        resp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(req_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(request_line, ensure_ascii=False) + "\n")
        with open(resp_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(response_line, ensure_ascii=False) + "\n")

        envelope = PracticeEventEnvelope(
            practice_id=self._practice_id,
            robot_id=self._robot_id,
            source="provider",
            event_type="provider.inference",
            timestamp_ns=ts_ns,
            payload={"request_id": request_id, "status": response_line["status"]},
            payload_ref={"requests_ref": req_rel, "responses_ref": resp_rel},
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

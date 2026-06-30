"""RuntimeBus — schema-aware wrapper around the legacy EventBus.

The RuntimeBus adds:
- RuntimeEvent envelope conversion
- Per-topic schema registry and validation
- Prefix subscriptions
- Replay from JsonlEventSink logs and in-memory history
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventBus, get_global_event_bus
from rosclaw.core.event_sink import JsonlEventSink
from rosclaw.runtime.event import RuntimeEvent

logger = logging.getLogger("rosclaw.runtime.bus")


class SchemaValidationError(Exception):
    """Raised when a published event fails schema validation."""


Callback = Callable[[RuntimeEvent], None]
AsyncCallback = Callable[[RuntimeEvent], Coroutine[Any, Any, None]]


class RuntimeBus:
    """Schema-aware event bus for the Runtime Kernel.

    Wraps the existing in-process EventBus and adds:
    - RuntimeEvent normalization
    - optional Pydantic payload schemas per event type
    - replay from the JSONL event sink
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        event_sink: JsonlEventSink | None = None,
    ):
        self._bus = event_bus or get_global_event_bus()
        self._sink = event_sink
        self._schemas: dict[str, Callable[[Any], Any]] = {}
        self._aliases: dict[str, str] = {
            # Backward-compatible aliases: legacy event name -> canonical v2 type.
            "skill.execution.complete": "skill.complete",
            "skill.execution.start": "skill.invoke",
            "practice.session_started": "practice.start",
            "practice.session_finished": "practice.stop",
            "provider.inference.completed": "provider.result",
        }
        # We keep our own callback bookkeeping so we can wrap legacy callbacks.
        self._wrapped: dict[Callback | AsyncCallback, Callable[[Event], None]] = {}

    # ------------------------------------------------------------------
    # Schema registry
    # ------------------------------------------------------------------
    def register_schema(self, event_type: str, schema_cls: Callable[[Any], Any]) -> None:
        """Register a schema callable (e.g. Pydantic model) for an event type."""
        self._schemas[event_type] = schema_cls

    def register_alias(self, alias: str, canonical_type: str) -> None:
        """Map a legacy event type/name to a canonical runtime type."""
        self._aliases[alias] = canonical_type

    def _canonical_type(self, event_type: str) -> str:
        if event_type.startswith("rosclaw."):
            event_type = event_type[len("rosclaw.") :]
        if event_type in self._schemas:
            return event_type
        return self._aliases.get(event_type, event_type)

    def _topic(self, event_type: str) -> str:
        canonical = self._canonical_type(event_type)
        return f"rosclaw.{canonical}"

    def _prefix_topic(self, prefix: str) -> str:
        """Return the raw bus topic pattern for a runtime type prefix."""
        if prefix in ("*", "**", "#"):
            return "#"
        if prefix.startswith("rosclaw."):
            return prefix
        topic_pattern = f"rosclaw.{prefix}"
        if not any(ch in topic_pattern for ch in "*?"):
            topic_pattern = f"{topic_pattern}.*"
        return topic_pattern

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish(self, event: RuntimeEvent) -> None:
        """Publish a RuntimeEvent after optional schema validation."""
        canonical = self._canonical_type(event.type)
        schema = self._schemas.get(canonical)
        if schema is not None and event.payload:
            try:
                event.payload = schema(**event.payload).model_dump()
            except Exception as exc:
                raise SchemaValidationError(
                    f"Payload validation failed for {canonical}: {exc}"
                ) from exc

        event.type = canonical
        bus_event = Event(
            topic=self._topic(canonical),
            payload=event.to_event_bus_payload(),
            source=event.source,
            trace_id=event.metadata.get("trace_id") or event.id,
            metadata=event.metadata,
        )
        self._bus.publish(bus_event)

    async def publish_async(self, event: RuntimeEvent) -> None:
        """Async wrapper around publish."""
        self.publish(event)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def subscribe(self, event_type: str, callback: Callback) -> None:
        """Subscribe to a specific event type (sync callback)."""
        topic = self._topic(event_type)

        def wrapper(bus_event: Event) -> None:
            try:
                runtime_event = RuntimeEvent.from_event_bus_payload(
                    bus_event.payload, topic=bus_event.topic
                )
            except Exception as exc:
                logger.warning("Failed to deserialize runtime event: %s", exc)
                return
            try:
                callback(runtime_event)
            except Exception as e:
                logger.warning("Error in runtime subscriber for %s: %s", event_type, e)

        self._wrapped[callback] = wrapper
        self._bus.subscribe(topic, wrapper)

    def subscribe_prefix(self, prefix: str, callback: Callback) -> None:
        """Subscribe to all event types matching a dotted prefix.

        ``camera.*`` matches ``camera.rgbd_frame``, ``camera.depth``, etc.
        ``*`` matches every runtime event.
        """
        topic_pattern = self._prefix_topic(prefix)

        def wrapper(bus_event: Event) -> None:
            try:
                runtime_event = RuntimeEvent.from_event_bus_payload(
                    bus_event.payload, topic=bus_event.topic
                )
            except Exception as exc:
                logger.warning("Failed to deserialize runtime event: %s", exc)
                return
            if fnmatch.fnmatch(runtime_event.type, prefix):
                try:
                    callback(runtime_event)
                except Exception as e:
                    logger.warning("Error in runtime subscriber for %s: %s", prefix, e)

        self._wrapped[callback] = wrapper
        self._bus.subscribe(topic_pattern, wrapper)

    def subscribe_async(self, event_type: str, callback: AsyncCallback) -> None:
        """Subscribe to a specific event type (async callback)."""
        topic = self._topic(event_type)

        async def wrapper(bus_event: Event) -> None:
            try:
                runtime_event = RuntimeEvent.from_event_bus_payload(
                    bus_event.payload, topic=bus_event.topic
                )
            except Exception as exc:
                logger.warning("Failed to deserialize runtime event: %s", exc)
                return
            try:
                await callback(runtime_event)
            except Exception as e:
                logger.warning("Error in async runtime subscriber for %s: %s", event_type, e)

        self._wrapped[callback] = wrapper  # type: ignore[assignment]
        self._bus.subscribe_async(topic, wrapper)

    def unsubscribe(
        self, event_type: str, callback: Callback | AsyncCallback, *, is_prefix: bool = False
    ) -> None:
        """Unsubscribe a callback from an event type or prefix."""
        wrapper = self._wrapped.pop(callback, None)
        if wrapper is None:
            return
        if is_prefix:
            topic = self._prefix_topic(event_type)
        else:
            topic = self._topic(event_type)
        self._bus.unsubscribe(topic, wrapper)

    # ------------------------------------------------------------------
    # Query / replay
    # ------------------------------------------------------------------
    def get_history(
        self, event_type: str | None = None, limit: int = 100
    ) -> list[RuntimeEvent]:
        """Return recent in-memory history as RuntimeEvents."""
        topic = self._topic(event_type) if event_type else None
        events = self._bus.get_history(topic=topic, limit=limit)
        result: list[RuntimeEvent] = []
        for ev in events:
            try:
                result.append(RuntimeEvent.from_event_bus_payload(ev.payload, topic=ev.topic))
            except Exception as exc:
                logger.debug("Skipping unparseable history event: %s", exc)
        return result

    def replay(
        self,
        event_type: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        trace_id: str | None = None,
        limit: int = 1000,
    ) -> list[RuntimeEvent]:
        """Replay events from the JSONL event sink, filtered by type/time/trace."""
        logs = []
        if self._sink is not None:
            logs = self._sink_log_records()
        # Also include in-memory history.
        for ev in self._bus.get_history(limit=self._bus._max_history):
            try:
                runtime_event = RuntimeEvent.from_event_bus_payload(ev.payload, topic=ev.topic)
                logs.append(runtime_event.model_dump())
            except Exception as exc:
                logger.debug("Skipping unparseable history event: %s", exc)

        results: list[RuntimeEvent] = []
        for record in logs:
            try:
                runtime_event = RuntimeEvent.from_event_bus_payload(record)
            except Exception as exc:
                logger.debug("Skipping unparseable log record: %s", exc)
                continue
            if event_type and not fnmatch.fnmatch(runtime_event.type, event_type):
                continue
            if start is not None and runtime_event.timestamp < start:
                continue
            if end is not None and runtime_event.timestamp > end:
                continue
            if trace_id is not None and runtime_event.metadata.get("trace_id") != trace_id:
                continue
            results.append(runtime_event)
            if len(results) >= limit:
                break
        return results

    def _sink_log_records(self) -> list[dict[str, Any]]:
        """Read all records from the active JSONL sink files."""
        if self._sink is None:
            return []
        home = getattr(self._sink, "_home", None)
        if home is None:
            return []
        events_dir = Path(home) / "events"
        if not events_dir.exists():
            return []
        records: list[dict[str, Any]] = []
        import json

        for path in sorted(events_dir.glob("live.jsonl*")):
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception as exc:
                logger.warning("Failed to read event log %s: %s", path, exc)
        return records

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        """Return bus statistics plus runtime schema info."""
        return {
            "schemas": list(self._schemas.keys()),
            "aliases": self._aliases,
            **self._bus.get_stats(),
        }

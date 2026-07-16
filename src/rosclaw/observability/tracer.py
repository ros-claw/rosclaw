"""Span lifecycle, EventBus streaming, and exporter fan-out."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from types import TracebackType
from typing import Any

from rosclaw.observability.context import (
    TraceContext,
    attach_trace_context,
    current_trace_context,
    detach_trace_context,
)
from rosclaw.observability.exporters.base import TraceExporter
from rosclaw.observability.redaction import TraceRedactor
from rosclaw.observability.schema import (
    ObservabilityConfig,
    SpanKind,
    SpanStatus,
    TraceRecord,
)

logger = logging.getLogger("rosclaw.observability.tracer")
_TRACER_LOCK = threading.Lock()


class Tracer:
    """Create structured spans and stream them without affecting runtime outcomes."""

    def __init__(
        self,
        event_bus: Any | None = None,
        config: ObservabilityConfig | None = None,
        exporters: list[TraceExporter] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.config = config or ObservabilityConfig()
        self.redactor = TraceRedactor(
            mode=self.config.capture_mode,
            max_text_chars=self.config.max_text_chars,
            max_collection_items=self.config.max_collection_items,
        )
        self._exporters: list[TraceExporter] = list(exporters or [])
        self._closed = False

    def configure(
        self,
        config: ObservabilityConfig,
        exporters: list[TraceExporter] | None = None,
    ) -> None:
        """Apply runtime configuration before traced operations begin."""

        self.config = config
        self.redactor = TraceRedactor(
            mode=config.capture_mode,
            max_text_chars=config.max_text_chars,
            max_collection_items=config.max_collection_items,
        )
        if exporters is not None:
            self._exporters = list(exporters)
        self._closed = False

    def start_span(
        self,
        name: str,
        kind: SpanKind | str,
        *,
        source: str = "rosclaw",
        operation: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
        mission_id: str | None = None,
        episode_id: str | None = None,
        robot_id: str | None = None,
        session_id: str | None = None,
    ) -> TraceSpan:
        parent = current_trace_context()
        # A child can never switch trace IDs while retaining the current span
        # as its parent. Explicit IDs only seed a root operation.
        resolved_trace_id = parent.trace_id if parent else trace_id
        return TraceSpan(
            tracer=self,
            name=name,
            kind=SpanKind(kind),
            source=source,
            operation=operation or name,
            trace_id=resolved_trace_id or f"trace_{uuid.uuid4().hex[:24]}",
            parent_span_id=(
                parent_span_id
                if parent_span_id is not None
                else (parent.span_id if parent else None)
            ),
            attributes=attributes or {},
            mission_id=mission_id,
            episode_id=episode_id,
            robot_id=robot_id,
            session_id=session_id,
        )

    def _publish(self, topic: str, record: TraceRecord) -> None:
        if not self.config.enabled or self.event_bus is None:
            return
        try:
            from rosclaw.core.event_bus import Event, EventPriority

            priority = (
                EventPriority.HIGH
                if record.status in {SpanStatus.ERROR, SpanStatus.BLOCKED}
                else EventPriority.LOW
            )
            self.event_bus.publish(
                Event(
                    topic=topic,
                    payload=record.to_dict(),
                    source="trace",
                    priority=priority,
                    trace_id=record.trace_id,
                    span_id=record.span_id,
                    parent_span_id=record.parent_span_id or "",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Trace EventBus publish failed: %s", exc)

    def _export(self, record: TraceRecord) -> None:
        if not self.config.enabled:
            return
        for exporter in list(self._exporters):
            try:
                exporter.export(record)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Trace exporter failed: %s", exc)

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        for exporter in list(self._exporters):
            try:
                exporter.close(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Trace exporter close failed: %s", exc)
        self._closed = True


class TraceSpan:
    """Sync/async context manager for one timed, causally-linked operation."""

    def __init__(
        self,
        *,
        tracer: Tracer,
        name: str,
        kind: SpanKind,
        source: str,
        operation: str,
        trace_id: str,
        parent_span_id: str | None,
        attributes: dict[str, Any],
        mission_id: str | None,
        episode_id: str | None,
        robot_id: str | None,
        session_id: str | None,
    ) -> None:
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.source = source
        self.operation = operation
        self.trace_id = trace_id
        self.span_id = f"span_{uuid.uuid4().hex[:16]}"
        self.parent_span_id = parent_span_id
        self.attributes = dict(attributes)
        self.mission_id = mission_id
        self.episode_id = episode_id
        self.robot_id = robot_id
        self.session_id = session_id
        self.started_at = 0.0
        self.ended_at: float | None = None
        self.status = SpanStatus.RUNNING
        self.severity = "INFO"
        self.input: Any | None = None
        self.output: Any | None = None
        self.evidence_refs: list[str] = []
        self.error: dict[str, Any] | None = None
        self._token: Any | None = None
        self._finished = False

    def __enter__(self) -> TraceSpan:
        self.started_at = time.time()
        self._token = attach_trace_context(
            TraceContext(
                trace_id=self.trace_id,
                span_id=self.span_id,
                parent_span_id=self.parent_span_id,
            )
        )
        self.tracer._publish("rosclaw.trace.span.started", self._record())
        return self

    async def __aenter__(self) -> TraceSpan:
        return self.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if exc is not None:
            self.status = (
                SpanStatus.CANCELLED if exc_type is asyncio_cancelled_error() else SpanStatus.ERROR
            )
            self.severity = "ERROR"
            self.error = {"type": type(exc).__name__, "message": str(exc)}
        elif self.status == SpanStatus.RUNNING:
            self.status = SpanStatus.OK
        self.finish()
        return False

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self.__exit__(exc_type, exc, traceback)

    def set_input(self, value: Any) -> TraceSpan:
        self.input = self.tracer.redactor.redact(value)
        return self

    def set_output(self, value: Any) -> TraceSpan:
        self.output = self.tracer.redactor.redact(value)
        return self

    def set_attribute(self, key: str, value: Any) -> TraceSpan:
        self.attributes[key] = self.tracer.redactor.redact(value)
        return self

    def set_status(self, status: SpanStatus | str, message: str | None = None) -> TraceSpan:
        self.status = SpanStatus(status)
        if self.status in {SpanStatus.ERROR, SpanStatus.BLOCKED}:
            self.severity = "ERROR" if self.status == SpanStatus.ERROR else "WARNING"
        if message:
            self.attributes["status.message"] = self.tracer.redactor.redact(message)
        return self

    def add_evidence(self, reference: str) -> TraceSpan:
        self.evidence_refs.append(reference)
        return self

    def finish(self) -> TraceRecord:
        if self._finished:
            return self._record()
        self._finished = True
        self.ended_at = time.time()
        record = self._record()
        topic = (
            "rosclaw.trace.span.failed"
            if self.status in {SpanStatus.ERROR, SpanStatus.BLOCKED, SpanStatus.CANCELLED}
            else "rosclaw.trace.span.completed"
        )
        self.tracer._publish(topic, record)
        self.tracer._export(record)
        if self._token is not None:
            detach_trace_context(self._token)
            self._token = None
        return record

    def _record(self) -> TraceRecord:
        attributes = self.tracer.redactor.redact(self.attributes)
        material = {"input": self.input, "output": self.output}
        try:
            payload_hash = hashlib.sha256(
                json.dumps(material, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
        except Exception:
            payload_hash = None
        duration_ms = (
            round((self.ended_at - self.started_at) * 1000, 3)
            if self.ended_at is not None and self.started_at
            else None
        )
        return TraceRecord(
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.parent_span_id,
            name=self.name,
            span_kind=self.kind.value,
            source=self.source,
            operation=self.operation,
            started_at=self.started_at or time.time(),
            ended_at=self.ended_at,
            duration_ms=duration_ms,
            status=self.status.value,
            severity=self.severity,
            mission_id=self.mission_id,
            episode_id=self.episode_id,
            robot_id=self.robot_id,
            session_id=self.session_id,
            input=self.input,
            output=self.output,
            attributes=attributes if isinstance(attributes, dict) else {},
            evidence_refs=list(self.evidence_refs),
            privacy_class=self.tracer.config.capture_mode.value,
            payload_hash=payload_hash,
            error=self.error,
        )


def asyncio_cancelled_error() -> type[BaseException]:
    """Import lazily to keep the tracer lightweight during module import."""

    import asyncio

    return asyncio.CancelledError


def get_tracer(event_bus: Any | None = None) -> Tracer:
    """Return the tracer shared by all components attached to one EventBus."""

    if event_bus is None:
        return Tracer()
    existing = getattr(event_bus, "_rosclaw_tracer", None)
    if isinstance(existing, Tracer):
        return existing
    with _TRACER_LOCK:
        existing = getattr(event_bus, "_rosclaw_tracer", None)
        if isinstance(existing, Tracer):
            return existing
        tracer = Tracer(event_bus=event_bus)
        event_bus._rosclaw_tracer = tracer
        return tracer

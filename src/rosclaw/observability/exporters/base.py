"""Trace exporter contract."""

from __future__ import annotations

from typing import Protocol

from rosclaw.observability.schema import TraceRecord


class TraceExporter(Protocol):
    """Non-blocking destination for completed trace records."""

    def export(self, record: TraceRecord) -> bool: ...

    def flush(self, timeout: float = 5.0) -> None: ...

    def close(self, timeout: float = 5.0) -> None: ...

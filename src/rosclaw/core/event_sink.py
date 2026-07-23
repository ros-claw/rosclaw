"""Persistent JSONL event sink for the ROSClaw EventBus.

Writes every published event to ``<ROSCLAW_HOME>/events/live.jsonl`` so that
cross-process consumers (e.g. rosclaw-dashboard) can tail the runtime event
stream without sharing the in-memory bus.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.firstboot.workspace import resolve_home

logger = logging.getLogger("rosclaw.core.event_sink")


def _summary_text(value: Any, limit: int = 256) -> str | int | float | bool | None:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)[:limit]


class JsonlEventSink:
    """Subscribe to all EventBus topics and append events to a JSONL file.

    Args:
        home: ROSClaw workspace root. Defaults to ``ROSCLAW_HOME`` or ``~/.rosclaw``.
        filename: Name of the JSONL file inside ``home/events/``.
        rotate_mb: Rotate the file when it exceeds this size.
    """

    def __init__(
        self,
        home: str | Path | None = None,
        filename: str = "live.jsonl",
        rotate_mb: float = 64.0,
        max_record_mb: float = 1.0,
    ):
        filename_value = str(filename)
        if not filename_value or Path(filename_value).name != filename_value:
            raise ValueError("Event sink filename must be a plain file name")
        for name, value in (("rotate_mb", rotate_mb), ("max_record_mb", max_record_mb)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        self._home = resolve_home(str(home) if home else None)
        self._events_dir = self._home / "events"
        self._events_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._events_dir / filename_value
        self._rotate_bytes = int(rotate_mb * 1024 * 1024)
        self._max_record_bytes = int(max_record_mb * 1024 * 1024)
        self._file: Any | None = None
        self._subscription: Any | None = None
        self._open()

    def _open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115

    def _rotate(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        suffix = 1
        while True:
            rotated = self._path.with_suffix(f".jsonl.{suffix:03d}")
            if not rotated.exists():
                self._path.rename(rotated)
                break
            suffix += 1
        self._open()

    def _write(self, record: dict[str, Any]) -> None:
        import json

        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("Failed to serialize event for persistence: %s", exc)
            return

        encoded_size = len((line + "\n").encode("utf-8"))
        if self._max_record_bytes and encoded_size > self._max_record_bytes:
            payload = record.get("payload")
            original_size = encoded_size
            line = json.dumps(
                {
                    "timestamp": _summary_text(record.get("timestamp")),
                    "topic": _summary_text(record.get("topic")),
                    "source": _summary_text(record.get("source")),
                    "event_id": _summary_text(record.get("event_id")),
                    "trace_id": _summary_text(record.get("trace_id")),
                    "span_id": _summary_text(record.get("span_id")),
                    "parent_span_id": _summary_text(record.get("parent_span_id")),
                    "priority": _summary_text(record.get("priority")),
                    "metadata": {
                        "persistence_truncated": True,
                        "original_size_bytes": original_size,
                    },
                    "payload": {
                        "persistence_truncated": True,
                        "original_size_bytes": original_size,
                        "keys": (
                            [str(key)[:128] for key in list(payload)[:50]]
                            if isinstance(payload, dict)
                            else []
                        ),
                    },
                },
                ensure_ascii=False,
                default=str,
            )
            encoded_size = len((line + "\n").encode("utf-8"))
            if encoded_size > self._max_record_bytes:
                line = json.dumps(
                    {
                        "persistence_truncated": True,
                        "original_size_bytes": original_size,
                    },
                    separators=(",", ":"),
                )
                encoded_size = len((line + "\n").encode("utf-8"))
            logger.warning(
                "Event persistence record exceeded %s bytes and was summarized",
                self._max_record_bytes,
            )

        if self._file is None:
            self._open()
        if (
            self._rotate_bytes
            and self._path.exists()
            and self._path.stat().st_size > 0
            and self._path.stat().st_size + encoded_size > self._rotate_bytes
        ):
            self._rotate()
        self._file.write(line + "\n")
        self._file.flush()

        if self._rotate_bytes and self._path.stat().st_size >= self._rotate_bytes:
            self._rotate()

    def _on_event(self, event: Event) -> None:
        record = {
            "timestamp": getattr(event, "timestamp", None),
            "topic": getattr(event, "topic", None),
            "source": getattr(event, "source", None),
            "event_id": getattr(event, "event_id", None),
            "trace_id": getattr(event, "trace_id", None),
            "span_id": getattr(event, "span_id", None),
            "parent_span_id": getattr(event, "parent_span_id", None),
            "priority": getattr(event, "priority", None),
            "metadata": getattr(event, "metadata", None),
            "payload": getattr(event, "payload", None),
        }
        try:
            self._write(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event persistence failed: %s", exc)

    def attach(self, event_bus: EventBus) -> None:
        """Subscribe to all topics on the given bus."""
        self.detach()
        event_bus.subscribe("#", self._on_event)
        self._subscription = (event_bus, self._on_event)
        logger.info("JsonlEventSink attached to %s", self._path)

    def detach(self) -> None:
        """Unsubscribe from the bus."""
        if self._subscription is not None:
            event_bus, callback = self._subscription
            try:
                event_bus.unsubscribe("#", callback)
            except Exception as exc:  # noqa: BLE001
                logger.debug("EventSink unsubscribe failed: %s", exc)
            self._subscription = None

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        self.detach()
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlEventSink:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

"""MCAP writer for practice events.

Writes ROSClaw practice events to an MCAP file using the ``mcap`` Python API.
Each event is serialized as JSON and recorded on a well-known channel so that
standard MCAP tooling (``mcap info``, ``mcap doctor``, Foxglove) can read it.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.practice.mcap_writer")

# Lazy import: mcap is an optional dependency.
mcap = None
McapWriterClass = None
CompressionType = None

try:
    from mcap.writer import CompressionType as _CompressionType
    from mcap.writer import Writer as _McapWriter

    mcap = True
    McapWriterClass = _McapWriter
    CompressionType = _CompressionType
except Exception as exc:  # noqa: BLE001
    logger.debug("mcap not available: %s", exc)


class McapWriter:
    """Append-only MCAP writer for ROSClaw practice events.

    Events are written as JSON messages on the ``/rosclaw/events`` channel.
    The schema is a minimal JSON Schema describing the envelope shape.
    """

    _EVENT_SCHEMA_NAME = "rosclaw.practice.PracticeEventEnvelope"
    _EVENT_TOPIC = "/rosclaw/events"
    _MESSAGE_ENCODING = "json"
    _SCHEMA_ENCODING = "jsonschema"

    def __init__(
        self,
        path: Path | str,
        compression: str = "zstd",
        chunk_size_bytes: int = 4 * 1024 * 1024,
    ):
        if McapWriterClass is None:
            raise ImportError(
                "mcap package is not installed; install it to use McapWriter "
                "(e.g. pip install mcap)"
            )

        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._compression = self._parse_compression(compression)
        self._chunk_size = chunk_size_bytes
        self._lock = threading.RLock()
        self._writer = McapWriterClass(
            str(self._path),
            compression=self._compression,
            chunk_size=self._chunk_size,
        )
        self._writer.start(profile="", library="rosclaw-mcap/1.0.0")

        self._schema_id: int | None = None
        self._channel_id: int | None = None
        self._sequence = 0
        self._started = True

    def _parse_compression(self, compression: str) -> Any:
        if CompressionType is None:
            raise ImportError("mcap not available")
        comp = (compression or "zstd").lower()
        mapping: dict[str, Any] = {
            "zstd": CompressionType.ZSTD,
            "lz4": CompressionType.LZ4,
            "none": CompressionType.NONE,
        }
        # Older ``mcap`` builds also expose ZLIB, but it is not present in all releases.
        if hasattr(CompressionType, "ZLIB"):
            mapping["zlib"] = CompressionType.ZLIB
            mapping["z"] = CompressionType.ZLIB
        if comp not in mapping:
            logger.warning("Unknown MCAP compression '%s', falling back to zstd", compression)
            return CompressionType.ZSTD
        return mapping[comp]

    def _ensure_channel(self) -> int:
        if self._channel_id is not None:
            return self._channel_id

        schema = {
            "type": "object",
            "properties": {
                "practice_id": {"type": "string"},
                "session_id": {"type": ["string", "null"]},
                "robot_id": {"type": "string"},
                "source": {"type": "string"},
                "event_type": {"type": "string"},
                "timestamp_ns": {"type": "integer"},
                "timestamp_utc": {"type": "string"},
                "payload": {"type": "object"},
            },
            "required": ["practice_id", "event_type", "timestamp_ns"],
        }
        self._schema_id = self._writer.register_schema(
            name=self._EVENT_SCHEMA_NAME,
            encoding=self._SCHEMA_ENCODING,
            data=json.dumps(schema, ensure_ascii=False).encode("utf-8"),
        )
        self._channel_id = self._writer.register_channel(
            topic=self._EVENT_TOPIC,
            message_encoding=self._MESSAGE_ENCODING,
            schema_id=self._schema_id,
            metadata={"source": "rosclaw.practice"},
        )
        return self._channel_id

    def write(self, record: dict[str, Any]) -> None:
        """Serialize *record* to JSON and append to the MCAP file."""
        if not isinstance(record, dict):
            logger.error(
                "McapWriter.write expected a dict, got %s; dropping record",
                type(record).__name__,
            )
            return
        try:
            data = json.dumps(record, ensure_ascii=False, default=str).encode("utf-8")
        except (TypeError, ValueError) as e:
            logger.error("Failed to serialize MCAP record: %s", e)
            return

        with self._lock:
            if not self._started:
                logger.warning("McapWriter is closed; dropping record")
                return
            channel_id = self._ensure_channel()
            self._sequence += 1
            timestamp_ns = record.get("timestamp_ns") or time.time_ns()
            self._writer.add_message(
                channel_id=channel_id,
                log_time=timestamp_ns,
                publish_time=timestamp_ns,
                data=data,
                sequence=self._sequence,
            )

    def flush(self) -> None:
        """MCAP Writer commits on finish; this is a no-op for API symmetry."""

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            self._started = False
            try:
                self._writer.finish()
            except Exception as e:
                logger.error("Failed to finish MCAP writer: %s", e)
                raise

    def __enter__(self) -> McapWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    @property
    def path(self) -> Path:
        return self._path

    @staticmethod
    def is_available() -> bool:
        """Return True if the ``mcap`` package is installed."""
        return McapWriterClass is not None

"""Non-blocking, bounded JSONL exporter for ROSClaw Trace."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from rosclaw.firstboot.workspace import resolve_home
from rosclaw.observability.schema import SpanStatus, TraceRecord

logger = logging.getLogger("rosclaw.observability.exporters.jsonl")


class JsonlTraceExporter:
    """Write trace records on a daemon thread so control paths never wait on disk."""

    def __init__(
        self,
        home: str | Path | None = None,
        filename: str = "live.jsonl",
        queue_size: int = 4096,
        rotate_mb: float = 64.0,
        output_path: str | Path | None = None,
    ) -> None:
        if output_path is not None:
            self._path = Path(output_path).expanduser()
        else:
            root = resolve_home(str(home) if home else None)
            self._path = root / "traces" / filename
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._rotate_bytes = int(rotate_mb * 1024 * 1024)
        self._queue: queue.Queue[TraceRecord | object] = queue.Queue(maxsize=queue_size)
        self._sentinel = object()
        self._closed = False
        self._dropped = 0
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="rosclaw-trace-writer",
            daemon=True,
        )
        self._thread.start()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dropped_records(self) -> int:
        return self._dropped

    def export(self, record: TraceRecord) -> bool:
        """Enqueue without blocking; terminal failures displace a normal record if full."""

        if self._closed:
            return False
        try:
            self._queue.put_nowait(record)
            return True
        except queue.Full:
            if record.status not in {SpanStatus.ERROR, SpanStatus.BLOCKED}:
                self._dropped += 1
                return False
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._dropped += 1
                self._queue.put_nowait(record)
                return True
            except (queue.Empty, queue.Full):
                self._dropped += 1
                return False

    def _writer_loop(self) -> None:
        file: Any | None = None
        try:
            file = self._path.open("a", encoding="utf-8")
            while True:
                item = self._queue.get()
                try:
                    if item is self._sentinel:
                        break
                    assert isinstance(item, TraceRecord)
                    file.write(json.dumps(item.to_dict(), ensure_ascii=False, default=str) + "\n")
                    file.flush()
                    if self._rotate_bytes and self._path.stat().st_size >= self._rotate_bytes:
                        file.close()
                        self._rotate()
                        file = self._path.open("a", encoding="utf-8")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Trace persistence failed: %s", exc)
                finally:
                    self._queue.task_done()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Trace writer stopped: %s", exc)
        finally:
            if file is not None and not file.closed:
                file.flush()
                file.close()

    def _rotate(self) -> None:
        suffix = 1
        while True:
            rotated = self._path.with_name(f"{self._path.name}.{suffix:03d}")
            if not rotated.exists():
                self._path.rename(rotated)
                return
            suffix += 1

    def flush(self, timeout: float = 5.0) -> None:
        """Wait briefly for already-enqueued records; never wait indefinitely."""

        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks and time.monotonic() < deadline:
            time.sleep(0.01)

    def close(self, timeout: float = 5.0) -> None:
        if self._closed:
            return
        self.flush(timeout=timeout)
        self._closed = True
        try:
            self._queue.put_nowait(self._sentinel)
        except queue.Full:
            # Flush above normally prevents this. If the writer is stalled,
            # discard one normal record without corrupting Queue accounting.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(self._sentinel)
            except (queue.Empty, queue.Full):
                pass
        self._thread.join(timeout=timeout)

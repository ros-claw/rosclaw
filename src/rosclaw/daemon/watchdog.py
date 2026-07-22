"""Database-independent watchdog scheduler for rosclawd safety checks."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("rosclaw.daemon.watchdog")


class RuntimeWatchdog:
    """Run one bounded safety callback and expose event-loop lag health."""

    def __init__(self, callback: Callable[[], None], *, interval_sec: float = 0.05):
        self._callback = callback
        self.interval_sec = max(0.01, float(interval_sec))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._ticks = 0
        self._failures = 0
        self._last_tick_monotonic: float | None = None
        self._max_lag_ms = 0.0
        self._last_error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="rosclawd-watchdog",
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout_sec: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout_sec))

    def status(self) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            age_ms = (
                None
                if self._last_tick_monotonic is None
                else max(0.0, (time.monotonic() - self._last_tick_monotonic) * 1000.0)
            )
            running = thread is not None and thread.is_alive() and not self._stop.is_set()
            healthy = running and age_ms is not None and age_ms <= self.interval_sec * 4000.0
            return {
                "running": running,
                "healthy": healthy,
                "interval_ms": round(self.interval_sec * 1000.0, 3),
                "ticks": self._ticks,
                "failures": self._failures,
                "last_tick_age_ms": round(age_ms, 3) if age_ms is not None else None,
                "max_lag_ms": round(self._max_lag_ms, 3),
                "last_error": self._last_error,
            }

    def _run(self) -> None:
        expected = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            lag_ms = max(0.0, (now - expected) * 1000.0)
            try:
                self._callback()
            except Exception as exc:  # noqa: BLE001
                logger.exception("rosclawd watchdog callback failed")
                with self._lock:
                    self._failures += 1
                    self._last_error = f"{type(exc).__name__}: {exc}"[:512]
            with self._lock:
                self._ticks += 1
                self._last_tick_monotonic = time.monotonic()
                self._max_lag_ms = max(self._max_lag_ms, lag_ms)
            expected += self.interval_sec
            delay = max(0.0, expected - time.monotonic())
            if self._stop.wait(delay):
                return


__all__ = ["RuntimeWatchdog"]

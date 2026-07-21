"""Supervise isolated Adapter worker processes with bounded restart policy."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.daemon.worker_manager")

WORKER_PROTOCOL_VERSION = "rosclaw.adapter.worker.v1"
MAX_WORKER_LINE_BYTES = 64 * 1024


class WorkerError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class WorkerState(StrEnum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"
    STOPPING = "STOPPING"


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    command: tuple[str, ...]
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout_sec: float = 5.0
    heartbeat_timeout_sec: float = 2.0
    stop_timeout_sec: float = 1.0
    restart_limit: int = 3
    restart_window_sec: float = 60.0

    def __post_init__(self) -> None:
        if (
            not self.worker_id
            or len(self.worker_id) > 128
            or any(ord(character) < 0x20 for character in self.worker_id)
        ):
            raise ValueError("worker_id must contain 1..128 printable characters")
        if not self.command or any(not item for item in self.command):
            raise ValueError("worker command must contain at least one non-empty argument")
        for name in (
            "startup_timeout_sec",
            "heartbeat_timeout_sec",
            "stop_timeout_sec",
            "restart_window_sec",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            isinstance(self.restart_limit, bool)
            or not isinstance(self.restart_limit, int)
            or self.restart_limit < 0
        ):
            raise ValueError("restart_limit must be a non-negative integer")


@dataclass
class _Worker:
    spec: WorkerSpec
    state: WorkerState = WorkerState.STOPPED
    process: subprocess.Popen[str] | None = None
    generation: int = 0
    connection_id: str | None = None
    started_at_monotonic: float | None = None
    last_heartbeat_monotonic: float | None = None
    last_error: str | None = None
    restart_times: deque[float] = field(default_factory=deque)
    desired_running: bool = False
    ready: threading.Event = field(default_factory=threading.Event)
    stdout_thread: threading.Thread | None = None
    stderr_thread: threading.Thread | None = None
    stderr_tail: deque[str] = field(default_factory=lambda: deque(maxlen=20))

    def to_dict(self) -> dict[str, Any]:
        process = self.process
        now = time.monotonic()
        heartbeat_age_ms = (
            None
            if self.last_heartbeat_monotonic is None
            else max(0.0, (now - self.last_heartbeat_monotonic) * 1000.0)
        )
        return {
            "worker_id": self.spec.worker_id,
            "state": self.state.value,
            "pid": process.pid if process is not None and process.poll() is None else None,
            "generation": self.generation,
            "connection_id": self.connection_id,
            "desired_running": self.desired_running,
            "heartbeat_age_ms": (
                round(heartbeat_age_ms, 3) if heartbeat_age_ms is not None else None
            ),
            "restart_count_window": len(self.restart_times),
            "restart_limit": self.spec.restart_limit,
            "last_error": self.last_error,
            "stderr_tail": list(self.stderr_tail),
        }


class WorkerManager:
    """Own worker process lifecycles; never resumes an old worker connection."""

    def __init__(
        self,
        *,
        monitor_interval_sec: float = 0.05,
        on_generation_change: Callable[[str, str | None, str], None] | None = None,
    ) -> None:
        self.monitor_interval_sec = max(0.01, float(monitor_interval_sec))
        self._on_generation_change = on_generation_change
        self._workers: dict[str, _Worker] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._monitor: threading.Thread | None = None

    def register(self, spec: WorkerSpec) -> None:
        with self._lock:
            if spec.worker_id in self._workers:
                raise WorkerError("WORKER_EXISTS", f"Worker {spec.worker_id!r} already exists")
            self._workers[spec.worker_id] = _Worker(spec=spec)

    def start(self) -> None:
        with self._lock:
            if self._monitor is not None and self._monitor.is_alive():
                return
            self._stop.clear()
            self._monitor = threading.Thread(
                target=self._monitor_loop,
                name="rosclawd-worker-monitor",
                daemon=True,
            )
            self._monitor.start()

    def start_worker(self, worker_id: str, *, wait_ready: bool = True) -> dict[str, Any]:
        self.start()
        generation_change: tuple[str, str | None, str] | None = None
        with self._lock:
            worker = self._get(worker_id)
            worker.desired_running = True
            if worker.process is None or worker.process.poll() is not None:
                generation_change = self._spawn_locked(
                    worker,
                    is_restart=worker.generation > 0,
                )
            ready = worker.ready
            timeout = worker.spec.startup_timeout_sec
        if generation_change is not None:
            self._notify_generation_change(*generation_change)
        if wait_ready and not ready.wait(timeout):
            self._fail_and_terminate(worker_id, "worker startup heartbeat timed out")
            raise WorkerError(
                "WORKER_START_TIMEOUT",
                f"Worker {worker_id!r} did not report ready within {timeout:.3f}s",
            )
        return self.get_status(worker_id)

    def stop_worker(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            worker = self._get(worker_id)
            worker.desired_running = False
            process = worker.process
            worker.state = WorkerState.STOPPING if process is not None else WorkerState.STOPPED
        self._terminate(worker, process)
        with self._lock:
            worker.state = WorkerState.STOPPED
            worker.process = None
            worker.connection_id = None
            worker.ready.clear()
            return worker.to_dict()

    def restart_worker(self, worker_id: str) -> dict[str, Any]:
        self.stop_worker(worker_id)
        return self.start_worker(worker_id)

    def get_status(self, worker_id: str) -> dict[str, Any]:
        with self._lock:
            return self._get(worker_id).to_dict()

    def status(self) -> dict[str, Any]:
        with self._lock:
            workers = [worker.to_dict() for worker in self._workers.values()]
        return {
            "registered": len(workers),
            "ready": sum(item["state"] == WorkerState.READY.value for item in workers),
            "failed": sum(item["state"] == WorkerState.FAILED.value for item in workers),
            "workers": workers,
        }

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            worker_ids = list(self._workers)
        for worker_id in worker_ids:
            try:
                self.stop_worker(worker_id)
            except Exception:  # noqa: BLE001
                logger.exception("failed to stop Adapter worker %s", worker_id)
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=2.0)

    def _spawn_locked(
        self,
        worker: _Worker,
        *,
        is_restart: bool,
    ) -> tuple[str, str | None, str]:
        now = time.monotonic()
        self._prune_restart_times(worker, now)
        if is_restart:
            if len(worker.restart_times) >= worker.spec.restart_limit:
                worker.state = WorkerState.FAILED
                worker.desired_running = False
                worker.last_error = "restart budget exhausted"
                raise WorkerError(
                    "WORKER_RESTART_EXHAUSTED",
                    f"Worker {worker.spec.worker_id!r} exhausted its restart budget",
                )
            worker.restart_times.append(now)
        old_connection_id = worker.connection_id
        environment = os.environ.copy()
        environment.update(worker.spec.env)
        try:
            process = subprocess.Popen(
                worker.spec.command,
                cwd=str(worker.spec.cwd) if worker.spec.cwd is not None else None,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            worker.state = WorkerState.FAILED
            worker.last_error = f"spawn failed: {exc}"[:512]
            raise WorkerError("WORKER_SPAWN_FAILED", worker.last_error) from exc
        worker.process = process
        worker.generation += 1
        new_connection_id = f"worker_{uuid.uuid4().hex}"
        worker.connection_id = new_connection_id
        worker.started_at_monotonic = now
        worker.last_heartbeat_monotonic = now
        worker.last_error = None
        worker.state = WorkerState.STARTING
        worker.ready.clear()
        worker.stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(worker.spec.worker_id, process),
            name=f"rosclawd-{worker.spec.worker_id}-stdout",
            daemon=True,
        )
        worker.stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(worker.spec.worker_id, process),
            name=f"rosclawd-{worker.spec.worker_id}-stderr",
            daemon=True,
        )
        worker.stdout_thread.start()
        worker.stderr_thread.start()
        return worker.spec.worker_id, old_connection_id, new_connection_id

    def _read_stdout(self, worker_id: str, process: subprocess.Popen[str]) -> None:
        stream = process.stdout
        if stream is None:
            return
        while True:
            line = stream.readline(MAX_WORKER_LINE_BYTES + 1)
            if not line:
                return
            if len(line.encode("utf-8")) > MAX_WORKER_LINE_BYTES or not line.endswith("\n"):
                self._fail_and_terminate(
                    worker_id,
                    "worker protocol line exceeded size limit",
                    expected_process=process,
                )
                return
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                self._fail_and_terminate(
                    worker_id,
                    "worker emitted invalid JSON protocol message",
                    expected_process=process,
                )
                return
            if (
                not isinstance(message, dict)
                or message.get("protocol_version") != WORKER_PROTOCOL_VERSION
            ):
                self._fail_and_terminate(
                    worker_id,
                    "worker emitted unsupported protocol message",
                    expected_process=process,
                )
                return
            message_type = message.get("type")
            if message_type not in {"ready", "heartbeat", "degraded"}:
                self._fail_and_terminate(
                    worker_id,
                    f"unsupported worker message type: {message_type!r}",
                    expected_process=process,
                )
                return
            with self._lock:
                worker = self._workers.get(worker_id)
                if worker is None or worker.process is not process:
                    return
                worker.last_heartbeat_monotonic = time.monotonic()
                if message_type == "ready":
                    worker.state = WorkerState.READY
                    worker.ready.set()
                elif message_type == "heartbeat":
                    if worker.state is WorkerState.STARTING:
                        worker.state = WorkerState.READY
                        worker.ready.set()
                elif message_type == "degraded":
                    worker.state = WorkerState.DEGRADED
                    worker.last_error = str(message.get("reason", "worker degraded"))[:512]

    def _read_stderr(self, worker_id: str, process: subprocess.Popen[str]) -> None:
        stream = process.stderr
        if stream is None:
            return
        for line in stream:
            with self._lock:
                worker = self._workers.get(worker_id)
                if worker is None or worker.process is not process:
                    return
                worker.stderr_tail.append(line.rstrip()[:512])

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.monitor_interval_sec):
            with self._lock:
                worker_ids = list(self._workers)
            for worker_id in worker_ids:
                self._monitor_worker(worker_id)

    def _monitor_worker(self, worker_id: str) -> None:
        generation_change: tuple[str, str | None, str] | None = None
        with self._lock:
            worker = self._workers.get(worker_id)
            if worker is None or not worker.desired_running:
                return
            process = worker.process
            if process is None:
                reason = "worker process is absent"
            else:
                return_code = process.poll()
                heartbeat_age = time.monotonic() - (worker.last_heartbeat_monotonic or 0.0)
                if return_code is not None:
                    reason = f"worker exited with status {return_code}"
                elif heartbeat_age > worker.spec.heartbeat_timeout_sec:
                    reason = f"worker heartbeat timed out after {heartbeat_age:.3f}s"
                else:
                    return
            worker.last_error = reason
            worker.state = WorkerState.DEGRADED
        if process is not None and process.poll() is None:
            self._terminate(worker, process)
        with self._lock:
            if not worker.desired_running or self._stop.is_set():
                return
            worker.process = None
            try:
                generation_change = self._spawn_locked(worker, is_restart=True)
            except WorkerError:
                worker.state = WorkerState.FAILED
                worker.desired_running = False
        if generation_change is not None:
            self._notify_generation_change(*generation_change)

    def _fail_and_terminate(
        self,
        worker_id: str,
        reason: str,
        *,
        expected_process: subprocess.Popen[str] | None = None,
    ) -> None:
        with self._lock:
            worker = self._get(worker_id)
            if expected_process is not None and worker.process is not expected_process:
                return
            if worker.state is WorkerState.FAILED and not worker.desired_running:
                return
            worker.last_error = reason
            worker.state = WorkerState.DEGRADED
            process = worker.process
        self._terminate(worker, process)

    @staticmethod
    def _terminate(worker: _Worker, process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=worker.spec.stop_timeout_sec)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            process.wait(timeout=worker.spec.stop_timeout_sec)

    def _notify_generation_change(
        self,
        worker_id: str,
        old_connection_id: str | None,
        new_connection_id: str,
    ) -> None:
        callback = self._on_generation_change
        if callback is None:
            return
        try:
            callback(worker_id, old_connection_id, new_connection_id)
        except Exception:  # noqa: BLE001
            logger.exception("Adapter worker generation callback failed for %s", worker_id)

    @staticmethod
    def _prune_restart_times(worker: _Worker, now: float) -> None:
        cutoff = now - worker.spec.restart_window_sec
        while worker.restart_times and worker.restart_times[0] < cutoff:
            worker.restart_times.popleft()

    def _get(self, worker_id: str) -> _Worker:
        worker = self._workers.get(worker_id)
        if worker is None:
            raise WorkerError("WORKER_NOT_FOUND", f"Worker {worker_id!r} is not registered")
        return worker


__all__ = [
    "MAX_WORKER_LINE_BYTES",
    "WORKER_PROTOCOL_VERSION",
    "WorkerError",
    "WorkerManager",
    "WorkerSpec",
    "WorkerState",
]

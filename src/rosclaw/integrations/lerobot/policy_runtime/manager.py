"""Managed persistent LeRobot policy runtime process.

The manager spawns a Python 3.12 subprocess running the LeRobot worker service,
talks to it over stdin/stdout JSONL, and handles graceful shutdown plus stale
process cleanup.

This module must not import torch or lerobot.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from rosclaw.integrations.lerobot.policy_runtime.protocol import (
    RUNTIME_PROTOCOL_VERSION,
    encode_request,
    parse_line,
)
from rosclaw.integrations.lerobot.policy_runtime.state import RuntimeState


DEFAULT_STARTUP_TIMEOUT_SEC = 60.0
DEFAULT_CALL_TIMEOUT_SEC = 120.0
DEFAULT_SHUTDOWN_TIMEOUT_SEC = 30.0


class PersistentRuntimeManager:
    """Manage a long-lived LeRobot policy worker subprocess."""

    def __init__(
        self,
        python_executable: Path | str,
        *,
        worker_module: str = "rosclaw.integrations.lerobot.policy_worker_runtime",
        policy_path: str | None = None,
        device: str = "cpu",
        dtype: str = "auto",
        allow_network: bool = False,
        timeout_sec: float = DEFAULT_CALL_TIMEOUT_SEC,
        startup_timeout_sec: float = DEFAULT_STARTUP_TIMEOUT_SEC,
        shutdown_timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT_SEC,
        env: dict[str, str] | None = None,
    ):
        self.python_executable = Path(python_executable)
        self.worker_module = worker_module
        self.policy_path = policy_path
        self.device = device
        self.dtype = dtype
        self.allow_network = allow_network
        self.timeout_sec = timeout_sec
        self.startup_timeout_sec = startup_timeout_sec
        self.shutdown_timeout_sec = shutdown_timeout_sec
        self.env = env

        self._process: subprocess.Popen[str] | None = None
        self._state = RuntimeState()
        self._request_id = 0
        self._lock = threading.Lock()
        self._pending: dict[str, threading.Event] = {}
        self._responses: dict[str, dict[str, Any]] = {}
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_queue: Queue[str] = Queue()
        self._stderr_lines: list[str] = []
        self._shutdown = False
        self._worker_generation = 0

    @property
    def state(self) -> RuntimeState:
        return self._state

    def start(self) -> RuntimeState:
        """Start the worker subprocess and complete the HELLO handshake."""
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return self._state

            self._state.transition("starting")
            self._shutdown = False
            self._responses.clear()
            self._pending.clear()
            self._stderr_lines.clear()
            self._worker_generation += 1

            env = dict(self.env or os.environ)
            # Ensure the worker can find the rosclaw package tree.
            rosclaw_src = Path(__file__).parents[4]
            python_path = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{rosclaw_src}{os.pathsep}{python_path}" if python_path else str(rosclaw_src)

            cmd = [
                str(self.python_executable),
                "-m",
                self.worker_module,
                "--protocol-version",
                RUNTIME_PROTOCOL_VERSION,
            ]
            if self.policy_path:
                cmd.extend(["--policy-path", self.policy_path])
            cmd.extend(["--device", self.device, "--dtype", self.dtype])
            if self.allow_network:
                cmd.append("--allow-network")

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=env,
                )
            except OSError as exc:
                self._state.transition("error", f"Failed to start worker: {exc}")
                return self._state

            self._state.pid = self._process.pid
            self._state.worker_generation = self._worker_generation

        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_thread.start()

        # Complete HELLO handshake to confirm protocol compatibility.
        hello = self.call(
            "HELLO",
            {
                "protocol_version": RUNTIME_PROTOCOL_VERSION,
                "rosclaw_python": sys.executable,
            },
            timeout_sec=self.startup_timeout_sec,
        )
        if hello.get("status") != "ok":
            self._state.transition(
                "error",
                hello.get("error", {}).get("message", "HELLO handshake failed"),
            )
            self._stop_process()
            return self._state

        self._state.transition("ready")
        return self._state

    def stop(self) -> None:
        """Gracefully stop the worker subprocess."""
        with self._lock:
            self._shutdown = True
        try:
            self.call("SHUTDOWN", {}, timeout_sec=self.shutdown_timeout_sec)
        except RuntimeError:
            pass
        self._stop_process()
        self._state.transition("stopped")

    def restart(self) -> RuntimeState:
        """Stop and start the runtime."""
        self.stop()
        return self.start()

    def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Send a JSONL request and block until the matching response arrives."""
        if timeout_sec is None:
            timeout_sec = self.timeout_sec

        request_id = self._next_request_id()
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = event

        line = encode_request(method, params, request_id)  # type: ignore[arg-type]
        try:
            self._send_line(line)
        except RuntimeError as exc:
            with self._lock:
                self._pending.pop(request_id, None)
            return {"status": "error", "error": {"code": "worker_died", "message": str(exc)}}

        if not event.wait(timeout=timeout_sec):
            with self._lock:
                self._pending.pop(request_id, None)
                process = self._process
            if process is not None and process.poll() is not None:
                self._state.transition("error", "Worker process died")
                return {"status": "error", "error": {"code": "worker_died", "message": "Worker process died"}}
            raise RuntimeError(f"Timeout waiting for {method} response")

        with self._lock:
            self._pending.pop(request_id, None)
            response = self._responses.pop(request_id, {})

        if response.get("error"):
            return {
                "status": "error",
                "error": response["error"],
            }
        return {"status": "ok", **response.get("result", {}), "worker_generation": self._worker_generation}

    def _next_request_id(self) -> str:
        with self._lock:
            self._request_id += 1
            return str(self._request_id)

    def _send_line(self, line: str) -> None:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("Worker process is not running")
            if self._process.stdin is None:
                raise RuntimeError("Worker stdin is not available")
            try:
                self._process.stdin.write(line)
                self._process.stdin.flush()
            except BrokenPipeError as exc:
                self._state.transition("error", f"Worker stdin closed: {exc}")
                raise RuntimeError("Worker stdin closed") from exc

    def _fail_all_pending(self, code: str, message: str) -> None:
        """Resolve all pending requests with a worker error."""
        with self._lock:
            pending = dict(self._pending)
            for request_id in pending:
                self._responses[request_id] = {"error": {"code": code, "message": message}}
                event = self._pending.pop(request_id, None)
                if event is not None:
                    event.set()

    def _read_stdout(self) -> None:
        """Background thread: read stdout lines and dispatch responses."""
        if self._process is None or self._process.stdout is None:
            self._fail_all_pending("worker_died", "Worker stdout not available")
            return
        try:
            for line in self._process.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    parsed = parse_line(line)
                except Exception:  # noqa: BLE001
                    # Ignore non-protocol stdout noise from the worker.
                    continue
                if parsed is None:
                    continue
                if hasattr(parsed, "id"):
                    request_id = parsed.id
                    with self._lock:
                        self._responses[request_id] = parsed.to_dict()
                        event = self._pending.pop(request_id, None)
                    if event is not None:
                        event.set()
        except Exception as exc:  # noqa: BLE001
            self._state.transition("error", f"stdout reader failed: {exc}")
            self._fail_all_pending("worker_died", f"stdout reader failed: {exc}")
        finally:
            self._fail_all_pending("worker_died", "Worker stdout closed")

    def _read_stderr(self) -> None:
        """Background thread: capture the last chunk of stderr for diagnostics."""
        if self._process is None or self._process.stderr is None:
            return
        try:
            for line in self._process.stderr:
                self._stderr_lines.append(line.rstrip("\n"))
                # Keep stderr bounded in memory.
                if len(self._stderr_lines) > 500:
                    self._stderr_lines = self._stderr_lines[-250:]
        except Exception:  # noqa: BLE001
            pass

    def _stop_process(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return

        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    pass

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass

        with self._lock:
            self._process = None

    def stderr_tail(self, limit: int = 50) -> list[str]:
        """Return the most recent stderr lines for debugging."""
        return self._stderr_lines[-limit:]

    def __enter__(self) -> "PersistentRuntimeManager":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

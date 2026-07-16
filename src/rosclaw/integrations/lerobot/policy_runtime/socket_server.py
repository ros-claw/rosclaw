"""Unix-socket front end for the persistent LeRobot policy runtime.

The socket server wraps a :class:`PersistentRuntimeManager` and forwards one
JSONL request/response at a time.  This keeps the LeRobot worker resident while
allowing separate ``rosclaw lerobot policy`` CLI invocations to talk to it.

This module is free of torch/lerobot imports.
"""

from __future__ import annotations

import contextlib
import socket
import threading
import time
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.policy_runtime.config import (
    get_policy_runtime_socket_path,
    load_policy_runtime_config,
    save_policy_runtime_config,
)
from rosclaw.integrations.lerobot.policy_runtime.manager import PersistentRuntimeManager
from rosclaw.integrations.lerobot.policy_runtime.protocol import (
    RUNTIME_PROTOCOL_VERSION,
    encode_request,
    encode_response,
    parse_line,
)

DEFAULT_SOCKET_TIMEOUT_SEC = 120.0


class RuntimeSocketServer:
    """Expose a :class:`PersistentRuntimeManager` over a Unix domain socket."""

    def __init__(
        self,
        manager: PersistentRuntimeManager,
        socket_path: Path | str | None = None,
    ):
        self.manager = manager
        self.socket_path = Path(socket_path or get_policy_runtime_socket_path())
        self._server: socket.socket | None = None
        self._shutdown = threading.Event()
        self._accept_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the manager and the socket listener."""
        self.manager.start()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove a stale socket file.
        if self.socket_path.exists():
            self.socket_path.unlink()
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(str(self.socket_path))
        self._server.listen(1)
        self._server.settimeout(0.5)
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        """Accept one client at a time and dispatch requests."""
        while not self._shutdown.is_set():
            try:
                conn, _ = self._server.accept()  # type: ignore[union-attr]
            except TimeoutError:
                continue
            except OSError:
                break
            self._handle_client(conn)

    def _handle_client(self, conn: socket.socket) -> None:
        """Read JSONL requests and respond until the client disconnects."""
        conn.settimeout(None)
        file = conn.makefile("r")
        try:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                request = parse_line(line)
                if request is None:
                    continue
                try:
                    result = self.manager.call(
                        request.method,
                        request.params,
                        timeout_sec=self.manager.timeout_sec,
                    )
                    response_line = encode_response(request.id, result=result)
                except Exception as exc:  # noqa: BLE001
                    response_line = encode_response(
                        request.id,
                        error={"code": "server_error", "message": str(exc)},
                    )
                try:
                    conn.sendall(response_line.encode("utf-8"))
                except BrokenPipeError:
                    break
        finally:
            file.close()
            conn.close()

    def stop(self) -> None:
        """Stop accepting clients and shut down the manager."""
        self._shutdown.set()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=2.0)
        self.manager.stop()
        if self.socket_path.exists():
            self.socket_path.unlink()

    def wait(self) -> None:
        """Block until the server is asked to stop."""
        try:
            while not self._shutdown.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


def send_request(
    socket_path: Path | str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout_sec: float = DEFAULT_SOCKET_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Send a single JSONL request to a runtime daemon and return the response."""
    params = params or {}
    request_id = "cli"
    line = encode_request(method, params, request_id)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout_sec)
    try:
        sock.connect(str(socket_path))
        sock.sendall(line.encode("utf-8"))
        file = sock.makefile("r")
        try:
            response_line = file.readline()
        finally:
            file.close()
    finally:
        sock.close()

    if not response_line:
        return {"status": "error", "error": {"code": "no_response", "message": "No response from runtime"}}

    parsed = parse_line(response_line.strip())
    if parsed is None:
        return {"status": "error", "error": {"code": "invalid_response", "message": "Invalid JSONL response"}}
    data = parsed.to_dict()
    if data.get("error"):
        return {"status": "error", "error": data["error"]}
    return {"status": "ok", **data.get("result", {})}


def try_send_request(
    socket_path: Path | str,
    method: str,
    params: dict[str, Any] | None = None,
    *,
    timeout_sec: float = 10.0,
) -> dict[str, Any] | None:
    """Best-effort request; returns None on connection failure."""
    try:
        return send_request(socket_path, method, params, timeout_sec=timeout_sec)
    except (TimeoutError, OSError):
        return None


def start_daemon(
    manager: PersistentRuntimeManager,
    socket_path: Path | str | None = None,
    *,
    policy_path: str | None = None,
    python_executable: str | None = None,
) -> RuntimeSocketServer:
    """Start a foreground socket server and persist its bookkeeping."""
    server = RuntimeSocketServer(manager, socket_path=socket_path)
    server.start()
    config = load_policy_runtime_config()
    config.update(
        {
            "socket_path": str(server.socket_path),
            "policy_path": policy_path,
            "device": manager.device,
            "dtype": manager.dtype,
            "python_executable": str(python_executable or manager.python_executable),
            "protocol_version": RUNTIME_PROTOCOL_VERSION,
        }
    )
    save_policy_runtime_config(config)
    return server


def stop_daemon(socket_path: Path | str | None = None) -> dict[str, Any]:
    """Ask the daemon to shut down and clear its bookkeeping."""
    from rosclaw.integrations.lerobot.policy_runtime.config import (
        clear_policy_runtime_config,
        get_daemon_status,
        remove_pid_file,
        terminate_process,
    )

    config = load_policy_runtime_config()
    resolved_socket = Path(socket_path or config.get("socket_path") or get_policy_runtime_socket_path())
    status = get_daemon_status(config)
    result: dict[str, Any] = {"running": status["running"], "pid": status["pid"]}

    if resolved_socket.exists():
        response = try_send_request(resolved_socket, "SHUTDOWN", timeout_sec=10.0)
        if response is not None:
            result["shutdown_response"] = response

    pid = status["pid"]
    if pid is not None:
        terminated = terminate_process(pid)
        result["terminated"] = terminated

    if resolved_socket.exists():
        resolved_socket.unlink()
    remove_pid_file()
    clear_policy_runtime_config()
    result["running"] = False
    return result

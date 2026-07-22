"""Minimal MCP stdio client for ``rosclaw mcp call`` and health handshakes.

This intentionally does not depend on the official MCP SDK so the core CLI can
call externally installed MCP servers without dragging in their dependency
constraints.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import yaml

from rosclaw.firstboot.workspace import resolve_home


class McpStdioError(Exception):
    """Raised when an MCP stdio interaction fails."""


MAX_MCP_STDIO_LINE_BYTES = 1024 * 1024
MAX_PENDING_RESPONSES = 128
MAX_MCP_STDERR_LINE_CHARS = 4096


class McpStdioClient:
    """Line-delimited JSON-RPC client for an MCP stdio server."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self.command = command
        self.args = args
        self.env = {**os.environ, **(env or {})}
        self._proc: subprocess.Popen[str] | None = None
        self._request_id = 0
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=MAX_PENDING_RESPONSES)
        self._pending: dict[int, dict[str, Any]] = {}
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._reader_done = threading.Event()
        self._reader_error: str | None = None
        self._stderr_tail: deque[str] = deque(maxlen=40)

    def start(self, timeout: float = 10.0) -> None:
        """Start the server subprocess."""
        if self._proc is not None:
            raise McpStdioError("MCP server is already started; stop it before restarting")
        self._responses = queue.Queue(maxsize=MAX_PENDING_RESPONSES)
        self._pending.clear()
        self._stderr_tail.clear()
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                close_fds=True,
                start_new_session=True,
            )
        except Exception as exc:
            raise McpStdioError(f"Failed to start MCP server: {exc}") from exc

        self._reader_done.clear()
        self._reader_error = None
        self._reader_thread = threading.Thread(
            target=self._read_stdout,
            name="rosclaw-mcp-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            name="rosclaw-mcp-stderr",
            daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread.start()
        try:
            init_id = self._next_id()
            self._send(
                {
                    "jsonrpc": "2.0",
                    "id": init_id,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "rosclaw-mcp-cli", "version": "1.0.0"},
                    },
                }
            )
            response = self._read_response(init_id, timeout=timeout)
            if "error" in response:
                raise McpStdioError(f"Initialize error: {response['error']}")
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            self.stop(timeout=min(1.0, timeout))
            raise

    def call_tool(
        self, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0
    ) -> dict[str, Any]:
        """Call an MCP tool and return the JSON-RPC result."""
        if self._proc is None:
            raise McpStdioError("MCP server not started")

        req_id = self._next_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments or {}},
            }
        )
        return self._read_response(req_id, timeout=timeout)

    def list_tools(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        """List available tools from the server."""
        if self._proc is None:
            raise McpStdioError("MCP server not started")

        req_id = self._next_id()
        self._send(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/list",
                "params": {},
            }
        )
        response = self._read_response(req_id, timeout=timeout)
        result = response.get("result", {})
        return list(result.get("tools", []))

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the server subprocess."""
        if self._proc is None:
            return
        process = self._proc
        stdin = process.stdin
        if stdin is not None:
            try:
                # The server may have already exited (e.g. after a tool crash or
                # natural shutdown). Writing to a closed stdin raises BrokenPipe;
                # that is expected and should not pollute stderr.
                stdin.write("\n")
                stdin.flush()
            except BrokenPipeError:
                pass
            except Exception:
                pass
        self._terminate_process(process, timeout=timeout)
        # Explicitly close stdio streams so the Popen finalizer does not emit
        # BrokenPipe warnings when the server has already exited.
        for stream_name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is not None:
                with contextlib.suppress(BrokenPipeError, OSError, ValueError):
                    stream.close()
        self._proc = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None or self._proc.poll() is not None:
            raise McpStdioError("MCP server not started")
        stdin = self._proc.stdin
        try:
            stdin.write(json.dumps(message, allow_nan=False) + "\n")
            stdin.flush()
        except Exception as exc:
            raise McpStdioError(f"Failed to send MCP request: {exc}") from exc

    def _read_response(self, expected_id: int, timeout: float) -> dict[str, Any]:
        if self._proc is None:
            raise McpStdioError("MCP server not started")
        pending = self._pending.pop(expected_id, None)
        if pending is not None:
            return pending
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process = self._proc
                if process is not None:
                    self._terminate_process(process, timeout=0.25)
                stderr = "\n".join(self._stderr_tail)
                detail = f" stderr: {stderr}" if stderr else ""
                raise McpStdioError(f"Timeout waiting for MCP response (id={expected_id}).{detail}")
            try:
                message = self._responses.get(timeout=min(remaining, 0.05))
            except queue.Empty:
                if self._reader_done.is_set():
                    detail = self._reader_error or f"server exited with {self._proc.poll()}"
                    raise McpStdioError(f"MCP response stream ended: {detail}") from None
                continue
            message_id = message.get("id")
            if (
                isinstance(message_id, int)
                and not isinstance(message_id, bool)
                and message_id == expected_id
            ):
                return message
            if isinstance(message_id, int) and not isinstance(message_id, bool):
                if len(self._pending) >= MAX_PENDING_RESPONSES:
                    process = self._proc
                    if process is not None:
                        self._terminate_process(process, timeout=0.25)
                    raise McpStdioError("MCP server exceeded the pending response limit")
                self._pending[message_id] = message

    def _read_stdout(self) -> None:
        process = self._proc
        if process is None or process.stdout is None:
            self._reader_error = "stdout is unavailable"
            self._reader_done.set()
            return
        stdout = process.stdout
        try:
            while True:
                line = stdout.readline(MAX_MCP_STDIO_LINE_BYTES + 1)
                if not line:
                    return
                if len(line.encode("utf-8")) > MAX_MCP_STDIO_LINE_BYTES or not line.endswith("\n"):
                    self._fail_reader(process, "MCP response exceeded the stdio line limit")
                    return
                stripped = line.strip()
                if not stripped or not stripped.startswith("{"):
                    continue
                try:
                    message = json.loads(stripped)
                except json.JSONDecodeError:
                    self._fail_reader(process, "MCP server emitted malformed JSON-RPC")
                    return
                if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
                    self._fail_reader(process, "MCP server emitted invalid JSON-RPC")
                    return
                message_id = message.get("id")
                if message_id is None:
                    continue
                if isinstance(message_id, bool) or not isinstance(message_id, int):
                    self._fail_reader(process, "MCP response id must be an integer")
                    return
                try:
                    self._responses.put_nowait(message)
                except queue.Full:
                    self._fail_reader(process, "MCP response queue exceeded its limit")
                    return
        except Exception as exc:  # noqa: BLE001
            self._reader_error = f"{type(exc).__name__}: {exc}"[:512]
        finally:
            self._reader_done.set()

    def _drain_stderr(self) -> None:
        process = self._proc
        stderr = process.stderr if process is not None else None
        if stderr is None:
            return
        while True:
            line = stderr.readline(MAX_MCP_STDERR_LINE_CHARS + 1)
            if not line:
                return
            self._stderr_tail.append(line.rstrip()[:1024])

    def _fail_reader(self, process: subprocess.Popen[str], message: str) -> None:
        self._reader_error = message
        self._terminate_process(process, timeout=0.25)

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str], *, timeout: float) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=max(0.01, timeout))
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=max(0.01, timeout))


def load_runtime_config(server_name: str, home: Path | str | None = None) -> dict[str, Any]:
    """Load the runtime YAML for an installed MCP server."""
    home_path = resolve_home(str(home) if home else None)
    path = home_path / "mcp" / "runtime" / f"{server_name}.yaml"
    if not path.exists():
        raise McpStdioError(f"Runtime config not found for server: {server_name}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def call_server_tool(
    server_name: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    home: Path | str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Call a tool on an installed MCP server by name."""
    config = load_runtime_config(server_name, home=home)
    transport = config.get("transport", {})
    command = transport.get("command")
    args = list(transport.get("args", []))
    env = transport.get("env", {})

    if not command:
        raise McpStdioError(f"No transport command configured for {server_name}")

    client: McpStdioClient | None = None
    try:
        client = McpStdioClient(command, args, env=env)
        client.start(timeout=min(timeout, 20.0))
        response = client.call_tool(tool_name, arguments or {}, timeout=timeout)
        if "error" in response:
            raise McpStdioError(f"Tool error: {response['error']}")
        return cast(dict[str, Any], response.get("result", {}))
    finally:
        if client is not None:
            client.stop()


def list_server_tools(
    server_name: str,
    home: Path | str | None = None,
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """List tools advertised by an installed MCP server."""
    config = load_runtime_config(server_name, home=home)
    transport = config.get("transport", {})
    command = transport.get("command")
    args = list(transport.get("args", []))
    env = transport.get("env", {})

    if not command:
        raise McpStdioError(f"No transport command configured for {server_name}")

    client: McpStdioClient | None = None
    try:
        client = McpStdioClient(command, args, env=env)
        client.start(timeout=timeout)
        return client.list_tools(timeout=timeout)
    finally:
        if client is not None:
            client.stop()


def health_smoke(
    server_name: str, home: Path | str | None = None, timeout: float = 20.0
) -> dict[str, Any]:
    """Run a lightweight health smoke test: handshake and list tools."""
    try:
        tools = list_server_tools(server_name, home=home, timeout=timeout)
        return {
            "healthy": True,
            "server_name": server_name,
            "tools_count": len(tools),
            "tools": [t.get("name") for t in tools],
        }
    except Exception as exc:
        return {
            "healthy": False,
            "server_name": server_name,
            "error": str(exc),
        }


class McpServerSession:
    """Stateful session with an installed MCP stdio server.

    Some tools (e.g. ``start_pipeline`` followed by ``capture_aligned_rgbd``)
    require state to persist across calls. This context manager keeps one
    server process alive for the duration of the ``with`` block.
    """

    def __init__(
        self, server_name: str, home: Path | str | None = None, start_timeout: float = 10.0
    ):
        config = load_runtime_config(server_name, home=home)
        transport = config.get("transport", {})
        self.command = transport.get("command")
        self.args = list(transport.get("args", []))
        self.env = transport.get("env", {})
        self.start_timeout = start_timeout
        self._client: McpStdioClient | None = None

    def __enter__(self) -> McpServerSession:
        if not self.command:
            raise McpStdioError("No transport command configured")
        self._client = McpStdioClient(self.command, self.args, env=self.env)
        self._client.start(timeout=self.start_timeout)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is not None:
            self._client.stop()
            self._client = None

    def call(
        self, tool_name: str, arguments: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        if self._client is None:
            raise McpStdioError("Session not started")
        response = self._client.call_tool(tool_name, arguments or {}, timeout=timeout)
        if "error" in response:
            raise McpStdioError(f"Tool error: {response['error']}")
        return cast(dict[str, Any], response.get("result", {}))

    def list_tools(self, timeout: float = 10.0) -> list[dict[str, Any]]:
        if self._client is None:
            raise McpStdioError("Session not started")
        return self._client.list_tools(timeout=timeout)

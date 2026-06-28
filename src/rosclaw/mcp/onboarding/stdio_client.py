"""Minimal MCP stdio client for ``rosclaw mcp call`` and health handshakes.

This intentionally does not depend on the official MCP SDK so the core CLI can
call externally installed MCP servers without dragging in their dependency
constraints.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from rosclaw.firstboot.workspace import resolve_home


class McpStdioError(Exception):
    """Raised when an MCP stdio interaction fails."""


class McpStdioClient:
    """Line-delimited JSON-RPC client for an MCP stdio server."""

    def __init__(self, command: str, args: list[str], env: dict[str, str] | None = None):
        self.command = command
        self.args = args
        self.env = {**os.environ, **(env or {})}
        self._proc: subprocess.Popen[str] | None = None
        self._request_id = 0

    def start(self, timeout: float = 10.0) -> None:
        """Start the server subprocess."""
        try:
            self._proc = subprocess.Popen(
                [self.command, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=self.env,
            )
        except Exception as exc:
            raise McpStdioError(f"Failed to start MCP server: {exc}") from exc

        # Perform initialize handshake.
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

        # Send initialized notification.
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, tool_name: str, arguments: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
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
        try:
            self._proc.stdin.write("\n")
            self._proc.stdin.flush()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise McpStdioError("MCP server not started")
        try:
            self._proc.stdin.write(json.dumps(message) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            raise McpStdioError(f"Failed to send MCP request: {exc}") from exc

    def _read_response(self, expected_id: int, timeout: float) -> dict[str, Any]:
        if self._proc is None or self._proc.stdout is None:
            raise McpStdioError("MCP server not started")

        import threading

        result: dict[str, Any] | None = None
        error: Exception | None = None

        def _read() -> None:
            nonlocal result, error
            try:
                for line in self._proc.stdout:
                    line = line.strip()
                    if not line or not line.startswith("{"):
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg_id = msg.get("id")
                    if msg_id == expected_id:
                        result = msg
                        return
            except Exception as exc:
                error = exc

        thread = threading.Thread(target=_read, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if error is not None:
            raise McpStdioError(f"Error reading MCP response: {error}")
        if result is None:
            # Try to capture stderr for diagnosis.
            stderr = ""
            if self._proc.stderr is not None:
                with contextlib.suppress(Exception):
                    stderr = self._proc.stderr.read1(4096).decode("utf-8", errors="ignore")
            raise McpStdioError(
                f"Timeout waiting for MCP response (id={expected_id}).{f' stderr: {stderr}' if stderr else ''}"
            )
        return result


def load_runtime_config(server_name: str, home: Path | str | None = None) -> dict[str, Any]:
    """Load the runtime YAML for an installed MCP server."""
    home = resolve_home(str(home) if home else None)
    path = home / "mcp" / "runtime" / f"{server_name}.yaml"
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
        client.start(timeout=min(timeout, 10.0))
        response = client.call_tool(tool_name, arguments or {}, timeout=timeout)
        if "error" in response:
            raise McpStdioError(f"Tool error: {response['error']}")
        return response.get("result", {})
    finally:
        if client is not None:
            client.stop()


def list_server_tools(
    server_name: str,
    home: Path | str | None = None,
    timeout: float = 10.0,
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


def health_smoke(server_name: str, home: Path | str | None = None, timeout: float = 10.0) -> dict[str, Any]:
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

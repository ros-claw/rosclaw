"""End-to-end smoke tests for the P0 MCP server over stdio and HTTP transports.

These tests start the real ``rosclaw-mcp-serve`` subprocess, connect with the
official MCP SDK clients, discover the seven P0 tools, and call each tool to
verify the JSON envelope shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client

# Ensure the subprocess server imports the branch code, not an installed copy.
_PROJECT_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _server_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _PROJECT_SRC if not existing else f"{_PROJECT_SRC}{os.pathsep}{existing}"
    return env


async def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"Server did not bind to {host}:{port} within {timeout}s")


async def _start_server(*args: str) -> asyncio.subprocess.Process:
    cmd = [sys.executable, "-m", "rosclaw.mcp.server", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_server_env(),
    )
    return proc


def _envelope(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    assert payload["ok"] is True, f"expected ok=True, got {payload}"
    assert payload["schema_version"].startswith("p0.")
    assert "trace_id" in payload
    assert "timestamp" in payload
    assert "data" in payload
    return payload


P0_TOOL_CALLS: list[tuple[str, dict[str, Any]]] = [
    ("get_robot_state", {}),
    ("list_skills", {"skill_type": None}),
    ("query_memory", {"instruction": "pick cup", "limit": 3}),
    ("practice_query", {"limit": 5}),
    (
        "validate_trajectory",
        {"trajectory": [[0.0] * 6, [0.1] * 6], "safety_level": "MODERATE"},
    ),
    ("sandbox_run", {"joint_positions": [0.1] * 6}),
    ("emergency_stop", {"reason": "e2e test halt"}),
    ("list_bodies", {}),
    ("get_body", {"body_id": "current"}),
    ("switch_body", {"body_id": "current"}),
    ("list_body_history", {"body_id": "current"}),
    ("check_skill_compatibility", {}),
    ("fleet_skill_compatibility", {}),
]

EXPECTED_TOOLS = {name for name, _ in P0_TOOL_CALLS}


@pytest.mark.asyncio
async def test_stdio_smoke(tmp_path: Path) -> None:
    """Discover and exercise every P0 tool through the stdio transport."""
    project_root = str(tmp_path)
    params = StdioServerParameters(
        command=sys.executable,
        args=[
            "-m",
            "rosclaw.mcp.server",
            "--transport",
            "stdio",
            "--log-level",
            "WARNING",
            "--project-root",
            project_root,
        ],
        env=_server_env(),
    )

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        discovered = {t.name for t in tools.tools}
        assert discovered == EXPECTED_TOOLS

        for tool_name, arguments in P0_TOOL_CALLS:
            result = await session.call_tool(tool_name, arguments=arguments)
            assert len(result.content) == 1
            _envelope(result.content[0].text)


@pytest.mark.asyncio
async def test_http_smoke(tmp_path: Path) -> None:
    """Discover and exercise every P0 tool through the streamable HTTP transport."""
    host = "127.0.0.1"
    port = _find_free_port(host)
    project_root = str(tmp_path)
    proc = await _start_server(
        "--transport",
        "http",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "WARNING",
        "--project-root",
        project_root,
    )
    try:
        await _wait_for_port(host, port)
        url = f"http://{host}:{port}/mcp"
        async with (
            streamable_http_client(url) as (read, write, _get_sid),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            discovered = {t.name for t in tools.tools}
            assert discovered == EXPECTED_TOOLS

            for tool_name, arguments in P0_TOOL_CALLS:
                result = await session.call_tool(tool_name, arguments=arguments)
                assert len(result.content) == 1
                _envelope(result.content[0].text)
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except TimeoutError:
                proc.kill()
                await proc.wait()


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the kernel for an ephemeral TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])

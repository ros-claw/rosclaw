"""End-to-end smoke tests for the P0 MCP server over stdio and HTTP transports.

These tests start the real ``rosclaw-mcp-serve`` subprocess, connect with the
official MCP SDK clients, discover the P0 tools, and verify success and
fail-closed error envelopes over both transports.
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

from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.body.service import BodyInstanceService

# Ensure the subprocess server imports the branch code, not an installed copy.
_PROJECT_SRC = str(Path(__file__).resolve().parents[2] / "src")


def _server_env(rosclaw_home: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = _PROJECT_SRC if not existing else f"{_PROJECT_SRC}{os.pathsep}{existing}"
    env["ROSCLAW_HOME"] = str(rosclaw_home)
    return env


async def _wait_for_port(host: str, port: int, timeout: float = 60.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return
        except OSError:
            await asyncio.sleep(0.1)
    raise TimeoutError(f"Server did not bind to {host}:{port} within {timeout}s")


async def _start_server(*args: str, rosclaw_home: Path) -> asyncio.subprocess.Process:
    cmd = [sys.executable, "-m", "rosclaw.mcp.server", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_server_env(rosclaw_home),
    )
    return proc


def _envelope(text: str, *, tool_name: str, expected_ok: bool) -> dict[str, Any]:
    payload = json.loads(text)
    assert payload["ok"] is expected_ok, {"tool": tool_name, "payload": payload}
    assert payload["schema_version"] == "rosclaw.mcp.v1"
    assert "trace_id" in payload
    assert "timestamp" in payload
    if expected_ok:
        assert "data" in payload
    else:
        assert payload["trust_level"] == "UNAVAILABLE"
        assert payload["usable_for_real_execution"] is False
        assert "error" in payload
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
    ("get_body_profile", {}),
    ("get_body_state", {"include_runtime": True}),
    ("list_body_capabilities", {"status": "all"}),
    ("query_body", {"question": "What robot body is this?"}),
    ("validate_body_action", {"action": "walk forward", "capability_id": "walk", "risk": "medium"}),
    ("get_calibration_status", {"component": "head_rgb_camera"}),
    ("get_product_status", {}),
    ("list_product_demos", {}),
]

EXPECTED_TOOLS = set(P0_AGENT_MCP_TOOLS)
EXPECTED_ERROR_TOOLS = {"get_robot_state"}


def _prepare_server_workspace(tmp_path: Path) -> tuple[Path, Path]:
    """Create deterministic project and body state for the subprocess server."""
    project_root = tmp_path / "project"
    rosclaw_home = tmp_path / "rosclaw-home"
    project_root.mkdir()
    BodyInstanceService(workspace=rosclaw_home).create_or_init(
        robot="ur5e",
        name="sim_ur5e",
        mode="single",
    )
    return project_root, rosclaw_home


async def _exercise_p0_tools(session: ClientSession) -> None:
    """Call every advertised P0 tool, including the receipt-dependent workflow."""
    called: set[str] = set()
    for tool_name, arguments in P0_TOOL_CALLS:
        result = await session.call_tool(tool_name, arguments=arguments)
        assert len(result.content) == 1
        _envelope(
            result.content[0].text,
            tool_name=tool_name,
            expected_ok=tool_name not in EXPECTED_ERROR_TOOLS,
        )
        called.add(tool_name)

    demo_result = await session.call_tool(
        "run_product_demo",
        arguments={"demo_id": "ur5e-reach"},
    )
    demo_payload = _envelope(
        demo_result.content[0].text,
        tool_name="run_product_demo",
        expected_ok=True,
    )
    run_id = demo_payload["data"]["receipt"]["action_id"]
    called.add("run_product_demo")

    for tool_name in ("get_execution_receipt", "explain_execution"):
        result = await session.call_tool(
            tool_name,
            arguments={"run_reference": run_id},
        )
        _envelope(result.content[0].text, tool_name=tool_name, expected_ok=True)
        called.add(tool_name)

    assert called == EXPECTED_TOOLS


@pytest.mark.asyncio
async def test_stdio_smoke(tmp_path: Path) -> None:
    """Discover and exercise every P0 tool through the stdio transport."""
    project_root, rosclaw_home = _prepare_server_workspace(tmp_path)
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
            str(project_root),
            "--robot-id",
            "sim_ur5e",
        ],
        env=_server_env(rosclaw_home),
    )

    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        discovered = {t.name for t in tools.tools}
        assert discovered == EXPECTED_TOOLS

        await _exercise_p0_tools(session)


@pytest.mark.asyncio
async def test_http_smoke(tmp_path: Path) -> None:
    """Discover and exercise every P0 tool through the streamable HTTP transport."""
    host = "127.0.0.1"
    port = _find_free_port(host)
    project_root, rosclaw_home = _prepare_server_workspace(tmp_path)
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
        str(project_root),
        "--robot-id",
        "sim_ur5e",
        rosclaw_home=rosclaw_home,
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

            await _exercise_p0_tools(session)
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

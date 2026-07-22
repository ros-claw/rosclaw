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
from rosclaw.daemon.client import DaemonClient, DaemonUnavailableError

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


async def _start_daemon(rosclaw_home: Path) -> tuple[asyncio.subprocess.Process, Path]:
    socket_path = rosclaw_home / "run" / "rosclawd.sock"
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "rosclaw.daemon.cli",
        "--socket",
        str(socket_path),
        "--robot-id",
        "sim_ur5e",
        "--log-level",
        "ERROR",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_server_env(rosclaw_home),
    )
    client = DaemonClient(socket_path=socket_path, timeout_sec=1.0)
    deadline = asyncio.get_running_loop().time() + 30.0
    while asyncio.get_running_loop().time() < deadline:
        if proc.returncode is not None:
            raise RuntimeError(f"rosclawd exited early with status {proc.returncode}")
        if socket_path.exists():
            try:
                await asyncio.to_thread(client.get_runtime_status)
                return proc, socket_path
            except DaemonUnavailableError:
                pass
        await asyncio.sleep(0.05)
    await _terminate_process(proc)
    raise TimeoutError("rosclawd did not become ready")


async def _terminate_process(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()


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

    runtime_result = await session.call_tool("get_runtime_status", arguments={})
    runtime_payload = _envelope(
        runtime_result.content[0].text,
        tool_name="get_runtime_status",
        expected_ok=True,
    )
    assert runtime_payload["data"]["southbound_owner"] == "rosclawd"
    called.add("get_runtime_status")

    action_result = await session.call_tool(
        "request_action",
        arguments={
            "capability_id": "robot.move_joints",
            "arguments": {"joint_positions": [0.0] * 6},
            "execution_mode": "REAL",
            "body_snapshot_hash": "sha256:e2e-body",
            "principal_id": "forged-operator",
            "approval_id": "forged-permit",
            "body_id": "sim_ur5e",
            "action_id": "action-mcp-e2e-forged",
            "required_evidence": "DRIVER_CONFIRMED",
            "wait_timeout_sec": 5.0,
        },
    )
    action_payload = _envelope(
        action_result.content[0].text,
        tool_name="request_action",
        expected_ok=True,
    )
    action_data = action_payload["data"]
    assert action_data["receipt"]["final_state"] == "BLOCKED"
    assert action_data["receipt"]["errors"][0]["code"] == "AUTHORIZATION_REQUIRED"
    action_id = action_data["action_id"]
    called.add("request_action")

    for tool_name in ("get_action_status", "cancel_action"):
        result = await session.call_tool(tool_name, arguments={"action_id": action_id})
        payload = _envelope(
            result.content[0].text,
            tool_name=tool_name,
            expected_ok=True,
        )
        assert payload["data"]["state"] == "FINISHED"
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
    daemon_proc, _socket_path = await _start_daemon(rosclaw_home)
    try:
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
    finally:
        await _terminate_process(daemon_proc)


@pytest.mark.asyncio
async def test_http_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Discover and exercise every P0 tool through the streamable HTTP transport."""
    # The MCP SDK delegates proxy handling to httpx.  Some developer machines
    # intentionally configure a loopback HTTP proxy, so make the local smoke
    # server an explicit direct-connection target for both conventional spellings.
    loopback_hosts = "127.0.0.1,localhost,::1"
    monkeypatch.setenv("NO_PROXY", loopback_hosts)
    monkeypatch.setenv("no_proxy", loopback_hosts)
    host = "127.0.0.1"
    port = _find_free_port(host)
    project_root, rosclaw_home = _prepare_server_workspace(tmp_path)
    daemon_proc, _socket_path = await _start_daemon(rosclaw_home)
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
        await _terminate_process(proc)
        await _terminate_process(daemon_proc)


def _find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the kernel for an ephemeral TCP port on ``host``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])

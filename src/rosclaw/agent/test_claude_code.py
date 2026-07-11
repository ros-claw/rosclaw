"""Command handler for `rosclaw agent test`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.agent.detectors import build_project_profile
from rosclaw.agent.install import AGENT_TARGETS
from rosclaw.agent.merge import read_json_if_exists
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS

_SHELL_DEFAULT_PATTERN = re.compile(r"\$\{([^}:]+):-([^}]+)\}")


@dataclass
class McpProbeResult:
    ok: bool
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _expand_config_value(value: Any, project_root: Path) -> str:
    text = str(value)
    text = text.replace("${PWD}", str(project_root))
    text = _SHELL_DEFAULT_PATTERN.sub(
        lambda match: os.environ.get(match.group(1), match.group(2)),
        text,
    )
    return os.path.expanduser(os.path.expandvars(text))


def _stdio_command(
    server_config: dict[str, Any], project_root: Path
) -> tuple[str, list[str], dict[str, str]]:
    command = _expand_config_value(server_config.get("command", "rosclaw"), project_root)
    args = [_expand_config_value(arg, project_root) for arg in server_config.get("args", [])]
    env = os.environ.copy()
    for key, value in server_config.get("env", {}).items():
        env[str(key)] = _expand_config_value(value, project_root)
    uses_config_command = os.environ.get("ROSCLAW_AGENT_PROBE_USE_CONFIG_COMMAND") == "1"
    if Path(command).name == "rosclaw" and (
        not uses_config_command or shutil.which(command) is None
    ):
        command = sys.executable
        args = ["-m", "rosclaw.cli", *args]
    return command, args, env


async def _probe_stdio_mcp(
    server_config: dict[str, Any],
    *,
    project_root: Path,
) -> McpProbeResult:
    """Start the configured stdio MCP server and verify key tools."""
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except Exception as exc:  # noqa: BLE001
        return McpProbeResult(False, errors=[f"MCP SDK unavailable: {exc}"])

    command, args, env = _stdio_command(server_config, project_root)
    params = StdioServerParameters(command=command, args=args, env=env)

    try:
        async with (
            stdio_client(params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            tools_response = await session.list_tools()
            discovered = sorted(tool.name for tool in tools_response.tools)
            missing = [tool for tool in P0_AGENT_MCP_TOOLS if tool not in discovered]
            errors = [f"Missing MCP tools: {missing}"] if missing else []

            probe_calls: list[tuple[str, dict[str, Any]]] = [
                ("get_robot_state", {}),
                (
                    "validate_trajectory",
                    {"trajectory": [[0.0] * 6], "safety_level": "MODERATE"},
                ),
                ("sandbox_run", {"joint_positions": [0.0] * 6}),
            ]
            for tool_name, arguments in probe_calls:
                result = await session.call_tool(tool_name, arguments=arguments)
                if not result.content:
                    errors.append(f"{tool_name} returned no content")
                    continue
                payload = json.loads(result.content[0].text)
                if payload.get("schema_version") != "rosclaw.mcp.v1":
                    errors.append(f"{tool_name} returned unexpected schema")
                if payload.get("ok") is not True:
                    errors.append(f"{tool_name} returned ok=false: {payload.get('error')}")

            return McpProbeResult(not errors, tools=discovered, errors=errors)
    except Exception as exc:  # noqa: BLE001
        return McpProbeResult(False, errors=[f"MCP stdio probe failed: {exc}"])


def _run_mcp_probe(server_config: dict[str, Any], *, project_root: Path) -> McpProbeResult:
    if server_config.get("type", "stdio") != "stdio":
        return McpProbeResult(
            False,
            errors=["MCP probe currently supports stdio transport only."],
        )
    return asyncio.run(_probe_stdio_mcp(server_config, project_root=project_root))


def _discover_tests(project_root: Path) -> list[Path]:
    """Return the ROSClaw-specific test directories that exist."""
    candidates = [
        project_root / "tests" / "agent",
        project_root / "tests" / "mcp",
        project_root / "tests" / "security",
    ]
    return [p for p in candidates if p.is_dir()]


def _run_pytest(test_dirs: list[Path], *, verbose: bool = False) -> int:
    """Invoke pytest on the discovered directories."""
    cmd = ["pytest", "-q" if not verbose else "-v"]
    cmd.extend(str(p) for p in test_dirs)
    print("Running:", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=os.getcwd())
    except FileNotFoundError:
        print("ERROR: pytest not found. Is it installed?", file=sys.stderr)
        return 1


def cmd_agent_test_claude_code(args: argparse.Namespace) -> int:
    """Implementation of `rosclaw agent test`."""
    project_root = Path(args.project_root) if args.project_root else None
    profile = build_project_profile(project_root=project_root)

    print(f"Agent target: {getattr(args, 'agent', 'claude-code')}")
    print(f"Project root: {profile.project_root}")
    print(f"Robot ID: {profile.robot_id or '(none detected)'}")
    print(f"MCP transport: {profile.default_transport}")
    print()

    generated_paths = {
        ".mcp.json": profile.project_root / ".mcp.json",
        "CLAUDE.md": profile.project_root / "CLAUDE.md",
        "ROSCLAW.md": profile.project_root / "ROSCLAW.md",
        "context.snapshot.json": profile.project_root
        / ".rosclaw"
        / "agent"
        / "context.snapshot.json",
    }

    all_ok = True
    for name, path in generated_paths.items():
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"{name}: {status}")

    mcp_data = read_json_if_exists(generated_paths[".mcp.json"])
    server_config = mcp_data.get("mcpServers", {}).get("rosclaw", {})
    transport = server_config.get("url") or server_config.get("command")
    print(f"MCP server config: {transport or 'not configured'}")

    snapshot = read_json_if_exists(generated_paths["context.snapshot.json"])
    tools = snapshot.get("tools", {}).get("available", [])
    print(f"Tools advertised: {len(tools)}")
    missing = [t for t in P0_AGENT_MCP_TOOLS if t not in tools]
    if missing:
        print(f"Missing tools: {missing}")
        all_ok = False

    if args.mcp_probe:
        probe = _run_mcp_probe(server_config, project_root=profile.project_root)
        print(f"MCP stdio probe: {'OK' if probe.ok else 'FAILED'}")
        if probe.tools:
            print(f"MCP tools discovered: {len(probe.tools)}")
        for error in probe.errors:
            print(f"  - {error}")
        if not probe.ok:
            all_ok = False

    print()

    if args.quick:
        print("Quick checks complete.")
        return 0 if all_ok else 1

    test_dirs = _discover_tests(profile.project_root)
    if not test_dirs:
        print("No ROSClaw tests discovered in tests/agent, tests/mcp, or tests/security.")
        return 0 if all_ok else 1

    pytest_exit = _run_pytest(test_dirs, verbose=args.verbose)
    if pytest_exit != 0:
        all_ok = False

    return 0 if all_ok else pytest_exit


def add_test_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "test",
        help="Run onboarding and MCP tests for ROSClaw agent integrations.",
    )
    parser.add_argument("agent", choices=AGENT_TARGETS, help="Agent target.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip running pytest and only check generated files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run pytest with verbose output.",
    )
    parser.add_argument(
        "--mcp-probe",
        action="store_true",
        help="Start the configured stdio MCP server and verify tool discovery plus envelopes.",
    )
    parser.set_defaults(func=cmd_agent_test_claude_code)


__all__ = ["cmd_agent_test_claude_code", "add_test_parser"]

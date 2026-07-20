"""Command handler for `rosclaw agent test`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rosclaw.agent.detectors import build_project_profile
from rosclaw.agent.harness_readiness import inspect_codex_project_trust
from rosclaw.agent.install import AGENT_TARGETS
from rosclaw.agent.merge import read_json_if_exists
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.agent.validate import agent_target_paths, validate_project

_SHELL_DEFAULT_PATTERN = re.compile(r"\$\{([^}:]+):-([^}]+)\}")
_EXPECTED_READINESS_CODES = {
    "BODY_NOT_FOUND",
    "RUNTIME_UNAVAILABLE",
    "STATE_PROVENANCE_UNAVAILABLE",
    "SYNTHETIC_SOURCE_NOT_ALLOWED",
}


@dataclass
class McpProbeResult:
    ok: bool
    tools: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    readiness_limits: list[str] = field(default_factory=list)
    verified_run_id: str | None = None


def _assess_probe_payload(
    tool_name: str,
    payload: Any,
) -> tuple[str | None, str | None]:
    """Classify a tool envelope as healthy, truthfully unavailable, or broken."""
    if not isinstance(payload, dict):
        return f"{tool_name} returned a non-object envelope", None
    if payload.get("schema_version") != "rosclaw.mcp.v1":
        return f"{tool_name} returned unexpected schema", None
    if payload.get("ok") is True:
        return None, None
    if payload.get("ok") is not False:
        return f"{tool_name} returned an envelope without boolean ok", None

    error = payload.get("error")
    if not isinstance(error, dict):
        return f"{tool_name} returned malformed error details", None
    code = str(error.get("code", ""))
    details = error.get("details")
    if (
        code in _EXPECTED_READINESS_CODES
        and isinstance(details, dict)
        and details.get("usable_for_real_execution") is False
    ):
        return None, f"{tool_name}: unavailable as expected ({code})"
    return f"{tool_name} returned ok=false: {error}", None


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
    use_current_python = os.environ.get("ROSCLAW_AGENT_PROBE_USE_CURRENT_PYTHON") == "1"
    if Path(command).name == "rosclaw" and use_current_python:
        command = sys.executable
        args = ["-m", "rosclaw.entrypoint", *args]
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
            unexpected = [tool for tool in discovered if tool not in P0_AGENT_MCP_TOOLS]
            errors = [f"Missing MCP tools: {missing}"] if missing else []
            if unexpected:
                errors.append(f"Unexpected MCP tools outside the P0 boundary: {unexpected}")
            readiness_limits: list[str] = []

            async def call_probe_tool(
                tool_name: str,
                arguments: dict[str, Any],
            ) -> dict[str, Any] | None:
                result = await session.call_tool(tool_name, arguments=arguments)
                if not result.content:
                    errors.append(f"{tool_name} returned no content")
                    return None
                try:
                    payload = json.loads(result.content[0].text)
                except (AttributeError, json.JSONDecodeError, TypeError) as exc:
                    errors.append(f"{tool_name} returned invalid JSON: {exc}")
                    return None
                error, readiness_limit = _assess_probe_payload(tool_name, payload)
                if error:
                    errors.append(error)
                if readiness_limit:
                    readiness_limits.append(readiness_limit)
                return payload

            for tool_name in ("get_product_status", "list_product_demos"):
                await call_probe_tool(tool_name, {})

            verified_run_id: str | None = None
            demo_payload = await call_probe_tool("run_product_demo", {})
            if demo_payload and demo_payload.get("ok") is True:
                data = demo_payload.get("data")
                receipt = data.get("receipt") if isinstance(data, dict) else None
                if not isinstance(receipt, dict):
                    errors.append("run_product_demo returned no receipt")
                else:
                    checks = {
                        "final_state": receipt.get("final_state") == "COMPLETED",
                        "evidence_level": receipt.get("evidence_level") == "TASK_VERIFIED",
                        "physics": (
                            isinstance(receipt.get("simulation_result"), dict)
                            and receipt["simulation_result"].get("has_physics") is True
                        ),
                        "verification": (
                            isinstance(receipt.get("verification_result"), dict)
                            and receipt["verification_result"].get("success") is True
                        ),
                        "simulation_boundary": (
                            demo_payload.get("usable_for_real_execution") is False
                        ),
                    }
                    failed_checks = [name for name, passed in checks.items() if not passed]
                    if failed_checks:
                        errors.append(
                            f"run_product_demo failed verified workflow checks: {failed_checks}"
                        )
                    action_id = receipt.get("action_id")
                    if isinstance(action_id, str) and action_id:
                        verified_run_id = action_id
                    else:
                        errors.append("run_product_demo receipt has no action_id")

            if verified_run_id:
                stored_payload = await call_probe_tool(
                    "get_execution_receipt",
                    {"run_reference": verified_run_id},
                )
                stored_data = (
                    stored_payload.get("data")
                    if stored_payload and stored_payload.get("ok") is True
                    else None
                )
                stored_receipt = (
                    stored_data.get("receipt") if isinstance(stored_data, dict) else None
                )
                if (
                    not isinstance(stored_receipt, dict)
                    or stored_receipt.get("action_id") != verified_run_id
                ):
                    errors.append("get_execution_receipt did not return the verified run")

                explanation_payload = await call_probe_tool(
                    "explain_execution",
                    {"run_reference": verified_run_id},
                )
                explanation_data = (
                    explanation_payload.get("data")
                    if explanation_payload and explanation_payload.get("ok") is True
                    else None
                )
                explanation = (
                    explanation_data.get("explanation")
                    if isinstance(explanation_data, dict)
                    else None
                )
                verification = (
                    explanation.get("verification") if isinstance(explanation, dict) else None
                )
                if (
                    not isinstance(explanation, dict)
                    or explanation.get("run_id") != verified_run_id
                    or not isinstance(verification, dict)
                    or verification.get("task_verified") is not True
                ):
                    errors.append("explain_execution did not verify the persisted run")

            readiness_probe_calls: list[tuple[str, dict[str, Any]]] = [
                ("get_robot_state", {}),
                (
                    "validate_trajectory",
                    {"trajectory": [[0.0] * 6], "safety_level": "MODERATE"},
                ),
                ("sandbox_run", {"joint_positions": [0.0] * 6}),
            ]
            for tool_name, arguments in readiness_probe_calls:
                await call_probe_tool(tool_name, arguments)

            return McpProbeResult(
                not errors,
                tools=discovered,
                errors=errors,
                readiness_limits=readiness_limits,
                verified_run_id=verified_run_id,
            )
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
    target = getattr(args, "agent", "claude-code")

    print(f"Agent target: {target}")
    print(f"Project root: {profile.project_root}")
    print(f"Robot ID: {profile.robot_id or '(none detected)'}")
    print(f"MCP transport: {profile.default_transport}")
    print()

    generated_paths = agent_target_paths(profile.project_root, target)

    all_ok = True
    for name, path in generated_paths.items():
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"{name}: {status}")
    validation = validate_project(
        profile.project_root,
        generated_paths,
        skip_secrets=True,
    )
    for error in validation.errors:
        print(f"Configuration error: {error}")
    if not validation.ok:
        all_ok = False

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

    if target == "codex":
        trust = inspect_codex_project_trust(profile.project_root)
        print(f"Codex project trust: {'yes' if trust.trusted else 'no'} ({trust.detail})")
        if not trust.trusted:
            print(
                "  - Codex will ignore .codex/config.toml until this exact repository is trusted."
            )
            all_ok = False

    if args.mcp_probe:
        probe = _run_mcp_probe(server_config, project_root=profile.project_root)
        print(f"MCP stdio probe: {'OK' if probe.ok else 'FAILED'}")
        if probe.tools:
            print(f"MCP tools discovered: {len(probe.tools)}")
        if probe.verified_run_id:
            print(f"MCP verified simulation run: {probe.verified_run_id}")
        for error in probe.errors:
            print(f"  - {error}")
        for readiness_limit in probe.readiness_limits:
            print(f"  - Readiness: {readiness_limit}")
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

    return 0 if all_ok else (pytest_exit or 1)


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
        help=(
            "Start the configured stdio MCP server and verify protocol, tool discovery, "
            "envelopes, and truthful readiness responses."
        ),
    )
    parser.set_defaults(func=cmd_agent_test_claude_code)


__all__ = ["cmd_agent_test_claude_code", "add_test_parser"]

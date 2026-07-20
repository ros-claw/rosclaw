"""Command handler for `rosclaw agent doctor`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rosclaw.agent.detectors import build_project_profile
from rosclaw.agent.install import AGENT_TARGETS
from rosclaw.agent.merge import read_json_if_exists
from rosclaw.agent.tool_catalog import P0_AGENT_MCP_TOOLS
from rosclaw.agent.validate import agent_target_paths, validate_project


def _check_server_reachable(profile: dict[str, Any]) -> tuple[bool, str]:
    """Best-effort check whether the configured MCP server is reachable."""
    transport = profile.get("type", "stdio")
    if transport == "stdio":
        return True, "stdio transport (no network reachability check)"

    url = profile.get("url", "http://127.0.0.1:9090/mcp")
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 9090
    try:
        import urllib.request

        health_url = f"http://{host}:{port}/health"
        with urllib.request.urlopen(health_url, timeout=2.0) as resp:  # noqa: S310
            return resp.status == 200, f"HTTP {resp.status} from {health_url}"
    except Exception as e:  # noqa: BLE001
        return False, f"could not reach {host}:{port}: {e}"


def cmd_agent_doctor_claude_code(args: argparse.Namespace) -> int:
    """Implementation of `rosclaw agent doctor`."""
    project_root = Path(args.project_root) if args.project_root else None
    profile = build_project_profile(project_root=project_root)
    target = getattr(args, "agent", "claude-code")

    generated_paths = agent_target_paths(profile.project_root, target)

    validation = validate_project(
        profile.project_root,
        generated_paths,
        skip_secrets=args.skip_secrets,
    )

    print(f"Agent target: {target}")
    print(f"Project root: {profile.project_root}")
    print(f"Robot ID: {profile.robot_id or '(none detected)'}")
    print(f"MCP transport: {profile.default_transport}")
    print()

    if validation.ok:
        print("Onboarding files: OK")
    else:
        print("Onboarding files: FAILED")
        for err in validation.errors:
            print(f"  - {err}")

    if validation.warnings:
        print("Warnings:")
        for warn in validation.warnings:
            print(f"  - {warn}")

    mcp_data = read_json_if_exists(generated_paths[".mcp.json"])
    server_config = mcp_data.get("mcpServers", {}).get("rosclaw", {})
    reachable, message = _check_server_reachable(server_config)
    print(f"MCP server reachable: {'yes' if reachable else 'no'} ({message})")

    snapshot_path = generated_paths["context.snapshot.json"]
    if snapshot_path.exists():
        snapshot = read_json_if_exists(snapshot_path)
        tools = snapshot.get("tools", {}).get("available", [])
        print(f"Expected tools: {len(tools)}")
        missing = [t for t in P0_AGENT_MCP_TOOLS if t not in tools]
        if missing:
            print(f"Missing tools: {missing}")

    return 0 if validation.ok else 1


def add_doctor_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "doctor",
        help="Diagnose ROSClaw agent onboarding configuration.",
    )
    parser.add_argument("agent", choices=AGENT_TARGETS, help="Agent target.")
    parser.add_argument("--project-root", type=str, default=None, help="Project root path.")
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip the heuristic secrets scan.",
    )
    parser.set_defaults(func=cmd_agent_doctor_claude_code)


__all__ = ["cmd_agent_doctor_claude_code", "add_doctor_parser"]

"""Command handler for `rosclaw agent test claude-code`."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from rosclaw.agent.detectors import build_project_profile
from rosclaw.agent.merge import read_json_if_exists


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
    """Implementation of `rosclaw agent test claude-code`."""
    project_root = Path(args.project_root) if args.project_root else None
    profile = build_project_profile(project_root=project_root)

    print(f"Project root: {profile.project_root}")
    print(f"Robot ID: {profile.robot_id or '(none detected)'}")
    print(f"MCP transport: {profile.default_transport}")
    print()

    generated_paths = {
        ".mcp.json": profile.project_root / ".mcp.json",
        "CLAUDE.md": profile.project_root / "CLAUDE.md",
        "ROSCLAW.md": profile.project_root / "ROSCLAW.md",
        "context.snapshot.json": profile.project_root / ".rosclaw" / "agent" / "context.snapshot.json",
    }

    all_ok = True
    for name, path in generated_paths.items():
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            all_ok = False
        print(f"{name}: {status}")

    mcp_data = read_json_if_exists(generated_paths[".mcp.json"])
    server_config = mcp_data.get("servers", {}).get("rosclaw-p0", {})
    transport = server_config.get("url") or server_config.get("command")
    print(f"MCP server config: {transport or 'not configured'}")

    snapshot = read_json_if_exists(generated_paths["context.snapshot.json"])
    tools = snapshot.get("tools", {}).get("available", [])
    print(f"Tools advertised: {len(tools)}")
    expected = [
        "get_robot_state",
        "list_skills",
        "query_memory",
        "validate_trajectory",
        "sandbox_run",
        "practice_query",
        "emergency_stop",
    ]
    missing = [t for t in expected if t not in tools]
    if missing:
        print(f"Missing tools: {missing}")
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
        help="Run onboarding and MCP tests for Claude Code.",
    )
    parser.add_argument("agent", choices=["claude-code"], help="Agent target.")
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
    parser.set_defaults(func=cmd_agent_test_claude_code)


__all__ = ["cmd_agent_test_claude_code", "add_test_parser"]

"""Command handler for universal ROSClaw agent installation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rosclaw.agent.detectors import build_project_profile
from rosclaw.agent.init_claude_code import _generate_files
from rosclaw.agent.validate import validate_project

AGENT_TARGETS = ("universal", "all", "claude-code", "codex", "openclaw")


def cmd_agent_install(args: argparse.Namespace) -> int:
    """Install cross-agent ROSClaw onboarding files into a project."""
    project_root = Path(args.project_root) if args.project_root else None
    profile = build_project_profile(
        project_root=project_root,
        profile=args.profile,
        robot=args.robot,
        transport=args.transport,
        host=args.host,
        port=args.port,
    )

    generated = _generate_files(
        profile.project_root,
        profile,
        check_mode=args.check,
        dry_run=args.dry_run,
        include_universal=True,
    )

    if args.dry_run:
        print("Would generate the following cross-agent ROSClaw files:")
        for key, path in generated.items():
            print(f"  {key}: {path}")
        return 0

    validation = validate_project(
        profile.project_root,
        generated,
        skip_secrets=args.skip_secrets,
    )

    if not validation.ok:
        print("ERROR: generated files failed validation:", file=sys.stderr)
        for err in validation.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    target = getattr(args, "target", "universal")
    print("ROSClaw universal agent integration installed.")
    print(f"Target: {target}")
    print(f"Project root: {profile.project_root}")
    print(f"MCP transport: {profile.default_transport}")
    print("Generated files:")
    for key in generated:
        print(f"  - {key}")
    print()
    print("One-line instruction for any agent:")
    print("  Run `rosclaw agent install --project-root . --skip-secrets`, then read ROSCLAW.md.")

    if validation.warnings:
        print("Warnings:")
        for warn in validation.warnings:
            print(f"  - {warn}")

    return 0


def add_install_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "install",
        help="Install cross-agent ROSClaw onboarding files.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        choices=AGENT_TARGETS,
        default="universal",
        help="Agent target to prepare; universal/all emits files useful to Codex, Claude Code, and OpenClaw.",
    )
    parser.add_argument("--project-root", type=str, default=None, help="Project root path.")
    parser.add_argument("--profile", type=str, default=None, help="Runtime profile YAML path.")
    parser.add_argument("--robot", type=str, default=None, help="Explicit robot ID.")
    parser.add_argument(
        "--transport", choices=["stdio", "http", "sse"], default=None, help="MCP transport."
    )
    parser.add_argument("--host", type=str, default=None, help="MCP HTTP/SSE host.")
    parser.add_argument("--port", type=int, default=None, help="MCP HTTP/SSE port.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Mark generated MCP metadata as check/audit mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be generated without writing them.",
    )
    parser.add_argument(
        "--skip-secrets",
        action="store_true",
        help="Skip the heuristic secrets scan.",
    )
    parser.set_defaults(func=cmd_agent_install)


__all__ = ["cmd_agent_install", "add_install_parser", "AGENT_TARGETS"]

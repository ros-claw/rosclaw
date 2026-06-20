"""Hardware MCP onboarding CLI commands.

Implements ``rosclaw mcp install``, ``rosclaw mcp list``, and
``rosclaw mcp health`` on top of the onboarding engine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root) if getattr(args, "project_root", None) else Path.cwd()


def _print_json(data: dict[str, Any] | list[Any]) -> None:
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def add_mcp_subparser(mcp_subparsers: Any) -> None:
    """Register ``install``, ``list``, and ``health`` under ``rosclaw mcp``."""
    install_parser = mcp_subparsers.add_parser(
        "install", help="Install/register a Hardware MCP server"
    )
    install_parser.add_argument("alias", help="Short name, alias, or canonical manifest ID")
    install_parser.add_argument("--version", default=None, help="Exact version to install")
    install_parser.add_argument("--dry-run", action="store_true", help="Show plan without writing files")
    install_parser.add_argument("--allow-dangerous", action="store_true", help="Auto-grant dangerous permissions")
    install_parser.add_argument(
        "--conflict",
        choices=["abort", "rename", "replace"],
        default="abort",
        help="Strategy for unmanaged .mcp.json collisions",
    )
    install_parser.add_argument("--offline", action="store_true", help="Prefer cache and built-in registry")
    install_parser.add_argument("--skip-body", action="store_true", help="Skip body.yaml binding")
    install_parser.add_argument("--skip-claude", action="store_true", help="Skip .mcp.json merge")
    install_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    install_parser.add_argument("--project-root", default=".", help="Project root path")
    install_parser.add_argument("--yes", action="store_true", help="Non-interactive mode (no-op)")
    install_parser.set_defaults(func=lambda args: dispatch_mcp_install(args))

    list_parser = mcp_subparsers.add_parser(
        "list", help="List installed and available Hardware MCP servers"
    )
    list_parser.add_argument("--installed", action="store_true", help="Show only installed servers")
    list_parser.add_argument("--available", action="store_true", help="Show only available servers")
    list_parser.add_argument("--type", dest="type_filter", default=None, help="Filter by hardware type")
    list_parser.add_argument("--bound", action="store_true", help="Show only bound installed servers")
    list_parser.add_argument("--unbound", action="store_true", help="Show only unbound installed servers")
    list_parser.add_argument("--offline", action="store_true", help="Use offline registry")
    list_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    list_parser.add_argument("--project-root", default=".", help="Project root path")
    list_parser.set_defaults(func=lambda args: dispatch_mcp_list(args))

    health_parser = mcp_subparsers.add_parser(
        "health", help="Run health checks for installed Hardware MCP servers"
    )
    health_parser.add_argument(
        "server_name",
        nargs="?",
        default=None,
        help="Specific server to check (default: all installed)",
    )
    health_parser.add_argument("--full", action="store_true", help="Run hardware/safety checks and MCP handshake")
    health_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    health_parser.add_argument("--project-root", default=".", help="Project root path")
    health_parser.set_defaults(func=lambda args: dispatch_mcp_health(args))


def dispatch_mcp_command(args: argparse.Namespace) -> int:
    """Dispatch ``rosclaw mcp <subcommand>``.

    Falls back to the serve handler when no onboarding subcommand was given.
    """
    command = getattr(args, "mcp_command", None)
    func = getattr(args, "func", None)
    if command == "serve" and func is not None:
        return func(args)
    if func is not None:
        return func(args)
    # Should not reach here because parse_args will have set func for valid commands.
    return 1


def dispatch_mcp_install(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp install``."""
    from rosclaw.mcp.onboarding.errors import OnboardingError
    from rosclaw.mcp.onboarding.installer import InstallEngine

    engine = InstallEngine(project_root=_project_root(args))

    if args.dry_run:
        try:
            plan = engine.plan(args.alias, version=args.version)
        except OnboardingError as exc:
            print(f"[ROSClaw MCP] ❌ Cannot plan install: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ROSClaw MCP] ❌ Unexpected error: {exc}", file=sys.stderr)
            return 1

        if args.json:
            _print_json(plan.to_dict())
            return 0

        solved = plan.solved
        print("=" * 60)
        print("Hardware MCP Install Plan")
        print("=" * 60)
        print(f"Alias:        {args.alias}")
        print(f"Manifest:     {solved.manifest_id}")
        print(f"Version:      {solved.version}")
        print(f"Source:       {solved.source}")
        print(f"Server name:  {plan.manifest.server_name}")
        print(f"Artifact:     {plan.installer_type}")
        print(f"Command:      {plan.install_command or 'N/A'}")
        if plan.body_patch:
            print("Body patch:")
            for key in sorted(plan.body_patch):
                print(f"  {key}: {plan.body_patch[key]}")
        if plan.permission_state:
            print("Permissions:")
            if plan.permission_state.granted:
                print(f"  granted: {', '.join(plan.permission_state.granted)}")
            if plan.permission_state.denied:
                print(f"  denied:  {', '.join(plan.permission_state.denied)}")
        print(f"Claude merge: {plan.claude_action}")
        print("-" * 60)
        print("Dry run: no changes will be written.")
        print("=" * 60)
        return 0

    try:
        result = engine.install(
            args.alias,
            version=args.version,
            dry_run=False,
            allow_dangerous=args.allow_dangerous,
            conflict=args.conflict,
            offline=args.offline,
            skip_body=args.skip_body,
            skip_claude=args.skip_claude,
        )
    except OnboardingError as exc:
        print(f"[ROSClaw MCP] ❌ Installation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw MCP] ❌ Unexpected error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(result.to_dict())
        return 0 if result.success else 1

    if result.success:
        print("=" * 60)
        print("Hardware MCP Installation")
        print("=" * 60)
        print(f"Server name:  {result.server_name}")
        print(f"Manifest:     {result.manifest_id}")
        print(f"Version:      {result.version}")
        if result.runtime_config_path:
            print(f"Runtime config: {result.runtime_config_path}")
        if result.runner_script_path:
            print(f"Runner script:  {result.runner_script_path}")
        if result.binding_result:
            print(f"Body binding:   {result.binding_result.binding_key}")
            if result.binding_result.patched_paths:
                print(f"  paths:        {', '.join(result.binding_result.patched_paths)}")
        if result.claude_result:
            print(f"Claude merge:   {result.claude_result.action}")
        print("Status:       installed")
        print("=" * 60)
        return 0

    print("=" * 60)
    print("Hardware MCP Installation")
    print("=" * 60)
    print(f"Server name:  {result.server_name}")
    print(f"Manifest:     {result.manifest_id}")
    print(f"Version:      {result.version}")
    print("Status:       failed")
    for error in result.errors:
        print(f"  • {error}")
    print("=" * 60)
    return 1


def dispatch_mcp_list(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp list``."""
    from rosclaw.mcp.onboarding.hub_client import HubClient
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    installed_registry = InstalledRegistry()
    installed = installed_registry.list()

    show_installed = not args.available or args.installed
    show_available = not args.installed or args.available

    # Build installed entries.
    installed_entries: list[dict[str, Any]] = []
    for record in installed:
        bound = bool(record.body_binding_key)
        if args.bound and not bound:
            continue
        if args.unbound and bound:
            continue
        entry = {
            "server_name": record.server_name,
            "manifest_id": record.manifest_id,
            "version": record.version,
            "artifact_type": record.artifact_type,
            "status": record.status,
            "bound": bound,
        }
        if args.type_filter is None:
            installed_entries.append(entry)
        else:
            try:
                hub = HubClient(offline=args.offline)
                manifest = hub.fetch_manifest(record.manifest_id, record.version)
                hw_type = manifest.hardware.type if manifest.hardware else None
            except Exception:  # noqa: BLE001
                hw_type = None
            if hw_type == args.type_filter:
                installed_entries.append(entry)

    # Build available entries.
    available_entries: list[dict[str, Any]] = []
    if show_available:
        hub = HubClient(offline=args.offline)
        index = hub.fetch_index()
        installed_ids = {r.manifest_id for r in installed}
        for manifest_id, meta in index.items():
            if manifest_id in installed_ids:
                continue
            try:
                manifest = hub.fetch_manifest(manifest_id)
            except Exception:  # noqa: BLE001
                continue
            if args.type_filter:
                hw_type = manifest.hardware.type if manifest.hardware else None
                if hw_type != args.type_filter:
                    continue
            available_entries.append({
                "manifest_id": manifest_id,
                "name": manifest.name,
                "display_name": manifest.display_name,
                "version": manifest.version,
                "description": manifest.description,
                "hardware_type": manifest.hardware.type if manifest.hardware else None,
                "versions": [v.get("version") for v in meta.get("versions", [])],
            })

    if args.json:
        _print_json({
            "installed": installed_entries,
            "available": available_entries,
        })
        return 0

    print("=" * 60)
    print("Hardware MCP Servers")
    print("=" * 60)

    if show_installed:
        print("\nInstalled:")
        if installed_entries:
            for entry in installed_entries:
                bound_marker = "bound" if entry["bound"] else "unbound"
                print(
                    f"  {entry['server_name']:20} "
                    f"{entry['manifest_id']:40} "
                    f"{entry['version']:10} "
                    f"{entry['artifact_type']:10} "
                    f"{bound_marker}"
                )
        else:
            print("  (none)")

    if show_available:
        print("\nAvailable:")
        if available_entries:
            for entry in available_entries:
                print(
                    f"  {entry['name']:20} "
                    f"{entry['manifest_id']:40} "
                    f"{entry['version']:10} "
                    f"{entry.get('hardware_type') or 'unknown':10}"
                )
                if entry.get("description"):
                    print(f"    {entry['description']}")
        else:
            print("  (none)")

    print("=" * 60)
    return 0


def dispatch_mcp_health(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp health``."""
    from rosclaw.mcp.onboarding.health import HealthRunner

    runner = HealthRunner()

    try:
        if args.server_name:
            reports = [runner.check(args.server_name, full=args.full)]
        else:
            reports = runner.check_all(full=args.full)
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw MCP] ❌ Health check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print_json([r.to_dict() for r in reports])
        return 1 if any(r.overall == "failed" for r in reports) else 0

    failed_any = False
    for report in reports:
        print("=" * 60)
        print(f"{report.server_name} — {report.overall.upper()}")
        if report.manifest_id:
            print(f"  manifest: {report.manifest_id} @ {report.version}")
        for check in report.checks:
            status = "PASS" if check.passed else "FAIL"
            req = "required" if check.required else "optional"
            print(f"  [{status}] {check.category:12} {check.check_id:24} ({req}) {check.message}")
        if report.skipped:
            print(f"  skipped: {', '.join(report.skipped)}")
        if report.overall == "failed":
            failed_any = True
        print("=" * 60)

    return 1 if failed_any else 0

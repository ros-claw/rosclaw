"""Hardware MCP onboarding CLI commands.

Implements ``rosclaw mcp install``, ``rosclaw mcp list``,
``rosclaw mcp health``, ``rosclaw mcp inspect``, and ``rosclaw mcp call``
on top of the onboarding engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from rosclaw.mcp.onboarding import source_installer, stdio_client


def _project_root(args: argparse.Namespace) -> Path:
    return Path(args.project_root) if getattr(args, "project_root", None) else Path.cwd()


def _print_json(data: dict[str, Any] | list[Any]) -> None:
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def add_mcp_subparser(mcp_subparsers: Any) -> None:
    """Register ``install``, ``list``, ``health``, ``inspect``, and ``call`` under ``rosclaw mcp``."""
    install_parser = mcp_subparsers.add_parser(
        "install", help="Install/register a Hardware MCP server"
    )
    install_parser.add_argument(
        "alias", nargs="?", default=None, help="Short name, alias, or canonical manifest ID"
    )
    install_parser.add_argument("--version", default=None, help="Exact version to install")
    install_parser.add_argument(
        "--from-git", dest="from_git", default=None, help="Install from a public git URL"
    )
    install_parser.add_argument(
        "--local-path", dest="local_path", default=None, help="Install from a local directory path"
    )
    install_parser.add_argument(
        "--python",
        dest="python",
        default=None,
        help="Python interpreter to use for dependency installation",
    )
    install_parser.add_argument(
        "--venv", dest="venv", default=None, help="Path to a virtualenv to use for the server"
    )
    install_parser.add_argument(
        "--no-install-deps",
        dest="no_install_deps",
        action="store_true",
        help="Skip dependency installation",
    )
    install_parser.add_argument(
        "--dry-run", action="store_true", help="Show plan without writing files"
    )
    install_parser.add_argument(
        "--allow-dangerous", action="store_true", help="Auto-grant dangerous permissions"
    )
    install_parser.add_argument(
        "--conflict",
        choices=["abort", "rename", "replace"],
        default="abort",
        help="Strategy for unmanaged .mcp.json collisions",
    )
    install_parser.add_argument(
        "--offline", action="store_true", help="Prefer cache and built-in registry"
    )
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
    list_parser.add_argument(
        "--type", dest="type_filter", default=None, help="Filter by hardware type"
    )
    list_parser.add_argument(
        "--bound", action="store_true", help="Show only bound installed servers"
    )
    list_parser.add_argument(
        "--unbound", action="store_true", help="Show only unbound installed servers"
    )
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
    health_parser.add_argument(
        "--full", action="store_true", help="Run hardware/safety checks and MCP handshake"
    )
    health_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    health_parser.add_argument("--project-root", default=".", help="Project root path")
    health_parser.set_defaults(func=lambda args: dispatch_mcp_health(args))

    inspect_parser = mcp_subparsers.add_parser(
        "inspect", help="Show details for an installed MCP server"
    )
    inspect_parser.add_argument("server_name", help="Installed server name")
    inspect_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    inspect_parser.add_argument("--project-root", default=".", help="Project root path")
    inspect_parser.set_defaults(func=lambda args: dispatch_mcp_inspect(args))

    call_parser = mcp_subparsers.add_parser("call", help="Call a tool on an installed MCP server")
    call_parser.add_argument("server_name", help="Installed server name")
    call_parser.add_argument("tool_name", help="Tool name to call")
    call_parser.add_argument(
        "--args", dest="tool_args", default="{}", help="JSON-encoded tool arguments"
    )
    call_parser.add_argument("--json", action="store_true", help="Output structured JSON")
    call_parser.add_argument("--project-root", default=".", help="Project root path")
    call_parser.set_defaults(func=lambda args: dispatch_mcp_call(args))


def dispatch_mcp_command(args: argparse.Namespace) -> int:
    """Dispatch ``rosclaw mcp <subcommand>``.

    Falls back to the serve handler when no onboarding subcommand was given.
    """
    command = getattr(args, "mcp_command", None)
    func = cast(Callable[[argparse.Namespace], int] | None, getattr(args, "func", None))
    if command == "serve" and func is not None:
        return func(args)
    if func is not None:
        return func(args)
    # Should not reach here because parse_args will have set func for valid commands.
    return 1


def dispatch_mcp_install(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp install``."""
    from rosclaw.mcp.onboarding.errors import OnboardingError
    from rosclaw.mcp.onboarding.hub_client import HubClient
    from rosclaw.mcp.onboarding.installer import InstallEngine

    # Source-based installs bypass the package registry.
    if args.from_git or args.local_path:
        if args.dry_run:
            plan = {
                "source_type": "git" if args.from_git else "local_path",
                "source_url": args.from_git or args.local_path,
                "server_name": args.alias
                or source_installer._default_name(args.from_git or args.local_path),
                "python": _resolve_python(args),
                "install_deps": not args.no_install_deps,
                "dry_run": True,
            }
            if args.json:
                _print_json(plan)
            else:
                print("=" * 60)
                print("Hardware MCP Source Install Plan")
                print("=" * 60)
                for key, value in plan.items():
                    print(f"{key:20} {value}")
                print("-" * 60)
                print("Dry run: no changes will be written.")
                print("=" * 60)
            return 0

        try:
            if args.from_git:
                source_result = source_installer.install_from_git(
                    url=args.from_git,
                    server_name=args.alias,
                    python=_resolve_python(args),
                    no_install_deps=args.no_install_deps,
                )
            else:
                source_result = source_installer.install_from_local_path(
                    path=Path(args.local_path),
                    server_name=args.alias,
                    python=_resolve_python(args),
                    no_install_deps=args.no_install_deps,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[ROSClaw MCP] ❌ Source install failed: {exc}", file=sys.stderr)
            return 1

        if args.json:
            _print_json(source_result.to_dict())
            return 0 if source_result.success else 1

        print("=" * 60)
        print("Hardware MCP Source Installation")
        print("=" * 60)
        print(f"Server name:  {source_result.server_name}")
        print(f"Manifest:     {source_result.manifest_id}")
        print(f"Version:      {source_result.version}")
        print(f"Source:       {source_result.source_url}")
        print(f"Local path:   {source_result.local_path}")
        if source_result.commit:
            print(f"Commit:       {source_result.commit}")
        print(f"Runtime:      {source_result.runtime_config_path}")
        print(f"Status:       {'installed' if source_result.success else 'failed'}")
        if source_result.errors:
            for error in source_result.errors:
                print(f"  • {error}")
        print("=" * 60)
        return 0 if source_result.success else 1

    if not args.alias:
        print(
            "[ROSClaw MCP] ❌ An alias or manifest ID is required for registry installs.",
            file=sys.stderr,
        )
        print(
            "Hint: use --from-git URL or --local-path PATH to install from source.",
            file=sys.stderr,
        )
        return 1

    hub = HubClient(offline=args.offline, cache_writes=not args.dry_run)
    engine = InstallEngine(project_root=_project_root(args), hub=hub)

    if args.dry_run:
        try:
            install_plan = engine.plan(args.alias, version=args.version)
        except OnboardingError as exc:
            print(f"[ROSClaw MCP] ❌ Cannot plan install: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ROSClaw MCP] ❌ Unexpected error: {exc}", file=sys.stderr)
            return 1

        if args.json:
            _print_json(install_plan.to_dict())
            return 0

        solved = install_plan.solved
        print("=" * 60)
        print("Hardware MCP Install Plan")
        print("=" * 60)
        print(f"Alias:        {args.alias}")
        print(f"Manifest:     {solved.manifest_id}")
        print(f"Version:      {solved.version}")
        print(f"Source:       {solved.source}")
        print(f"Server name:  {install_plan.manifest.server_name}")
        print(f"Artifact:     {install_plan.installer_type}")
        print(f"Command:      {install_plan.install_command or 'N/A'}")
        if install_plan.body_patch:
            print("Body patch:")
            for key in sorted(install_plan.body_patch):
                print(f"  {key}: {install_plan.body_patch[key]}")
        if install_plan.permission_state:
            print("Permissions:")
            if install_plan.permission_state.granted:
                print(f"  granted: {', '.join(install_plan.permission_state.granted)}")
            if install_plan.permission_state.denied:
                print(f"  denied:  {', '.join(install_plan.permission_state.denied)}")
        print(f"Claude merge: {install_plan.claude_action}")
        print("-" * 60)
        print("Dry run: no changes will be written.")
        print("=" * 60)
        return 0

    try:
        install_result = engine.install(
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
        _print_json(install_result.to_dict())
        return 0 if install_result.success else 1

    if install_result.success:
        print("=" * 60)
        print("Hardware MCP Installation")
        print("=" * 60)
        print(f"Server name:  {install_result.server_name}")
        print(f"Manifest:     {install_result.manifest_id}")
        print(f"Version:      {install_result.version}")
        if install_result.runtime_config_path:
            print(f"Runtime config: {install_result.runtime_config_path}")
        if install_result.runner_script_path:
            print(f"Runner script:  {install_result.runner_script_path}")
        if install_result.binding_result:
            print(f"Body binding:   {install_result.binding_result.binding_key}")
            if install_result.binding_result.patched_paths:
                print(f"  paths:        {', '.join(install_result.binding_result.patched_paths)}")
        if install_result.claude_result:
            print(f"Claude merge:   {install_result.claude_result.action}")
        print("Status:       installed")
        print("=" * 60)
        return 0

    print("=" * 60)
    print("Hardware MCP Installation")
    print("=" * 60)
    print(f"Server name:  {install_result.server_name}")
    print(f"Manifest:     {install_result.manifest_id}")
    print(f"Version:      {install_result.version}")
    print("Status:       failed")
    for error in install_result.errors:
        print(f"  • {error}")
    print("=" * 60)
    return 1


def _resolve_python(args: argparse.Namespace) -> str | None:
    """Return the effective Python interpreter for source installs."""
    if args.venv:
        venv_python = Path(args.venv) / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        # Also allow a Windows Scripts directory if present.
        venv_python_win = Path(args.venv) / "Scripts" / "python.exe"
        if venv_python_win.exists():
            return str(venv_python_win)
        print(
            f"[ROSClaw MCP] ⚠️  Venv not found at {args.venv}; falling back to system Python.",
            file=sys.stderr,
        )
    return cast(str | None, args.python)


def dispatch_mcp_list(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp list``."""
    from rosclaw.mcp.onboarding.hub_client import HubClient
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    installed_registry = InstalledRegistry()
    installed = installed_registry.list()

    show_installed = not args.available or args.installed
    show_available = not args.installed or args.available

    # Avoid remote network timeouts when no registry is explicitly configured.
    # Users can still set ROSCLAW_MCP_HUB or pass --online to fetch remote index.
    offline = args.offline or not os.environ.get("ROSCLAW_MCP_HUB")

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
            "source_type": record.extra.get("source_type"),
            "source_url": record.extra.get("source_url"),
        }
        if args.type_filter is None:
            installed_entries.append(entry)
        else:
            try:
                hub = HubClient(offline=offline)
                manifest = hub.fetch_manifest(record.manifest_id, record.version)
                hw_type = manifest.hardware.type if manifest.hardware else None
            except Exception:  # noqa: BLE001
                hw_type = None
            if hw_type == args.type_filter:
                installed_entries.append(entry)

    # Build available entries.
    available_entries: list[dict[str, Any]] = []
    hub_error: str | None = None
    if show_available:
        try:
            hub = HubClient(offline=offline)
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
                available_entries.append(
                    {
                        "manifest_id": manifest_id,
                        "name": manifest.name,
                        "display_name": manifest.display_name,
                        "version": manifest.version,
                        "description": manifest.description,
                        "hardware_type": manifest.hardware.type if manifest.hardware else None,
                        "versions": [v.get("version") for v in meta.get("versions", [])],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            hub_error = str(exc)
            available_entries = []

    if args.json:
        payload: dict[str, Any] = {
            "installed": installed_entries,
            "available": available_entries,
        }
        if hub_error:
            payload["available_error"] = hub_error
        _print_json(payload)
        return 0

    print("=" * 60)
    print("Hardware MCP Servers")
    print("=" * 60)

    if show_installed:
        print("\nInstalled:")
        if installed_entries:
            for entry in installed_entries:
                bound_marker = "bound" if entry["bound"] else "unbound"
                source_marker = f" ({entry.get('source_type')})" if entry.get("source_type") else ""
                print(
                    f"  {entry['server_name']:20} "
                    f"{entry['manifest_id']:40} "
                    f"{entry['version']:10} "
                    f"{entry['artifact_type']:10} "
                    f"{bound_marker}{source_marker}"
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
        elif hub_error:
            print(f"  (registry unavailable: {hub_error})")
            print("  Hint: install from source with rosclaw mcp install --from-git <url>")
        else:
            print("  (none)")

    print("=" * 60)
    return 0


def dispatch_mcp_health(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp health``."""
    from rosclaw.mcp.onboarding.health import HealthRunner
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    registry = InstalledRegistry()

    def _check_server(server_name: str) -> dict[str, Any]:
        record = registry.get(server_name)
        source_type = record.extra.get("source_type") if record else None
        if source_type in ("git", "local_path"):
            return stdio_client.health_smoke(server_name, timeout=20.0)
        runner = HealthRunner()
        report = runner.check(server_name, full=args.full)
        return {
            "server_name": report.server_name,
            "overall": report.overall,
            "manifest_id": report.manifest_id,
            "version": report.version,
            "checks": [c.to_dict() for c in report.checks],
            "skipped": report.skipped,
        }

    try:
        if args.server_name:
            reports = [_check_server(args.server_name)]
        else:
            installed = registry.list()
            if not installed:
                if args.json:
                    _print_json(
                        {
                            "servers": [],
                            "count": 0,
                            "message": "No installed Hardware MCP servers.",
                        }
                    )
                else:
                    print("[ROSClaw MCP] No installed Hardware MCP servers.")
                    print(
                        "Install one with `rosclaw mcp install <alias>` or inspect options with `rosclaw mcp list`."
                    )
                return 0
            reports = [_check_server(r.server_name) for r in installed]
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw MCP] ❌ Health check failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(reports)
        return (
            1
            if any(r.get("overall") == "failed" or not r.get("healthy", True) for r in reports)
            else 0
        )

    failed_any = False
    for report in reports:
        overall = report.get("overall")
        if overall is None:
            overall = "healthy" if report.get("healthy") else "failed"
        print("=" * 60)
        print(f"{report.get('server_name')} — {overall.upper()}")
        if report.get("manifest_id"):
            print(f"  manifest: {report['manifest_id']} @ {report.get('version')}")
        if "tools_count" in report:
            print(f"  tools: {report['tools_count']}")
            if report.get("tools"):
                print(f"    {', '.join(report['tools'])}")
        if report.get("error"):
            print(f"  error: {report['error']}")
            failed_any = True
        for check in report.get("checks", []):
            status = "PASS" if check.get("passed") else "FAIL"
            req = "required" if check.get("required") else "optional"
            print(
                f"  [{status}] {check.get('category', ''):12} "
                f"{check.get('check_id', ''):24} ({req}) {check.get('message', '')}"
            )
        if report.get("skipped"):
            print(f"  skipped: {', '.join(report['skipped'])}")
        if overall == "failed":
            failed_any = True
        print("=" * 60)

    return 1 if failed_any else 0


def dispatch_mcp_inspect(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp inspect``."""
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    record = InstalledRegistry().get(args.server_name)
    if record is None:
        print(f"[ROSClaw MCP] ❌ Server not installed: {args.server_name}", file=sys.stderr)
        return 1

    data = record.to_dict()
    if args.json:
        _print_json(data)
        return 0

    print("=" * 60)
    print(f"MCP Server: {record.server_name}")
    print("=" * 60)
    print(f"  manifest_id:      {record.manifest_id}")
    print(f"  name:             {record.name}")
    print(f"  version:          {record.version}")
    print(f"  artifact_type:    {record.artifact_type}")
    print(f"  status:           {record.status}")
    print(f"  installed_at:     {record.installed_at}")
    print(f"  server_dir:       {record.server_dir}")
    if record.runtime_config_path:
        print(f"  runtime_config:   {record.runtime_config_path}")
    if record.body_binding_key:
        print(f"  body_binding_key: {record.body_binding_key}")
    if record.eurdf_profile:
        print(f"  eurdf_profile:    {record.eurdf_profile}")
    if record.extra:
        print("  extra:")
        for key, value in record.extra.items():
            print(f"    {key}: {value}")
    print("=" * 60)
    return 0


def dispatch_mcp_call(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp call``."""
    try:
        tool_args = json.loads(args.tool_args)
    except json.JSONDecodeError as exc:
        print(f"[ROSClaw MCP] ❌ Invalid --args JSON: {exc}", file=sys.stderr)
        return 1

    try:
        result = stdio_client.call_server_tool(
            server_name=args.server_name,
            tool_name=args.tool_name,
            arguments=tool_args,
            timeout=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw MCP] ❌ Tool call failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        _print_json(result)
        return 0

    print("=" * 60)
    print(f"MCP Tool Result: {args.server_name}/{args.tool_name}")
    print("=" * 60)
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    print("=" * 60)
    return 0

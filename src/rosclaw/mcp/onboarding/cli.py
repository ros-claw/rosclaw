"""Hardware MCP onboarding CLI commands.

Implements ``rosclaw mcp install``, ``rosclaw mcp list``, and
``rosclaw mcp health`` on top of the onboarding engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast


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

    # mcp call subcommand
    call_parser = mcp_subparsers.add_parser(
        "call", help="Call an MCP tool on a built-in server directly"
    )
    call_parser.add_argument("server", help="Server name (e.g., realsense-d405, realsense-d435i)")
    call_parser.add_argument("tool", help="Tool name to call")
    call_parser.add_argument(
        "--arg", action="append", default=[], help="Tool argument as key=value (repeatable)"
    )
    call_parser.add_argument("--json", action="store_true", help="Output structured JSON")
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


def _infer_value(v: str) -> Any:
    """Infer bool/int/float from a string argument value."""
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def _collect_tool_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """Collect tool arguments from --arg key=value and free-form --key value pairs."""
    tool_args: dict[str, Any] = {}
    for raw in getattr(args, "arg", []):
        if "=" not in raw:
            print(f"[ROSClaw MCP] Invalid --arg (expected key=value): {raw}", file=sys.stderr)
            return None
        k, v = raw.split("=", 1)
        tool_args[k] = _infer_value(v)

    extra = getattr(args, "extra", [])
    i = 0
    while i < len(extra):
        token = extra[i]
        if token.startswith("--"):
            key = token[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                tool_args[k] = _infer_value(v)
                i += 1
                continue
            if i + 1 >= len(extra):
                print(f"[ROSClaw MCP] Missing value for argument --{key}", file=sys.stderr)
                return None
            tool_args[key] = _infer_value(extra[i + 1])
            i += 2
        elif token.startswith("-") and len(token) == 2 and token[1].isalnum():
            # short option -k value
            key = token[1:]
            if i + 1 >= len(extra):
                print(f"[ROSClaw MCP] Missing value for argument -{key}", file=sys.stderr)
                return None
            tool_args[key] = _infer_value(extra[i + 1])
            i += 2
        else:
            print(f"[ROSClaw MCP] Unexpected argument: {token}", file=sys.stderr)
            return None
    return tool_args


def dispatch_mcp_call(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp call <server> <tool>``.

    Supports both ``--arg key=value`` and free-form ``--key value`` arguments.
    For built-in RealSense servers, resolves the server name to the Python
    module and invokes ``run_tool`` directly (no subprocess).  For external
    servers, falls back to spawning the transport command and speaking JSON-RPC
    over stdio.
    """
    server_name: str = args.server
    tool_name: str = args.tool
    tool_args = _collect_tool_args(args)
    if tool_args is None:
        return 1

    # Built-in server short-cut
    builtin_map = {
        "realsense-d405": "rosclaw.mcp.servers.realsense_d405",
        "realsense_d405": "rosclaw.mcp.servers.realsense_d405",
        "realsense-d435i": "rosclaw.mcp.servers.realsense_d435i",
        "realsense_d435i": "rosclaw.mcp.servers.realsense_d435i",
    }
    module_name = builtin_map.get(server_name)
    if module_name:
        # Normalize server alias (realsense-d435i -> realsense_d435i) so
        # built-in helpers can resolve the correct CameraSpec.
        normalized_key = server_name.replace("-", "_")
        tool_args.setdefault("camera_key", normalized_key)
        try:
            import importlib

            mod = importlib.import_module(module_name)
            run_tool = getattr(mod, "run_tool", None)
            if run_tool is None:
                # Fallback to the shared helper in servers.__init__
                from rosclaw.mcp.servers import run_tool as _run_tool

                run_tool = _run_tool
            result = run_tool(tool_name, **tool_args)
        except Exception as exc:
            print(f"[ROSClaw MCP] Tool call failed: {exc}", file=sys.stderr)
            return 1
        if args.json:
            _print_json(result if isinstance(result, dict) else {"result": result})
        else:
            print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
        return 0

    # External server: read runtime config and speak JSON-RPC over stdio
    from rosclaw.mcp.onboarding.installed import InstalledRegistry

    registry = InstalledRegistry()
    record = registry.get(server_name)
    if record is None or not record.runtime_config_path:
        print(f"[ROSClaw MCP] Server '{server_name}' not installed or has no runtime config.", file=sys.stderr)
        return 1

    config_path = Path(record.runtime_config_path)
    if not config_path.exists():
        print(f"[ROSClaw MCP] Runtime config not found: {config_path}", file=sys.stderr)
        return 1

    import yaml

    try:
        runtime_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ROSClaw MCP] Failed to read runtime config: {exc}", file=sys.stderr)
        return 1

    transport = runtime_cfg.get("transport", {})
    cmd = transport.get("command")
    cmd_args = transport.get("args", [])
    env = transport.get("env", {})
    if not cmd:
        print(f"[ROSClaw MCP] No transport command in runtime config.", file=sys.stderr)
        return 1

    import subprocess

    full_env = {**os.environ, **env}
    try:
        proc = subprocess.Popen(
            [cmd] + cmd_args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=full_env,
        )
    except Exception as exc:
        print(f"[ROSClaw MCP] Failed to spawn server: {exc}", file=sys.stderr)
        return 1

    # JSON-RPC helpers
    def _send(obj: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def _recv() -> dict[str, Any] | None:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            return None
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None

    # initialize
    _send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    init_resp = _recv()
    if init_resp is None or "error" in init_resp:
        print(f"[ROSClaw MCP] Initialize failed: {init_resp}", file=sys.stderr)
        proc.terminate()
        return 1

    # tools/list
    _send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    list_resp = _recv()
    if list_resp is None or "error" in list_resp:
        print(f"[ROSClaw MCP] tools/list failed: {list_resp}", file=sys.stderr)
        proc.terminate()
        return 1

    # tools/call
    _send(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": tool_args},
        }
    )
    call_resp = _recv()
    proc.stdin.close()
    proc.terminate()

    if call_resp is None or "error" in call_resp:
        print(f"[ROSClaw MCP] Tool call failed: {call_resp}", file=sys.stderr)
        return 1

    result = call_resp.get("result", {})
    if args.json:
        _print_json(result if isinstance(result, dict) else {"result": result})
    else:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    return 0


def dispatch_mcp_install(args: argparse.Namespace) -> int:
    """Handle ``rosclaw mcp install``."""
    from rosclaw.mcp.onboarding.errors import OnboardingError
    from rosclaw.mcp.onboarding.installer import InstallEngine

    engine = InstallEngine(project_root=_project_root(args))

    if args.dry_run:
        try:
            plan = engine.plan(args.alias, version=args.version)
        except OnboardingError as exc:
            print(f"[ROSClaw MCP] Cannot plan install: {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[ROSClaw MCP] Unexpected error: {exc}", file=sys.stderr)
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
        print(f"[ROSClaw MCP] Installation failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"[ROSClaw MCP] Unexpected error: {exc}", file=sys.stderr)
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

    if args.json:
        _print_json(
            {
                "installed": installed_entries,
                "available": available_entries,
            }
        )
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
        print(f"[ROSClaw MCP] Health check failed: {exc}", file=sys.stderr)
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

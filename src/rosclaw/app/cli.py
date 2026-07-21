"""Thin CLI for installing, authoring, validating, and running Apps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from rosclaw.app.runner import AppRunner
from rosclaw.app.schema import APP_API_VERSION, APP_KIND, AppManifest
from rosclaw.app.store import AppStore, AppStoreError
from rosclaw.daemon.client import DaemonClient, DaemonClientError
from rosclaw.kernel import ExecutionMode


def dispatch_app_argv(argv: list[str]) -> int | None:
    if not argv or argv[0] != "app":
        return None
    parser = _build_parser()
    args = parser.parse_args(argv)
    return dispatch_app_command(args)


def dispatch_app_command(args: argparse.Namespace) -> int:
    handler = getattr(args, "app_handler", None)
    if not callable(handler):
        return 1
    try:
        return int(handler(args))
    except (AppStoreError, DaemonClientError, OSError, ValueError) as exc:
        error_code = str(getattr(exc, "code", "APP_COMMAND_FAILED"))
        payload = {
            "ok": False,
            "error": {
                "code": error_code,
                "message": str(exc),
            },
        }
        if getattr(args, "json", False):
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(f"[ROSClaw] {error_code}: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rosclaw")
    app = parser.add_subparsers(dest="command").add_parser("app", help="Capability App commands")
    commands = app.add_subparsers(dest="app_command")
    add_app_subparsers(commands)
    return parser


def add_app_subparsers(commands: Any) -> None:
    """Add capability-only App commands to an argparse subparser."""

    install = commands.add_parser("install", help="Install a local or bundled App")
    install.add_argument("source")
    install.add_argument("--home", default=None)
    install.add_argument("--force", action="store_true")
    install.add_argument("--json", action="store_true")
    install.set_defaults(app_handler=_cmd_install)

    listing = commands.add_parser("list", help="List installed Apps")
    listing.add_argument("--home", default=None)
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(app_handler=_cmd_list)

    init = commands.add_parser("init", help="Create a minimal App manifest")
    init.add_argument("name")
    init.add_argument("--path", default=".")
    init.add_argument("--force", action="store_true")
    init.add_argument("--json", action="store_true")
    init.set_defaults(app_handler=_cmd_init)

    add = commands.add_parser("add", help="Add a Capability step to an App manifest")
    add.add_argument("capability")
    add.add_argument("--app", default="app.yaml")
    add.add_argument("--save-as", default=None)
    add.add_argument("--input", default="{}", help="Step input JSON")
    add.add_argument("--json", action="store_true")
    add.set_defaults(app_handler=_cmd_add)

    validate = commands.add_parser("validate", help="Validate an App manifest")
    validate.add_argument("target", nargs="?", default="app.yaml")
    validate.add_argument("--home", default=None)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(app_handler=_cmd_validate)

    run = commands.add_parser("run", help="Run an installed App through rosclawd")
    run.add_argument("name")
    run.add_argument("--home", default=None)
    run.add_argument("--body", required=True)
    run.add_argument("--snapshot", default="")
    run.add_argument(
        "--mode",
        choices=[mode.value for mode in ExecutionMode],
        default=ExecutionMode.SHADOW.value,
    )
    run.add_argument("--principal", default="")
    run.add_argument(
        "--permit",
        action="append",
        default=[],
        help="Capability-bound daemon permit as CAPABILITY=PERMIT_ID",
    )
    run.add_argument("--input", default="{}", help="App input JSON")
    run.add_argument("--socket", default=None)
    run.add_argument("--timeout", type=float, default=5.0)
    run.add_argument("--json", action="store_true")
    run.set_defaults(app_handler=_cmd_run)


def _cmd_install(args: argparse.Namespace) -> int:
    record = AppStore(args.home).install(args.source, force=args.force)
    payload = {"ok": True, "kind": "App", **record.__dict__}
    return _print(args, payload)


def _cmd_list(args: argparse.Namespace) -> int:
    records = AppStore(args.home).list_installed()
    payload = {"ok": True, "apps": [record.__dict__ for record in records]}
    if args.json:
        return _print(args, payload)
    if not records:
        print("No Apps installed.")
    for record in records:
        print(f"{record.name} {record.version} {record.manifest_digest}")
    return 0


def _cmd_init(args: argparse.Namespace) -> int:
    directory = Path(args.path).expanduser().resolve() / args.name
    manifest_path = directory / "app.yaml"
    if manifest_path.exists() and not args.force:
        raise ValueError(f"App manifest already exists: {manifest_path}")
    payload = {
        "apiVersion": APP_API_VERSION,
        "kind": APP_KIND,
        "metadata": {"name": args.name, "version": "0.1.0"},
        "requires": {"capabilities": ["app.placeholder"]},
        "workflow": [{"call": "app.placeholder", "input": {}}],
        "verification": {"require": []},
    }
    AppManifest.model_validate(payload)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return _print(args, {"ok": True, "path": str(manifest_path)})


def _cmd_add(args: argparse.Namespace) -> int:
    path = Path(args.app).expanduser().resolve()
    if path.is_dir():
        path = path / "app.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("App manifest must be a mapping")
    step_input = json.loads(args.input)
    if not isinstance(step_input, dict):
        raise ValueError("--input must be a JSON object")
    capabilities = raw.setdefault("requires", {}).setdefault("capabilities", [])
    workflow = raw.setdefault("workflow", [])
    if capabilities == ["app.placeholder"] and workflow == [
        {"call": "app.placeholder", "input": {}}
    ]:
        capabilities.clear()
        workflow.clear()
    if args.capability not in capabilities:
        capabilities.append(args.capability)
    step: dict[str, Any] = {"call": args.capability, "input": step_input}
    if args.save_as:
        step["save_as"] = args.save_as
    workflow.append(step)
    manifest = AppManifest.model_validate(raw)
    path.write_text(yaml.safe_dump(manifest.to_dict(), sort_keys=False), encoding="utf-8")
    return _print(args, {"ok": True, "path": str(path), "capability": args.capability})


def _cmd_validate(args: argparse.Namespace) -> int:
    candidate = Path(args.target).expanduser()
    if candidate.exists():
        manifest = AppManifest.from_path(candidate)
        source = str(candidate.resolve())
    else:
        _record, manifest = AppStore(args.home).resolve(args.target)
        source = args.target
    payload = {
        "ok": True,
        "valid": True,
        "kind": manifest.kind,
        "name": manifest.metadata.name,
        "version": manifest.metadata.version,
        "source": source,
        "capabilities": manifest.requires.capabilities,
        "steps": len(manifest.workflow),
    }
    return _print(args, payload)


def _cmd_run(args: argparse.Namespace) -> int:
    _record, manifest = AppStore(args.home).resolve(args.name)
    inputs = json.loads(args.input)
    if not isinstance(inputs, dict):
        raise ValueError("--input must be a JSON object")
    permits: dict[str, str] = {}
    for item in args.permit:
        capability, separator, permit_id = item.partition("=")
        if not separator or not capability or not permit_id:
            raise ValueError("--permit must use CAPABILITY=PERMIT_ID")
        permits[capability] = permit_id
    client = DaemonClient(socket_path=args.socket, timeout_sec=args.timeout)
    result = AppRunner(client).run(
        manifest,
        body_id=args.body,
        body_snapshot_hash=args.snapshot,
        execution_mode=ExecutionMode(args.mode),
        principal_id=args.principal,
        permits=permits,
        inputs=inputs,
    )
    _print(args, result.to_dict())
    return 0 if result.status == "success" else 3


def _print(args: argparse.Namespace, payload: dict[str, Any]) -> int:
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if payload.get("ok") is True:
            print("App command completed.")
        else:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


__all__ = ["add_app_subparsers", "dispatch_app_argv", "dispatch_app_command"]

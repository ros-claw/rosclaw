"""CLI lifecycle for discover/add/configure/verify Robot Pack workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rosclaw.robot_pack.catalog import RobotPackCatalog, RobotPackNotFoundError
from rosclaw.robot_pack.discovery import discover_realsense_devices
from rosclaw.robot_pack.instance import RobotInstanceError, configure_robot_instance
from rosclaw.robot_pack.store import RobotPackStore, RobotPackStoreError
from rosclaw.robot_pack.verification import verify_installed_robot_pack

_ROBOT_PACK_COMMANDS = frozenset({"discover", "add", "configure", "verify"})


def add_robot_pack_subparsers(subparsers: Any) -> None:
    """Add new Pack commands to an existing ``robot`` subparser collection."""

    discover = subparsers.add_parser("discover", help="Discover supported physical devices")
    discover.add_argument("--type", choices=["camera"], default="camera")
    discover.add_argument("--backend", choices=["auto", "sdk", "sysfs"], default="auto")
    discover.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    discover.set_defaults(robot_pack_handler=cmd_robot_discover)

    add = subparsers.add_parser("add", help="Install a signed Robot Pack")
    add.add_argument("source", help="Pack ref, alias, manifest, or local directory")
    add.add_argument("--home", default=None, help="ROSCLAW_HOME override")
    add.add_argument("--force", action="store_true", help="Replace the same locked Pack version")
    add.add_argument(
        "--install-adapter",
        action="store_true",
        help="Install the Pack's hardware MCP at its locked git revision",
    )
    add.add_argument("--adapter-python", default=None, help="Python for the isolated MCP adapter")
    add.add_argument("--no-install-adapter-deps", action="store_true")
    add.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    add.set_defaults(robot_pack_handler=cmd_robot_add)

    configure = subparsers.add_parser("configure", help="Bind a Pack to a device and Body")
    configure.add_argument("pack", help="Installed Pack ref, name, or alias")
    configure.add_argument("--instance", default=None, help="Stable Body instance id")
    configure.add_argument("--serial", default=None, help="Exact hardware serial")
    configure.add_argument("--model", default=None, help="Exact supported model")
    configure.add_argument("--stable-uri", default=None, help="Offline stable device URI")
    configure.add_argument("--allow-offline", action="store_true")
    configure.add_argument("--force", action="store_true")
    configure.add_argument("--switch-active", action="store_true")
    configure.add_argument("--home", default=None, help="ROSCLAW_HOME override")
    configure.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    configure.set_defaults(robot_pack_handler=cmd_robot_configure)

    verify = subparsers.add_parser("verify", help="Verify Pack contracts or read-only hardware")
    verify.add_argument("target", help="Installed Pack ref/name or configured instance id")
    verify.add_argument("--stage", choices=["contract", "read-only"], default="contract")
    verify.add_argument("--instance", default=None, help="Configured instance id")
    verify.add_argument("--receipt", default=None, help="Canonical rosclawd receipt JSON")
    verify.add_argument("--output", default=None, help="Evidence report output path")
    verify.add_argument("--home", default=None, help="ROSCLAW_HOME override")
    verify.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    verify.set_defaults(robot_pack_handler=cmd_robot_verify)


def dispatch_robot_pack_argv(argv: list[str]) -> int | None:
    """Handle only new Robot Pack commands without importing the legacy CLI."""

    if len(argv) < 2 or argv[0] != "robot" or argv[1] not in _ROBOT_PACK_COMMANDS:
        return None
    parser = argparse.ArgumentParser(prog="rosclaw")
    robot = parser.add_subparsers(dest="command").add_parser("robot")
    subparsers = robot.add_subparsers(dest="robot_command")
    add_robot_pack_subparsers(subparsers)
    args = parser.parse_args(argv)
    return dispatch_robot_pack_command(args)


def dispatch_robot_pack_command(args: argparse.Namespace) -> int:
    handler = getattr(args, "robot_pack_handler", None)
    if not callable(handler):
        return 1
    return int(handler(args))


def cmd_robot_discover(args: argparse.Namespace) -> int:
    try:
        entry = RobotPackCatalog().resolve("realsense")
        report = discover_realsense_devices(entry.manifest, backend=args.backend)
    except Exception as exc:  # noqa: BLE001 - hardware discovery boundary
        return _print_error(args, "DISCOVERY_FAILED", str(exc))
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("ROSClaw Device Discovery")
        print(f"Pack: {entry.manifest.canonical_ref}")
        print(f"Backends: {', '.join(report.attempted_backends) or 'none'}")
        if not report.devices:
            print("Devices: 0")
        for device in report.devices:
            completeness = "complete" if device.identity_complete else "partial"
            print(
                f"- {device.model} serial={device.serial or 'unknown'} "
                f"usb={device.usb_speed} backend={device.backend} identity={completeness}"
            )
            print(f"  vid:pid={device.vendor_id}:{device.product_id} uri={device.stable_uri}")
            print(f"  firmware={device.firmware} stream_profiles={len(device.stream_profiles)}")
        for warning in report.warnings:
            print(f"WARN: {warning}")
        for error in report.errors:
            print(f"FAIL: {error}")
    return 0 if report.ok else 2


def cmd_robot_add(args: argparse.Namespace) -> int:
    try:
        store = RobotPackStore(args.home)
        record = store.install(args.source, force=args.force)
        from rosclaw.robot_pack.schema import RobotPackManifest

        manifest = RobotPackManifest.from_path(Path(record.path) / "robot-pack.yaml")
    except (RobotPackNotFoundError, RobotPackStoreError, ValueError) as exc:
        return _print_error(args, "ROBOT_PACK_INSTALL_FAILED", str(exc))
    adapter_result: dict[str, Any] | None = None
    adapter_error: str | None = None
    if args.install_adapter:
        component = manifest.component(manifest.adapter.component_id)
        try:
            from rosclaw.mcp.onboarding.source_installer import install_from_git

            installed = install_from_git(
                component.ref,
                server_name=manifest.adapter.component_id,
                home=store.home,
                python=args.adapter_python,
                no_install_deps=args.no_install_adapter_deps,
                revision=component.version,
            )
            adapter_result = installed.to_dict()
            if not installed.success or installed.commit != component.version:
                adapter_error = "; ".join(installed.errors) or (
                    f"adapter commit {installed.commit!r} does not match {component.version!r}"
                )
        except Exception as exc:  # noqa: BLE001 - git/native dependency boundary
            adapter_error = str(exc)
    adapter_status = _installed_adapter_status(manifest, store.home)
    body_profiles = [variant.body_profile for variant in manifest.device.variants]
    real_actuation = "forbidden" if manifest.safety.actuation == "forbidden" else "locked"
    payload = {
        "ok": adapter_error is None,
        "status": "installed" if adapter_error is None else "pack_installed_adapter_failed",
        "pack_installed": True,
        "pack_ref": record.ref,
        "manifest_digest": record.manifest_digest,
        "signature_status": record.signature_status,
        "trusted": record.trusted,
        "support_tier": record.support_tier,
        "install_path": record.path,
        "body_profiles": body_profiles,
        "transport_adapter": {
            "component_id": manifest.adapter.component_id,
            "locked_revision": manifest.component(manifest.adapter.component_id).version,
            "status": adapter_status,
            "install_result": adapter_result,
            "error": adapter_error,
        },
        "calibration": "factory-device-calibration",
        "verification_stages": [stage.id for stage in manifest.verification],
        "real_actuation": real_actuation,
        "agent_blackbox": "not_verified",
    }
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Robot Pack installed")
        print(f"Pack: {record.ref}")
        print(f"Signature: {record.signature_status} (trusted={str(record.trusted).lower()})")
        print(f"Support tier: {record.support_tier}")
        print(f"Body profiles: {', '.join(body_profiles)}")
        print(f"Transport adapter: {adapter_status} ({manifest.adapter.component_id})")
        print("Calibration: factory device calibration")
        print("Read-only verification: available")
        print(f"Real actuation: {real_actuation}")
        print("Agent black-box: not verified")
        print(f"Next: rosclaw robot configure {manifest.pack.name}")
    if adapter_error:
        if args.json:
            return 2
        print(f"[ROSClaw] ADAPTER_INSTALL_FAILED: {adapter_error}", file=sys.stderr)
        return 2
    return 0


def cmd_robot_configure(args: argparse.Namespace) -> int:
    try:
        config, path = configure_robot_instance(
            args.pack,
            home=args.home,
            instance_id=args.instance,
            serial=args.serial,
            model=args.model,
            stable_uri=args.stable_uri,
            allow_offline=args.allow_offline,
            force=args.force,
            switch_active=args.switch_active,
        )
    except (RobotInstanceError, RobotPackStoreError) as exc:
        return _print_error(args, "ROBOT_PACK_CONFIGURE_FAILED", str(exc))
    payload = {"ok": True, "config_path": str(path), "instance": config.model_dump(mode="json")}
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Robot Pack configured")
        print(f"Instance: {config.instance_id}")
        print(f"Pack: {config.pack.ref}")
        print(f"Device: {config.device.model} serial={config.device.serial}")
        print(f"Stable URI: {config.device.stable_uri}")
        print(f"Body snapshot: {config.body_snapshot_hash}")
        print(f"Adapter: {config.adapter.status}")
        if config.device.offline_configured:
            print("WARN: offline binding is not hardware verification")
        print(
            f"Next: rosclaw robot verify {config.instance_id} --stage read-only --receipt RECEIPT.json"
        )
    return 0


def cmd_robot_verify(args: argparse.Namespace) -> int:
    try:
        report = verify_installed_robot_pack(
            args.target,
            stage=args.stage,
            instance_id=args.instance,
            home=args.home,
            receipt_path=args.receipt,
            output_path=args.output,
        )
    except (RobotPackStoreError, RobotInstanceError, ValueError) as exc:
        return _print_error(args, "ROBOT_PACK_VERIFY_FAILED", str(exc))
    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Robot Pack verification: {'PASS' if report.passed else 'FAIL'}")
        print(f"Evidence ID: {report.evidence_id}")
        print(f"Pack: {report.pack_ref}")
        print(f"Stage: {report.stage}")
        print(f"Support tier: {report.support_tier}")
        if report.observed_candidate_tier:
            print(f"Observed candidate: {report.observed_candidate_tier}")
        for check in report.checks:
            print(f"[{check.status.upper()}] {check.id}: {check.message}")
        for blocker in report.promotion_blockers:
            print(f"BLOCKER: {blocker}")
        print(f"Evidence report: {report.report_path}")
    return 0 if report.passed else 3


def _print_error(args: argparse.Namespace, code: str, message: str) -> int:
    payload = {"ok": False, "error": {"code": code, "message": message}}
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"[ROSClaw] {code}: {message}", file=sys.stderr)
    return 2


def _installed_adapter_status(manifest: Any, home: Path) -> str:
    from rosclaw.robot_pack.instance import resolve_adapter_binding

    return resolve_adapter_binding(manifest, home).status


__all__ = [
    "add_robot_pack_subparsers",
    "dispatch_robot_pack_argv",
    "dispatch_robot_pack_command",
]

"""CLI dispatchers for the LeRobot integration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.capabilities import get_lerobot_capabilities
from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor
from rosclaw.integrations.lerobot.installer import install_lerobot
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider
from rosclaw.integrations.lerobot.subprocess_runner import run_command, which
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def cmd_setup_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw setup lerobot`."""
    report = install_lerobot(
        profile=args.profile,
        dry_run=args.dry_run,
        upgrade=args.upgrade,
    )
    print(f"[rosclaw-lerobot] Profile: {report.profile}")
    print(f"[rosclaw-lerobot] OK: {report.ok}")
    print(f"[rosclaw-lerobot] Message: {report.message}")
    if report.lerobot_version:
        print(f"[rosclaw-lerobot] LeRobot version: {report.lerobot_version}")
    if args.json:
        print(json.dumps(report.details, indent=2, default=str))
    return 0 if report.ok else 1


def cmd_lerobot_doctor(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot doctor`."""
    report = run_lerobot_doctor(
        registry_check={
            "provider_type_lerobot_policy": True,
            "dataset_export_lerobot": True,
        }
    )
    if args.json:
        print(
            json.dumps(
                {
                    "name": report.name,
                    "status": report.status,
                    "version": report.version,
                    "message": report.message,
                    "capabilities": [
                        {
                            "name": c.name,
                            "kind": c.kind,
                            "enabled": c.enabled,
                            "experimental": c.experimental,
                            "description": c.description,
                        }
                        for c in report.capabilities
                    ],
                    "python_version": report.python_version,
                    "python_executable": report.python_executable,
                    "lerobot_importable": report.lerobot_importable,
                    "lerobot_version": report.lerobot_version,
                    "lerobot_info_path": report.lerobot_info_path,
                    "lerobot_info_ok": report.lerobot_info_ok,
                    "torch_available": report.torch_available,
                    "torch_version": report.torch_version,
                    "cuda_available": report.cuda_available,
                    "hf_endpoint": report.hf_endpoint,
                    "config_enabled": report.config_enabled,
                    "provider_type_registered": report.provider_type_registered,
                    "exporter_registered": report.exporter_registered,
                },
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"[rosclaw-lerobot] Integration: {report.name}")
    print(f"[rosclaw-lerobot] Status: {report.status}")
    print(f"[rosclaw-lerobot] Version: {report.version or 'N/A'}")
    print(f"[rosclaw-lerobot] Message: {report.message}")
    print("[rosclaw-lerobot] Capabilities:")
    for cap in report.capabilities:
        flag = "enabled" if cap.enabled else "disabled"
        exp = " (experimental)" if cap.experimental else ""
        print(f"  - {cap.name} ({cap.kind}): {flag}{exp}")
        if cap.description:
            print(f"      {cap.description}")
    print(f"[rosclaw-lerobot] Python: {report.python_executable} ({report.python_version})")
    print(f"[rosclaw-lerobot] LeRobot importable: {report.lerobot_importable}")
    print(f"[rosclaw-lerobot] LeRobot info: {report.lerobot_info_path or 'not found'}")
    print(f"[rosclaw-lerobot] Torch available: {report.torch_available}")
    if report.cuda_available is not None:
        print(f"[rosclaw-lerobot] CUDA available: {report.cuda_available}")
    print(f"[rosclaw-lerobot] HF endpoint: {report.hf_endpoint}")
    print(f"[rosclaw-lerobot] Config enabled: {report.config_enabled}")
    return 0


def cmd_lerobot_info(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot info`."""
    info_path = which("lerobot-info")
    if info_path is None:
        print(
            "[rosclaw-lerobot] lerobot-info not found on PATH. "
            "Install LeRobot first: rosclaw setup lerobot --profile core",
            file=sys.stderr,
        )
        return 1

    cmd = [info_path]
    if args.args:
        cmd.extend(args.args)
    result = run_command(cmd, timeout=120.0)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return 0 if result.ok else result.returncode


def cmd_lerobot_capabilities(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot capabilities`."""
    capabilities = get_lerobot_capabilities()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "name": c.name,
                        "kind": c.kind,
                        "enabled": c.enabled,
                        "experimental": c.experimental,
                        "description": c.description,
                    }
                    for c in capabilities
                ],
                indent=2,
            )
        )
        return 0

    print("[rosclaw-lerobot] Capabilities:")
    for cap in capabilities:
        flag = "enabled" if cap.enabled else "disabled"
        exp = " (experimental)" if cap.experimental else ""
        print(f"  - {cap.name} ({cap.kind}): {flag}{exp}")
        if cap.description:
            print(f"      {cap.description}")
    return 0


def cmd_capability_list(args: argparse.Namespace, registry: Any) -> int:
    """Dispatch `rosclaw capability list`."""
    reports = registry.list_integrations()
    all_caps: list[dict[str, Any]] = []
    for report in reports:
        for cap in report.capabilities:
            all_caps.append(
                {
                    "integration": report.name,
                    "name": cap.name,
                    "kind": cap.kind,
                    "enabled": cap.enabled,
                    "experimental": cap.experimental,
                    "description": cap.description,
                    "integration_status": report.status,
                }
            )

    # Also include static LeRobot capabilities even if not installed.
    lerobot_names = {c["name"] for c in all_caps if c["integration"] == "lerobot"}
    for cap in get_lerobot_capabilities():
        if cap.name not in lerobot_names:
            all_caps.append(
                {
                    "integration": "lerobot",
                    "name": cap.name,
                    "kind": cap.kind,
                    "enabled": cap.enabled,
                    "experimental": cap.experimental,
                    "description": cap.description,
                    "integration_status": "not_installed",
                }
            )

    if args.json:
        print(json.dumps(all_caps, indent=2))
        return 0

    print("[rosclaw] Capabilities:")
    for cap in all_caps:
        flag = "enabled" if cap["enabled"] else "disabled"
        exp = " (experimental)" if cap["experimental"] else ""
        print(
            f"  - {cap['name']} ({cap['kind']}) [{cap['integration']}/{cap['integration_status']}]: "
            f"{flag}{exp}"
        )
        if cap.get("description"):
            print(f"      {cap['description']}")
    return 0


def cmd_provider_infer_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw provider infer --type lerobot_policy ...`."""
    from rosclaw.core.async_utils import run_sync

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[rosclaw-lerobot] Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    try:
        manifest = ProviderManifest.from_yaml(manifest_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[rosclaw-lerobot] Failed to load manifest: {exc}", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[rosclaw-lerobot] Input not found: {input_path}", file=sys.stderr)
        return 1

    try:
        inputs = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[rosclaw-lerobot] Invalid input JSON: {exc}", file=sys.stderr)
        return 1

    dry_run = getattr(args, "dry_run", False)
    inputs["dry_run"] = dry_run

    provider = LeRobotPolicyProvider(manifest)
    request = ProviderRequest(
        request_id=f"lerobot_policy_{int(time.time())}",
        capability="lerobot.policy.infer",
        inputs=inputs,
        context={"manifest": str(manifest_path)},
    )

    try:
        response = run_sync(provider.infer(request))
    except Exception as exc:  # noqa: BLE001
        print(f"[rosclaw-lerobot] Inference failed: {exc}", file=sys.stderr)
        return 1

    result = {
        "provider": response.provider,
        "capability": response.capability,
        "status": response.status,
        "result": response.result,
        "latency_ms": response.latency_ms,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[rosclaw-lerobot] Result written to {out_path}")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if response.status == "ok" else 1

"""CLI dispatchers for the LeRobot integration."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from rosclaw.integrations.lerobot.capabilities import get_lerobot_capabilities
from rosclaw.integrations.lerobot.compatibility import (
    build_compatibility_report,
    format_compatibility_text,
)
from rosclaw.integrations.lerobot.config import (
    get_configured_lerobot_runtime,
)
from rosclaw.integrations.lerobot.dataset_feature_infer import infer_features
from rosclaw.integrations.lerobot.dataset_profile import resolve_profile
from rosclaw.integrations.lerobot.dataset_report import (
    report_from_worker_response,
    write_dataset_export_report,
)
from rosclaw.integrations.lerobot.dataset_validator import (
    format_validation_result,
    validate_dataset,
)
from rosclaw.integrations.lerobot.dataset_worker_runner import (
    LeRobotDatasetWorkerRunner,
    run_dataset_api_inspect,
    run_dataset_dataloader_smoke,
    run_dataset_export,
)
from rosclaw.integrations.lerobot.doctor import run_lerobot_doctor
from rosclaw.integrations.lerobot.installer import install_lerobot
from rosclaw.integrations.lerobot.practice_normalizer import (
    NormalizationError,
    normalize_practice_episode,
    write_normalized_episode,
)
from rosclaw.integrations.lerobot.provider import LeRobotPolicyProvider
from rosclaw.integrations.lerobot.runtime import LEROBOT_INFO_MODULE, LeRobotRuntime
from rosclaw.integrations.lerobot.smoke_policy import (
    DEFAULT_SMOKE_POLICY,
    SmokePolicyOptions,
    run_smoke_policy_sync,
)
from rosclaw.integrations.lerobot.smoke_report import get_validation_status
from rosclaw.integrations.lerobot.subprocess_runner import run_command, which
from rosclaw.provider.core.manifest import ProviderManifest
from rosclaw.provider.core.request import ProviderRequest


def _build_lerobot_provider_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Map CLI args to provider inputs."""
    inputs: dict[str, Any] = {
        "policy.path": getattr(args, "policy_path", None) or getattr(args, "policy_path", None),
        "revision": getattr(args, "revision", "main"),
        "device": getattr(args, "device", "cpu"),
        "allow_network": getattr(args, "allow_network", False),
        "timeout_sec": getattr(args, "timeout_sec", 120),
    }
    if getattr(args, "dry_run", False):
        inputs["dry_run"] = True
    return {k: v for k, v in inputs.items() if v is not None}


def cmd_setup_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw setup lerobot`."""
    report = install_lerobot(
        profile=args.profile,
        mode=args.mode,
        python=args.python,
        runtime_path=args.runtime_path,
        upgrade=args.upgrade,
        force=args.force,
        dry_run=args.dry_run,
        index_url=args.index_url,
        extra_index_url=args.extra_index_url,
    )
    if args.json:
        payload = {
            "ok": report.ok,
            "profile": report.profile,
            "dry_run": report.dry_run,
            "mode": report.mode,
            "message": report.message,
            "error_code": report.error_code,
            "lerobot_version": report.lerobot_version,
            "python_executable": report.python_executable,
            "pip_executable": report.pip_executable,
            "runtime": _serialize_runtime(report.runtime),
            "details": report.details,
        }
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(f"[rosclaw-lerobot] Profile: {report.profile}")
        print(f"[rosclaw-lerobot] Mode: {report.mode or 'N/A'}")
        print(f"[rosclaw-lerobot] OK: {report.ok}")
        print(f"[rosclaw-lerobot] Message: {report.message}")
        if report.lerobot_version:
            print(f"[rosclaw-lerobot] LeRobot version: {report.lerobot_version}")
        if report.runtime and report.runtime.python_executable:
            print(f"[rosclaw-lerobot] Runtime Python: {report.runtime.python_executable}")
        if report.error_code:
            print(f"[rosclaw-lerobot] Error code: {report.error_code}")

    if report.ok:
        return 0
    # User/environment errors: exit code 2; install failures: exit code 1.
    if report.error_code in {
        "python_too_old",
        "python312_not_found",
        "external_python_not_found",
        "external_python_too_old",
        "lerobot_version_unsupported",
    }:
        return 2
    return 1


def cmd_lerobot_doctor(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot doctor`."""
    report = run_lerobot_doctor(
        registry_check={
            "provider_type_lerobot_policy": True,
            "dataset_export_lerobot": True,
        }
    )
    validation = report.validation_status or get_validation_status()
    if args.json:
        payload: dict[str, Any] = {
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
            "rosclaw_runtime": {
                "python_executable": report.rosclaw_python_executable,
                "python_version": report.rosclaw_python_version,
                "in_process_lerobot_import": report.lerobot_importable,
            },
            "lerobot_runtime": _serialize_runtime(report.lerobot_runtime),
            "bridge_capabilities": {
                "provider_type_lerobot_policy": report.provider_type_registered,
                "dataset_export_lerobot": report.exporter_registered,
                "worker_subprocess": report.worker_subprocess_available,
                "worker_in_process": report.worker_in_process_available,
            },
            "validation": validation,
            "dataset_export_status": report.dataset_export_status,
            "hf_endpoint": report.hf_endpoint,
            "config_enabled": report.config_enabled,
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print("ROSClaw × LeRobot Bridge Doctor")
    print()
    print("ROSClaw Runtime")
    print(f"  Python executable: {report.rosclaw_python_executable}")
    print(f"  Python version:    {report.rosclaw_python_version or 'N/A'}")
    print(f"  In-process LeRobot import: {'yes' if report.lerobot_importable else 'no'}")
    if not report.lerobot_importable:
        print("  Reason: ROSClaw Python < 3.12 or LeRobot not installed")

    print()
    print("LeRobot Runtime")
    if report.lerobot_runtime is not None:
        rt = report.lerobot_runtime
        print(f"  Mode:              {rt.mode}")
        print(f"  Runtime path:      {rt.runtime_path or 'N/A'}")
        print(f"  Python executable: {rt.python_executable}")
        print(f"  Python version:    {rt.python_version or 'N/A'}")
        print(f"  LeRobot version:   {rt.lerobot_version or 'N/A'}")
        print(f"  lerobot-info:      {'ok' if rt.subprocess_available else 'not available'}")
        print(f"  Torch:             {rt.torch_version or 'N/A'}")
        cuda_text = (
            "available"
            if rt.cuda_available
            else "not available"
            if rt.cuda_available is not None
            else "N/A"
        )
        print(f"  CUDA:              {cuda_text}")
    else:
        print("  Not configured")

    print()
    print("Bridge Capabilities")
    for cap in report.capabilities:
        flag = "enabled" if cap.enabled else "disabled"
        exp = " (experimental)" if cap.experimental else ""
        print(f"  {cap.name + ':':<34} {flag}{exp}")

    print()
    print("Real Policy Smoke Validation")
    state = validation.get("state", "not_configured")
    print(f"  Status:            {state}")
    if validation.get("last_policy"):
        print(f"  Last policy:       {validation['last_policy']}")
    if validation.get("policy_type"):
        print(f"  Policy type:       {validation['policy_type']}")
    if validation.get("lerobot_version"):
        print(f"  LeRobot version:   {validation['lerobot_version']}")
    if validation.get("device"):
        print(f"  Device:            {validation['device']}")
    if validation.get("action_shape") is not None:
        print(f"  Action shape:      {validation['action_shape']}")
    if validation.get("time"):
        print(f"  Time:              {validation['time']}")
    if validation.get("safety"):
        print(f"  Safety labels:     {', '.join(validation['safety'])}")
    perf = validation.get("performance_warning")
    if perf:
        print(f"  Performance:       {perf}")
    for reason in validation.get("stale_reasons", []):
        print(f"  Stale reason:      {reason}")
    if state == "not_configured":
        print("  Hint: Run `rosclaw lerobot smoke-policy` to validate a real policy.")

    print()
    print("Dataset Export Validation")
    ds_status = report.dataset_export_status or {"state": "not_configured"}
    ds_state = ds_status.get("state", "not_configured")
    print(f"  Status:            {ds_state}")
    if ds_status.get("last_output_dir"):
        print(f"  Last output:       {ds_status['last_output_dir']}")
    if ds_status.get("repo_id"):
        print(f"  Repo ID:           {ds_status['repo_id']}")
    if ds_status.get("num_frames") is not None:
        print(f"  Frames:            {ds_status['num_frames']}")
    if ds_status.get("num_episodes") is not None:
        print(f"  Episodes:          {ds_status['num_episodes']}")
    if ds_status.get("features"):
        print(f"  Features:          {', '.join(ds_status['features'])}")
    visual = ds_status.get("visual", {})
    if visual.get("storage_mode"):
        print(f"  Visual mode:       {visual['storage_mode']}")
    if visual.get("camera_keys"):
        print(f"  Cameras:           {', '.join(visual['camera_keys'])}")
    if ds_status.get("load_ok") is not None:
        print(f"  Load OK:           {ds_status['load_ok']}")
    if ds_status.get("index_ok") is not None:
        print(f"  Index OK:          {ds_status['index_ok']}")
    gates = ds_status.get("quality_gates", {})
    if gates.get("dataloader_ok") is not None:
        print(f"  Dataloader OK:     {gates['dataloader_ok']}")
    profile = ds_status.get("profile", {})
    if profile.get("name"):
        print(f"  Profile:           {profile['name']}")
    scope = profile.get("scope", {})
    if scope.get("validated"):
        print(f"  Validated groups:  {', '.join(scope['validated'])}")
    if scope.get("planned"):
        print(f"  Planned groups:    {', '.join(scope['planned'])}")
    if scope.get("missing"):
        print(f"  Missing groups:    {', '.join(scope['missing'])}")
    if ds_state == "not_configured":
        print("  Hint: Run `rosclaw lerobot export-dataset` to validate a real dataset export.")
    for reason in ds_status.get("stale_reasons", []):
        print(f"  Stale reason:      {reason}")

    print()
    print(f"Status: {report.status.upper()}")
    if report.status_detail:
        print(report.status_detail)
    return 0


def _serialize_runtime(runtime: LeRobotRuntime | None) -> dict[str, Any] | None:
    if runtime is None:
        return None
    return {
        "mode": runtime.mode,
        "runtime_path": str(runtime.runtime_path) if runtime.runtime_path else None,
        "python_executable": str(runtime.python_executable),
        "pip_executable": str(runtime.pip_executable) if runtime.pip_executable else None,
        "lerobot_info_executable": (
            str(runtime.lerobot_info_executable) if runtime.lerobot_info_executable else None
        ),
        "python_version": runtime.python_version,
        "lerobot_version": runtime.lerobot_version,
        "torch_version": runtime.torch_version,
        "cuda_available": runtime.cuda_available,
        "state": runtime.state,
        "in_process_available": runtime.in_process_available,
        "subprocess_available": runtime.subprocess_available,
        "error": runtime.error,
    }


def cmd_lerobot_info(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot info`."""
    runtime_cfg = get_configured_lerobot_runtime()
    info_path: Path | str | None = None

    if runtime_cfg and runtime_cfg.get("lerobot_info_executable"):
        candidate = Path(runtime_cfg["lerobot_info_executable"])
        if candidate.exists() and candidate != Path(runtime_cfg.get("python_executable", "")):
            info_path = candidate
        else:
            # If the stored executable is the python interpreter itself, use
            # python -m lerobot_info as a fallback.
            info_path = Path(runtime_cfg["python_executable"])

    if info_path is None:
        from_path = which("lerobot-info")
        if from_path is not None:
            info_path = from_path

    if info_path is None:
        print(
            "[rosclaw-lerobot] lerobot-info not found. "
            "Install or register LeRobot first: rosclaw setup lerobot --profile core",
            file=sys.stderr,
        )
        return 1

    cmd: list[str]
    if isinstance(info_path, Path) and info_path.name.startswith("python"):
        cmd = [str(info_path), "-m", LEROBOT_INFO_MODULE]
    else:
        cmd = [str(info_path)]
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


def cmd_provider_inspect_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw provider inspect --type lerobot_policy ...`."""
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

    if not getattr(args, "policy_path", None):
        print("[rosclaw-lerobot] --policy.path is required.", file=sys.stderr)
        return 1

    provider = LeRobotPolicyProvider(manifest)
    request = ProviderRequest(
        request_id=f"lerobot_policy_inspect_{int(time.time())}",
        capability="lerobot.policy.inspect",
        inputs=_build_lerobot_provider_inputs(args),
        context={"manifest": str(manifest_path)},
    )

    try:
        response = run_sync(provider.infer(request))
    except Exception as exc:  # noqa: BLE001
        print(f"[rosclaw-lerobot] Inspect failed: {exc}", file=sys.stderr)
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


def cmd_provider_load_test_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw provider load-test --type lerobot_policy ...`."""
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

    if not getattr(args, "policy_path", None):
        print("[rosclaw-lerobot] --policy.path is required.", file=sys.stderr)
        return 1

    provider = LeRobotPolicyProvider(manifest)
    request = ProviderRequest(
        request_id=f"lerobot_policy_load_test_{int(time.time())}",
        capability="lerobot.policy.load_test",
        inputs=_build_lerobot_provider_inputs(args),
        context={"manifest": str(manifest_path)},
    )

    try:
        response = run_sync(provider.infer(request))
    except Exception as exc:  # noqa: BLE001
        print(f"[rosclaw-lerobot] Load-test failed: {exc}", file=sys.stderr)
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
    if isinstance(inputs, dict):
        inputs.setdefault("_base_dir", str(input_path.parent))

    inputs.update(_build_lerobot_provider_inputs(args))

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




def _resolve_episode_path(args: argparse.Namespace) -> Path:
    """Resolve the practice episode path from CLI args."""
    episode_id = args.episode_dir or args.episode_id
    if not episode_id:
        raise ValueError("Episode identifier or path is required.")
    candidate = Path(episode_id)
    if candidate.is_dir():
        return candidate
    data_root = Path(getattr(args, "data_root", "/data/rosclaw/practice"))
    from_data_root = data_root / episode_id
    if from_data_root.is_dir():
        return from_data_root
    return candidate


def cmd_lerobot_export_dataset(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot export-dataset`."""
    import tempfile

    try:
        episode_path = _resolve_episode_path(args)
    except ValueError as exc:
        print(f"[rosclaw-lerobot] {exc}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    repo_id = args.repo_id
    fps = float(getattr(args, "fps", 10.0))
    timeout_sec = int(getattr(args, "timeout_sec", 300))
    profile = getattr(args, "profile", "minimal")
    visual_storage_mode = getattr(args, "visual_storage_mode", "auto")
    use_videos = bool(getattr(args, "use_videos", False))
    include_body_snapshot = bool(getattr(args, "include_body_snapshot", False))
    body_snapshot_mode = getattr(args, "body_snapshot_mode", "sanitized")
    acknowledge_sensitive_body_data = bool(getattr(args, "acknowledge_sensitive_body_data", False))
    dry_run = bool(getattr(args, "dry_run", False))
    dataloader = bool(getattr(args, "dataloader", False))
    allow_partial = bool(getattr(args, "allow_partial", False))
    missing_policy = getattr(args, "missing_policy", "nan")

    try:
        normalized = normalize_practice_episode(
            episode_path,
            task=getattr(args, "task", None),
            robot_id=getattr(args, "robot_id", None),
            body_profile=getattr(args, "body_profile", None),
            fps=fps if getattr(args, "fps", None) is not None else None,
        )
    except NormalizationError as exc:
        print(f"[rosclaw-lerobot] Normalization failed ({exc.code}): {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"[rosclaw-lerobot] Details: {exc.details}", file=sys.stderr)
        return 1

    if dry_run:
        try:
            prof = resolve_profile(
                profile,
                include_body_snapshot=include_body_snapshot,
                body_snapshot_mode=body_snapshot_mode,
            )
            features = infer_features(normalized, feature_groups=sorted(prof.feature_groups))
            warnings: list[str] = []
            if prof.feature_groups - {"safety", "failure", "intervention", "action", "outcome"}:
                warnings.append("Some requested feature groups are not implemented in Gate A.")
            payload = {
                "status": "dry_run",
                "profile": profile,
                "feature_groups": sorted(prof.feature_groups),
                "num_frames": len(normalized.frames),
                "features": {k: {"dtype": v["dtype"], "shape": v["shape"]} for k, v in features.items()},
                "warnings": warnings,
            }
            if args.json:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
            else:
                print("[rosclaw-lerobot] Dry run -- no dataset will be written")
                print(f"  Profile:       {profile}")
                print(f"  Feature groups: {', '.join(sorted(prof.feature_groups)) or '(none)'}")
                print(f"  Frames:        {len(normalized.frames)}")
                print("  Features:")
                for key, value in features.items():
                    print(f"    {key}: dtype={value['dtype']}, shape={value['shape']}")
                if warnings:
                    print("  Warnings:")
                    for w in warnings:
                        print(f"    - {w}")
            return 0
        except ValueError as exc:
            print(f"[rosclaw-lerobot] {exc}", file=sys.stderr)
            return 1
        except Exception as exc:  # noqa: BLE001
            print(f"[rosclaw-lerobot] Dry-run inference failed: {exc}", file=sys.stderr)
            return 1

    temp_dir = Path(tempfile.mkdtemp(prefix="rosclaw_lerobot_export_"))
    normalized_path = temp_dir / "normalized_episode.json"
    try:
        write_normalized_episode(normalized, normalized_path)

        try:
            response = run_dataset_export(
                normalized_episode_path=str(normalized_path),
                output_dir=str(output_dir),
                repo_id=repo_id,
                fps=fps,
                use_videos=use_videos,
                visual_storage_mode=visual_storage_mode,
                profile=profile,
                include_body_snapshot=include_body_snapshot,
                body_snapshot_mode=body_snapshot_mode,
                acknowledge_sensitive_body_data=acknowledge_sensitive_body_data,
                dataloader=dataloader,
                timeout_sec=timeout_sec,
                allow_partial=allow_partial,
                missing_policy=missing_policy,
            )
        except ValueError as exc:
            print(f"[rosclaw-lerobot] {exc}", file=sys.stderr)
            return 1

        report = report_from_worker_response(
            episode_path=str(episode_path),
            episode_id=normalized.episode_id,
            output_dir=str(output_dir),
            repo_id=repo_id,
            response_dict=response.to_dict(),
        )
        write_dataset_export_report(report, output_dir)

        if args.json:
            payload = {
                "status": report.status,
                "output_dir": str(output_dir),
                "repo_id": repo_id,
                "dataset": report.dataset,
                "validation": report.validation,
                "feature_groups": report.feature_groups,
                "sidecar_files": response.sidecar_files,
                "warnings": response.warnings,
                "error": report.error,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print("[rosclaw-lerobot] Dataset export complete")
            print(f"  Status:      {report.status}")
            print(f"  Output:      {output_dir}")
            print(f"  Repo ID:     {repo_id}")
            print(f"  Frames:      {report.dataset.get('num_frames', 0)}")
            print(f"  Episodes:    {report.dataset.get('num_episodes', 0)}")
            print(f"  Features:    {', '.join(report.dataset.get('features', {}).keys())}")
            visual = report.dataset.get('visual', {})
            print(f"  Visual mode: {visual.get('storage_mode', 'images')}")
            print(f"  Load OK:     {report.validation.get('load_ok', False)}")
            print(f"  Index OK:    {report.validation.get('index_ok', False)}")
            if report.validation.get('dataloader_ok') is not None:
                print(f"  Dataloader OK: {report.validation['dataloader_ok']}")
            if report.feature_groups:
                print(f"  ROSClaw groups: {', '.join(report.feature_groups)}")
            if response.sidecar_files:
                print(f"  Sidecar files: {', '.join(response.sidecar_files)}")
            if response.warnings:
                for w in response.warnings:
                    print(f"  Warning:     {w}")
            if report.error:
                print(f"  Error:       {report.error.get('message', '')}", file=sys.stderr)

        return 0 if report.status == "ok" else 1
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def cmd_lerobot_validate_dataset(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot validate-dataset`."""
    dataset_dir = Path(args.dataset)
    repo_id = args.repo_id
    level = getattr(args, "level", "load")
    result = validate_dataset(dataset_dir, repo_id, level=level)
    formatted = format_validation_result(result)
    ok = bool(result.load_ok and result.index_ok and not result.error)
    if args.json:
        print(json.dumps(formatted, indent=2, ensure_ascii=False))
    else:
        print("[rosclaw-lerobot] Dataset validation")
        print(f"  Level:       {level}")
        print(f"  Status:      {'ok' if ok else 'failed'}")
        print(f"  Load OK:     {result.load_ok}")
        print(f"  Index OK:    {result.index_ok}")
        print(f"  Frames:      {result.num_frames or 'N/A'}")
        print(f"  Episodes:    {result.num_episodes or 'N/A'}")
        if result.sample_keys:
            print(f"  Sample keys: {', '.join(result.sample_keys)}")
        if result.sample_image_keys:
            print(f"  Image keys:  {', '.join(result.sample_image_keys)}")
        if result.dataloader_ok is not None:
            print(f"  Dataloader OK: {result.dataloader_ok}")
        if result.batch_keys:
            print(f"  Batch keys:  {', '.join(result.batch_keys)}")
        if result.error:
            print(f"  Error:       {result.error.get('message', '')}")
    return 0 if ok else 1


def cmd_lerobot_dataset_api(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot dataset-api`."""
    response = run_dataset_api_inspect(timeout_sec=getattr(args, "timeout_sec", 120))
    api_info = response.api_info
    if api_info is None:
        print("[rosclaw-lerobot] Could not introspect LeRobotDataset API.", file=sys.stderr)
        if response.error:
            print(f"[rosclaw-lerobot] {response.error.code}: {response.error.message}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(api_info.to_dict() if api_info else {}, indent=2, ensure_ascii=False))
    else:
        print("[rosclaw-lerobot] LeRobotDataset API")
        print(f"  create signature: {api_info.create_signature}")
        print(f"  add_frame:        {'yes' if api_info.has_add_frame else 'no'}")
        print(f"  save_episode:     {'yes' if api_info.has_save_episode else 'no'}")
        print(f"  consolidate:      {'yes' if api_info.has_consolidate else 'no'}")
        print(f"  finalize:         {'yes' if api_info.has_finalize else 'no'}")
        if api_info.lerobot_version:
            print(f"  LeRobot version:  {api_info.lerobot_version}")
    return 0


def cmd_lerobot_smoke_dataloader(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot smoke-dataloader`."""
    response = run_dataset_dataloader_smoke(
        output_dir=str(args.dataset),
        repo_id=args.repo_id,
        batch_size=int(getattr(args, "batch_size", 2)),
        num_workers=int(getattr(args, "num_workers", 0)),
        timeout_sec=int(getattr(args, "timeout_sec", 300)),
    )
    result = response.validation
    if args.json:
        print(json.dumps(format_validation_result(result), indent=2, ensure_ascii=False))
    else:
        print("[rosclaw-lerobot] DataLoader smoke")
        print(f"  Dataloader OK: {result.dataloader_ok}")
        print(f"  Frames:        {result.num_frames or 'N/A'}")
        print(f"  Episodes:      {result.num_episodes or 'N/A'}")
        if result.batch_keys:
            print(f"  Batch keys:    {', '.join(result.batch_keys)}")
        if result.batch_shapes:
            print("  Batch shapes:")
            for key, shape in result.batch_shapes.items():
                print(f"    {key}: {shape}")
        if result.error:
            print(f"  Error:         {result.error.get('message', '')}", file=sys.stderr)
    return 0 if result.dataloader_ok else 1


_DATASET_COMPATIBILITY_MATRIX = [
    {
        "feature": "observation.state / action / task",
        "status": "supported",
        "since": "P2",
        "notes": "Core LeRobotDataset features.",
    },
    {
        "feature": "Single RGB camera",
        "status": "supported",
        "since": "P2",
        "notes": "observation.images.<camera>",
    },
    {
        "feature": "Safety / sandbox metadata",
        "status": "supported",
        "since": "P2.1 Gate A",
        "notes": "rosclaw.sandbox.* int8/float32 features.",
    },
    {
        "feature": "Failure metadata",
        "status": "supported",
        "since": "P2.1 Gate A",
        "notes": "rosclaw.failure.* features.",
    },
    {
        "feature": "Intervention metadata",
        "status": "supported",
        "since": "P2.1 Gate A",
        "notes": "rosclaw.intervention.* features.",
    },
    {
        "feature": "Action provenance",
        "status": "supported",
        "since": "P2.1 Gate A",
        "notes": "rosclaw.action.* features.",
    },
    {
        "feature": "Episode sidecar (parquet/jsonl)",
        "status": "supported",
        "since": "P2.1 Gate A",
        "notes": "meta/rosclaw/episodes.parquet",
    },
    {
        "feature": "Multi-camera / depth",
        "status": "planned",
        "since": "P2.1 Gate C",
        "notes": "Multiple RGB/depth cameras.",
    },
    {
        "feature": "Physical telemetry (current/force/temp)",
        "status": "planned",
        "since": "P2.1 Gate B",
        "notes": "Motor current, force/torque, temperature.",
    },
]


def cmd_lerobot_dataset_compatibility(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot dataset-compatibility`."""
    if args.json:
        print(json.dumps({"matrix": _DATASET_COMPATIBILITY_MATRIX}, indent=2, ensure_ascii=False))
        return 0

    print("ROSClaw × LeRobot Dataset Compatibility Matrix")
    print("")
    print(f"{'Feature':<42} {'Status':<12} {'Since':<18} Notes")
    print("-" * 110)
    for row in _DATASET_COMPATIBILITY_MATRIX:
        print(f"{row['feature']:<42} {row['status']:<12} {row['since']:<18} {row['notes']}")
    return 0


def cmd_smoke_policy_lerobot(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot smoke-policy`."""
    options = SmokePolicyOptions(
        policy_path=getattr(args, "policy_path", DEFAULT_SMOKE_POLICY) or DEFAULT_SMOKE_POLICY,
        revision=getattr(args, "revision", "main"),
        device=getattr(args, "device", "cpu"),
        dtype=getattr(args, "dtype", "auto"),
        allow_network=getattr(args, "allow_network", False),
        timeout_sec=getattr(args, "timeout_sec", 300),
        output=Path(args.output) if getattr(args, "output", None) else None,
        keep_worker_files=getattr(args, "keep_worker_files", False),
        force_download=getattr(args, "force_download", False),
        skip_infer=getattr(args, "skip_infer", False),
        observation_file=Path(args.observation_file) if getattr(args, "observation_file", None) else None,
        json_output=getattr(args, "json", False),
    )

    report = run_smoke_policy_sync(options)
    payload = report.to_dict()

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.json:
            print(f"[rosclaw-lerobot] Smoke report written to {out_path}")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if report.status == "ok" else 1


def cmd_lerobot_compatibility(args: argparse.Namespace) -> int:
    """Dispatch `rosclaw lerobot compatibility`."""
    policy_type: str | None = getattr(args, "policy_type", None) or None
    report = build_compatibility_report(policy_type=policy_type)

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print(format_compatibility_text(report))
    return 0


__all__ = [
    "LeRobotDatasetWorkerRunner",
]

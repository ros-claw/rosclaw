"""Fail-closed daemon-side executors loaded from configured Robot Packs."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from rosclaw.body.resolver import BodyResolver
from rosclaw.firstboot.workspace import resolve_home
from rosclaw.kernel import (
    ActionEnvelope,
    ActionExecutionResult,
    ActionState,
    EvidenceLevel,
    ExecutionMode,
)
from rosclaw.robot_pack.instance import RobotInstanceConfig, resolve_adapter_binding
from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.store import RobotPackStore
from rosclaw.robot_pack.verifier import verify_robot_pack


class RobotPackRuntimeError(RuntimeError):
    """Raised when daemon startup encounters a configured but unsafe Pack."""


class _ArtifactDirectoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


class RealSenseCaptureExecutor:
    """Daemon-owned adapter from ActionEnvelope to the existing MCP-only skill runner."""

    def __init__(self, instance: RobotInstanceConfig, *, home: Path) -> None:
        self.instance = instance
        self.home = home.resolve()
        self.artifacts_root = self.home / "artifacts" / "robot-packs"

    def __call__(self, action: ActionEnvelope) -> ActionExecutionResult:
        if action.body_id != self.instance.instance_id:
            return _failed_result(
                "ROBOT_PACK_BODY_MISMATCH",
                f"Action body {action.body_id!r} is not Pack instance {self.instance.instance_id!r}",
            )
        if action.body_snapshot_hash != self.instance.body_snapshot_hash:
            return _failed_result(
                "ROBOT_PACK_BODY_SNAPSHOT_MISMATCH",
                "Action Body snapshot does not match the configured Robot Pack instance",
            )
        if action.capability_id != "camera.capture_rgbd":
            return _failed_result(
                "ROBOT_PACK_CAPABILITY_MISMATCH",
                f"Unsupported RealSense Pack capability: {action.capability_id}",
            )
        if (
            not action.authorization.approved
            or not action.authorization.approval_id
            or action.capability_id not in action.authorization.scopes
        ):
            return _failed_result(
                "ROBOT_PACK_AUTHORIZATION_REQUIRED",
                "RealSense REAL execution requires daemon-authored capability authorization",
            )
        requested_serial = action.arguments.get("serial")
        if requested_serial and str(requested_serial) != self.instance.device.serial:
            return _failed_result(
                "ROBOT_PACK_DEVICE_IDENTITY_MISMATCH",
                "Action attempted to substitute a different device serial",
            )

        try:
            output_dir = self._output_directory(action)
        except _ArtifactDirectoryError as exc:
            return _failed_result(
                exc.code,
                str(exc),
            )
        params = {
            **action.arguments,
            "workspace": str(self.home),
            "body_id": self.instance.instance_id,
            "serial": self.instance.device.serial,
            "server_name": self.instance.adapter.server_name,
            "output_dir": str(output_dir),
        }
        capture_started_at = datetime.now(UTC)
        try:
            from rosclaw.skill.builtins.realsense_capture_rgbd.runner import run

            result = run(params)
        except Exception as exc:  # noqa: BLE001 - native/MCP failures become receipts
            return _failed_result("ROBOT_PACK_ADAPTER_ERROR", str(exc), output_dir=output_dir)
        capture_finished_at = datetime.now(UTC)

        if not isinstance(result, dict):
            return _failed_result(
                "ROBOT_PACK_ADAPTER_PROTOCOL_ERROR",
                "RealSense adapter returned a non-mapping response",
                output_dir=output_dir,
            )
        if result.get("status") != "success":
            return _failed_result(
                "ROBOT_PACK_CAPTURE_FAILED",
                str(result.get("reason") or "RealSense MCP capture failed"),
                output_dir=output_dir,
            )
        captured_at = str(result.get("timestamp") or "")
        captured_timestamp = _parse_timestamp(captured_at)
        mcp_result = result.get("mcp_result")
        metadata_ok = bool(
            result.get("serial") == self.instance.device.serial
            and result.get("server_name") == self.instance.adapter.server_name
            and result.get("tool") in {"capture_aligned_rgbd", "capture_frames"}
            and captured_timestamp is not None
            and capture_started_at - timedelta(seconds=5)
            <= captured_timestamp
            <= capture_finished_at + timedelta(seconds=5)
            and isinstance(mcp_result, dict)
            and _is_positive_int(mcp_result.get("width"))
            and _is_positive_int(mcp_result.get("height"))
            and mcp_result.get("aligned") is True
        )
        if not metadata_ok:
            return _failed_result(
                "ROBOT_PACK_CAPTURE_METADATA_INVALID",
                "Capture must report the exact serial, timestamp, positive dimensions, and RGB-D alignment",
                output_dir=output_dir,
            )
        assert isinstance(mcp_result, dict)
        artifacts = _resolve_rgbd_artifacts(result, output_dir)
        missing = [name for name in ("color", "depth") if name not in artifacts]
        if missing:
            return _failed_result(
                "ROBOT_PACK_ARTIFACT_MISSING",
                f"Capture did not produce required artifacts: {', '.join(missing)}",
                output_dir=output_dir,
            )
        hashes = {name: f"sha256:{_hash_file(path)}" for name, path in artifacts.items()}
        artifact_uris = [path.as_uri() for path in artifacts.values()]
        observation = {
            "kind": "rgbd_capture",
            "device_identity": {
                "model": self.instance.device.model,
                "serial": self.instance.device.serial,
                "stable_uri": self.instance.device.stable_uri,
                "firmware": result.get("firmware") or self.instance.device.firmware_at_configure,
            },
            "captured_at": captured_at,
            "artifact_hashes": hashes,
            "artifacts": {name: path.as_uri() for name, path in artifacts.items()},
            "metrics": {
                **(result.get("metrics") if isinstance(result.get("metrics"), dict) else {}),
                "width": mcp_result["width"],
                "height": mcp_result["height"],
                "aligned": True,
            },
        }
        return ActionExecutionResult(
            final_state=ActionState.COMPLETED,
            evidence_level=EvidenceLevel.PHYSICALLY_OBSERVED,
            policy_decision={
                "allowed": True,
                "policy": "robot-pack/perception-only",
                "reason": "read-only Pack capability",
            },
            authorization_decision={
                "authorized": action.authorization.approved,
                "approval_id": action.authorization.approval_id,
            },
            dispatch_result={
                "accepted": True,
                "adapter": self.instance.adapter.component_id,
                "server_name": result.get("server_name"),
                "tool": result.get("tool"),
            },
            driver_ack={"acknowledged": True, "captured_at": captured_at},
            observations=[observation],
            verification_result={
                "success": True,
                "predicate": "aligned RGB-D artifacts exist and match recorded hashes",
                "artifact_hashes": hashes,
            },
            artifacts=artifact_uris,
            artifact_directory=str(output_dir),
        )

    def _output_directory(self, action: ActionEnvelope) -> Path:
        artifacts_parent = self.artifacts_root.parent
        if artifacts_parent.is_symlink() or self.artifacts_root.is_symlink():
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "Robot Pack artifact path cannot contain a symbolic link",
            )
        artifacts_parent.mkdir(parents=True, exist_ok=True)
        self.artifacts_root.mkdir(exist_ok=True)
        if (
            artifacts_parent.is_symlink()
            or self.artifacts_root.is_symlink()
            or self.artifacts_root.resolve() != self.artifacts_root
        ):
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "Robot Pack artifact path cannot contain a symbolic link",
            )
        configured = action.arguments.get("output_dir")
        candidate = (
            Path(str(configured)).expanduser()
            if configured
            else self.artifacts_root / action.action_id
        )
        if not candidate.is_absolute():
            candidate = self.artifacts_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(self.artifacts_root)
        except ValueError:
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_PATH_DENIED",
                "RealSense artifacts must remain under ROSCLAW_HOME/artifacts/robot-packs",
            ) from None
        candidate.parent.mkdir(parents=True, exist_ok=True)
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError as exc:
            raise _ArtifactDirectoryError(
                "ROBOT_PACK_ARTIFACT_COLLISION",
                "A fresh artifact directory is required for every action id",
            ) from exc
        return candidate


def load_daemon_robot_pack(
    runtime: Any,
    *,
    robot_id: str,
    home: str | Path | None = None,
) -> dict[str, Any] | None:
    """Load exactly one configured Pack instance into a daemon Runtime."""

    resolved_home = resolve_home(str(home) if home is not None else None)
    instances_root = resolved_home / "robots" / "instances"
    config_path = instances_root / f"{robot_id}.yaml"
    if instances_root.is_symlink() or config_path.is_symlink():
        raise RobotPackRuntimeError("Configured Robot Pack instance cannot be a symbolic link")
    if not config_path.is_file():
        return None
    try:
        instance = RobotInstanceConfig.from_path(config_path)
        store = RobotPackStore(resolved_home)
        record, manifest = store.resolve_installed(instance.pack.ref)
    except Exception as exc:  # noqa: BLE001 - configured state must fail daemon startup
        raise RobotPackRuntimeError(f"Configured Robot Pack cannot be loaded: {exc}") from exc

    verification = verify_robot_pack(record.path)
    if not verification.ok or not verification.trusted:
        raise RobotPackRuntimeError(
            "Configured Robot Pack failed trusted integrity verification: "
            + "; ".join(verification.errors)
        )
    if verification.manifest_digest != instance.pack.manifest_digest:
        raise RobotPackRuntimeError("Robot instance Pack digest does not match installed content")
    _validate_instance_contract(instance, manifest, robot_id=robot_id)
    if manifest.safety.agent_southbound_access != "daemon_only":
        raise RobotPackRuntimeError("Robot Pack does not require daemon-only Agent access")
    if manifest.safety.actuation == "forbidden" and any(
        capability.safety_class != "read_only" for capability in manifest.capabilities
    ):
        raise RobotPackRuntimeError("Actuation-forbidden Pack exposes a non-read-only capability")

    try:
        resolver = BodyResolver(workspace=resolved_home, body_id=instance.instance_id)
        body = resolver.get_current_body_yaml()
        effective = resolver.get_effective_body()
    except Exception as exc:  # noqa: BLE001 - configured Body must be complete
        raise RobotPackRuntimeError(f"Configured Robot Pack Body cannot be loaded: {exc}") from exc
    body_matches = bool(
        effective.effective_body_hash == instance.body_snapshot_hash
        and body.body_instance.get("serial_number") == instance.device.serial
        and body.metadata.get("robot_pack_ref") == instance.pack.ref
        and body.metadata.get("robot_pack_manifest_digest") == instance.pack.manifest_digest
        and body.metadata.get("device_stable_uri") == instance.device.stable_uri
        and body.metadata.get("perception_only") is manifest.safety.perception_only
        and body.metadata.get("no_actuation") is (manifest.safety.actuation == "forbidden")
        and sorted(body.capabilities.get("enabled", [])) == sorted(instance.capabilities)
        and body.agent_policy.get("direct_real_robot_execution_allowed") is False
        and body.agent_policy.get("robot_pack_gateway") == "rosclawd"
    )
    if not body_matches:
        raise RobotPackRuntimeError(
            "Configured Robot Pack Body snapshot or device binding no longer matches the instance"
        )

    current_adapter = resolve_adapter_binding(manifest, resolved_home)
    if (
        instance.adapter.status != "installed"
        or current_adapter.status != "installed"
        or not current_adapter.server_name
        or current_adapter.server_name != instance.adapter.server_name
    ):
        raise RobotPackRuntimeError(
            "Configured Robot Pack adapter binding is missing or no longer matches its locked revision"
        )

    registered: list[str] = []
    for capability in manifest.capabilities:
        if capability.id != "camera.capture_rgbd" or capability.safety_class != "read_only":
            raise RobotPackRuntimeError(
                f"No daemon-side executor is implemented for Pack capability {capability.id!r}"
            )
        if "REAL" not in capability.execution_modes:
            raise RobotPackRuntimeError("RealSense capture capability must declare REAL mode")
        runtime.action_gateway.register_executor(
            capability.id,
            ExecutionMode.REAL,
            RealSenseCaptureExecutor(instance, home=resolved_home),
        )
        registered.append(f"{capability.id}:REAL")

    status = {
        "loaded": True,
        "instance_id": instance.instance_id,
        "pack_ref": record.ref,
        "manifest_digest": record.manifest_digest,
        "signature_status": verification.signature_status,
        "support_tier": record.support_tier,
        "device": {
            "type": instance.device.type,
            "model": instance.device.model,
            "stable_uri": instance.device.stable_uri,
        },
        "registered_executors": registered,
        "safety": instance.safety,
    }
    runtime.robot_pack_status = status
    return status


def _resolve_rgbd_artifacts(result: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    raw = result.get("artifacts")
    if not isinstance(raw, dict):
        raw = {}
    candidates = {
        "color": raw.get("color") or raw.get("color_path") or raw.get("save_path"),
        "depth": raw.get("depth") or raw.get("depth_path"),
    }
    resolved: dict[str, Path] = {}
    for name, value in candidates.items():
        if not value:
            fallback = output_dir / f"{name}.png"
            value = fallback if fallback.is_file() else None
        if value:
            unresolved = Path(str(value)).expanduser()
            if unresolved.is_symlink():
                continue
            path = unresolved.resolve()
            try:
                path.relative_to(output_dir)
            except ValueError:
                continue
            if path.is_file() and path.stat().st_size > 0:
                resolved[name] = path
    return resolved


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_instance_contract(
    instance: RobotInstanceConfig,
    manifest: RobotPackManifest,
    *,
    robot_id: str,
) -> None:
    if instance.instance_id != robot_id:
        raise RobotPackRuntimeError("Robot instance id does not match its configured file name")

    variant = next(
        (
            candidate
            for candidate in manifest.device.variants
            if instance.device.product_id in candidate.product_ids
        ),
        None,
    )
    model = instance.device.model.casefold()
    parsed_uri = urlparse(instance.device.stable_uri)
    device_ok = bool(
        instance.device.type == manifest.device.type
        and instance.device.vendor_id in manifest.device.vendor_ids
        and variant is not None
        and any(pattern.casefold() in model for pattern in variant.model_patterns)
        and variant.body_profile == instance.body_profile
        and parsed_uri.scheme == "realsense"
        and parsed_uri.netloc == instance.device.serial
    )
    if not device_ok:
        raise RobotPackRuntimeError(
            "Robot instance device contract does not match the signed Robot Pack"
        )

    declared_capabilities = sorted(capability.id for capability in manifest.capabilities)
    if sorted(instance.capabilities) != declared_capabilities:
        raise RobotPackRuntimeError(
            "Robot instance capability contract does not match the signed Robot Pack"
        )

    expected_safety = {
        "perception_only": manifest.safety.perception_only,
        "actuation": manifest.safety.actuation,
        "direct_driver_access": manifest.safety.direct_driver_access,
        "agent_southbound_access": manifest.safety.agent_southbound_access,
    }
    if instance.safety != expected_safety:
        raise RobotPackRuntimeError(
            "Robot instance safety contract does not match the signed Robot Pack"
        )
    if instance.adapter.component_id != manifest.adapter.component_id:
        raise RobotPackRuntimeError(
            "Robot instance adapter contract does not match the signed Robot Pack"
        )


def _failed_result(
    code: str,
    message: str,
    *,
    output_dir: Path | None = None,
) -> ActionExecutionResult:
    return ActionExecutionResult(
        final_state=ActionState.FAILED,
        evidence_level=EvidenceLevel.REQUESTED,
        policy_decision={"allowed": False, "reason": message},
        dispatch_result={"accepted": False},
        errors=[{"code": code, "message": message}],
        artifact_directory=str(output_dir) if output_dir else None,
    )


__all__ = [
    "RealSenseCaptureExecutor",
    "RobotPackRuntimeError",
    "load_daemon_robot_pack",
]

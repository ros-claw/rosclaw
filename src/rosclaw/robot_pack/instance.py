"""Bind an installed Robot Pack to one immutable device and Body identity."""

from __future__ import annotations

import re
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rosclaw.body.registry import BodyRegistryError
from rosclaw.body.resolver import BodyResolver
from rosclaw.body.service import BodyInstanceService
from rosclaw.mcp.onboarding.installed import InstalledRegistry
from rosclaw.robot_pack.discovery import (
    DiscoveredDevice,
    DiscoveryReport,
    discover_realsense_devices,
    match_device_variant,
)
from rosclaw.robot_pack.schema import RobotPackManifest
from rosclaw.robot_pack.store import RobotPackStore

_INSTANCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")


class RobotInstanceError(RuntimeError):
    """Raised when a Pack cannot be bound without ambiguity or unsafe assumptions."""


class _InstanceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PackBinding(_InstanceModel):
    ref: str = Field(pattern=r"^rosclaw://robot_pack/[a-z0-9_-]+/[a-z0-9_.-]+@[^/]+$")
    manifest_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    signature_status: Literal["valid"]


class DeviceBinding(_InstanceModel):
    type: str = Field(min_length=1)
    vendor_id: str = Field(pattern=r"^[0-9a-f]{4}$")
    product_id: str = Field(pattern=r"^[0-9a-f]{4}$")
    model: str = Field(min_length=1)
    serial: str = Field(min_length=1, max_length=128)
    firmware_at_configure: str = Field(min_length=1)
    usb_speed_at_configure: str = Field(min_length=1)
    stable_uri: str = Field(min_length=1)
    stable_path: str | None = None
    discovery_backend: str
    offline_configured: bool = False


class AdapterBinding(_InstanceModel):
    component_id: str
    server_name: str | None = None
    status: Literal["installed", "not_installed", "version_mismatch", "unknown"]


class RobotInstanceConfig(_InstanceModel):
    schema_version: Literal["rosclaw.robot_instance.v1"]
    instance_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    configured_at: str
    pack: PackBinding
    device: DeviceBinding
    body_profile: str
    body_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    capabilities: list[str]
    adapter: AdapterBinding
    safety: dict[str, Any]

    @classmethod
    def from_path(cls, path: str | Path) -> RobotInstanceConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise RobotInstanceError(f"Robot instance config must be a mapping: {path}")
        return cls.model_validate(raw)


def configure_robot_instance(
    pack_identifier: str,
    *,
    home: str | Path | None = None,
    instance_id: str | None = None,
    serial: str | None = None,
    model: str | None = None,
    stable_uri: str | None = None,
    allow_offline: bool = False,
    force: bool = False,
    switch_active: bool = False,
    discovery_report: DiscoveryReport | None = None,
) -> tuple[RobotInstanceConfig, Path]:
    """Configure one Pack instance and materialize its canonical Body workspace."""

    store = RobotPackStore(home)
    record, manifest = store.resolve_installed(pack_identifier)
    selected, offline = _select_device(
        manifest,
        serial=serial,
        model=model,
        stable_uri=stable_uri,
        allow_offline=allow_offline,
        report=discovery_report,
    )
    variant = match_device_variant(
        manifest,
        product_id=selected.product_id,
        model=selected.model,
    )
    if variant is None:
        raise RobotInstanceError(
            f"Device {selected.model!r} ({selected.product_id}) is not supported by {record.ref}"
        )
    _require_exact_device_identity(selected, manifest, variant.product_ids)

    resolved_id = instance_id or _default_instance_id(variant.body_profile, selected.serial)
    if _INSTANCE_ID_RE.fullmatch(resolved_id) is None:
        raise RobotInstanceError(
            "instance id must use 2-64 lowercase letters, digits, underscores, or dashes"
        )
    instance_path = store.home / "robots" / "instances" / f"{resolved_id}.yaml"
    if instance_path.exists() and not force:
        if instance_path.is_symlink():
            raise RobotInstanceError(
                f"Robot instance config cannot be a symbolic link: {instance_path}"
            )
        existing = RobotInstanceConfig.from_path(instance_path)
        if _existing_instance_matches(
            existing,
            record_ref=record.ref,
            manifest_digest=record.manifest_digest,
            selected=selected,
            body_profile=variant.body_profile,
            manifest=manifest,
            home=store.home,
        ):
            current_adapter = resolve_adapter_binding(manifest, store.home)
            if current_adapter != existing.adapter:
                refreshed = RobotInstanceConfig.model_validate(
                    {
                        **existing.model_dump(mode="json"),
                        "configured_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                        "adapter": current_adapter.model_dump(mode="json"),
                    }
                )
                _write_instance_atomic(instance_path, refreshed)
                return refreshed, instance_path
            return existing, instance_path
        raise RobotInstanceError(
            f"Robot instance already exists but no longer matches its signed binding: "
            f"{resolved_id}; inspect it or use --force"
        )

    try:
        BodyInstanceService(workspace=store.home).create_or_init(
            robot=variant.body_profile,
            name=resolved_id,
            nickname=f"{selected.model} {selected.serial}",
            mode="registry",
            version="1.0.0",
            force=force,
            update_registry=True,
            switch_active=switch_active,
            render_agent_view=True,
        )
    except BodyRegistryError as exc:
        raise RobotInstanceError(str(exc)) from exc

    resolver = BodyResolver(workspace=store.home, body_id=resolved_id)
    enabled_capabilities = sorted(capability.id for capability in manifest.capabilities)
    resolver.update_body_yaml(
        {
            "body_instance.serial_number": selected.serial,
            "capabilities.enabled": enabled_capabilities,
            "metadata.robot_pack_ref": record.ref,
            "metadata.robot_pack_manifest_digest": record.manifest_digest,
            "metadata.device_stable_uri": selected.stable_uri,
            "metadata.perception_only": manifest.safety.perception_only,
            "metadata.no_actuation": manifest.safety.actuation == "forbidden",
            "agent_policy.direct_real_robot_execution_allowed": False,
            "agent_policy.robot_pack_gateway": "rosclawd",
        }
    )
    effective, _compatibility = resolver.refresh_all_artifacts(reason="robot_pack_configure")
    resolver.create_snapshot(effective)
    adapter = resolve_adapter_binding(manifest, store.home)
    config = RobotInstanceConfig(
        schema_version="rosclaw.robot_instance.v1",
        instance_id=resolved_id,
        configured_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        pack=PackBinding(
            ref=record.ref,
            manifest_digest=record.manifest_digest,
            signature_status="valid",
        ),
        device=DeviceBinding(
            type=selected.device_type,
            vendor_id=selected.vendor_id,
            product_id=selected.product_id,
            model=selected.model,
            serial=selected.serial,
            firmware_at_configure=selected.firmware,
            usb_speed_at_configure=selected.usb_speed,
            stable_uri=selected.stable_uri,
            stable_path=selected.stable_path,
            discovery_backend=selected.backend,
            offline_configured=offline,
        ),
        body_profile=variant.body_profile,
        body_snapshot_hash=effective.effective_body_hash,
        capabilities=enabled_capabilities,
        adapter=adapter,
        safety={
            "perception_only": manifest.safety.perception_only,
            "actuation": manifest.safety.actuation,
            "direct_driver_access": manifest.safety.direct_driver_access,
            "agent_southbound_access": manifest.safety.agent_southbound_access,
        },
    )
    _write_instance_atomic(instance_path, config)
    return config, instance_path


def load_robot_instance(
    identifier: str,
    *,
    home: str | Path | None = None,
) -> tuple[RobotInstanceConfig, Path]:
    store = RobotPackStore(home)
    direct = store.home / "robots" / "instances" / f"{identifier}.yaml"
    if _INSTANCE_ID_RE.fullmatch(identifier) and direct.is_file() and not direct.is_symlink():
        return RobotInstanceConfig.from_path(direct), direct

    matches: list[tuple[RobotInstanceConfig, Path]] = []
    instances_root = store.home / "robots" / "instances"
    if instances_root.is_dir():
        for path in sorted(instances_root.glob("*.yaml")):
            if path.is_symlink():
                continue
            try:
                config = RobotInstanceConfig.from_path(path)
            except Exception:
                continue
            pack_name = config.pack.ref.split("/")[-1].split("@", 1)[0]
            if identifier in {config.pack.ref, pack_name}:
                matches.append((config, path))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RobotInstanceError(
            f"Multiple configured instances match {identifier!r}; specify an instance id"
        )
    raise RobotInstanceError(f"Robot instance is not configured: {identifier}")


def _select_device(
    manifest: RobotPackManifest,
    *,
    serial: str | None,
    model: str | None,
    stable_uri: str | None,
    allow_offline: bool,
    report: DiscoveryReport | None,
) -> tuple[DiscoveredDevice, bool]:
    discovery = report or discover_realsense_devices(manifest)
    candidates = list(discovery.devices)
    if serial:
        candidates = [device for device in candidates if device.serial == serial]
    if model:
        candidates = [
            device for device in candidates if model.casefold() in device.model.casefold()
        ]
    if len(candidates) == 1:
        return candidates[0], False
    if len(candidates) > 1:
        identities = ", ".join(f"{device.model} sn={device.serial}" for device in candidates)
        raise RobotInstanceError(
            f"Multiple devices match; pass --serial to choose one: {identities}"
        )
    if not allow_offline:
        detail = "; ".join((*discovery.errors, *discovery.warnings))
        raise RobotInstanceError(
            "No matching live device was discovered. Attach it or use --allow-offline with "
            f"both --model and --serial. {detail}".strip()
        )
    if not serial or not model:
        raise RobotInstanceError("Offline configuration requires both --model and --serial")
    variant = match_device_variant(manifest, model=model)
    if variant is None:
        raise RobotInstanceError(f"Unsupported offline model: {model}")
    product_id = variant.product_ids[0]
    return (
        DiscoveredDevice(
            device_type=manifest.device.type,
            vendor_id=manifest.device.vendor_ids[0],
            product_id=product_id,
            model=variant.model,
            serial=serial,
            firmware="unknown",
            usb_speed="unknown",
            stable_uri=stable_uri or f"realsense://{serial}",
            stable_path=None,
            backend="offline_operator_input",
            body_profile=variant.body_profile,
            pack_ref=manifest.canonical_ref,
            identity_complete=False,
        ),
        True,
    )


def resolve_adapter_binding(manifest: RobotPackManifest, home: Path) -> AdapterBinding:
    patterns = [pattern.casefold() for pattern in manifest.adapter.server_name_patterns]
    try:
        records = InstalledRegistry(home=home).list()
    except Exception:
        return AdapterBinding(
            component_id=manifest.adapter.component_id,
            status="unknown",
        )
    candidates = [
        record
        for record in records
        if record.server_name == manifest.adapter.component_id
        or record.server_name.casefold() in patterns
    ]
    exact = [record for record in candidates if record.server_name == manifest.adapter.component_id]
    if exact:
        candidates = exact
    if len(candidates) > 1:
        return AdapterBinding(
            component_id=manifest.adapter.component_id,
            status="unknown",
        )
    for record in candidates:
        component = manifest.component(manifest.adapter.component_id)
        if not _adapter_source_matches_lock(record, component.ref, component.version, home):
            return AdapterBinding(
                component_id=manifest.adapter.component_id,
                server_name=record.server_name,
                status="version_mismatch",
            )
        return AdapterBinding(
            component_id=manifest.adapter.component_id,
            server_name=record.server_name,
            status="installed" if record.status == "installed" else "unknown",
        )
    return AdapterBinding(
        component_id=manifest.adapter.component_id,
        status="not_installed",
    )


def _adapter_source_matches_lock(
    record: Any,
    expected_source: str,
    expected_revision: str | None,
    home: Path,
) -> bool:
    if not expected_revision or record.extra.get("repo_commit") != expected_revision:
        return False
    if record.artifact_type != "git":
        return False
    if _normalize_git_url(str(record.extra.get("source_url") or "")) != _normalize_git_url(
        expected_source
    ):
        return False
    if record.extra.get("requested_revision") != expected_revision:
        return False

    source = Path(record.server_dir).expanduser()
    expected_path = home / "mcp" / "installed" / record.server_name / "source"
    if source.is_symlink() or source.absolute() != expected_path.absolute() or not source.is_dir():
        return False
    try:
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if head.returncode != 0 or head.stdout.strip() != expected_revision:
        return False
    try:
        tracked_diff = subprocess.run(
            ["git", "-C", str(source), "diff", "--quiet", "--no-ext-diff", "HEAD", "--"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if tracked_diff.returncode != 0:
        return False
    try:
        worktree_status = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    if worktree_status.returncode != 0:
        return False
    for entry in worktree_status.stdout.split("\0"):
        if not entry:
            continue
        if not entry.startswith("?? "):
            return False
        if not _allowed_generated_adapter_path(entry[3:]):
            return False
    try:
        ignored_files = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "-z",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return ignored_files.returncode == 0 and not any(
        path and not _allowed_generated_adapter_path(path)
        for path in ignored_files.stdout.split("\0")
    )


def _normalize_git_url(value: str) -> str:
    return value.strip().rstrip("/").removesuffix(".git")


def _allowed_generated_adapter_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(
        ("__pycache__" in path.parts and path.suffix == ".pyc")
        or ".pytest_cache" in path.parts
        or any(part.endswith(".egg-info") for part in path.parts)
    )


def _require_exact_device_identity(
    selected: DiscoveredDevice,
    manifest: RobotPackManifest,
    product_ids: list[str],
) -> None:
    exact = bool(
        selected.vendor_id in manifest.device.vendor_ids
        and selected.product_id in product_ids
        and selected.model.strip()
        and selected.serial.strip()
        and selected.stable_uri.strip()
    )
    parsed = urlparse(selected.stable_uri)
    stable_uri_matches = manifest.discovery.backend != "realsense" or (
        parsed.scheme == "realsense" and parsed.netloc == selected.serial
    )
    if not exact or not stable_uri_matches:
        raise RobotInstanceError(
            "Device binding requires exact vendor, product, model, serial, and stable URI identity"
        )


def _existing_instance_matches(
    existing: RobotInstanceConfig,
    *,
    record_ref: str,
    manifest_digest: str,
    selected: DiscoveredDevice,
    body_profile: str,
    manifest: RobotPackManifest,
    home: Path,
) -> bool:
    expected_safety = {
        "perception_only": manifest.safety.perception_only,
        "actuation": manifest.safety.actuation,
        "direct_driver_access": manifest.safety.direct_driver_access,
        "agent_southbound_access": manifest.safety.agent_southbound_access,
    }
    expected_capabilities = sorted(capability.id for capability in manifest.capabilities)
    binding_matches = bool(
        existing.pack.ref == record_ref
        and existing.pack.manifest_digest == manifest_digest
        and existing.pack.signature_status == "valid"
        and existing.device.type == selected.device_type
        and existing.device.vendor_id == selected.vendor_id
        and existing.device.product_id == selected.product_id
        and existing.device.serial == selected.serial
        and existing.device.model.casefold() == selected.model.casefold()
        and existing.device.stable_uri == selected.stable_uri
        and existing.body_profile == body_profile
        and sorted(existing.capabilities) == expected_capabilities
        and existing.safety == expected_safety
        and existing.adapter.component_id == manifest.adapter.component_id
    )
    if not binding_matches:
        return False
    try:
        resolver = BodyResolver(workspace=home, body_id=existing.instance_id)
        body = resolver.get_current_body_yaml()
        effective = resolver.get_effective_body()
    except Exception:
        return False
    return bool(
        effective.effective_body_hash == existing.body_snapshot_hash
        and body.body_instance.get("serial_number") == existing.device.serial
        and body.metadata.get("robot_pack_ref") == record_ref
        and body.metadata.get("robot_pack_manifest_digest") == manifest_digest
        and body.metadata.get("device_stable_uri") == existing.device.stable_uri
        and sorted(body.capabilities.get("enabled", [])) == expected_capabilities
        and body.agent_policy.get("direct_real_robot_execution_allowed") is False
        and body.agent_policy.get("robot_pack_gateway") == "rosclawd"
    )


def _default_instance_id(profile: str, serial: str) -> str:
    suffix = re.sub(r"[^a-z0-9]", "", serial.casefold())[-8:] or uuid.uuid4().hex[:8]
    base = profile.replace("_", "-")
    return f"{base}-{suffix}"[:64]


def _write_instance_atomic(path: Path, config: RobotInstanceConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RobotInstanceError(f"Robot instance config cannot be a symbolic link: {path}")
    temporary = path.with_suffix(f".yaml.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "RobotInstanceConfig",
    "RobotInstanceError",
    "configure_robot_instance",
    "load_robot_instance",
    "resolve_adapter_binding",
]

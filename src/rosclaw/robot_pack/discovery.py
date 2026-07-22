"""Read-only device discovery for Robot Pack onboarding."""

from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw.robot_pack.schema import DeviceVariant, RobotPackManifest

DEFAULT_USB_SYSFS_ROOT = Path("/sys/bus/usb/devices")
DEFAULT_V4L_BY_ID_ROOT = Path("/dev/v4l/by-id")


@dataclass(frozen=True)
class StreamProfile:
    stream: str
    format: str
    fps: int | None = None
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class DiscoveredDevice:
    device_type: str
    vendor_id: str
    product_id: str
    model: str
    serial: str
    firmware: str
    usb_speed: str
    stable_uri: str
    stable_path: str | None
    backend: str
    stream_profiles: tuple[StreamProfile, ...] = ()
    physical_port: str | None = None
    product_line: str | None = None
    device_nodes: tuple[str, ...] = ()
    body_profile: str | None = None
    pack_ref: str | None = None
    identity_complete: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["stream_profiles"] = [asdict(profile) for profile in self.stream_profiles]
        value["device_nodes"] = list(self.device_nodes)
        return value


@dataclass(frozen=True)
class DiscoveryReport:
    device_type: str
    devices: tuple[DiscoveredDevice, ...]
    attempted_backends: tuple[str, ...]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "rosclaw.robot_discovery.v1",
            "ok": self.ok,
            "device_type": self.device_type,
            "count": len(self.devices),
            "attempted_backends": list(self.attempted_backends),
            "devices": [device.to_dict() for device in self.devices],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def discover_realsense_devices(
    manifest: RobotPackManifest,
    *,
    backend: str = "auto",
    sdk: Any | None = None,
    sysfs_root: str | Path = DEFAULT_USB_SYSFS_ROOT,
    by_id_root: str | Path = DEFAULT_V4L_BY_ID_ROOT,
) -> DiscoveryReport:
    """Discover supported RealSense cameras without opening a stream."""

    if manifest.discovery.backend != "realsense" or manifest.device.type != "camera":
        return DiscoveryReport(
            device_type=manifest.device.type,
            devices=(),
            attempted_backends=(),
            errors=(f"Pack {manifest.canonical_ref} does not define RealSense discovery",),
        )
    if backend not in {"auto", "sdk", "sysfs"}:
        return DiscoveryReport(
            device_type="camera",
            devices=(),
            attempted_backends=(),
            errors=(f"Unsupported RealSense discovery backend: {backend}",),
        )

    attempted: list[str] = []
    warnings: list[str] = []
    sdk_error: str | None = None
    if backend in {"auto", "sdk"}:
        attempted.append("pyrealsense2")
        try:
            loaded_sdk = sdk if sdk is not None else importlib.import_module("pyrealsense2")
            devices = _discover_with_sdk(
                loaded_sdk,
                manifest,
                by_id_root=Path(by_id_root),
            )
            if devices or backend == "sdk":
                return DiscoveryReport(
                    device_type="camera",
                    devices=tuple(devices),
                    attempted_backends=tuple(attempted),
                    warnings=tuple(warnings),
                )
            warnings.append("pyrealsense2 reported no supported D405 or D435i device")
        except Exception as exc:  # noqa: BLE001 - optional native SDK boundary
            sdk_error = str(exc)
            warnings.append(f"pyrealsense2 discovery unavailable: {exc}")
            if backend == "sdk":
                return DiscoveryReport(
                    device_type="camera",
                    devices=(),
                    attempted_backends=tuple(attempted),
                    errors=(f"pyrealsense2 discovery failed: {exc}",),
                )

    attempted.append("linux_sysfs")
    try:
        devices = _discover_with_sysfs(
            manifest,
            sysfs_root=Path(sysfs_root),
            by_id_root=Path(by_id_root),
        )
    except Exception as exc:  # noqa: BLE001 - platform filesystem boundary
        detail = f"linux sysfs discovery failed: {exc}"
        errors = (detail,) if sdk_error else ()
        warnings.append(detail)
        return DiscoveryReport(
            device_type="camera",
            devices=(),
            attempted_backends=tuple(attempted),
            errors=errors,
            warnings=tuple(warnings),
        )
    if devices:
        warnings.append(
            "sysfs fallback cannot prove firmware or stream-profile readiness; use pyrealsense2 "
            "or the Pack MCP adapter before read-only verification"
        )
    return DiscoveryReport(
        device_type="camera",
        devices=tuple(devices),
        attempted_backends=tuple(attempted),
        warnings=tuple(warnings),
    )


def _discover_with_sdk(
    sdk: Any,
    manifest: RobotPackManifest,
    *,
    by_id_root: Path,
) -> list[DiscoveredDevice]:
    context = sdk.context()
    discovered: list[DiscoveredDevice] = []
    for raw_device in list(context.query_devices()):
        info = {
            "model": _device_info(sdk, raw_device, "camera_info_name"),
            "serial": _device_info(sdk, raw_device, "camera_info_serial_number"),
            "firmware": _device_info(sdk, raw_device, "camera_info_firmware_version"),
            "usb_speed": _device_info(sdk, raw_device, "camera_info_usb_type_descriptor"),
            "physical_port": _device_info(sdk, raw_device, "camera_info_physical_port"),
            "product_line": _device_info(sdk, raw_device, "camera_info_product_line"),
            "product_id": _device_info(sdk, raw_device, "camera_info_product_id").lower(),
        }
        variant = _match_variant(
            manifest,
            product_id=info["product_id"],
            model=info["model"],
        )
        if variant is None:
            continue
        serial = info["serial"]
        nodes = _stable_device_nodes(serial, by_id_root)
        profiles = tuple(_stream_profiles(raw_device))
        stable_uri = f"realsense://{serial}" if serial else ""
        required: dict[str, object] = {
            "vendor_id": manifest.device.vendor_ids[0],
            "product_id": info["product_id"],
            "model": info["model"],
            "serial": serial,
            "firmware": info["firmware"],
            "usb_speed": info["usb_speed"],
            "stable_uri": stable_uri,
            "stream_profiles": profiles,
            "backend": "pyrealsense2",
        }
        discovered.append(
            DiscoveredDevice(
                device_type="camera",
                vendor_id=manifest.device.vendor_ids[0],
                product_id=info["product_id"],
                model=info["model"] or variant.model,
                serial=serial,
                firmware=info["firmware"],
                usb_speed=info["usb_speed"],
                stable_uri=stable_uri,
                stable_path=nodes[0] if nodes else None,
                backend="pyrealsense2",
                stream_profiles=profiles,
                physical_port=info["physical_port"] or None,
                product_line=info["product_line"] or None,
                device_nodes=tuple(nodes),
                body_profile=variant.body_profile,
                pack_ref=manifest.canonical_ref,
                identity_complete=all(
                    bool(required.get(field))
                    for field in manifest.discovery.required_identity_fields
                ),
            )
        )
    return sorted(discovered, key=lambda item: (item.model, item.serial))


def _discover_with_sysfs(
    manifest: RobotPackManifest,
    *,
    sysfs_root: Path,
    by_id_root: Path,
) -> list[DiscoveredDevice]:
    if not sysfs_root.is_dir():
        return []
    vendor_ids = set(manifest.device.vendor_ids)
    discovered: list[DiscoveredDevice] = []
    for entry in sorted(sysfs_root.iterdir()):
        vendor_id = _read_text(entry / "idVendor").lower()
        if vendor_id not in vendor_ids:
            continue
        product_id = _read_text(entry / "idProduct").lower()
        model = _read_text(entry / "product")
        variant = _match_variant(manifest, product_id=product_id, model=model)
        if variant is None:
            continue
        serial = _read_text(entry / "serial")
        speed = _read_text(entry / "speed")
        usb_speed = f"{speed} Mbps" if speed else "unknown"
        nodes = _stable_device_nodes(serial, by_id_root)
        stable_uri = (
            f"realsense://{serial}" if serial else f"usb://{vendor_id}:{product_id}/{entry.name}"
        )
        discovered.append(
            DiscoveredDevice(
                device_type="camera",
                vendor_id=vendor_id,
                product_id=product_id,
                model=model or variant.model,
                serial=serial,
                firmware="unknown",
                usb_speed=usb_speed,
                stable_uri=stable_uri,
                stable_path=nodes[0] if nodes else None,
                backend="linux_sysfs",
                device_nodes=tuple(nodes),
                physical_port=entry.name,
                body_profile=variant.body_profile,
                pack_ref=manifest.canonical_ref,
                identity_complete=False,
            )
        )
    return sorted(discovered, key=lambda item: (item.model, item.serial))


def _device_info(sdk: Any, device: Any, attribute: str) -> str:
    key = getattr(sdk, attribute, None)
    if key is None and attribute.startswith("camera_info_"):
        camera_info = getattr(sdk, "camera_info", None)
        key = getattr(camera_info, attribute.removeprefix("camera_info_"), None)
    if key is None:
        return ""
    try:
        supports = getattr(device, "supports", None)
        if callable(supports) and not supports(key):
            return ""
        return str(device.get_info(key)).strip()
    except Exception:
        return ""


def _stream_profiles(device: Any) -> list[StreamProfile]:
    profiles: dict[tuple[Any, ...], StreamProfile] = {}
    try:
        sensors = list(device.query_sensors())
    except Exception:
        return []
    for sensor in sensors:
        try:
            raw_profiles = list(sensor.get_stream_profiles())
        except Exception:
            continue
        for raw in raw_profiles:
            stream = _enum_label(_call_or_value(raw, "stream_type"))
            format_name = _enum_label(_call_or_value(raw, "format"))
            fps_value = _call_or_value(raw, "fps")
            fps = int(fps_value) if isinstance(fps_value, (int, float)) else None
            width: int | None = None
            height: int | None = None
            try:
                video = raw.as_video_stream_profile()
                width = int(video.width())
                height = int(video.height())
            except Exception:
                pass
            profile = StreamProfile(
                stream=stream,
                format=format_name,
                fps=fps,
                width=width,
                height=height,
            )
            key = (stream, format_name, fps, width, height)
            profiles[key] = profile
    return sorted(
        profiles.values(),
        key=lambda item: (
            item.stream,
            item.width or 0,
            item.height or 0,
            item.fps or 0,
            item.format,
        ),
    )


def _call_or_value(value: Any, attribute: str) -> Any:
    member = getattr(value, attribute, None)
    try:
        return member() if callable(member) else member
    except Exception:
        return None


def _enum_label(value: Any) -> str:
    label = str(value or "unknown")
    return label.rsplit(".", 1)[-1].lower()


def _match_variant(
    manifest: RobotPackManifest,
    *,
    product_id: str,
    model: str,
) -> DeviceVariant | None:
    normalized_pid = product_id.lower().removeprefix("0x")
    normalized_model = model.casefold()
    for variant in manifest.device.variants:
        if normalized_pid and normalized_pid in variant.product_ids:
            return variant
        if any(pattern.casefold() in normalized_model for pattern in variant.model_patterns):
            return variant
    return None


def match_device_variant(
    manifest: RobotPackManifest,
    *,
    product_id: str = "",
    model: str = "",
) -> DeviceVariant | None:
    """Resolve one discovered or operator-provided device to a Pack variant."""

    return _match_variant(manifest, product_id=product_id, model=model)


def _stable_device_nodes(serial: str, by_id_root: Path) -> list[str]:
    if not serial or not by_id_root.is_dir():
        return []
    result: list[str] = []
    try:
        entries = sorted(by_id_root.iterdir())
    except OSError:
        return []
    serial_folded = serial.casefold()
    for entry in entries:
        if serial_folded in entry.name.casefold():
            result.append(str(entry))
    return result


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


__all__ = [
    "DiscoveredDevice",
    "DiscoveryReport",
    "StreamProfile",
    "discover_realsense_devices",
    "match_device_variant",
]

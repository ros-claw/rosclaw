from __future__ import annotations

from pathlib import Path
from typing import Any

from rosclaw.robot_pack.discovery import discover_realsense_devices
from rosclaw.robot_pack.schema import RobotPackManifest


class _CameraInfo:
    name = "name"
    serial_number = "serial"
    firmware_version = "firmware"
    usb_type_descriptor = "usb"
    physical_port = "port"
    product_line = "line"
    product_id = "pid"


class _Profile:
    def __init__(self, stream: str, fmt: str, width: int, height: int, fps: int) -> None:
        self._stream = stream
        self._fmt = fmt
        self._width = width
        self._height = height
        self._fps = fps

    def stream_type(self) -> str:
        return f"stream.{self._stream}"

    def format(self) -> str:
        return f"format.{self._fmt}"

    def fps(self) -> int:
        return self._fps

    def as_video_stream_profile(self) -> _Profile:
        return self

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _Sensor:
    def get_stream_profiles(self) -> list[_Profile]:
        return [
            _Profile("depth", "z16", 640, 480, 30),
            _Profile("color", "rgb8", 640, 480, 30),
        ]


class _Device:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def supports(self, key: str) -> bool:
        return key in self.values

    def get_info(self, key: str) -> str:
        return self.values[key]

    def query_sensors(self) -> list[_Sensor]:
        return [_Sensor()]


class _Context:
    def __init__(self, devices: list[_Device]) -> None:
        self.devices = devices

    def query_devices(self) -> list[_Device]:
        return self.devices


class _Sdk:
    camera_info = _CameraInfo

    def __init__(self, devices: list[_Device]) -> None:
        self.devices = devices

    def context(self) -> _Context:
        return _Context(self.devices)


def _sdk_device() -> _Device:
    return _Device(
        {
            "name": "Intel RealSense D405",
            "serial": "RS123456",
            "firmware": "5.16.0.1",
            "usb": "3.2",
            "port": "2-1.4",
            "line": "D400",
            "pid": "0B5B",
        }
    )


def test_sdk_discovery_returns_complete_identity_and_streams(
    builtin_pack_root: Path,
    tmp_path: Path,
) -> None:
    manifest = RobotPackManifest.from_path(builtin_pack_root / "robot-pack.yaml")

    report = discover_realsense_devices(
        manifest,
        sdk=_Sdk([_sdk_device()]),
        by_id_root=tmp_path / "missing-by-id",
    )

    assert report.ok
    assert report.attempted_backends == ("pyrealsense2",)
    assert len(report.devices) == 1
    device = report.devices[0]
    assert device.model == "Intel RealSense D405"
    assert device.serial == "RS123456"
    assert device.product_id == "0b5b"
    assert device.body_profile == "realsense_d405"
    assert device.stable_uri == "realsense://RS123456"
    assert device.identity_complete
    assert {profile.stream for profile in device.stream_profiles} == {"color", "depth"}


def test_sdk_discovery_ignores_unlisted_product(builtin_pack_root: Path, tmp_path: Path) -> None:
    manifest = RobotPackManifest.from_path(builtin_pack_root / "robot-pack.yaml")
    unknown = _sdk_device()
    unknown.values["name"] = "Other USB Camera"
    unknown.values["pid"] = "ffff"

    report = discover_realsense_devices(
        manifest,
        sdk=_Sdk([unknown]),
        backend="sdk",
        by_id_root=tmp_path,
    )

    assert report.ok
    assert report.devices == ()


def test_sysfs_fallback_is_truthfully_partial(
    builtin_pack_root: Path,
    tmp_path: Path,
) -> None:
    manifest = RobotPackManifest.from_path(builtin_pack_root / "robot-pack.yaml")
    usb = tmp_path / "sysfs" / "2-1"
    usb.mkdir(parents=True)
    values: dict[str, Any] = {
        "idVendor": "8086",
        "idProduct": "0b3a",
        "product": "Intel RealSense D435I",
        "serial": "RSI987",
        "speed": "5000",
    }
    for name, value in values.items():
        (usb / name).write_text(str(value), encoding="utf-8")

    report = discover_realsense_devices(
        manifest,
        backend="sysfs",
        sysfs_root=tmp_path / "sysfs",
        by_id_root=tmp_path / "missing",
    )

    assert report.ok
    assert len(report.devices) == 1
    assert report.devices[0].model == "Intel RealSense D435I"
    assert report.devices[0].firmware == "unknown"
    assert not report.devices[0].identity_complete
    assert any("cannot prove firmware" in warning for warning in report.warnings)

"""Runtime doctor plugin for RealSense devices."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctorPlugin

_SYSFS_USB = Path("/sys/bus/usb/devices")


def _read_sysfs_autosuspend() -> str | None:
    """Return the ``power/control`` value of the Intel RealSense USB device.

    ``auto`` means runtime autosuspend is enabled for the port; ``on`` keeps
    the device permanently powered.  Returns ``None`` when no RealSense USB
    device or sysfs entry is found (e.g. non-Linux hosts).
    """
    try:
        for entry in _SYSFS_USB.iterdir():
            try:
                if (entry / "idVendor").read_text(encoding="utf-8").strip() != "8086":
                    continue
                product = (entry / "idProduct").read_text(encoding="utf-8").strip()
                if product not in {"0b3a", "0b3d", "0b5c", "0b64"}:  # D435i/D405/D455/D456
                    continue
                control = entry / "power" / "control"
                if control.exists():
                    return control.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    except OSError:
        return None
    return None


class RealSenseDoctor(RuntimeDoctorPlugin):
    """Check RealSense software availability and optionally enumerate devices."""

    name = "realsense"

    def check(self) -> Iterable[DoctorCheck]:
        try:
            import pyrealsense2 as rs
        except Exception as exc:
            yield DoctorCheck(
                plugin=self.name,
                check="pyrealsense2_import",
                status="warn",
                message=f"pyrealsense2 is not importable: {exc}",
            )
            return

        yield DoctorCheck(
            plugin=self.name,
            check="pyrealsense2_import",
            status="ok",
            message="pyrealsense2 is available",
        )

        try:
            ctx = rs.context()
            devices = list(ctx.query_devices())
            if devices:
                yield DoctorCheck(
                    plugin=self.name,
                    check="device_count",
                    status="ok",
                    message=f"Found {len(devices)} RealSense device(s)",
                    details={"count": len(devices)},
                )
            else:
                yield DoctorCheck(
                    plugin=self.name,
                    check="device_count",
                    status="warn",
                    message="No RealSense devices detected",
                    details={"count": 0},
                )
                return
        except Exception as exc:
            yield DoctorCheck(
                plugin=self.name,
                check="device_count",
                status="error",
                message=f"Failed to enumerate RealSense devices: {exc}",
            )
            return

        # Device details: firmware + USB link speed for the first device.
        dev = devices[0]
        info: dict[str, str] = {}
        for key, attr in (
            ("name", "camera_info_name"),
            ("serial", "camera_info_serial_number"),
            ("firmware", "camera_info_firmware_version"),
            ("usb_speed", "camera_info_usb_type_descriptor"),
        ):
            try:
                info[key] = dev.get_info(getattr(rs, attr))
            except Exception:
                info[key] = "unknown"
        usb_speed = str(info.get("usb_speed", "unknown"))
        degraded = usb_speed not in ("unknown", "") and "3." not in usb_speed
        yield DoctorCheck(
            plugin=self.name,
            check="device_info",
            status="warn" if degraded else "ok",
            message=(
                f"{info.get('name')} sn={info.get('serial')} "
                f"fw={info.get('firmware')} usb={usb_speed}"
                + (" — USB2 link degrades RGB-D streams" if degraded else "")
            ),
            details=info,
        )

        # USB autosuspend hygiene: not the wedge root cause, but keeping the
        # port powered avoids suspend/resume flakiness on Jetson hosts.
        autosuspend = _read_sysfs_autosuspend()
        if autosuspend is not None:
            yield DoctorCheck(
                plugin=self.name,
                check="usb_autosuspend",
                status="ok" if autosuspend == "on" else "warn",
                message=(
                    f"USB power/control={autosuspend}"
                    + (
                        ""
                        if autosuspend == "on"
                        else " — consider: echo on | sudo tee "
                        "/sys/bus/usb/devices/<port>/power/control"
                    )
                ),
                details={"power_control": autosuspend},
            )

        yield DoctorCheck(
            plugin=self.name,
            check="session_ritual",
            status="ok",
            message=(
                "Between camera sessions call hardware_reset() before the next "
                "pipe.start(); repeated starts without a reset wedge this D435i "
                "(UVC GET_CUR -110 -> USB disconnect). Never retry pipe.start() "
                "after a UVC timeout — recover by physical re-plug + 10s power drain."
            ),
        )

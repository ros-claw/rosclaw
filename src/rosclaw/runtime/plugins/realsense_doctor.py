"""Runtime doctor plugin for RealSense devices."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from rosclaw.runtime.doctor import DoctorCheck, RuntimeDoctorPlugin


class RealSenseDoctor(RuntimeDoctorPlugin):
    """Check RealSense software availability and optionally enumerate devices."""

    name = "realsense"

    def check(self) -> Iterable[DoctorCheck]:
        try:
            import pyrealsense2 as rs  # type: ignore[import-untyped]
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
        except Exception as exc:
            yield DoctorCheck(
                plugin=self.name,
                check="device_count",
                status="error",
                message=f"Failed to enumerate RealSense devices: {exc}",
            )

"""D435i capture with the field-proven wedge-safe lifecycle (PR-EVO-HW-2,
updated 2026-08-04 per the ros-claw/realsense_ops discipline).

The D435i on this rig wedges when a pipeline starts on a device whose
firmware state is stale (UVC GET_CUR -110 → USB disconnect, sometimes
unrecoverable without a physical re-plug).  The 2026-07 lesson was
"hardware_reset FIRST, always"; the 2026-08 lesson is the other half of
the story: each hardware_reset is a stress event, and ~20/day plus
pipeline stop/start churn killed the xHCI host controller (NVDA8000:00,
dmesg `Host halt failed, -110 → HC died`, 2026-08-03 — and AGAIN after a
reboot on 2026-08-04).  The lifecycle is therefore the escalation ladder:

    enumerate → plain pipe.start (no reset) → on failure ONLY:
    ONE hardware_reset → wait re-enumeration → ONE retry

and the forbidden moves are: preventive hardware_reset, pipe.start retry
loops on a wedged control channel, sysfs bus resets (kills the xHCI
controller), and SIGTERM on a streaming pipeline.  ``D435iCapture``
implements exactly that lifecycle and nothing else.  pyrealsense2 is an
optional dependency: its absence means "camera unavailable", never a mock.
"""

from __future__ import annotations

import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REENUMERATE_TIMEOUT_S = 20.0
FRAME_TIMEOUT_MS = 5000


class CameraUnavailableError(RuntimeError):
    """No D435i (or pyrealsense2) — the caller decides BLOCKED, never mock."""


class CameraWedgeError(RuntimeError):
    """The device wedged at pipeline start (frame timeout / UVC error)."""


@dataclass
class CapturedFrame:
    session_id: str
    sequence: int
    captured_mono: float
    captured_wall: float
    color: Any = None  # np.ndarray when numpy available
    depth: Any = None
    color_timestamp_ms: float = 0.0
    depth_timestamp_ms: float = 0.0

    @property
    def rgb_depth_delta_ms(self) -> float:
        return abs(self.color_timestamp_ms - self.depth_timestamp_ms)

    def age_ms(self, now_mono: float | None = None) -> float:
        now = now_mono if now_mono is not None else time.monotonic()
        return (now - self.captured_mono) * 1000.0


@dataclass
class D435iCapture:
    """One wedge-safe capture session.  Context manager: graceful stop only."""

    width: int = 640
    height: int = 480
    fps: int = 30
    session_id: str = field(default_factory=lambda: f"cam_{uuid.uuid4().hex[:12]}")
    _pipeline: Any = field(default=None, repr=False)
    _sequence: int = field(default=0, repr=False)

    # ------------------------------------------------------------------

    @staticmethod
    def _rs() -> Any:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraUnavailableError("pyrealsense2_not_installed") from exc
        return rs

    @staticmethod
    def enumerate_devices() -> list[dict[str, str]]:
        rs = D435iCapture._rs()
        devices = []
        for dev in rs.context().query_devices():
            devices.append(
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "firmware": dev.get_info(rs.camera_info.firmware_version),
                }
            )
        return devices

    def start(self, *, serial: str | None = None) -> dict[str, Any]:
        """Plain pipe.start FIRST; ONE hardware_reset only on actual failure.

        hardware_reset is a full device re-enumeration — a stress event for
        the host controller.  Reset churn (~20/day) plus pipeline stop/start
        cycles killed xHCI host controllers twice on this rig (NVDA8000:00,
        2026-08-03 and again 2026-08-04).  The escalation ladder: try the
        plain start; only a real start failure earns the single reset.
        """
        rs = self._rs()
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise CameraUnavailableError("no_device_enumerated")
        target = None
        for dev in devices:
            if serial is None or dev.get_info(rs.camera_info.serial_number) == serial:
                target = dev
                break
        if target is None:
            raise CameraUnavailableError(f"device serial {serial} not found")
        identity = {
            "name": target.get_info(rs.camera_info.name),
            "serial": target.get_info(rs.camera_info.serial_number),
            "firmware": target.get_info(rs.camera_info.firmware_version),
        }

        def _start_once() -> None:
            config = rs.config()
            config.enable_device(identity["serial"])
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self._pipeline = rs.pipeline()
            self._pipeline.start(config)
            # First frame must actually arrive — a timeout here is the
            # wedge signature; do NOT retry pipe.start (field notes).
            self._pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

        try:
            _start_once()
        except Exception as first_exc:
            with suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None
            # Escalation rung 3: ONE hardware_reset after a real start
            # failure, wait re-enumeration, ONE retry.  Never more.
            target.hardware_reset()
            deadline = time.monotonic() + REENUMERATE_TIMEOUT_S
            reenumerated = None
            while time.monotonic() < deadline:
                time.sleep(1.0)
                for dev in rs.context().query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == identity["serial"]:
                        reenumerated = dev
                        break
                if reenumerated is not None:
                    break
            if reenumerated is None:
                raise CameraUnavailableError(
                    "device did not re-enumerate after hardware_reset "
                    "(physical re-plug + 10s power drain required)"
                ) from first_exc
            try:
                _start_once()
            except Exception as exc:
                with suppress(Exception):
                    self._pipeline.stop()
                self._pipeline = None
                raise CameraWedgeError(f"pipeline start wedged: {exc}") from exc
        return {"session_id": self.session_id, **identity}

    # ------------------------------------------------------------------

    def read(self, *, timeout_ms: int = FRAME_TIMEOUT_MS) -> CapturedFrame:
        if self._pipeline is None:
            raise CameraUnavailableError("capture not started")
        frames = self._pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        self._sequence += 1
        return CapturedFrame(
            session_id=self.session_id,
            sequence=self._sequence,
            captured_mono=time.monotonic(),
            captured_wall=time.time(),
            color=_as_array(color),
            depth=_as_array(depth),
            color_timestamp_ms=float(color.get_timestamp()) if color else 0.0,
            depth_timestamp_ms=float(depth.get_timestamp()) if depth else 0.0,
        )

    def stop(self) -> None:
        """Graceful stop only — never SIGTERM a streaming pipeline."""
        if self._pipeline is not None:
            with suppress(Exception):
                self._pipeline.stop()
            self._pipeline = None

    def __enter__(self) -> D435iCapture:
        return self

    def __exit__(self, *_: Any) -> None:
        self.stop()


def _as_array(frame: Any) -> Any:
    if frame is None:
        return None
    try:
        import numpy as np

        return np.asanyarray(frame.get_data())
    except ImportError:
        return None


def write_artifacts(
    frame: CapturedFrame, directory: Path, *, prefix: str | None = None
) -> dict[str, str]:
    """Persist color/depth artifacts; returns artifact:// refs.

    PNG via OpenCV when available, raw .npy otherwise — the artifact is
    honest about its format either way.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"{frame.session_id}_{frame.sequence:06d}"
    refs: dict[str, str] = {}
    color_path = directory / f"{stem}_color"
    depth_path = directory / f"{stem}_depth"
    try:
        import cv2

        if frame.color is not None:
            cv2.imwrite(str(color_path.with_suffix(".png")), frame.color)
            refs["color"] = f"artifact://{color_path.with_suffix('.png')}"
        if frame.depth is not None:
            cv2.imwrite(str(depth_path.with_suffix(".png")), frame.depth)
            refs["depth"] = f"artifact://{depth_path.with_suffix('.png')}"
    except ImportError:
        import numpy as np

        if frame.color is not None:
            np.save(str(color_path.with_suffix(".npy")), frame.color)
            refs["color"] = f"artifact://{color_path.with_suffix('.npy')}"
        if frame.depth is not None:
            np.save(str(depth_path.with_suffix(".npy")), frame.depth)
            refs["depth"] = f"artifact://{depth_path.with_suffix('.npy')}"
    return refs

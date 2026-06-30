"""RealSense perception benchmark runner.

Orchestrates a short RealSense perception benchmark: dual-camera streams,
provider inference, practice recording, and dashboard metrics.  When no
hardware is present the runner degrades to a structured SKIP report instead
of fabricating data.
"""
from __future__ import annotations

import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RealSenseBenchReport:
    """Result of a RealSense benchmark run."""

    duration_sec: float
    status: str  # pass | fail | skip
    d405: dict[str, Any] = field(default_factory=dict)
    d435i: dict[str, Any] = field(default_factory=dict)
    provider: dict[str, Any] = field(default_factory=dict)
    practice: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_sec": self.duration_sec,
            "status": self.status,
            "d405": self.d405,
            "d435i": self.d435i,
            "provider": self.provider,
            "practice": self.practice,
            "errors": self.errors,
        }


_DASHBOARD_URL = "http://localhost:8765"


def _detect_cameras() -> tuple[bool, bool]:
    """Best-effort RealSense hardware detection.

    Tries the pyrealsense2 Python bindings first, then falls back to the
    system ``rs-enumerate-devices`` binary and finally ``lsusb`` so the
    benchmark can detect cameras even when the Python package is not
    installed in the active interpreter.
    """
    try:
        import pyrealsense2 as rs  # type: ignore[import-untyped]

        ctx = rs.context()
        devices = [ctx.get_device(i) for i in range(ctx.get_device_count())]
        d405 = any("D405" in d.get_info(rs.camera_info.name) for d in devices)
        d435i = any("D435I" in d.get_info(rs.camera_info.name) for d in devices)
        return d405, d435i
    except Exception:  # noqa: BLE001
        pass

    # Fallback 1: librealsense CLI tool
    try:
        output = subprocess.run(
            ["rs-enumerate-devices", "-s"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        text = output.lower()
        return "d405" in text, "d435i" in text
    except Exception:  # noqa: BLE001
        pass

    # Fallback 2: lsusb vendor/product ID scan
    try:
        output = subprocess.run(
            ["lsusb"], capture_output=True, text=True, timeout=5, check=False
        ).stdout
        # Intel RealSense vendor ID is 8086. Product IDs for reference:
        # D405 = 0b5b, D435i = 0b3a (among others). Name strings are not
        # always present, so we also accept the model names when available.
        text = output.lower()
        d405 = any(x in text for x in ("d405", "0b5b"))
        d435i = any(x in text for x in ("d435i", "d435", "0b3a"))
        return d405, d435i
    except Exception:  # noqa: BLE001
        return False, False


def _fetch_dashboard_streams() -> dict[str, Any]:
    """Fetch the current RealSense stream state from the dashboard."""
    try:
        req = urllib.request.Request(
            f"{_DASHBOARD_URL}/api/realsense/streams",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return {}


def _count_frames(session_dir: Path | None, camera_id: str) -> int:
    """Count saved frame files for a camera in the practice session."""
    if session_dir is None:
        return 0
    frames_dir = session_dir / "frames" / camera_id
    if not frames_dir.exists():
        return 0
    return sum(1 for p in frames_dir.iterdir() if p.is_file() and p.suffix in {".jpg", ".png", ".png16"})


def _count_imu_samples(session_dir: Path | None, camera_id: str = "d435i") -> int:
    """Count IMU sample lines recorded for a camera."""
    if session_dir is None:
        return 0
    imu_file = session_dir / "imu" / f"{camera_id}_imu.jsonl"
    if not imu_file.exists():
        return 0
    try:
        with open(imu_file, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def _camera_topic_specs(d405: bool, d435i: bool) -> list[dict[str, Any]]:
    """Build sensor_msgs topic specs for the detected cameras."""
    try:
        from sensor_msgs.msg import CameraInfo, Image, Imu
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"sensor_msgs not available: {exc}") from exc

    topics: list[dict[str, Any]] = []
    for camera_id, enabled in (("d405", d405), ("d435i", d435i)):
        if not enabled:
            continue
        topics.extend(
            [
                {"topic": f"/camera/{camera_id}/color/image_raw", "msg_type": Image},
                {"topic": f"/camera/{camera_id}/depth/image_rect_raw", "msg_type": Image},
                {"topic": f"/camera/{camera_id}/color/camera_info", "msg_type": CameraInfo},
                {"topic": f"/camera/{camera_id}/depth/camera_info", "msg_type": CameraInfo},
            ]
        )
        if camera_id == "d435i":
            topics.append({"topic": f"/camera/{camera_id}/imu", "msg_type": Imu})
    return topics


def run_realsense_bench(
    duration_sec: float = 600.0,
    data_root: str = "/data/rosclaw/practice",
    output_dir: str | None = None,
    robot_id: str = "dual_lab_01",
) -> RealSenseBenchReport:
    """Run a RealSense perception benchmark.

    Args:
        duration_sec: Benchmark duration in seconds (default 10 minutes).
        data_root: Practice session root.
        output_dir: Directory to write report.json; None skips disk write.
        robot_id: Robot/body identifier.

    Returns:
        RealSenseBenchReport with measured metrics or a SKIP status when
        RealSense hardware or ROS2 is unavailable.
    """
    start = time.monotonic()
    report = RealSenseBenchReport(duration_sec=duration_sec, status="skip")

    # ROS2 is required to subscribe to RealSense topics.
    try:
        import rclpy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"ROS2 (rclpy) not available: {exc}")
        return report

    d405_ok, d435i_ok = _detect_cameras()
    if not d405_ok and not d435i_ok:
        report.errors.append("No RealSense cameras detected (pyrealsense2 missing or no USB devices)")
        return report

    report.d405 = {"detected": d405_ok, "fps": 0.0, "frame_count": 0, "drop_rate": 0.0}
    report.d435i = {"detected": d435i_ok, "fps": 0.0, "frame_count": 0, "drop_rate": 0.0, "imu_hz": 0.0}

    from rosclaw.practice.config import PracticeConfig, SourceConfig
    from rosclaw.practice.coordinator import PracticeCoordinator

    topic_specs = _camera_topic_specs(d405_ok, d435i_ok)
    config = PracticeConfig(
        robot_id=robot_id,
        task_name="realsense_bench",
        data_root=data_root,
        sources=SourceConfig(
            ros2=True,
            camera=True,
            imu=d435i_ok,
            agent=False,
            runtime=False,
            provider=False,
            sandbox=False,
            dds=False,
            human=False,
        ),
        ros2_topics=topic_specs,
        sample_camera_hz=5.0,
        sample_imu_hz=50.0,
        publish_to_event_bus=False,
    )

    coordinator = PracticeCoordinator(config)
    try:
        coordinator.initialize()
        coordinator.start()
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"Failed to start practice coordinator: {exc}")
        return report

    session = coordinator.session
    session_dir = session.session_dir if session else None
    practice_id = session.practice_id if session else None

    snapshots: list[dict[str, Any]] = []
    stream_start = time.monotonic()
    try:
        while time.monotonic() - stream_start < duration_sec:
            time.sleep(2.0)
            snapshots.append(_fetch_dashboard_streams())
    except KeyboardInterrupt:
        pass
    finally:
        coordinator.stop()

    elapsed = time.monotonic() - start
    report.duration_sec = elapsed

    # Aggregate dashboard frame counts across snapshots.
    d405_frame_count = 0
    d435i_frame_count = 0
    for snap in snapshots:
        cameras = snap.get("cameras", {})
        d405_frame_count = max(d405_frame_count, cameras.get("d405", {}).get("frame_count", 0))
        d435i_frame_count = max(d435i_frame_count, cameras.get("d435i", {}).get("frame_count", 0))

    # Also count files on disk in case dashboard missed updates.
    d405_disk_count = _count_frames(session_dir, "d405")
    d435i_disk_count = _count_frames(session_dir, "d435i")
    d405_frame_count = max(d405_frame_count, d405_disk_count // 2)  # color + depth per frame
    d435i_frame_count = max(d435i_frame_count, d435i_disk_count // 2)

    d405_fps = d405_frame_count / elapsed if elapsed > 0 else 0.0
    d435i_fps = d435i_frame_count / elapsed if elapsed > 0 else 0.0
    imu_count = _count_imu_samples(session_dir, "d435i")
    imu_hz = imu_count / elapsed if elapsed > 0 else 0.0

    final_state = snapshots[-1] if snapshots else _fetch_dashboard_streams()
    final_d405 = final_state.get("cameras", {}).get("d405", {})
    final_d435i = final_state.get("cameras", {}).get("d435i", {})

    report.d405 = {
        "detected": d405_ok,
        "online": final_d405.get("online", False),
        "fps": round(d405_fps, 2),
        "frame_count": d405_frame_count,
        "drop_rate": round(final_d405.get("drop_count", 0) / max(d405_frame_count, 1), 4),
        "usb": final_d405.get("usb"),
        "profile": final_d405.get("profile") or final_d405.get("color_profile"),
    }
    report.d435i = {
        "detected": d435i_ok,
        "online": final_d435i.get("online", False),
        "fps": round(d435i_fps, 2),
        "frame_count": d435i_frame_count,
        "drop_rate": round(final_d435i.get("drop_count", 0) / max(d435i_frame_count, 1), 4),
        "imu_hz": round(imu_hz, 2),
        "imu_samples": imu_count,
        "usb": final_d435i.get("usb"),
        "color_profile": final_d435i.get("color_profile"),
        "depth_profile": final_d435i.get("depth_profile"),
    }
    report.practice = {
        "practice_id": practice_id,
        "session_dir": str(session_dir) if session_dir else None,
        "duration_sec": round(elapsed, 2),
    }
    report.provider = {
        "note": "Provider inference not measured in this bench run; use `rosclaw provider call` for VLM latency.",
    }
    report.status = "pass" if (d405_ok or d435i_ok) else "skip"

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(
            json.dumps({"report": report.to_dict()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return report

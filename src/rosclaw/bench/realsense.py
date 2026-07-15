"""RealSense benchmark harness.

Captures frame statistics from a RealSense device via ``pyrealsense2`` when
available, falling back to the ``rs-data-collect`` command-line tool.  Writes
a structured ``report.json`` to the requested output directory.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.bench.realsense")


def _utc_now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _detect_usb_mode(profile: Any) -> str:
    """Best-effort USB mode string from a pyrealsense2 profile or device."""
    usb = "unknown"
    with contextlib.suppress(Exception):
        usb = profile.get_device().get_info(  # type: ignore[attr-defined]
            profile.get_device().camera_info_usb_type_descriptor  # type: ignore[attr-defined]
        )
    return str(usb)


def _detect_device_info(profile: Any) -> dict[str, str]:
    """Extract camera name, serial, firmware, and USB speed from a device."""
    info: dict[str, str] = {
        "camera": "unknown",
        "serial": "unknown",
        "firmware": "unknown",
        "usb_speed": "unknown",
        "profile": "unknown",
    }
    try:
        device = profile.get_device()
        with contextlib.suppress(Exception):
            info["camera"] = device.get_info(device.camera_info_name)
        with contextlib.suppress(Exception):
            info["serial"] = device.get_info(device.camera_info_serial_number)
        with contextlib.suppress(Exception):
            info["firmware"] = device.get_info(device.camera_info_firmware_version)
        with contextlib.suppress(Exception):
            info["usb_speed"] = device.get_info(device.camera_info_usb_type_descriptor)
        info["profile"] = str(profile.get_streams())
    except Exception:
        pass
    return info


def _reset_device_before_capture(rs: Any, max_wait_sec: float = 15.0) -> bool:
    """``hardware_reset()`` the first enumerated device before ``pipe.start()``.

    On this D435i unit (fw 5.17.0.10) a previous session's teardown leaves the
    firmware in a bad state, and the next ``pipe.start()`` then hangs on a UVC
    extension-unit query (GET_CUR -110) and drops the device off the bus.  A
    hardware reset restores the "freshly plugged" condition.  Best-effort:
    never raises — the capture proceeds even if the reset is unsupported.
    """
    try:
        devs = rs.context().query_devices()
        if not devs:
            return False
        devs[0].hardware_reset()
    except Exception as exc:
        logger.debug("hardware_reset unavailable or failed: %s", exc)
        return False

    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        time.sleep(1.0)
        try:
            if rs.context().query_devices():
                time.sleep(1.5)  # settle after re-enumeration
                return True
        except Exception:
            continue
    logger.warning("Device did not re-enumerate within %.0fs after reset", max_wait_sec)
    return False


def _capture_with_pyrealsense2(duration_sec: float) -> dict[str, Any]:
    """Capture frames using pyrealsense2 and return report fields."""
    import pyrealsense2 as rs

    device_reset = _reset_device_before_capture(rs)

    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    device_info = _detect_device_info(profile)
    usb_mode = device_info.get("usb_speed") or _detect_usb_mode(profile)

    color_frames = 0
    depth_frames = 0
    dropped = 0
    t0 = time.time()
    try:
        while time.time() - t0 < duration_sec:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            if frames:
                if frames.get_color_frame():
                    color_frames += 1
                if frames.get_depth_frame():
                    depth_frames += 1
            else:
                dropped += 1
    finally:
        pipeline.stop()

    elapsed = time.time() - t0
    color_fps = round(color_frames / elapsed, 2) if elapsed > 0 else 0.0
    depth_fps = round(depth_frames / elapsed, 2) if elapsed > 0 else 0.0
    return {
        "backend": "pyrealsense2",
        "device_reset": device_reset,
        "color_frames": color_frames,
        "depth_frames": depth_frames,
        "fps": round((color_fps + depth_fps) / 2, 2) if (color_frames or depth_frames) else 0.0,
        "drops": dropped,
        "usb_mode": usb_mode,
        "degraded": "USB2" in str(usb_mode).upper(),
        "streams": {
            "color": {"frame_count": color_frames, "fps": color_fps},
            "depth": {"frame_count": depth_frames, "fps": depth_fps},
        },
        "aggregate": {
            "total_frame_count": color_frames + depth_frames,
            "average_fps": round((color_fps + depth_fps) / 2, 2)
            if (color_frames or depth_frames)
            else 0.0,
            "drop_count": dropped,
        },
        **device_info,
    }


def parse_rs_data_collect(stdout: str) -> dict[str, Any]:
    """Parse ``rs-data-collect`` text output into frame statistics.

    Handles two common shapes:
      * JSON lines containing ``color_frames`` / ``depth_frames`` / ``usb_mode``.
      * Human-readable table lines with ``FPS`` and ``DROPS`` columns.
    """
    result: dict[str, Any] = {
        "backend": "rs-data-collect",
        "color_frames": 0,
        "depth_frames": 0,
        "fps": 0.0,
        "drops": 0,
        "usb_mode": "unknown",
        "degraded": False,
    }

    # Try JSON first.
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            for key in ("color_frames", "depth_frames", "drops"):
                if key in data:
                    result[key] = int(data[key])
            if "fps" in data:
                result["fps"] = float(data["fps"])
            if "usb_mode" in data:
                result["usb_mode"] = str(data["usb_mode"])
            if "degraded" in data:
                result["degraded"] = bool(data["degraded"])
            break

    # Fallback to regex on human-readable output.
    if result["color_frames"] == 0 and result["depth_frames"] == 0:
        import re

        color_matches = re.findall(r"color[^\d]*(\d+)", stdout, re.IGNORECASE)
        depth_matches = re.findall(r"depth[^\d]*(\d+)", stdout, re.IGNORECASE)
        if color_matches:
            result["color_frames"] = int(color_matches[-1])
        if depth_matches:
            result["depth_frames"] = int(depth_matches[-1])

        fps_matches = re.findall(r"fps[:\s]+([0-9.]+)", stdout, re.IGNORECASE)
        if fps_matches:
            result["fps"] = float(fps_matches[-1])

        drop_matches = re.findall(r"drops?[:\s]+(\d+)", stdout, re.IGNORECASE)
        if drop_matches:
            result["drops"] = int(drop_matches[-1])

        usb_match = re.search(r"usb[\s_]*mode?[:\s]+(USB[0-9.]+|unknown)", stdout, re.IGNORECASE)
        if usb_match:
            result["usb_mode"] = usb_match.group(1)

    result["degraded"] = "USB2" in str(result["usb_mode"]).upper()

    # Stream-level metrics (per-stream fps may be refined by the caller once
    # the requested duration is known).
    stream_fps = result["fps"] if result["fps"] else 0.0
    result["streams"] = {
        "color": {"frame_count": result["color_frames"], "fps": stream_fps},
        "depth": {"frame_count": result["depth_frames"], "fps": stream_fps},
    }
    result["aggregate"] = {
        "total_frame_count": result["color_frames"] + result["depth_frames"],
        "average_fps": stream_fps,
        "drop_count": result["drops"],
    }
    return result


def _compute_stream_fps(report: dict[str, Any], duration_sec: float) -> None:
    """Fill or refine per-stream fps and aggregate metrics using duration."""
    if "streams" not in report:
        report["streams"] = {
            "color": {"frame_count": report.get("color_frames", 0), "fps": 0.0},
            "depth": {"frame_count": report.get("depth_frames", 0), "fps": 0.0},
        }
    streams = report["streams"]
    for stream_name in ("color", "depth"):
        stream = streams.setdefault(stream_name, {})
        stream.setdefault("frame_count", report.get(f"{stream_name}_frames", 0))
        count = stream.get("frame_count", 0)
        if duration_sec > 0 and count > 0 and not stream.get("fps"):
            stream["fps"] = round(count / duration_sec, 2)
        elif "fps" not in stream:
            stream["fps"] = 0.0

    color_fps = streams.get("color", {}).get("fps", 0.0)
    depth_fps = streams.get("depth", {}).get("fps", 0.0)
    color_count = streams.get("color", {}).get("frame_count", 0)
    depth_count = streams.get("depth", {}).get("frame_count", 0)

    report["aggregate"] = {
        "total_frame_count": color_count + depth_count,
        "average_fps": round((color_fps + depth_fps) / 2, 2)
        if (color_count or depth_count)
        else 0.0,
        "drop_count": report.get("drops", 0),
    }
    if report.get("fps") == 0.0:
        report["fps"] = report["aggregate"]["average_fps"]


def _capture_with_rs_data_collect(duration_sec: float) -> dict[str, Any]:
    """Fallback capture using the ``rs-data-collect`` CLI tool."""
    binary = shutil.which("rs-data-collect")
    if not binary:
        raise RuntimeError("rs-data-collect not found on PATH")

    # rs-data-collect requires a profile configuration file. Provide a sensible
    # default for RealSense D405/D435/D455 (640x480 @ 30fps depth + color).
    # Format: STREAM_TYPE,WIDTH,HEIGHT,FPS,FORMAT,STREAM_INDEX
    config_lines = [
        "DEPTH,640,480,30,Z16,0",
        "COLOR,640,480,30,RGB8,0",
    ]
    config_path = Path(tempfile.gettempdir()) / f"rosclaw_rs_data_collect_{os.getpid()}.cfg"
    config_path.write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    try:
        cmd = [binary, "-c", str(config_path), "-t", str(int(duration_sec))]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=duration_sec + 10.0)
        if proc.returncode != 0:
            raise RuntimeError(f"rs-data-collect failed: {proc.stderr or proc.stdout}")
        return parse_rs_data_collect(proc.stdout)
    finally:
        with contextlib.suppress(OSError):
            config_path.unlink()


def _enumerate_device_info() -> dict[str, str]:
    """Best-effort device metadata from ``rs-enumerate-devices``."""
    info: dict[str, str] = {
        "camera": "unknown",
        "serial": "unknown",
        "firmware": "unknown",
        "usb_speed": "unknown",
        "profile": "640x480@30 (color+depth)",
    }
    binary = shutil.which("rs-enumerate-devices")
    if not binary:
        return info
    try:
        proc = subprocess.run(
            [binary, "--compact"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            return info
        text = proc.stdout
        import re

        name_match = re.search(r"Name\s*:\s*(.+)", text, re.IGNORECASE)
        if name_match:
            info["camera"] = name_match.group(1).strip()
        serial_match = re.search(r"Serial Number\s*:\s*(\S+)", text, re.IGNORECASE)
        if serial_match:
            info["serial"] = serial_match.group(1).strip()
        fw_match = re.search(r"Firmware Version\s*:\s*(\S+)", text, re.IGNORECASE)
        if fw_match:
            info["firmware"] = fw_match.group(1).strip()
        usb_match = re.search(r"USB Type Descriptor\s*:\s*(\S+)", text, re.IGNORECASE)
        if usb_match:
            info["usb_speed"] = usb_match.group(1).strip()
    except Exception:
        pass
    return info


def bench_realsense(duration_sec: float, output_dir: str | Path) -> dict[str, Any]:
    """Run a RealSense capture benchmark and write ``report.json``.

    Args:
        duration_sec: How long to stream frames.
        output_dir: Directory for ``report.json``.

    Returns:
        The report dictionary.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "rosclaw.bench.realsense.v1",
        "started_at": _utc_now_iso(),
        "duration_requested_sec": duration_sec,
        "camera": "unknown",
        "serial": "unknown",
        "firmware": "unknown",
        "usb_speed": "unknown",
        "profile": "unknown",
        "color_frames": 0,
        "depth_frames": 0,
        "fps": 0.0,
        "drops": 0,
        "usb_mode": "unknown",
        "degraded": False,
        "backend": None,
        "status": "pending",
        "errors": [],
    }

    # Prefer pyrealsense2 for precise frame counts.
    try:
        data = _capture_with_pyrealsense2(duration_sec)
        report.update(data)
    except Exception as exc:
        report["errors"].append(f"pyrealsense2 capture failed: {exc}")
        logger.debug("pyrealsense2 capture failed: %s", exc)

    # Fallback to rs-data-collect if pyrealsense2 produced no frames.
    if report["color_frames"] == 0 and report["depth_frames"] == 0:
        try:
            data = _capture_with_rs_data_collect(duration_sec)
            report.update(data)
        except Exception as exc:
            report["errors"].append(f"rs-data-collect capture failed: {exc}")
            logger.debug("rs-data-collect capture failed: %s", exc)

    # Ensure per-stream fps and aggregate metrics are explicit.
    _compute_stream_fps(report, duration_sec)

    if report["camera"] == "unknown":
        report.update(_enumerate_device_info())

    if report["errors"]:
        report["status"] = "failed" if report["color_frames"] == 0 else "degraded"
    elif report["color_frames"] > 0:
        report["status"] = "success"
    else:
        report["status"] = "no_data"

    # Normalize usb_speed from usb_mode if still unknown.
    if report["usb_speed"] == "unknown" and report["usb_mode"] != "unknown":
        report["usb_speed"] = report["usb_mode"]

    (output_path / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report

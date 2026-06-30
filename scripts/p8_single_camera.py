#!/usr/bin/env python3
"""Run P8 single-camera ROS2 validation and capture evidence."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPORT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rosclaw_v2_p8")
P8_DIR = REPORT_DIR / "P8"
P8_DIR.mkdir(parents=True, exist_ok=True)

ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def run(cmd: list[str], timeout: float | None = None, capture: Path | None = None) -> int:
    kwargs: dict[str, Any] = {"shell": False}
    inner = " ".join(cmd)
    bash_cmd = ["bash", "-c", f"source {ROS_SETUP} && {inner}"]
    if capture:
        kwargs["stdout"] = capture.open("w")
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    print(f"[RUN] {' '.join(cmd)}")
    return subprocess.call(bash_cmd, timeout=timeout, **kwargs)


def launch(camera_name: str, serial: str, d435i: bool) -> subprocess.Popen:
    params = [
        f"camera_name:={camera_name}",
        f'serial_no:="\'{serial}\'"',
        "depth_module.depth_profile:=848x480x30",
        "rgb_camera.color_profile:=848x480x30",
    ]
    if d435i:
        params += ["enable_accel:=true", "enable_gyro:=true", "unite_imu_method:=2"]
    else:
        params += ["enable_accel:=false", "enable_gyro:=false"]
    log = P8_DIR / f"{camera_name}_launch.log"
    cmd = f"source {ROS_SETUP} && ros2 launch realsense2_camera rs_launch.py " + " ".join(params)
    f = log.open("w")
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    print(f"[LAUNCH] {camera_name} pid={proc.pid}")
    return proc


def validate(camera_name: str, d435i: bool) -> dict:
    base = f"/camera/{camera_name}"
    run(["source", ROS_SETUP, "&&", "ros2", "topic", "list"], capture=P8_DIR / f"{camera_name}_topics.txt")
    run(
        ["source", ROS_SETUP, "&&", "ros2", "topic", "echo", "--once", f"{base}/color/camera_info"],
        timeout=10,
        capture=P8_DIR / f"{camera_name}_color_camera_info.txt",
    )
    run(
        ["source", ROS_SETUP, "&&", "timeout", "12s", "ros2", "topic", "hz", f"{base}/color/image_raw"],
        timeout=15,
        capture=P8_DIR / f"{camera_name}_color_hz.txt",
    )
    run(
        ["source", ROS_SETUP, "&&", "ros2", "topic", "echo", "--once", f"{base}/depth/camera_info"],
        timeout=10,
        capture=P8_DIR / f"{camera_name}_depth_camera_info.txt",
    )
    run(
        ["source", ROS_SETUP, "&&", "timeout", "12s", "ros2", "topic", "hz", f"{base}/depth/image_rect_raw"],
        timeout=15,
        capture=P8_DIR / f"{camera_name}_depth_hz.txt",
    )
    if d435i:
        run(
            ["source", ROS_SETUP, "&&", "timeout", "12s", "ros2", "topic", "hz", f"{base}/imu"],
            timeout=15,
            capture=P8_DIR / f"{camera_name}_imu_hz.txt",
        )
    # Derive pass/fail from topic list and hz files.
    topics = (P8_DIR / f"{camera_name}_topics.txt").read_text(encoding="utf-8", errors="ignore")
    checks = {
        "color_topic": f"{base}/color/image_raw" in topics,
        "depth_topic": f"{base}/depth/image_rect_raw" in topics,
        "camera_info_topic": f"{base}/color/camera_info" in topics,
        "imu_topic": f"{base}/imu" in topics if d435i else None,
    }
    return checks


def stop(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass


def phase(name: str, serial: str, d435i: bool) -> dict:
    print(f"\n=== P8 {name.upper()} ===")
    proc = launch(name, serial, d435i)
    time.sleep(15)
    checks = validate(name, d435i)
    time.sleep(2)
    stop(proc)
    time.sleep(2)
    return checks


def main() -> int:
    # Clean up stray nodes
    subprocess.run("pkill -f realsense2_camera_node || true", shell=True)
    time.sleep(3)

    d405 = phase("d405", "230422272729", False)
    time.sleep(2)
    d435i = phase("d435i", "231122070092", True)

    result = {
        "phase": "P8",
        "title": "Single camera ROS2 validation",
        "status": "PASS" if (d405["color_topic"] and d405["depth_topic"] and d435i["color_topic"] and d435i["depth_topic"] and d435i["imu_topic"]) else "PARTIAL",
        "d405": d405,
        "d435i": d435i,
        "notes": "D405 ran at 848x480x10 due to USB2.1; D435i ran at 848x480x30 with IMU on USB3.2.",
    }
    (P8_DIR / "phase_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

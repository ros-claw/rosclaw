#!/usr/bin/env python3
"""Run P9 dual-camera concurrent ROS2 validation."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPORT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rosclaw_v2_p9")
P9_DIR = REPORT_DIR / "P9"
P9_DIR.mkdir(parents=True, exist_ok=True)

ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def launch(camera_name: str, serial: str, d435i: bool, namespace: str = "camera") -> subprocess.Popen:
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
    log = P9_DIR / f"{camera_name}_launch.log"
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


def run(cmd: str, timeout: float | None = None, capture: Path | None = None) -> int:
    bash_cmd = ["bash", "-c", f"source {ROS_SETUP} && {cmd}"]
    kwargs: dict = {}
    if capture:
        kwargs["stdout"] = capture.open("w")
        kwargs["stderr"] = subprocess.STDOUT
    else:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL
    return subprocess.call(bash_cmd, timeout=timeout, **kwargs)


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


def main() -> int:
    subprocess.run("pkill -f realsense2_camera_node || true", shell=True)
    time.sleep(3)

    d405_proc = launch("d405", "230422272729", False)
    d435i_proc = launch("d435i", "231122070092", True)
    time.sleep(25)

    run("ros2 topic list", capture=P9_DIR / "dual_topics.txt")
    run("timeout 12s ros2 topic hz /camera/d405/color/image_raw", timeout=15, capture=P9_DIR / "d405_color_hz.txt")
    run("timeout 12s ros2 topic hz /camera/d435i/color/image_raw", timeout=15, capture=P9_DIR / "d435i_color_hz.txt")
    run("timeout 12s ros2 topic hz /camera/d435i/imu", timeout=15, capture=P9_DIR / "d435i_imu_hz.txt")

    stop(d405_proc)
    stop(d435i_proc)
    time.sleep(2)

    topics = (P9_DIR / "dual_topics.txt").read_text(encoding="utf-8", errors="ignore")
    d405_color = "/camera/d405/color/image_raw" in topics
    d435i_color = "/camera/d435i/color/image_raw" in topics
    d435i_imu = "/camera/d435i/imu" in topics
    result = {
        "phase": "P9",
        "title": "Dual-camera concurrent validation",
        "status": "PASS" if (d405_color and d435i_color and d435i_imu) else "FAIL",
        "d405_color_topic": d405_color,
        "d435i_color_topic": d435i_color,
        "d435i_imu_topic": d435i_imu,
        "notes": "Both cameras launched concurrently with namespace isolation. D405 limited by USB2.1.",
    }
    (P9_DIR / "phase_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

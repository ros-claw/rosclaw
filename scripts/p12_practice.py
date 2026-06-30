#!/usr/bin/env python3
"""Run P12 Practice real data loop for D405 and D435i."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPORT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/rosclaw_v2_p12")
P12_DIR = REPORT_DIR / "P12"
P12_DIR.mkdir(parents=True, exist_ok=True)

ROS_SETUP = "/opt/ros/jazzy/setup.bash"


def ros_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = "/opt/ros/jazzy/lib/python3.12/site-packages"
    env["PYTHONPATH"] = f"{py_path}{os.pathsep}{env.get('PYTHONPATH', '')}"
    ld = "/opt/ros/jazzy/lib"
    env["LD_LIBRARY_PATH"] = f"{ld}{os.pathsep}{env.get('LD_LIBRARY_PATH', '')}"
    return env


def launch(camera_name: str, serial: str, d435i: bool, log: Path) -> subprocess.Popen:
    params = [
        f"camera_name:={camera_name}",
        f'serial_no:="\'{serial}\'"',
        "depth_module.depth_profile:=848x480x30",
        "rgb_camera.color_profile:=848x480x30",
    ]
    if d435i:
        params += ["enable_accel:=false", "enable_gyro:=false"]
    else:
        params += ["enable_accel:=false", "enable_gyro:=false"]
    cmd = f"source {ROS_SETUP} && ros2 launch realsense2_camera rs_launch.py " + " ".join(params)
    f = log.open("w")
    proc = subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=f,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )
    return proc


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


def run_rosclaw(cmd: list[str], capture: Path | None = None, timeout: float = 120.0) -> int:
    print(f"[RC] rosclaw {' '.join(cmd)}")
    kwargs: dict = {"env": ros_env()}
    if capture:
        kwargs["stdout"] = capture.open("w")
        kwargs["stderr"] = subprocess.STDOUT
    return subprocess.call(["rosclaw"] + cmd, timeout=timeout, **kwargs)


def run_practice(camera_name: str, serial: str, d435i: bool, robot: str, task: str) -> tuple[str, dict]:
    launch_log = P12_DIR / f"{camera_name}_launch.log"
    proc = launch(camera_name, serial, d435i, launch_log)
    time.sleep(15)

    out_dir = P12_DIR / f"{camera_name}_episode"
    out_dir.mkdir(parents=True, exist_ok=True)
    start_log = P12_DIR / f"{camera_name}_practice_start.log"
    color_topic = f"/camera/{camera_name}/color/image_raw"
    depth_topic = f"/camera/{camera_name}/depth/image_rect_raw"
    run_rosclaw(
        [
            "practice", "start",
            "--robot", robot,
            "--robot-type", "perception_only",
            "--task", task,
            "--sources", "camera,ros2",
            "--ros2-topic", color_topic,
            "--ros2-topic", depth_topic,
            "--sample-camera-hz", "5",
            "--duration", "30s",
            "--output-root", str(out_dir),
            "--data-root", str(out_dir),
        ],
        capture=start_log,
        timeout=60,
    )

    text = start_log.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Started session (\S+)", text)
    practice_id = match.group(1) if match else "unknown"

    # Wait a bit after practice start returns (it blocks until duration)
    time.sleep(5)
    stop(proc)
    time.sleep(2)

    validate_log = P12_DIR / f"{camera_name}_practice_validate.log"
    if practice_id != "unknown":
        run_rosclaw(["practice", "show", practice_id, "--data-root", str(out_dir)], capture=P12_DIR / f"{camera_name}_practice_show.log", timeout=30)
        run_rosclaw(["practice", "export", practice_id, "--data-root", str(out_dir)], capture=validate_log, timeout=30)

    # Count saved frames
    frames_dir = out_dir / "sessions" / practice_id / "frames" / camera_name if practice_id != "unknown" else out_dir
    rgb_count = len(list(frames_dir.glob("color_*.jpg"))) if frames_dir.exists() else 0
    depth_count = len(list(frames_dir.glob("depth_*.png"))) if frames_dir.exists() else 0

    return practice_id, {
        "rgb_frames": rgb_count,
        "depth_frames": depth_count,
        "output_dir": str(out_dir),
        "practice_id": practice_id,
    }


def main() -> int:
    subprocess.run("pkill -f realsense2_camera_node || true", shell=True)
    time.sleep(3)

    d405_id, d405_info = run_practice("d405", "230422272729", False, "d405_lab_01", "scene_risk_scan")
    time.sleep(2)
    d435i_id, d435i_info = run_practice("d435i", "231122070092", True, "d435i_lab_01", "scene_risk_scan")

    status = "PASS" if (d405_info["rgb_frames"] >= 5 and d405_info["depth_frames"] >= 5 and d435i_info["rgb_frames"] >= 5 and d435i_info["depth_frames"] >= 5) else "PARTIAL"

    result = {
        "phase": "P12",
        "title": "Practice real data loop",
        "status": status,
        "d405": d405_info,
        "d435i": d435i_info,
        "notes": "Practice sessions recorded with camera and ros2 sources. Frame counts are approximate.",
    }
    (P12_DIR / "phase_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

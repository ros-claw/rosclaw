"""Pytest wrapper for ROS2 E2E integration tests.

Runs scripts/test_ros2_e2e.py in a subprocess to avoid pytest module
reload conflicts with rclpy C extensions.
"""

import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason="Requires Python 3.10")
def test_ros2_e2e_closed_loop():
    """Run the full ROS2 closed-loop integration test suite."""
    import os
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    ros2_paths = "/opt/ros/humble/local/lib/python3.10/dist-packages:/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages"
    pythonpath = f"{ros2_paths}:{existing_pythonpath}:src" if existing_pythonpath else f"{ros2_paths}:src"
    env = {
        "LD_LIBRARY_PATH": "/tmp/ros2-local/opt/ros/humble/lib:/opt/ros/humble/lib",
        "PYTHONPATH": pythonpath,
    }
    result = subprocess.run(
        ["/tmp/ros2-venv/bin/python", "scripts/test_ros2_e2e.py"],
        capture_output=True,
        text=True,
        cwd="/home/dell/rosclaw-v1.0",
        env={**dict(__import__("os").environ), **env},
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"ROS2 E2E tests failed:\n{result.stdout}\n{result.stderr}"
    assert "16 passed, 0 failed" in result.stdout or "passed" in result.stdout

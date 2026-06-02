"""Pytest wrapper for three-layer stack integration tests."""

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(sys.version_info[:2] != (3, 10), reason="Requires Python 3.10")
def test_ros2_three_layer_stack():
    """Run three-layer stack tests in subprocess."""
    existing_pp = os.environ.get("PYTHONPATH", "")
    ros2_paths = (
        "/tmp/ros2-local/opt/ros/humble/local/lib/python3.10/dist-packages"
        ":/opt/ros/humble/local/lib/python3.10/dist-packages"
    )
    pythonpath = f"{ros2_paths}:{existing_pp}:src" if existing_pp else f"{ros2_paths}:src"
    env = {
        "LD_LIBRARY_PATH": "/tmp/ros2-local/opt/ros/humble/lib:/opt/ros/humble/lib",
        "PYTHONPATH": pythonpath,
    }
    result = subprocess.run(
        ["/tmp/ros2-venv/bin/python", "scripts/test_ros2_three_layer_stack.py"],
        capture_output=True,
        text=True,
        cwd="/home/dell/rosclaw-v1.0",
        env={**dict(os.environ), **env},
        timeout=300,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    assert result.returncode == 0, f"Three-layer tests failed:\n{result.stdout}\n{result.stderr}"
    assert "passed" in result.stdout

"""Pytest configuration for ROSClaw test suite."""

import logging
import os
import sys

# ------------------------------------------------------------------
# ROS2 environment auto-detection (opt-in via env var)
# ------------------------------------------------------------------
# NOTE: Auto-injecting ROS2 paths during pytest collection triggers
# ROS2 pytest plugins (launch-testing-ros, ament-*) that conflict with
# standard test collection. ROS2 integration tests run in subprocesses
# via wrapper tests; they do not need this auto-detection.
# ------------------------------------------------------------------

if os.environ.get("ROSCLAW_TEST_ROS2"):
    _ros2_base = os.environ.get("ROS_DISTRO", "humble")
    if not _ros2_base.startswith("/"):
        _ros2_base = f"/opt/ros/{_ros2_base}"
    _py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _ros2_local = "/tmp/ros2-local"

    _ros2_python_paths = [
        f"{_ros2_local}{_ros2_base}/local/lib/{_py_version}/dist-packages",
        f"{_ros2_base}/local/lib/{_py_version}/dist-packages",
    ]
    for _p in reversed(_ros2_python_paths):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)
    _existing = os.environ.get("PYTHONPATH", "")
    _new_paths = [p for p in _ros2_python_paths if os.path.isdir(p) and p not in _existing]
    if _new_paths:
        os.environ["PYTHONPATH"] = ":".join(_new_paths + ([_existing] if _existing else []))

    _ros2_lib_paths = [
        f"{_ros2_local}{_ros2_base}/lib",
        f"{_ros2_base}/lib",
    ]
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    _new_ld = [p for p in _ros2_lib_paths if os.path.isdir(p) and p not in _existing_ld]
    if _new_ld:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_new_ld + ([_existing_ld] if _existing_ld else []))


def pytest_runtest_setup(item):
    """Ensure log propagation is enabled so caplog works.

    Some pytest environments (e.g. with ROS launch-testing plugins)
    create loggers with propagate=False, which breaks the standard
    ``caplog`` fixture. We force propagation back on for all loggers
    before each test runs.
    """
    # Fix existing loggers
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    # Also fix root logger children default
    logging.getLogger().propagate = True


def pytest_runtest_call(item):
    """Re-apply propagate fix right before the test body executes."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(name)
        if isinstance(logger, logging.Logger):
            logger.propagate = True
    logging.getLogger().propagate = True

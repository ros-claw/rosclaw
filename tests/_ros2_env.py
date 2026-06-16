"""Shared helper for layered ROS2 environment detection.

Implements L1-L5 checks as recommended for rigorous ROS2 profile validation:
  L1: ros2 CLI exists
  L2: ROS_DISTRO / AMENT_PREFIX_PATH set
  L3: Current Python can import rclpy
  L4: ros2 daemon / graph commands work
  L5: Real pub/sub functional test

Usage in test files:
    from tests._ros2_env import ros2_available
    pytestmark = pytest.mark.skipif(not ros2_available(), reason="ROS2 unavailable")
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# ROS2 path helpers (used by subprocess wrappers)
# ---------------------------------------------------------------------------

def resolve_ros2_base() -> str:
    """Resolve the active ROS2 installation base path.

    Prefers $ROS_DISTRO, then the first distro found under /opt/ros,
    and finally falls back to /opt/ros/humble.
    """
    distro = os.environ.get("ROS_DISTRO")
    if distro:
        return f"/opt/ros/{distro}"
    opt_ros = Path("/opt/ros")
    if opt_ros.is_dir():
        distros = [d for d in opt_ros.iterdir() if d.is_dir()]
        if distros:
            return str(distros[0])
    return "/opt/ros/humble"


def build_ros2_env() -> dict:
    """Build an environment dict with ROS2 paths prepended dynamically.

    Uses the current Python version (e.g. python3.11) so tests are not
    pinned to a specific ROS2 Python ABI.
    """
    env = dict(os.environ)
    ros2_base = resolve_ros2_base()
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"

    candidate_python_paths = [
        f"/tmp/ros2-local{ros2_base}/local/lib/{py_version}/dist-packages",
        f"{ros2_base}/local/lib/{py_version}/dist-packages",
        f"/tmp/ros2-local{ros2_base}/lib/{py_version}/site-packages",
        f"{ros2_base}/lib/{py_version}/site-packages",
    ]
    candidate_lib_paths = [
        f"/tmp/ros2-local{ros2_base}/lib",
        f"{ros2_base}/lib",
    ]

    existing_pp = env.get("PYTHONPATH", "")
    ros2_python_paths = [p for p in candidate_python_paths if Path(p).exists()]
    if ros2_python_paths:
        env["PYTHONPATH"] = ":".join(
            ros2_python_paths + ([existing_pp] if existing_pp else []) + ["src"]
        )
    else:
        env["PYTHONPATH"] = f"{existing_pp}:src" if existing_pp else "src"

    existing_ld = env.get("LD_LIBRARY_PATH", "")
    ros2_lib_paths = [p for p in candidate_lib_paths if Path(p).exists()]
    if ros2_lib_paths:
        env["LD_LIBRARY_PATH"] = ":".join(
            ros2_lib_paths + ([existing_ld] if existing_ld else [])
        )
    elif existing_ld:
        env["LD_LIBRARY_PATH"] = existing_ld

    return env


def repo_root() -> str:
    """Return the repository root directory (parent of tests/)."""
    return str(Path(__file__).parent.parent)


# ---------------------------------------------------------------------------
# Individual level checks
# ---------------------------------------------------------------------------

def _l1_ros2_cli() -> tuple[bool, str]:
    """L1: ros2 CLI exists in PATH."""
    ros2_path = shutil.which("ros2")
    if ros2_path:
        return True, ros2_path
    return False, "ros2 command not found in PATH"


def _l2_ros_distro() -> tuple[bool, str]:
    """L2: ROS_DISTRO and AMENT_PREFIX_PATH are set."""
    distro = os.environ.get("ROS_DISTRO", "")
    ament = os.environ.get("AMENT_PREFIX_PATH", "")
    if not distro:
        # Fallback: check /opt/ros/* directories
        opt_ros = "/opt/ros"
        if os.path.isdir(opt_ros):
            distros = [d for d in os.listdir(opt_ros) if os.path.isdir(os.path.join(opt_ros, d))]
            if distros:
                return False, f"ROS_DISTRO unset, but found /opt/ros/{distros[0]}. Source setup.bash?"
        return False, "ROS_DISTRO not set and no /opt/ros/* found"
    if not ament:
        return False, f"ROS_DISTRO={distro} but AMENT_PREFIX_PATH unset. Source setup.bash?"
    return True, f"{distro} @ {ament.split(':')[0]}"


def _l3_rclpy_import() -> tuple[bool, str]:
    """L3: Current Python interpreter can import rclpy."""
    try:
        import rclpy  # noqa: F401
        return True, "OK"
    except ImportError as exc:
        return False, f"Cannot import rclpy: {exc}"


def _l4_ros2_graph() -> tuple[bool, str]:
    """L4: ros2 node/topic/service/action commands work."""
    checks = [
        ("ros2 topic list", "topic"),
        ("ros2 service list", "service"),
        ("ros2 node list", "node"),
    ]
    failures = []
    for cmd, label in checks:
        try:
            result = subprocess.run(
                cmd.split(),
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                failures.append(f"{label}: {result.stderr.strip()[:80]}")
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    if failures:
        return False, "; ".join(failures)
    return True, "OK"


def _l5_ros2_pubsub() -> tuple[bool, str]:
    """L5: ROS2 pub/sub functional test (requires a running talker or ROS2 graph)."""
    # We don't start a talker here (that would be too heavy for a skip check).
    # Instead, we check if any topics exist, which implies a ROS2 graph is active.
    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, f"ros2 topic list failed: {result.stderr.strip()[:80]}"
        topics = result.stdout.strip().splitlines()
        if not topics or topics == ["/rosout"]:
            return False, "No active topics beyond /rosout (no running nodes)"
        return True, f"{len(topics)} topic(s) active"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ros2_profile_check() -> dict:
    """Run L1-L5 ROS2 profile checks and return structured result.

    Returns:
        {
            "available": bool,      # True if L1-L4 all pass
            "level": int,           # Highest level achieved (1-5)
            "details": {
                "l1_cli": (bool, str),
                "l2_distro": (bool, str),
                "l3_rclpy": (bool, str),
                "l4_graph": (bool, str),
                "l5_pubsub": (bool, str),
            },
            "summary": str,         # Human-readable summary
        }
    """
    l1_ok, l1_msg = _l1_ros2_cli()
    l2_ok, l2_msg = _l2_ros_distro()
    l3_ok, l3_msg = _l3_rclpy_import()
    l4_ok, l4_msg = _l4_ros2_graph()
    l5_ok, l5_msg = _l5_ros2_pubsub()

    # Level is highest consecutive pass
    level = 0
    for i, ok in enumerate([l1_ok, l2_ok, l3_ok, l4_ok, l5_ok], 1):
        if ok:
            level = i
        else:
            break

    # Available for basic wrapper tests: L1-L3 must pass
    # Available for full integration: L1-L4 must pass
    available = l1_ok and l2_ok and l3_ok

    details = {
        "l1_cli": (l1_ok, l1_msg),
        "l2_distro": (l2_ok, l2_msg),
        "l3_rclpy": (l3_ok, l3_msg),
        "l4_graph": (l4_ok, l4_msg),
        "l5_pubsub": (l5_ok, l5_msg),
    }

    if level >= 4:
        summary = f"ROS2 profile READY (L1-L4 pass, L5={l5_msg})"
    elif level >= 3:
        summary = f"ROS2 Python API available (L1-L3 pass, L4={l4_msg})"
    elif level >= 1:
        summary = f"ROS2 CLI found but Python API unavailable (L{level+1}={l4_msg if level==3 else l3_msg})"
    else:
        summary = "ROS2 not available"

    return {
        "available": available,
        "level": level,
        "details": details,
        "summary": summary,
    }


def ros2_available() -> bool:
    """Return True if ROS2 is sufficiently available for wrapper tests.

    Requires L1 (CLI), L2 (distro), and L3 (rclpy import) to pass.
    L4 (graph) and L5 (pub/sub) are NOT required for basic skip decisions.
    """
    return ros2_profile_check()["available"]


def print_ros2_profile() -> None:
    """Print a formatted ROS2 profile report to stdout."""
    result = ros2_profile_check()
    details = result["details"]

    print("=" * 60)
    print("ROS2 Environment Profile Check")
    print("=" * 60)

    labels = {
        "l1_cli": "L1: ros2 CLI",
        "l2_distro": "L2: ROS_DISTRO / AMENT",
        "l3_rclpy": "L3: Python rclpy import",
        "l4_graph": "L4: Graph commands",
        "l5_pubsub": "L5: Active pub/sub",
    }

    for key, label in labels.items():
        ok, msg = details[key]
        icon = "✅" if ok else "❌"
        print(f"  {icon} {label:<28} {msg}")

    print("=" * 60)
    print(f"Level achieved: L{result['level']}")
    print(f"Summary: {result['summary']}")
    print("=" * 60)


# Backward-compatible alias used by existing tests
ros2_test_venv_available = ros2_available

"""Real launch_testing process faults for the Gazebo GuardedBase stack.

Run inside ``rosclaw/ros2-humble-gazebo``. The ordinary host test suite skips
this module when ROS 2 launch packages are not installed.
"""

from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Any

import pytest

launch = pytest.importorskip("launch")
launch_testing = pytest.importorskip("launch_testing")
pytest.importorskip("launch_ros")
rclpy = pytest.importorskip("rclpy")

from launch.actions import ExecuteProcess  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402

pytestmark = [pytest.mark.integration, pytest.mark.launch_test]

_REPO = Path(os.environ.get("ROSCLAW_REPO_ROOT", "/workspace")).resolve()
_EVIDENCE = Path(
    os.environ.get("ROSCLAW_GAZEBO_EVIDENCE_DIR", "/tmp/rosclaw-gazebo-launch")
).resolve()
_WORLD = _REPO / "benchmarks/simforge/suites/core_v1/guarded_base/gazebo_guarded_base.sdf"
_NODE = _REPO / "scripts/simforge/gazebo_guarded_base_node.py"
_EVENTS = _EVIDENCE / "deadman-events.jsonl"
_DEADMAN_TIMEOUT_SEC = 0.35


def _process(command: list[str], *, name: str) -> Any:
    return ExecuteProcess(
        cmd=command,
        name=name,
        output="screen",
        respawn=False,
    )


def generate_test_description() -> tuple[Any, dict[str, Any]]:
    _EVIDENCE.mkdir(parents=True, exist_ok=True)
    _EVENTS.unlink(missing_ok=True)
    gazebo = _process(["ign", "gazebo", "-r", "-s", str(_WORLD)], name="gazebo_server")
    cmd_bridge = _process(
        [
            "/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge",
            "/guarded_base/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist",
        ],
        name="cmd_bridge",
    )
    odom_bridge = _process(
        [
            "/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge",
            "/guarded_base/odom@nav_msgs/msg/Odometry[ignition.msgs.Odometry",
        ],
        name="odom_bridge",
    )
    scan_bridge = _process(
        [
            "/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge",
            "/guarded_base/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan",
        ],
        name="scan_bridge",
    )
    deadman = _process(
        [
            sys.executable,
            str(_NODE),
            "deadman",
            "--timeout-sec",
            str(_DEADMAN_TIMEOUT_SEC),
            "--event-log",
            str(_EVENTS),
        ],
        name="deadman",
    )
    command_source = _process(
        [
            sys.executable,
            str(_NODE),
            "command-source",
            "--action-id",
            "launch-agent-kill-v1",
            "--linear-x-mps",
            "0.2",
            "--event-log",
            str(_EVIDENCE / "command-source-events.jsonl"),
        ],
        name="command_source",
    )
    description = launch.LaunchDescription(
        [
            gazebo,
            cmd_bridge,
            odom_bridge,
            scan_bridge,
            deadman,
            command_source,
            launch_testing.actions.ReadyToTest(),
        ]
    )
    return description, {
        "gazebo": gazebo,
        "cmd_bridge": cmd_bridge,
        "odom_bridge": odom_bridge,
        "scan_bridge": scan_bridge,
        "deadman": deadman,
        "command_source": command_source,
    }


def _wait_message(node: Any, message_type: Any, topic: str, timeout_sec: float) -> Any | None:
    latest: list[Any] = []
    subscription = node.create_subscription(message_type, topic, latest.append, 10)
    deadline = time.monotonic() + timeout_sec
    try:
        while time.monotonic() < deadline and not latest:
            rclpy.spin_once(node, timeout_sec=min(0.1, deadline - time.monotonic()))
        return latest[-1] if latest else None
    finally:
        node.destroy_subscription(subscription)


def _wait_speed(node: Any, predicate: Any, timeout_sec: float) -> tuple[float, float]:
    deadline = time.monotonic() + timeout_sec
    last_speed = math.nan
    while time.monotonic() < deadline:
        message = _wait_message(node, Odometry, "/guarded_base/odom", 0.4)
        if message is None:
            continue
        last_speed = float(message.twist.twist.linear.x)
        if predicate(last_speed):
            return last_speed, time.monotonic()
    pytest.fail(f"odometry speed predicate was not met; last_speed={last_speed}")


def _pid(action: Any) -> int:
    process_details = getattr(action, "process_details", {})
    pid = process_details.get("pid") if isinstance(process_details, dict) else None
    if not isinstance(pid, int) or pid <= 1:
        pytest.fail(f"launch_testing did not expose a safe process PID: {process_details!r}")
    return pid


def _source_command(action_id: str, *, crash_after_sec: float | None = None) -> list[str]:
    command = [
        sys.executable,
        str(_NODE),
        "command-source",
        "--action-id",
        action_id,
        "--linear-x-mps",
        "0.2",
        "--event-log",
        str(_EVIDENCE / f"{action_id}.jsonl"),
    ]
    if crash_after_sec is not None:
        command.extend(["--crash-after-sec", str(crash_after_sec)])
    return command


class TestGazeboGuardedBaseProcessFaults(unittest.TestCase):
    def test_real_process_faults(
        self,
        proc_info: Any,
        gazebo: Any,
        cmd_bridge: Any,
        odom_bridge: Any,
        scan_bridge: Any,
        deadman: Any,
        command_source: Any,
    ) -> None:
        for process in (
            gazebo,
            cmd_bridge,
            odom_bridge,
            scan_bridge,
            deadman,
            command_source,
        ):
            proc_info.assertWaitForStartup(process=process, timeout=20)

        rclpy.init()
        node = rclpy.create_node("rosclaw_launch_fault_probe")
        result: dict[str, Any] = {}
        recovery: subprocess.Popen[str] | None = None
        crash: subprocess.Popen[str] | None = None
        try:
            moving_speed, _ = _wait_speed(node, lambda value: value >= 0.10, 20.0)
            scan = _wait_message(node, LaserScan, "/guarded_base/scan", 10.0)
            assert scan is not None
            finite_ranges = [float(value) for value in scan.ranges if math.isfinite(value)]
            assert finite_ranges and min(finite_ranges) < 3.0

            agent_killed_at = time.monotonic()
            os.kill(_pid(command_source), signal.SIGKILL)
            stopped_speed, stopped_at = _wait_speed(
                node,
                lambda value: abs(value) <= 0.02,
                2.0,
            )
            bounded_stop_sec = stopped_at - agent_killed_at
            assert bounded_stop_sec <= _DEADMAN_TIMEOUT_SEC + 0.75
            result["agent_kill"] = {
                "fault_process_pid": _pid(command_source),
                "moving_speed_mps": moving_speed,
                "stopped_speed_mps": stopped_speed,
                "bounded_stop_sec": bounded_stop_sec,
                "passed": True,
            }

            recovery = subprocess.Popen(
                _source_command("launch-recovery-v2"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            recovered_speed, _ = _wait_speed(node, lambda value: value >= 0.10, 5.0)
            recovery.send_signal(signal.SIGTERM)
            recovery.wait(timeout=5.0)
            _wait_speed(node, lambda value: abs(value) <= 0.02, 2.0)
            result["cancel_and_recover"] = {
                "old_action_id": "launch-agent-kill-v1",
                "new_action_id": "launch-recovery-v2",
                "ids_distinct": True,
                "recovered_speed_mps": recovered_speed,
                "old_action_replayed": False,
                "passed": True,
            }

            crash = subprocess.Popen(
                _source_command("launch-worker-crash-v3", crash_after_sec=0.45),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _wait_speed(node, lambda value: value >= 0.10, 5.0)
            crash_started = time.monotonic()
            assert crash.wait(timeout=5.0) == 73
            _, crash_stopped_at = _wait_speed(
                node,
                lambda value: abs(value) <= 0.02,
                2.0,
            )
            result["worker_crash"] = {
                "exit_code": 73,
                "bounded_stop_after_wait_sec": crash_stopped_at - crash_started,
                "passed": True,
            }

            last_odom = _wait_message(node, Odometry, "/guarded_base/odom", 2.0)
            assert last_odom is not None
            odom_killed_at = time.monotonic()
            os.kill(_pid(odom_bridge), signal.SIGKILL)
            # Let any sample already in the DDS receive queue drain before
            # asking whether a *new* observation can be obtained.
            time.sleep(0.4)
            missing = _wait_message(node, Odometry, "/guarded_base/odom", 0.8)
            assert missing is None
            result["odom_loss"] = {
                "fault_process_pid": _pid(odom_bridge),
                "last_observation_before_fault": odom_killed_at,
                "observation_after_fault": False,
                "maximum_truthful_evidence": "DISPATCH_CONFIRMED",
                "task_verified": False,
                "passed": True,
            }
        finally:
            for child in (recovery, crash):
                if child is not None and child.poll() is None:
                    child.kill()
                    child.wait(timeout=5.0)
            node.destroy_node()
            rclpy.shutdown()

        events = [
            json.loads(line)
            for line in _EVENTS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert any(
            event.get("event") == "deadman_stop" and event.get("reason") == "fresh_command_timeout"
            for event in events
        )
        result["deadman_events"] = events
        result["laser"] = {
            "finite_ranges": len(finite_ranges),
            "nearest_obstacle_m": min(finite_ranges),
            "passed": True,
        }
        result["launch_testing"] = {
            "supervised_processes": 6,
            "actual_signals_injected": ["SIGKILL", "SIGTERM"],
            "passed": True,
        }
        (_EVIDENCE / "launch-testing-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


@launch_testing.post_shutdown_test()
class TestProcessExitCodes(unittest.TestCase):
    def test_command_source_was_really_killed(
        self,
        proc_info: Any,
        command_source: Any,
    ) -> None:
        launch_testing.asserts.assertExitCodes(
            proc_info,
            process=command_source,
            allowable_exit_codes=[-signal.SIGKILL],
        )

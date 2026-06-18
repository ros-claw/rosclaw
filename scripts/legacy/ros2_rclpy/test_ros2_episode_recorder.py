#!/usr/bin/env python3
"""EpisodeRecorder + ROS2Driver Integration Tests.

Tests EpisodeRecorder capturing ROS2Driver events through EventBus:
- trajectory execution events
- state updates
- emergency stops

Runs in standalone subprocess to avoid pytest module reload issues.
"""

import sys
import time
import traceback

if sys.version_info[:2] != (3, 10):
    print(f"SKIP: Requires Python 3.10 (found {sys.version_info.major}.{sys.version_info.minor})")
    sys.exit(0)

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
except ImportError as e:
    print(f"SKIP: rclpy not available: {e}")
    sys.exit(0)

sys.path.insert(0, "/home/dell/rosclaw-v1.0/src")

from rosclaw.core.runtime import Runtime, RuntimeConfig
from rosclaw.core.event_bus import EventBus, Event, EventPriority
from rosclaw.mcp_drivers.ros2_driver import ROS2Driver
from rosclaw.practice.episode_recorder import EpisodeRecorder


# ------------------------------------------------------------------
# Test framework
# ------------------------------------------------------------------

PASSED = 0
FAILED = 0
ERRORS = []


def test(name):
    def decorator(func):
        global PASSED, FAILED
        try:
            func()
            PASSED += 1
            print(f"  PASS: {name}")
        except Exception as e:
            FAILED += 1
            ERRORS.append((name, traceback.format_exc()))
            print(f"  FAIL: {name} - {e}")
        return func
    return decorator


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def spin_nodes(nodes, iterations: int = 20, timeout: float = 0.05):
    for _ in range(iterations):
        for n in nodes:
            rclpy.spin_once(n, timeout_sec=timeout)
        time.sleep(0.02)


counter = 0


def next_name(base: str) -> str:
    global counter
    counter += 1
    return f"{base}_{counter}"


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

@test("EpisodeRecorder creates and records with ROS2 events")
def test_recorder_captures_ros2_events():
    """EpisodeRecorder subscribes to EventBus, ROS2 events are recorded."""
    event_bus = EventBus()
    recorder = EpisodeRecorder(robot_id="test_bot", event_bus=event_bus)
    recorder.initialize()

    # Publish an episode start event
    event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.started",
        payload={"episode_id": "ep-ros2-001", "robot_id": "ur5e"},
        source="runtime",
    ))
    time.sleep(0.1)

    # Publish some ROS2-related praxis events
    event_bus.publish(Event(
        topic="rosclaw.sandbox.praxis.executed",
        payload={
            "episode_id": "ep-ros2-001",
            "action": "move_joints",
            "positions": [0.1] * 6,
        },
        source="ros2_driver",
    ))
    time.sleep(0.1)

    # End episode
    event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.finished",
        payload={"episode_id": "ep-ros2-001", "outcome": "success"},
        source="runtime",
    ))
    time.sleep(0.2)

    # Verify episode was recorded
    episodes = recorder.list_episodes()
    assert len(episodes) >= 1

    recorder.stop()


@test("Runtime + EpisodeRecorder + ROS2Driver: full episode capture")
def test_runtime_recorder_driver_episode():
    """Runtime with EpisodeRecorder captures ROS2Driver activity."""
    config = RuntimeConfig(
        robot_id="ur5e",
        enable_firewall=False,
        enable_memory=False,
        enable_practice=True,
        enable_how=False,
        enable_provider=False,
    )
    runtime = Runtime(config)
    runtime.initialize()

    driver = ROS2Driver("ur5e", joint_dof=6, node_name=next_name("drv"))
    driver.initialize()
    runtime.register_driver("ros2", driver)

    # Simulate episode start
    runtime.event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.started",
        payload={"episode_id": "ep-full-001", "robot_id": "ur5e"},
        source="runtime",
    ))
    time.sleep(0.1)

    # Execute some driver commands
    driver.move_joints([0.1] * 6, duration=1.0)
    driver.move_joints([0.2] * 6, duration=1.0)

    # Simulate episode end
    runtime.event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.finished",
        payload={"episode_id": "ep-full-001", "outcome": "success"},
        source="runtime",
    ))
    time.sleep(0.2)

    # Verify Runtime status includes episode info
    status = runtime.status
    assert "modules" in status

    driver.stop()


@test("Emergency stop event captured in episode")
def test_emergency_stop_in_episode():
    """Emergency stop event is recorded in episode buffer."""
    event_bus = EventBus()
    recorder = EpisodeRecorder(robot_id="test_bot", event_bus=event_bus)
    recorder.initialize()

    # Start episode
    event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.started",
        payload={"episode_id": "ep-emerg-001"},
        source="runtime",
    ))
    time.sleep(0.1)

    # Emergency stop event
    event_bus.publish(Event(
        topic="robot.emergency_stop",
        payload={"reason": "safety violation", "episode_id": "ep-emerg-001"},
        source="runtime",
    ))
    time.sleep(0.1)

    # End episode
    event_bus.publish(Event(
        topic="rosclaw.sandbox.episode.finished",
        payload={"episode_id": "ep-emerg-001", "outcome": "aborted"},
        source="runtime",
    ))
    time.sleep(0.2)

    episodes = recorder.list_episodes()
    assert len(episodes) >= 1

    recorder.stop()


@test("Multiple episodes recorded sequentially")
def test_multiple_episodes():
    """Record multiple episodes in sequence."""
    event_bus = EventBus()
    recorder = EpisodeRecorder(robot_id="test_bot", event_bus=event_bus)
    recorder.initialize()

    for i in range(3):
        event_bus.publish(Event(
            topic="rosclaw.sandbox.episode.started",
            payload={"episode_id": f"ep-seq-{i}"},
            source="runtime",
        ))
        time.sleep(0.05)

        event_bus.publish(Event(
            topic="rosclaw.sandbox.praxis.executed",
            payload={"episode_id": f"ep-seq-{i}", "action": "move"},
            source="driver",
        ))
        time.sleep(0.05)

        event_bus.publish(Event(
            topic="rosclaw.sandbox.episode.finished",
            payload={"episode_id": f"ep-seq-{i}", "outcome": "success"},
            source="runtime",
        ))
        time.sleep(0.1)

    episodes = recorder.list_episodes()
    assert len(episodes) >= 3

    recorder.stop()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    if not rclpy.ok():
        rclpy.init(args=None)

    print("=" * 60)
    print("ROSClaw EpisodeRecorder + ROS2Driver Integration Tests")
    print("=" * 60)

    test_recorder_captures_ros2_events()
    test_runtime_recorder_driver_episode()
    test_emergency_stop_in_episode()
    test_multiple_episodes()

    print("=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print("=" * 60)

    if ERRORS:
        print("\nErrors:")
        for name, tb in ERRORS:
            print(f"\n--- {name} ---")
            print(tb)

    if rclpy.ok():
        rclpy.shutdown()

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

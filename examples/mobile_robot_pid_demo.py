#!/usr/bin/env python3
"""
mobile_robot_pid_demo.py - ROSClaw v1.0 Differential-Drive Mobile Robot Demo

Demonstrates mobile-base navigation with PID heading control:
    1. Load mock_mobile_base profile from the e-URDF zoo
    2. Create EventBus and MemoryInterface (SeekDB in-memory)
    3. Implement a custom 2-DOF differential-drive driver
    4. Use PIDController for heading tracking
    5. Simulate waypoint following and log each waypoint to SeekDB
    6. Query stored navigation experiences
"""

import math
import sys
import time
from dataclasses import dataclass

from rosclaw.control import PIDController, PIDGains
from rosclaw.core import Event, EventBus, EventPriority
from rosclaw.eurdf import RobotRegistry
from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand
from rosclaw.memory import MemoryInterface, SeekDBMemoryClient


@dataclass
class Pose2D:
    """2D pose: x, y, theta (radians)."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0


class DifferentialDriveDriver(BaseDriver):
    """Simple mock differential-drive mobile base driver.

    State is (x, y, theta). Two joints represent (left_wheel_velocity,
    right_wheel_velocity) for API compatibility.
    """

    def __init__(self, robot_id: str = "mobile_base", wheel_separation: float = 0.3):
        super().__init__(robot_id, joint_dof=2)
        self._pose = Pose2D()
        self._wheel_separation = wheel_separation
        self._wheel_radius = 0.05
        self._target_linear = 0.0
        self._target_angular = 0.0

    def _do_initialize(self) -> None:
        self._driver_state.connected = True
        self._driver_state.joint_positions = [0.0, 0.0]

    def _do_stop(self) -> None:
        self._driver_state.connected = False

    def set_command(self, linear: float, angular: float) -> None:
        """Set (linear m/s, angular rad/s) command."""
        self._target_linear = linear
        self._target_angular = angular

    def step(self, dt: float) -> None:
        """Integrate pose forward by dt using unicycle model."""
        v = self._target_linear
        w = self._target_angular

        # Bicycle/unicycle integration
        if abs(w) < 1e-6:
            self._pose.x += v * dt * math.cos(self._pose.theta)
            self._pose.y += v * dt * math.sin(self._pose.theta)
        else:
            r = v / w
            theta0 = self._pose.theta
            self._pose.x += r * (math.sin(theta0 + w * dt) - math.sin(theta0))
            self._pose.y += r * (math.cos(theta0) - math.cos(theta0 + w * dt))
            self._pose.theta += w * dt

        # Normalize angle
        self._pose.theta = math.atan2(math.sin(self._pose.theta), math.cos(self._pose.theta))

        # Update driver state (joint positions as wheel angles)
        left_w = (v - 0.5 * w * self._wheel_separation) / self._wheel_radius
        right_w = (v + 0.5 * w * self._wheel_separation) / self._wheel_radius
        self._driver_state.joint_positions[0] += left_w * dt
        self._driver_state.joint_positions[1] += right_w * dt

    def get_pose(self) -> Pose2D:
        return Pose2D(self._pose.x, self._pose.y, self._pose.theta)

    def get_joint_positions(self) -> list[float]:
        return list(self._driver_state.joint_positions)

    def get_joint_velocities(self) -> list[float]:
        return [0.0, 0.0]

    def get_joint_torques(self) -> list[float]:
        return [0.0, 0.0]

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._validate_joint_positions(positions)
        self._driver_state.joint_positions = list(positions)
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        self._validate_trajectory(trajectory)
        for wp in trajectory.waypoints:
            self.move_joints(wp)
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        return True

    def emergency_stop(self) -> None:
        self._target_linear = 0.0
        self._target_angular = 0.0
        self._driver_state.error_code = 99

    def get_state(self) -> DriverState:
        return self._driver_state


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def main() -> int:
    print("=== ROSClaw Mobile Robot PID Navigation Demo ===\n")

    robot_id = "mobile_pid_demo"
    waypoints = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]

    # 1. Load mobile base e-URDF profile
    registry = RobotRegistry()
    try:
        profile = registry.install("mock_mobile_base")
        print(f"1. Loaded e-URDF profile: {profile.name} (DOF={profile.embodiment.dof})")
    except Exception as exc:
        print(f"1. Could not load mobile base profile: {exc}")
        profile = None

    # 2. EventBus + MemoryInterface
    bus = EventBus()
    bus.subscribe(
        "telemetry.pose",
        lambda e: print(
            f"   [Bus] pose: x={e.payload['x']:.2f}, y={e.payload['y']:.2f}, "
            f"theta={math.degrees(e.payload['theta']):.1f}°"
        ),
    )

    memory = MemoryInterface(
        robot_id=robot_id,
        event_bus=bus,
        seekdb_client=SeekDBMemoryClient(),
    )
    memory.initialize()
    memory.start()
    print("2. MemoryInterface started (SeekDB in-memory)")

    # 3. Differential-drive driver
    driver = DifferentialDriveDriver(robot_id=robot_id, wheel_separation=0.3)
    driver.initialize()
    driver.start()
    print("3. DifferentialDriveDriver started")

    # 4. PID heading controller + linear speed
    heading_pid = PIDController(PIDGains(kp=2.0, ki=0.0, kd=0.2))
    heading_pid.set_output_limit(-1.5, 1.5)

    dt = 0.05
    linear_speed = 0.2
    reached_threshold = 0.1

    print("4. Following waypoints...")
    for idx, (wx, wy) in enumerate(waypoints):
        print(f"   Navigating to waypoint {idx + 1}/{len(waypoints)}: ({wx}, {wy})")
        steps = 0
        max_steps = 400
        while steps < max_steps:
            pose = driver.get_pose()
            dx = wx - pose.x
            dy = wy - pose.y
            distance = math.hypot(dx, dy)

            if distance < reached_threshold:
                break

            desired_heading = math.atan2(dy, dx)
            heading_error = normalize_angle(desired_heading - pose.theta)
            angular_cmd = heading_pid.update(heading_error, dt)

            # Slow down near waypoint
            v = min(linear_speed, distance * 0.5)
            driver.set_command(v, angular_cmd)
            driver.step(dt)
            steps += 1

            if steps % 20 == 0:
                bus.publish(
                    Event(
                        topic="telemetry.pose",
                        payload={
                            "x": pose.x,
                            "y": pose.y,
                            "theta": pose.theta,
                            "waypoint": (wx, wy),
                            "distance": distance,
                        },
                        source="mobile_pid_demo",
                        priority=EventPriority.NORMAL,
                    )
                )

        final_pose = driver.get_pose()
        memory.store_experience(
            event_id=f"{robot_id}-wp{idx}-{int(time.time() * 1000)}",
            event_type="waypoint_reached",
            instruction=f"Navigate to waypoint ({wx}, {wy})",
            outcome="success",
            duration_sec=steps * dt,
            tags=["mobile", "pid", "navigation"],
            metadata={
                "waypoint": (wx, wy),
                "final_pose": {
                    "x": final_pose.x,
                    "y": final_pose.y,
                    "theta": final_pose.theta,
                },
                "steps": steps,
            },
        )

    # 5. Query navigation experiences
    experiences = memory.seekdb_client.query(
        "experience_graph",
        filters={"robot_id": robot_id, "event_type": "waypoint_reached"},
        limit=10,
    )
    print(f"5. Stored {len(experiences)} waypoint experiences for {robot_id}")

    # 6. Final pose summary
    final = driver.get_pose()
    print(
        f"6. Final pose: x={final.x:.2f}, y={final.y:.2f}, "
        f"theta={math.degrees(final.theta):.1f}°"
    )

    # Cleanup
    driver.stop()
    memory.stop()
    print("\n=== Mobile Robot PID Demo complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

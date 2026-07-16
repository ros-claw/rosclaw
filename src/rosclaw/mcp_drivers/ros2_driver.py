"""ROS2 Driver - Real robot control via ROS 2 / rclpy."""

import logging
from typing import Any

from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand

logger = logging.getLogger("rosclaw.mcp_drivers.ros2")


class ROS2Driver(BaseDriver):
    """
    Hardware driver for ROS 2 controlled robots.

    Communicates with the robot via ROS 2 topics and services.
    Requires rclpy and ROS 2 environment to be available.
    """

    def __init__(
        self,
        robot_id: str,
        joint_dof: int = 6,
        node_name: str = "rosclaw_driver",
        *,
        fixture_mode: bool = False,
    ):
        super().__init__(robot_id, joint_dof, fixture_mode=fixture_mode)
        self._node_name = node_name
        self._rclpy: Any | None = None
        self._node: Any | None = None
        self._pub_joint_cmd: Any | None = None
        self._sub_joint_state: Any | None = None
        self._latest_joint_state: dict | None = None

    def _do_initialize(self) -> None:
        if self.fixture_mode:
            self._activate_fixture("Explicit ROS2 driver fixture mode; no ROS graph connected.")
            return
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import JointState
            from trajectory_msgs.msg import JointTrajectory

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)
            self._node = Node(self._node_name)

            self._pub_joint_cmd = self._node.create_publisher(
                JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
            )
            self._sub_joint_state = self._node.create_subscription(
                JointState, "/joint_states", self._on_joint_state, 10
            )
            self._driver_state.execution_mode = "REAL"
            self._driver_state.trust_level = "UNVERIFIED"
            self._driver_state.implementation_kind = "ros2"
            self._driver_state.connection_evidence = (
                "ROS2 node initialized; waiting for joint-state feedback."
            )
            logger.info("ROS 2 node '%s' initialized; waiting for feedback", self._node_name)
        except Exception as e:
            self._rclpy = None
            self._node = None
            self._activate_fixture(f"ROS2 initialization failed: {e}")

    def _do_stop(self) -> None:
        if self._node:
            self._node.destroy_node()
            self._node = None
        # NOTE: Do NOT call rclpy.shutdown() here — it's global state.
        # Multiple test instances share the same rclpy context.
        self._driver_state.connected = False

    def _on_joint_state(self, msg: Any) -> None:
        self._latest_joint_state = {
            "positions": list(msg.position),
            "velocities": list(msg.velocity),
            "efforts": list(msg.effort),
        }
        self._driver_state.joint_positions = list(msg.position)[: self.joint_dof]
        self._driver_state.joint_velocities = list(msg.velocity)[: self.joint_dof]
        self._driver_state.joint_torques = list(msg.effort)[: self.joint_dof]
        self._mark_backend_ready(
            execution_mode="REAL",
            trust_level="OBSERVED",
            implementation_kind="ros2",
            connection_evidence="joint_states feedback received",
            usable_for_real_execution=True,
        )

    def get_joint_positions(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["positions"][: self.joint_dof]
        return [0.0] * self.joint_dof

    def get_joint_velocities(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["velocities"][: self.joint_dof]
        return [0.0] * self.joint_dof

    def get_joint_torques(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["efforts"][: self.joint_dof]
        return [0.0] * self.joint_dof

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._ensure_ready("move_joints")
        self._validate_joint_positions(positions)
        self._validate_duration(duration)
        if not self._driver_state.connected:
            return False

        if self._rclpy is None:
            self._driver_state.joint_positions = list(positions)
            return True

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        traj = JointTrajectory()
        traj.joint_names = [f"joint_{i}" for i in range(self.joint_dof)]
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration)
        traj.points.append(point)
        self._pub_joint_cmd.publish(traj)
        self._driver_state.joint_positions = list(positions)
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        if not self._driver_state.connected:
            return False
        self._ensure_ready("execute_trajectory")
        self._validate_trajectory(trajectory)
        if self._rclpy is None:
            # Mock mode: update state and return success
            if trajectory.waypoints:
                self._driver_state.joint_positions = list(trajectory.waypoints[-1])
            return True
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        traj = JointTrajectory()
        traj.joint_names = [f"joint_{i}" for i in range(self.joint_dof)]
        for wp, t in zip(trajectory.waypoints, trajectory.times, strict=False):
            point = JointTrajectoryPoint()
            point.positions = wp
            point.time_from_start.sec = int(t)
            traj.points.append(point)
        self._pub_joint_cmd.publish(traj)
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        self._driver_state.gripper_state = position
        return True

    def emergency_stop(self) -> dict[str, Any]:
        self._driver_state.error_code = 99
        self._driver_state.error_message = "Emergency stop triggered"
        if self.fixture_mode or self._rclpy is None or self._pub_joint_cmd is None:
            return {
                "acknowledged": False,
                "physical_stop_observed": False,
                "execution_mode": "FIXTURE" if self.fixture_mode else "UNKNOWN",
                "trust_level": "SYNTHETIC" if self.fixture_mode else "UNAVAILABLE",
            }

        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        trajectory = JointTrajectory()
        trajectory.joint_names = [f"joint_{i}" for i in range(self.joint_dof)]
        point = JointTrajectoryPoint()
        point.positions = self.get_joint_positions()
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 100_000_000
        trajectory.points.append(point)
        self._pub_joint_cmd.publish(trajectory)
        return {
            "acknowledged": False,
            "physical_stop_observed": False,
            "observed_joint_velocity": list(self._driver_state.joint_velocities),
            "verification_source": None,
            "execution_mode": "REAL",
            "trust_level": "UNVERIFIED",
            "note": "ROS publish completed; controller ACK and stopped feedback were not observed.",
        }

    def get_state(self) -> DriverState:
        return self._driver_state

"""ROS2 Driver - Real robot control via ROS 2 / rclpy."""

from typing import Any, Optional

from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand


class ROS2Driver(BaseDriver):
    """
    Hardware driver for ROS 2 controlled robots.

    Communicates with the robot via ROS 2 topics and services.
    Requires rclpy and ROS 2 environment to be available.
    """

    def __init__(self, robot_id: str, joint_dof: int = 6, node_name: str = "rosclaw_driver"):
        super().__init__(robot_id, joint_dof)
        self._node_name = node_name
        self._rclpy: Optional[Any] = None
        self._node: Optional[Any] = None
        self._pub_joint_cmd: Optional[Any] = None
        self._sub_joint_state: Optional[Any] = None
        self._latest_joint_state: Optional[dict] = None

    def _do_initialize(self) -> None:
        try:
            import rclpy
            from rclpy.node import Node
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from sensor_msgs.msg import JointState
        except ImportError:
            print(f"[ROS2Driver] rclpy not available, running in mock mode")
            self._driver_state.connected = True
            return

        self._rclpy = rclpy
        rclpy.init(args=None)
        self._node = Node(self._node_name)

        self._pub_joint_cmd = self._node.create_publisher(
            JointTrajectory, "/joint_trajectory_controller/joint_trajectory", 10
        )
        self._sub_joint_state = self._node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10
        )
        self._driver_state.connected = True
        print(f"[ROS2Driver] ROS 2 node '{self._node_name}' initialized")

    def _do_stop(self) -> None:
        if self._node:
            self._node.destroy_node()
        if self._rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
        self._driver_state.connected = False

    def _on_joint_state(self, msg: Any) -> None:
        self._latest_joint_state = {
            "positions": list(msg.position),
            "velocities": list(msg.velocity),
            "efforts": list(msg.effort),
        }
        self._driver_state.joint_positions = list(msg.position)[:self.joint_dof]
        self._driver_state.joint_velocities = list(msg.velocity)[:self.joint_dof]
        self._driver_state.joint_torques = list(msg.effort)[:self.joint_dof]

    def get_joint_positions(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["positions"][:self.joint_dof]
        return [0.0] * self.joint_dof

    def get_joint_velocities(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["velocities"][:self.joint_dof]
        return [0.0] * self.joint_dof

    def get_joint_torques(self) -> list[float]:
        if self._latest_joint_state:
            return self._latest_joint_state["efforts"][:self.joint_dof]
        return [0.0] * self.joint_dof

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._ensure_ready("move_joints")
        self._validate_joint_positions(positions)
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
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        if not self._driver_state.connected or self._rclpy is None:
            return False
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        traj = JointTrajectory()
        traj.joint_names = [f"joint_{i}" for i in range(self.joint_dof)]
        for wp, t in zip(trajectory.waypoints, trajectory.times):
            point = JointTrajectoryPoint()
            point.positions = wp
            point.time_from_start.sec = int(t)
            traj.points.append(point)
        self._pub_joint_cmd.publish(traj)
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        self._driver_state.gripper_state = position
        return True

    def emergency_stop(self) -> None:
        self._driver_state.error_code = 99
        self._driver_state.error_message = "Emergency stop triggered"
        self.move_joints(self.get_joint_positions(), duration=0.1)

    def get_state(self) -> DriverState:
        return self._state

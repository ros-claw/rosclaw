"""MuJoCo Simulation Driver - Simulated robot control via MuJoCo."""

from pathlib import Path
from typing import Any, Optional

import numpy as np

from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand


class MuJoCoSimDriver(BaseDriver):
    """
    Hardware driver for MuJoCo-simulated robots.

    Provides identical interface to real hardware but runs in simulation.
    """

    def __init__(self, robot_id: str = "default_robot", model_path: str = "", joint_dof: int = 6):
        super().__init__(robot_id, joint_dof)
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._data: Optional[Any] = None
        self._sim_step = 0.002

    def _do_initialize(self) -> None:
        try:
            import mujoco
        except ImportError:
            print("[MuJoCoSimDriver] mujoco not available, running in mock mode")
            self._driver_state.connected = True
            return

        if not Path(self._model_path).exists():
            print(f"[MuJoCoSimDriver] Model not found, running in mock mode: {self._model_path}")
            self._driver_state.connected = True
            return

        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._data = mujoco.MjData(self._model)
        self._driver_state.connected = True
        print(f"[MuJoCoSimDriver] MuJoCo model loaded: {self._model_path}")

    def _do_stop(self) -> None:
        self._driver_state.connected = False
        self._model = None
        self._data = None

    def _step(self) -> None:
        if self._model is not None:
            import mujoco
            mujoco.mj_step(self._model, self._data)

    def get_joint_positions(self) -> list[float]:
        if self._data is not None:
            qpos = np.array(self._data.qpos)[:self.joint_dof]
            return qpos.tolist()
        return [0.0] * self.joint_dof

    def get_joint_velocities(self) -> list[float]:
        if self._data is not None:
            qvel = np.array(self._data.qvel)[:self.joint_dof]
            return qvel.tolist()
        return [0.0] * self.joint_dof

    def get_joint_torques(self) -> list[float]:
        if self._data is not None:
            ctrl = np.array(self._data.ctrl)[:self.joint_dof]
            return ctrl.tolist()
        return [0.0] * self.joint_dof

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._ensure_ready("move_joints")
        self._validate_joint_positions(positions)
        if not self._driver_state.connected:
            return False

        if self._data is None:
            self._driver_state.joint_positions = list(positions)
            return True

        target = np.array(positions)
        current = np.array(self.get_joint_positions())
        steps = int(duration / self._sim_step)
        if steps <= 0:
            steps = 1
        for i in range(steps):
            interp = current + (target - current) * (i / steps)
            self._data.ctrl[:self.joint_dof] = interp
            self._step()

        self._driver_state.joint_positions = self.get_joint_positions()
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        if not self._driver_state.connected:
            return False
        for wp in trajectory.waypoints:
            if len(wp) != self.joint_dof:
                self._driver_state.error_code = 1
                return False
            self.move_joints(wp, duration=trajectory.times[0] if trajectory.times else 1.0)
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        self._driver_state.gripper_state = position
        return True

    def emergency_stop(self) -> None:
        self._driver_state.error_code = 99
        self._driver_state.error_message = "Emergency stop triggered"
        if self._data is not None:
            self._data.ctrl[:] = 0.0

    def get_state(self) -> DriverState:
        self._driver_state.joint_positions = self.get_joint_positions()
        self._driver_state.joint_velocities = self.get_joint_velocities()
        self._driver_state.joint_torques = self.get_joint_torques()
        return self._driver_state

    def get_mujoco_data(self) -> Optional[Any]:
        """Access underlying MuJoCo data for Digital Twin validation."""
        return self._data

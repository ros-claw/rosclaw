"""MuJoCo Simulation Driver - Simulated robot control via MuJoCo."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw.mcp_drivers.base import BaseDriver, DriverState, TrajectoryCommand

logger = logging.getLogger("rosclaw.mcp_drivers.mujoco_sim")


class MuJoCoSimDriver(BaseDriver):
    """
    Hardware driver for MuJoCo-simulated robots.

    Provides identical interface to real hardware but runs in simulation.
    """

    def __init__(
        self,
        robot_id: str = "default_robot",
        model_path: str = "",
        joint_dof: int = 6,
        *,
        fixture_mode: bool = False,
    ):
        super().__init__(robot_id, joint_dof, fixture_mode=fixture_mode)
        self._model_path = model_path
        self._model: Any | None = None
        self._data: Any | None = None
        self._sim_step = 0.002

    def _do_initialize(self) -> None:
        if self.fixture_mode:
            self._activate_fixture("Explicit MuJoCo driver fixture mode; no physics loaded.")
            return
        try:
            import mujoco
        except ImportError:
            self._activate_fixture("MuJoCo is not installed.")

        if not self._model_path or not self._model_path.strip():
            self._activate_fixture("MuJoCo model_path is required.")

        if not Path(self._model_path).exists():
            self._activate_fixture(f"MuJoCo model not found: {self._model_path}")

        self._model = mujoco.MjModel.from_xml_path(self._model_path)
        self._data = mujoco.MjData(self._model)
        self._mark_backend_ready(
            execution_mode="SIMULATION",
            trust_level="SIMULATED",
            implementation_kind="mujoco",
            connection_evidence=f"loaded:{Path(self._model_path).resolve()}",
            usable_for_real_execution=False,
        )
        logger.info("MuJoCo model loaded: %s", self._model_path)

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
            qpos = np.array(self._data.qpos)[: self.joint_dof]
            return qpos.tolist()
        return [0.0] * self.joint_dof

    def get_joint_velocities(self) -> list[float]:
        if self._data is not None:
            qvel = np.array(self._data.qvel)[: self.joint_dof]
            return qvel.tolist()
        return [0.0] * self.joint_dof

    def get_joint_torques(self) -> list[float]:
        if self._data is not None:
            ctrl = np.array(self._data.ctrl)[: self.joint_dof]
            return ctrl.tolist()
        return [0.0] * self.joint_dof

    def move_joints(self, positions: list[float], duration: float = 2.0) -> bool:
        self._ensure_ready("move_joints")
        self._validate_joint_positions(positions)
        self._validate_duration(duration)
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
            self._data.ctrl[: self.joint_dof] = interp
            self._step()

        self._driver_state.joint_positions = self.get_joint_positions()
        return True

    def execute_trajectory(self, trajectory: TrajectoryCommand) -> bool:
        if not self._driver_state.connected:
            return False
        self._ensure_ready("execute_trajectory")
        self._validate_trajectory(trajectory)
        for wp, t in zip(trajectory.waypoints, trajectory.times, strict=False):
            self.move_joints(wp, duration=t)
        return True

    def set_gripper(self, position: float, force: float = 0.5) -> bool:
        self._driver_state.gripper_state = position
        return True

    def emergency_stop(self) -> dict[str, Any]:
        self._driver_state.error_code = 99
        self._driver_state.error_message = "Emergency stop triggered"
        if self._data is None or self._model is None:
            return {
                "acknowledged": False,
                "physical_stop_observed": False,
                "execution_mode": "FIXTURE" if self.fixture_mode else "UNKNOWN",
                "trust_level": "SYNTHETIC" if self.fixture_mode else "UNAVAILABLE",
                "note": "No MuJoCo physics state exists to stop or observe.",
            }

        import mujoco

        actuator_count = min(self._model.nu, self._model.nq)
        self._data.ctrl[:actuator_count] = self._data.qpos[:actuator_count]
        for _ in range(100):
            mujoco.mj_step(self._model, self._data)
        joint_velocity = self.get_joint_velocities()
        observed_velocity = max((abs(value) for value in joint_velocity), default=0.0)
        observed = observed_velocity <= 1e-3
        return {
            "acknowledged": True,
            "physical_stop_observed": observed,
            "observed_velocity": observed_velocity,
            "observed_joint_velocity": joint_velocity,
            "verification_source": "mujoco.qvel",
            "execution_mode": "SIMULATION",
            "trust_level": "SIMULATED",
        }

    def get_state(self) -> DriverState:
        self._driver_state.joint_positions = self.get_joint_positions()
        self._driver_state.joint_velocities = self.get_joint_velocities()
        self._driver_state.joint_torques = self.get_joint_torques()
        return self._driver_state

    def get_mujoco_data(self) -> Any | None:
        """Access underlying MuJoCo data for Digital Twin validation."""
        return self._data

"""Wrapper for mjlab environments with ROSClaw integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch

# Lazy imports to avoid loading heavy dependencies unless needed
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
    from mjlab.scene import SceneCfg


@dataclass
class MjlabEnvConfig:
    """Configuration for mjlab environment wrapper.

    Attributes:
        num_envs: Number of parallel simulation environments.
        device: Device to run simulation on ("cuda" or "cpu").
        physics_dt: Physics timestep in seconds (default: 0.002 = 500Hz).
        decimation: Number of physics steps per environment step.
        episode_length_s: Maximum episode length in seconds.
        enable_rendering: Whether to enable RGB rendering.
        seed: Random seed for reproducibility.
    """

    num_envs: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    physics_dt: float = 0.002  # 500Hz physics
    decimation: int = 4  # 50Hz control (dt = 0.008)
    episode_length_s: float = 10.0
    enable_rendering: bool = False
    seed: int | None = None

    # Scene configuration
    env_spacing: float = 2.0
    terrain_type: str | None = None  # "plane", "trimesh", None

    # Domain randomization
    enable_domain_randomization: bool = False
    randomization_config: dict[str, Any] = field(default_factory=dict)


class MjlabEnv:
    """ROSClaw wrapper for mjlab ManagerBasedRlEnv.

    Provides a simplified interface for running robot simulations with
    MuJoCo Warp (GPU-accelerated) physics. Supports parallel environments
    for data collection and training.

    Example:
        >>> config = MjlabEnvConfig(num_envs=4, device="cuda")
        >>> env = MjlabEnv(config)
        >>> env.load_robot("so101.urdf")
        >>> obs = env.reset()
        >>> for _ in range(100):
        ...     action = policy(obs)
        ...     obs, reward, done, info = env.step(action)
    """

    def __init__(self, config: MjlabEnvConfig | None = None):
        self.config = config or MjlabEnvConfig()
        self._env: ManagerBasedRlEnv | None = None
        self._env_cfg: ManagerBasedRlEnvCfg | None = None
        self._robot_name: str | None = None
        self._obs_space = None
        self._action_space = None

    @property
    def is_initialized(self) -> bool:
        """Check if environment is initialized."""
        return self._env is not None

    @property
    def num_envs(self) -> int:
        """Number of parallel environments."""
        return self.config.num_envs

    @property
    def device(self) -> str:
        """Device being used for simulation."""
        return self.config.device

    @property
    def physics_dt(self) -> float:
        """Physics timestep."""
        return self.config.physics_dt

    @property
    def step_dt(self) -> float:
        """Environment step duration (physics_dt * decimation)."""
        return self.config.physics_dt * self.config.decimation

    def load_robot(
        self,
        robot_path: str | Path,
        robot_name: str = "robot",
        fix_base: bool = False,
        initial_joint_pos: dict[str, float] | None = None,
    ) -> None:
        """Load a robot model into the simulation.

        Args:
            robot_path: Path to robot URDF or MJCF file.
            robot_name: Name for the robot entity in simulation.
            fix_base: Whether to fix the robot base.
            initial_joint_pos: Initial joint positions (joint_name -> position).
        """
        from rosclaw_sim.robot_loader import RobotLoader

        robot_path = Path(robot_path)
        if not robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {robot_path}")

        # Use RobotLoader to convert URDF to mjlab scene
        loader = RobotLoader()
        scene_cfg = loader.load(
            robot_path=robot_path,
            robot_name=robot_name,
            fix_base=fix_base,
            initial_joint_pos=initial_joint_pos,
            num_envs=self.config.num_envs,
            env_spacing=self.config.env_spacing,
        )

        self._robot_name = robot_name
        self._scene_cfg = scene_cfg

    def initialize(self) -> None:
        """Initialize the simulation environment.

        Must be called after load_robot() and before reset()/step().
        """
        if self._scene_cfg is None:
            raise RuntimeError("Must call load_robot() before initialize()")

        from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
        from mjlab.sim import SimulationCfg, MujocoCfg

        # Create simulation configuration
        sim_cfg = SimulationCfg(
            mujoco=MujocoCfg(
                timestep=self.config.physics_dt,
                integrator="implicitfast",
                solver="newton",
            )
        )

        # Create environment configuration
        self._env_cfg = ManagerBasedRlEnvCfg(
            decimation=self.config.decimation,
            scene=self._scene_cfg,
            sim=sim_cfg,
            episode_length_s=self.config.episode_length_s,
            seed=self.config.seed,
            viewer=None,
        )

        # Initialize environment
        render_mode = "rgb_array" if self.config.enable_rendering else None
        self._env = ManagerBasedRlEnv(
            cfg=self._env_cfg,
            device=self.config.device,
            render_mode=render_mode,
        )

        # Cache spaces
        self._obs_space = self._env.observation_space
        self._action_space = self._env.action_space

    def reset(
        self, env_ids: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Reset environment(s).

        Args:
            env_ids: Specific environment IDs to reset. If None, reset all.

        Returns:
            Observation dictionary.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        obs, info = self._env.reset(env_ids=env_ids)
        return obs

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict]:
        """Step the environment.

        Args:
            action: Action tensor of shape (num_envs, action_dim).

        Returns:
            Tuple of (observations, rewards, dones, info).
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call initialize() first.")

        # Convert numpy to torch if needed
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.config.device)

        # Ensure correct device and dtype
        action = action.to(self.config.device, dtype=torch.float32)

        obs, reward, terminated, truncated, info = self._env.step(action)

        # Combine terminated and truncated for simpler interface
        done = terminated | truncated

        return obs, reward, done, info

    def get_robot_state(self) -> dict[str, torch.Tensor]:
        """Get current robot state.

        Returns:
            Dictionary containing:
                - joint_pos: Joint positions
                - joint_vel: Joint velocities
                - base_pos: Base position (if floating)
                - base_quat: Base orientation quaternion (if floating)
        """
        if self._env is None or self._robot_name is None:
            raise RuntimeError("Environment or robot not initialized")

        robot = self._env.scene[self._robot_name]
        return {
            "joint_pos": robot.data.joint_pos,
            "joint_vel": robot.data.joint_vel,
            "base_pos": robot.data.root_pos_w if hasattr(robot.data, "root_pos_w") else None,
            "base_quat": robot.data.root_quat_w if hasattr(robot.data, "root_quat_w") else None,
        }

    def set_robot_state(
        self,
        joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Set robot state.

        Args:
            joint_pos: Joint positions to set.
            joint_vel: Joint velocities to set.
            env_ids: Specific environments to set. If None, apply to all.
        """
        if self._env is None or self._robot_name is None:
            raise RuntimeError("Environment or robot not initialized")

        robot = self._env.scene[self._robot_name]

        if joint_pos is not None:
            robot.write_joint_state_to_sim(
                joint_pos, robot.data.joint_vel if joint_vel is None else joint_vel, env_ids
            )
        elif joint_vel is not None:
            robot.write_joint_state_to_sim(robot.data.joint_pos, joint_vel, env_ids)

    def render(self) -> np.ndarray | None:
        """Render the environment.

        Returns:
            RGB image array of shape (H, W, 3) if rendering enabled, else None.
        """
        if self._env is None:
            return None
        return self._env.render()

    def close(self) -> None:
        """Close the environment and free resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get joint position limits.

        Returns:
            Tuple of (lower_limits, upper_limits) tensors.
        """
        if self._env is None:
            raise RuntimeError("Environment not initialized")

        return (
            self._env.sim.mj_model.jnt_range[:, 0],
            self._env.sim.mj_model.jnt_range[:, 1],
        )

    def get_joint_names(self) -> list[str]:
        """Get list of joint names."""
        if self._env is None or self._robot_name is None:
            raise RuntimeError("Environment or robot not initialized")

        robot = self._env.scene[self._robot_name]
        return list(robot.joint_names)

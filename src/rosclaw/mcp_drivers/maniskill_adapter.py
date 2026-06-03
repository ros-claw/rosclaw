"""ManiSkill adapter for ROSClaw — wrap ManiSkill simulation as ROSClaw tasks.

Turns ManiSkill benchmark rollouts into auditable ROSClaw Practice/Memory traces.

Supported tasks:
    - PickCube
    - StackCube
    - PegInsertionSide

Usage:
    from rosclaw.mcp_drivers.maniskill_adapter import ManiSkillAdapter
    adapter = ManiSkillAdapter(task="PickCube")
    adapter.reset()
    obs, reward, terminated, truncated, info = adapter.step(action)
    adapter.record_episode(success=info["success"])
"""

from pathlib import Path
from typing import Any


class ManiSkillAdapter:
    """Adapter that wraps a ManiSkill environment as a ROSClaw-compatible task.

    The adapter bridges:
        ManiSkill env.step() → ROSClaw Sandbox/Runtime execution
        ManiSkill obs/reward  → ROSClaw Practice episode recording
        ManiSkill info        → ROSClaw Memory failure/success traces
    """

    _SUPPORTED_TASKS = {
        "PickCube": "PickCube-v1",
        "StackCube": "StackCube-v1",
        "PegInsertion": "PegInsertionSide-v1",
    }

    def __init__(
        self,
        task: str = "PickCube",
        robot: str = "panda",
        control_mode: str = "pd_joint_pos",
        render_mode: str | None = None,
        record_dir: str | None = None,
    ) -> None:
        """Initialize ManiSkill adapter.

        Args:
            task: Task name — PickCube, StackCube, or PegInsertion
            robot: Robot urdf name (panda, fetch, etc.)
            control_mode: ManiSkill control mode
            render_mode: "rgb_array" for recording, None for headless
            record_dir: Directory to save episode videos/traces
        """
        self.task = task
        self.robot = robot
        self.control_mode = control_mode
        self.render_mode = render_mode
        self.record_dir = Path(record_dir) if record_dir else None

        self._env: Any = None
        self._episode_steps: list[dict[str, Any]] = []
        self._episode_reward: float = 0.0
        self._episode_count: int = 0

        self._init_env()

    def _init_env(self) -> None:
        """Lazy-import and create ManiSkill environment."""
        try:
            import gymnasium as gym
            import mani_skill.envs  # noqa: F401 — register envs
        except ImportError as exc:
            raise ImportError(
                "ManiSkill not installed. Run: pip install mani-skill"
            ) from exc

        env_id = self._SUPPORTED_TASKS.get(self.task)
        if env_id is None:
            raise ValueError(
                f"Unknown task: {self.task}. "
                f"Supported: {list(self._SUPPORTED_TASKS.keys())}"
            )

        self._env = gym.make(
            env_id,
            obs_mode="state_dict",
            control_mode=self.control_mode,
            render_mode=self.render_mode,
        )

    # ------------------------------------------------------------------
    # ROSClaw-compatible interface
    # ------------------------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """Reset environment and start new episode recording."""
        self._episode_steps = []
        self._episode_reward = 0.0
        obs, info = self._env.reset(seed=self._episode_count)
        self._episode_count += 1
        return {"observation": obs, "info": info}

    def step(self, action: Any) -> dict[str, Any]:
        """Execute one step and record to episode trace.

        Returns:
            dict with observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self._env.step(action)

        self._episode_reward += reward
        self._episode_steps.append(
            {
                "action": action,
                "reward": reward,
                "success": info.get("success", False),
            }
        )

        return {
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        }

    def get_state(self) -> dict[str, Any]:
        """Return current episode state for ROSClaw Practice/Memory."""
        return {
            "task": self.task,
            "robot": self.robot,
            "steps": len(self._episode_steps),
            "total_reward": self._episode_reward,
            "success": (
                self._episode_steps[-1]["success"]
                if self._episode_steps
                else False
            ),
        }

    def record_episode(self, success: bool | None = None) -> dict[str, Any]:
        """Finalize episode and return trace for ROSClaw Practice.

        Args:
            success: Override success flag (uses last step if None)

        Returns:
            Episode trace dict compatible with PracticeRecorder
        """
        if success is None and self._episode_steps:
            success = self._episode_steps[-1]["success"]
        else:
            success = bool(success)

        trace = {
            "task": self.task,
            "robot": self.robot,
            "episode_id": f"maniskill_{self.task}_{self._episode_count:04d}",
            "steps": len(self._episode_steps),
            "total_reward": round(self._episode_reward, 4),
            "success": success,
            "control_mode": self.control_mode,
            "step_traces": self._episode_steps,
        }

        # Write video if render_mode was set
        if self.render_mode and self.record_dir:
            self.record_dir.mkdir(parents=True, exist_ok=True)

        return trace

    def close(self) -> None:
        """Close ManiSkill environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    # ------------------------------------------------------------------
    # ROSClaw Provider integration helpers
    # ------------------------------------------------------------------

    def get_observation_space(self) -> dict[str, Any]:
        """Return observation space info for Provider routing."""
        if self._env is None:
            return {}
        return {
            "obs_mode": "state_dict",
            "control_mode": self.control_mode,
            "action_space": str(self._env.action_space),
        }

    def sample_action(self) -> Any:
        """Sample random action — useful for baseline policy."""
        if self._env is None:
            raise RuntimeError("Environment not initialized")
        return self._env.action_space.sample()

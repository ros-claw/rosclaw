"""sandbox_api.py — Minimal MuJoCo sandbox for ROSClaw v1.0.

Provides real physics stepping via MuJoCo mj_step for trajectory validation.
Falls back gracefully when no model is found.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional


class SandboxSession:
    """Lightweight session handle."""

    def __init__(self, session_id: str):
        self.session_id = session_id


class Sandbox:
    """MuJoCo-backed physics sandbox.

    Loads a robot's MuJoCo XML from e-urdf-zoo and steps physics.
    """

    def __init__(
        self,
        robot_id: str,
        world_id: str,
        engine: str = "mujoco",
        publisher: Optional[Any] = None,
    ):
        self._robot_id = robot_id
        self._world_id = world_id
        self._engine = engine
        self._publisher = publisher
        self.session = SandboxSession(str(uuid.uuid4()))
        self._model: Optional[Any] = None
        self._data: Optional[Any] = None
        self._load_model()

    @classmethod
    def create(
        cls,
        robot_id: str,
        world_id: str,
        engine: str = "mujoco",
        publisher: Optional[Any] = None,
    ) -> "Sandbox":
        """Factory method used by SandboxRuntimeAdapter."""
        return cls(robot_id, world_id, engine, publisher)

    def _load_model(self) -> None:
        """Attempt to load a MuJoCo XML for the robot."""
        try:
            import mujoco
        except ImportError:
            print("[Sandbox] MuJoCo not installed — physics disabled")
            return

        # Locate project root (repo top-level)
        project_root = Path(__file__).parent.parent.parent.parent
        zoo_path = project_root / "e-urdf-zoo"

        # Canonical robot directory lookup
        robot_dir = zoo_path / self._robot_id
        if not robot_dir.exists():
            aliases = {
                "ur5e": "ur5e",
                "universal_robots_ur5e": "ur5e",
                "g1": "g1",
                "unitree_g1": "g1",
                "go2": "unitree_go2",
                "unitree_go2": "unitree_go2",
            }
            robot_dir = zoo_path / aliases.get(self._robot_id, self._robot_id)

        if not robot_dir.exists():
            print(f"[Sandbox] No robot directory for {self._robot_id}")
            return

        # Candidate MuJoCo XML files
        candidates = [
            robot_dir / "scene.xml",
            robot_dir / "robot.mjcf.xml",
            robot_dir / f"{self._robot_id}.xml",
        ]
        # Also try directory-name-based xml
        if robot_dir.name != self._robot_id:
            candidates.append(robot_dir / f"{robot_dir.name}.xml")

        for candidate in candidates:
            if candidate.exists():
                try:
                    self._model = mujoco.MjModel.from_xml_path(str(candidate))
                    self._data = mujoco.MjData(self._model)
                    print(f"[Sandbox] MuJoCo model loaded: {candidate.name} "
                          f"(njoints={self._model.njnt}, nq={self._model.nq}, nv={self._model.nv})")
                    return
                except Exception as e:
                    print(f"[Sandbox] Failed to load {candidate}: {e}")

        print(f"[Sandbox] No MuJoCo model found for {self._robot_id}")

    def reset(self) -> None:
        """Reset physics state."""
        if self._data is not None and self._model is not None:
            import mujoco
            mujoco.mj_resetData(self._model, self._data)

    def close(self) -> None:
        """Cleanup (no-op for in-memory MuJoCo)."""
        self._model = None
        self._data = None

    def step(self, joint_positions: list[float]) -> Optional[dict[str, Any]]:
        """Step physics with given joint positions (position control).

        Returns state dict with qpos, qvel, time or None if no model.
        """
        if self._data is None or self._model is None:
            return None

        import mujoco

        # Position control: set ctrl to target positions
        nact = min(len(joint_positions), self._model.nu)
        self._data.ctrl[:nact] = joint_positions[:nact]

        # Step physics
        mujoco.mj_step(self._model, self._data)

        return {
            "qpos": self._data.qpos.copy().tolist(),
            "qvel": self._data.qvel.copy().tolist(),
            "time": float(self._data.time),
        }

    def get_state(self) -> Optional[dict[str, Any]]:
        """Get current physics state without stepping."""
        if self._data is None or self._model is None:
            return None
        return {
            "qpos": self._data.qpos.copy().tolist(),
            "qvel": self._data.qvel.copy().tolist(),
            "time": float(self._data.time),
        }

    @property
    def has_physics(self) -> bool:
        return self._model is not None and self._data is not None

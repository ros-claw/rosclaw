"""sandbox_api.py — Minimal MuJoCo sandbox for ROSClaw v1.0.

Provides real physics stepping via MuJoCo mj_step for trajectory validation.
Falls back gracefully when no model is found.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.sandbox.sandbox_api")


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
        publisher: Any | None = None,
    ):
        self._robot_id = robot_id
        self._world_id = world_id
        self._engine = engine
        self._publisher = publisher
        self.session = SandboxSession(str(uuid.uuid4()))
        self._model: Any | None = None
        self._data: Any | None = None
        self._load_model()

    @classmethod
    def create(
        cls,
        robot_id: str,
        world_id: str,
        engine: str = "mujoco",
        publisher: Any | None = None,
    ) -> Sandbox:
        """Factory method used by SandboxRuntimeAdapter."""
        return cls(robot_id, world_id, engine, publisher)

    def _load_model(self) -> None:
        """Attempt to load a MuJoCo XML for the robot."""
        try:
            import mujoco
        except ImportError:
            logger.warning("MuJoCo not installed — physics disabled")
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
            logger.warning("No robot directory for %s", self._robot_id)
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
                    logger.info("MuJoCo model loaded: %s (njoints=%s, nq=%s, nv=%s)",
                                candidate.name, self._model.njnt, self._model.nq, self._model.nv)
                    return
                except Exception as e:
                    logger.warning("Failed to load %s: %s", candidate, e)

        logger.warning("No MuJoCo model found for %s", self._robot_id)

    def reset(self) -> None:
        """Reset physics state."""
        if self._data is not None and self._model is not None:
            import mujoco
            mujoco.mj_resetData(self._model, self._data)

    def close(self) -> None:
        """Cleanup (no-op for in-memory MuJoCo)."""
        self._model = None
        self._data = None

    def step(self, joint_positions: list[float]) -> dict[str, Any] | None:
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

    def get_state(self) -> dict[str, Any] | None:
        """Get current physics state without stepping."""
        if self._data is None or self._model is None:
            return None
        return {
            "qpos": self._data.qpos.copy().tolist(),
            "qvel": self._data.qvel.copy().tolist(),
            "time": float(self._data.time),
        }

    def get_observation(self, normalize: bool = True) -> dict[str, Any] | None:
        """Get rich normalized observation from MuJoCo scene state.

        Returns a dict with:
        - joint_positions: raw qpos
        - joint_positions_normalized: mapped to [-1, 1] via jnt_range
        - joint_velocities: raw qvel
        - body_positions: named body positions in world frame
        - contacts: active contact pairs with distance
        - time: simulation time

        Args:
            normalize: If True, joint positions are normalized to [-1, 1]
                       using MuJoCo jnt_range. Velocities are clipped to
                       [-1, 1] via tanh for stability.
        """
        if self._data is None or self._model is None:
            return None

        import mujoco
        import numpy as np

        qpos = self._data.qpos.copy()
        qvel = self._data.qvel.copy()

        # Joint position normalization using model limits
        if normalize and self._model.njnt > 0:
            qpos_norm = np.zeros_like(qpos)
            jnt_range = self._model.jnt_range
            for i in range(min(len(qpos), self._model.njnt)):
                lo, hi = jnt_range[i]
                if hi > lo:
                    qpos_norm[i] = 2.0 * (qpos[i] - lo) / (hi - lo) - 1.0
                else:
                    qpos_norm[i] = 0.0
            # Velocity: tanh clipping for stability
            qvel_norm = np.tanh(qvel)
        else:
            qpos_norm = qpos.copy()
            qvel_norm = qvel.copy()

        # Body positions (scene objects)
        body_positions: dict[str, list[float]] = {}
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                body_positions[name] = self._data.xpos[i].copy().tolist()

        # Active contacts
        contacts: list[dict[str, Any]] = []
        for i in range(self._data.ncon):
            c = self._data.contact[i]
            if c.dist < 0.01:  # near contact
                g1 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom{c.geom1}"
                g2 = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom{c.geom2}"
                contacts.append({
                    "geom1": g1,
                    "geom2": g2,
                    "distance": float(c.dist),
                })

        return {
            "joint_positions": qpos.tolist(),
            "joint_positions_normalized": qpos_norm.tolist(),
            "joint_velocities": qvel.tolist(),
            "joint_velocities_normalized": qvel_norm.tolist(),
            "body_positions": body_positions,
            "contacts": contacts,
            "time": float(self._data.time),
        }

    @property
    def has_physics(self) -> bool:
        return self._model is not None and self._data is not None

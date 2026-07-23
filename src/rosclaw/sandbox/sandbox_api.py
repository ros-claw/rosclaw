"""Minimal, truthful MuJoCo sandbox for ROSClaw v1.0."""

from __future__ import annotations

import logging
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

logger = logging.getLogger("rosclaw.sandbox.sandbox_api")
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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
        self._model_path: Path | None = None
        self._load_error: str | None = None
        self._world_metadata: dict[str, Any] = {}
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
        if not isinstance(self._robot_id, str) or not _SAFE_IDENTIFIER_RE.fullmatch(self._robot_id):
            self._load_error = "Invalid robot identifier."
            logger.warning("%s", self._load_error)
            return
        if self._engine.lower() != "mujoco":
            self._load_error = f"Engine '{self._engine}' does not execute physics."
            return
        try:
            import mujoco
        except ImportError:
            self._load_error = "MuJoCo is not installed."
            logger.warning("%s", self._load_error)
            return

        if self._world_id not in {"empty", "tabletop"}:
            self._load_error = f"Unsupported MuJoCo world '{self._world_id}'."
            logger.warning("%s", self._load_error)
            return

        # Resolve both editable-source and wheel-installed robot data.
        from rosclaw.runtime.eurdf_loader import _default_zoo_path

        zoo_path = _default_zoo_path().expanduser().resolve()

        # Canonical robot directory lookup
        robot_dir = (zoo_path / self._robot_id).resolve()
        if not robot_dir.exists():
            aliases = {
                "ur5e": "ur5e",
                "sim_ur5e": "ur5e",
                "universal_robots_ur5e": "ur5e",
                "g1": "g1",
                "unitree_g1": "g1",
                "go2": "unitree_go2",
                "unitree_go2": "unitree_go2",
            }
            robot_dir = (zoo_path / aliases.get(self._robot_id, self._robot_id)).resolve()

        if not robot_dir.is_relative_to(zoo_path) or not robot_dir.exists():
            self._load_error = f"No robot model directory for '{self._robot_id}'."
            logger.warning("%s", self._load_error)
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
                    if self._world_id == "tabletop":
                        self._model = self._load_tabletop_model(candidate)
                    else:
                        self._model = mujoco.MjModel.from_xml_path(str(candidate))
                    self._data = mujoco.MjData(self._model)
                    self._model_path = candidate.resolve()
                    self._load_error = None
                    logger.info(
                        "MuJoCo model loaded: %s world=%s (njoints=%s, nq=%s, nv=%s)",
                        candidate.name,
                        self._world_id,
                        self._model.njnt,
                        self._model.nq,
                        self._model.nv,
                    )
                    return
                except Exception as e:
                    self._load_error = f"Failed to load '{candidate}': {e}"
                    logger.warning("Failed to load %s: %s", candidate, e)

        if self._load_error is None:
            self._load_error = f"No MuJoCo XML found for '{self._robot_id}'."
        logger.warning("%s", self._load_error)

    def _load_tabletop_model(self, model_path: Path) -> Any:
        """Compose a deterministic tabletop world into a robot MJCF model."""
        import mujoco

        tree = ET.parse(model_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        mesh_dir = model_path.parent / compiler.get("meshdir", "")
        compiler.set("meshdir", str(mesh_dir.resolve()))

        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"MJCF model '{model_path}' has no worldbody")

        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "world_floor",
                "type": "plane",
                "size": "2 2 0.05",
                "pos": "0 0 -0.01",
                "rgba": "0.82 0.84 0.86 1",
                "friction": "1 0.005 0.0001",
            },
        )
        table_center = [-0.3, 0.5, 0.18]
        table_half_size = [0.5, 0.35, 0.02]
        table = ET.SubElement(
            worldbody,
            "body",
            {
                "name": "tabletop_table",
                "pos": " ".join(str(value) for value in table_center),
            },
        )
        ET.SubElement(
            table,
            "geom",
            {
                "name": "tabletop_surface",
                "type": "box",
                "size": " ".join(str(value) for value in table_half_size),
                "rgba": "0.36 0.39 0.42 1",
                "friction": "1 0.005 0.0001",
            },
        )
        ET.SubElement(
            worldbody,
            "site",
            {
                "name": "default_reach_target",
                "type": "sphere",
                "size": "0.018",
                "pos": "-0.24 0.51 0.47",
                "rgba": "0.1 0.8 0.2 0.7",
                "group": "4",
            },
        )
        self._world_metadata = {
            "world_id": "tabletop",
            "table": {
                "center": table_center,
                "half_size": table_half_size,
                "top_z": table_center[2] + table_half_size[2],
                "geom": "tabletop_surface",
            },
            "default_target": [-0.24, 0.51, 0.47],
        }
        return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))

    def reset(self, keyframe: str | None = "home") -> None:
        """Reset physics state, preferring the robot's named home keyframe."""
        if self._data is not None and self._model is not None:
            import mujoco

            keyframe_id = -1
            if keyframe:
                keyframe_id = mujoco.mj_name2id(
                    self._model,
                    mujoco.mjtObj.mjOBJ_KEY,
                    keyframe,
                )
            if keyframe_id >= 0:
                mujoco.mj_resetDataKeyframe(self._model, self._data, keyframe_id)
            else:
                mujoco.mj_resetData(self._model, self._data)
            mujoco.mj_forward(self._model, self._data)

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
                g1 = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
                    or f"geom{c.geom1}"
                )
                g2 = (
                    mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                    or f"geom{c.geom2}"
                )
                contacts.append(
                    {
                        "geom1": g1,
                        "geom2": g2,
                        "distance": float(c.dist),
                    }
                )

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

    @property
    def load_error(self) -> str | None:
        """Explain why physics is unavailable, if applicable."""

        return self._load_error

    @property
    def model_path(self) -> Path | None:
        """Resolved MJCF path used by this sandbox."""

        return self._model_path

    @property
    def world_metadata(self) -> dict[str, Any]:
        """Structured metadata for the composed world."""

        return dict(self._world_metadata)

    @property
    def physics_model(self) -> Any | None:
        """Low-level MuJoCo model for trusted in-process task executors."""

        return self._model

    @property
    def physics_data(self) -> Any | None:
        """Low-level MuJoCo data for trusted in-process task executors."""

        return self._data

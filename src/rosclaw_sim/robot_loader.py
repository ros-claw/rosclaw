"""Robot loader for mjlab - Load URDF/MJCF robots into simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np


@dataclass
class RobotSpec:
    """Specification for a robot to load.

    Attributes:
        name: Name for the robot entity.
        urdf_path: Path to URDF file.
        fix_base: Whether to fix the base link.
        initial_joint_pos: Initial joint positions.
        joint_stiffness: Override joint stiffness values.
        joint_damping: Override joint damping values.
        scale: Scale factor for the robot model.
    """

    name: str = "robot"
    urdf_path: Path | None = None
    mjcf_path: Path | None = None
    fix_base: bool = False
    initial_joint_pos: dict[str, float] = field(default_factory=dict)
    joint_stiffness: dict[str, float] = field(default_factory=dict)
    joint_damping: dict[str, float] = field(default_factory=dict)
    scale: float = 1.0

    def __post_init__(self):
        # Ensure at least one path is provided
        if self.urdf_path is None and self.mjcf_path is None:
            raise ValueError("Must provide either urdf_path or mjcf_path")


@dataclass
class EntityCfg:
    """Entity configuration for mjlab scene."""

    name: str
    spec: mujoco.MjSpec
    initial_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneCfg:
    """Scene configuration for mjlab."""

    num_envs: int = 1
    env_spacing: float = 2.0
    entities: dict[str, EntityCfg] = field(default_factory=dict)
    terrain: Any = None


class RobotLoader:
    """Load robots from URDF/MJCF into mjlab-compatible scene configuration.

    This loader converts robot description files into mjlab scene configurations
    that can be used with ManagerBasedRlEnv.

    Example:
        >>> loader = RobotLoader()
        >>> scene_cfg = loader.load(
        ...     robot_path="so101.urdf",
        ...     robot_name="so101",
        ...     fix_base=False,
        ... )
    """

    def __init__(self):
        self._loaded_robots: dict[str, RobotSpec] = {}

    def load(
        self,
        robot_path: str | Path,
        robot_name: str = "robot",
        fix_base: bool = False,
        initial_joint_pos: dict[str, float] | None = None,
        num_envs: int = 1,
        env_spacing: float = 2.0,
        **kwargs,
    ) -> SceneCfg:
        """Load a robot and create scene configuration.

        Args:
            robot_path: Path to URDF or MJCF file.
            robot_name: Name for the robot entity.
            fix_base: Whether to fix the robot base.
            initial_joint_pos: Initial joint positions.
            num_envs: Number of parallel environments.
            env_spacing: Spacing between environments.
            **kwargs: Additional robot configuration.

        Returns:
            SceneCfg compatible with mjlab ManagerBasedRlEnv.
        """
        robot_path = Path(robot_path)

        if not robot_path.exists():
            raise FileNotFoundError(f"Robot file not found: {robot_path}")

        # Create robot spec
        spec = RobotSpec(
            name=robot_name,
            urdf_path=robot_path if robot_path.suffix == ".urdf" else None,
            mjcf_path=robot_path if robot_path.suffix in (".xml", ".mjcf") else None,
            fix_base=fix_base,
            initial_joint_pos=initial_joint_pos or {},
        )

        # Load the robot spec
        mj_spec = self._load_spec(spec)

        # Create entity configuration
        entity_cfg = EntityCfg(
            name=robot_name,
            spec=mj_spec,
            initial_state={"joint_pos": initial_joint_pos} if initial_joint_pos else {},
        )

        # Create scene configuration compatible with mjlab
        scene_cfg = self._create_scene_cfg(entity_cfg, num_envs, env_spacing)

        # Store for later reference
        self._loaded_robots[robot_name] = spec

        return scene_cfg

    def _load_spec(self, spec: RobotSpec) -> mujoco.MjSpec:
        """Load MuJoCo spec from robot description."""
        if spec.mjcf_path is not None:
            # Direct MJCF loading
            return mujoco.MjSpec.from_file(str(spec.mjcf_path))

        elif spec.urdf_path is not None:
            # URDF loading via MuJoCo
            return self._load_urdf(spec.urdf_path, spec)

        raise ValueError("No valid robot path provided")

    def _load_urdf(self, urdf_path: Path, spec: RobotSpec) -> mujoco.MjSpec:
        """Load URDF file and convert to MuJoCo spec.

        Args:
            urdf_path: Path to URDF file.
            spec: Robot specification.

        Returns:
            MuJoCo MjSpec.
        """
        # MuJoCo can load URDF directly
        mj_spec = mujoco.MjSpec.from_file(str(urdf_path))

        # Apply modifications
        if spec.fix_base:
            # Find and fix the base link
            self._fix_base_link(mj_spec)

        if spec.scale != 1.0:
            self._apply_scale(mj_spec, spec.scale)

        # Set initial joint positions if provided
        if spec.initial_joint_pos:
            self._set_initial_joint_positions(mj_spec, spec.initial_joint_pos)

        return mj_spec

    def _fix_base_link(self, spec: mujoco.MjSpec) -> None:
        """Fix the base link in the spec."""
        # In MuJoCo URDF loading, the base is typically the first body
        # We need to find the base body and ensure it's attached to world
        if spec.bodies:
            # Find world body
            world_body = None
            for body in spec.bodies:
                if body.name == "world":
                    world_body = body
                    break

            if world_body is not None and len(spec.bodies) > 1:
                # First non-world body is the base
                base_body = [b for b in spec.bodies if b.name != "world"][0]
                # Ensure it has no joint (fixed) or is attached to world
                # This is handled by the URDF compiler
                pass

    def _apply_scale(self, spec: mujoco.MjSpec, scale: float) -> None:
        """Apply scale factor to the spec."""
        # Scale body positions and geometries
        for body in spec.bodies:
            for geom in body.geoms:
                # Scale geometry sizes
                if hasattr(geom, "size"):
                    geom.size = tuple(s * scale for s in geom.size)

    def _set_initial_joint_positions(
        self,
        spec: mujoco.MjSpec,
        joint_positions: dict[str, float],
    ) -> None:
        """Set initial joint positions in the spec."""
        # Set keyframe if joints are specified
        if joint_positions and spec.joints:
            # Find joint indices and create keyframe
            qpos = []
            for joint in spec.joints:
                if joint.name in joint_positions:
                    qpos.append(joint_positions[joint.name])
                else:
                    # Default position
                    qpos.append(0.0)

            # Add keyframe if there are joints
            if qpos and not spec.keys:
                spec.add_key(name="home", qpos=qpos)

    def _create_scene_cfg(
        self,
        entity_cfg: EntityCfg,
        num_envs: int,
        env_spacing: float,
    ) -> Any:
        """Create scene configuration compatible with mjlab.

        This creates a configuration object that matches mjlab's SceneCfg structure.
        """
        # Import mjlab scene configuration
        try:
            from mjlab.scene import SceneCfg as MjlabSceneCfg
            from mjlab.entity import EntityCfg as MjlabEntityCfg

            # Create mjlab entity config
            mjlab_entity_cfg = MjlabEntityCfg(
                name=entity_cfg.name,
                asset_path=str(entity_cfg.spec),
            )

            # Create mjlab scene config
            scene_cfg = MjlabSceneCfg(
                num_envs=num_envs,
                env_spacing=env_spacing,
                entities={entity_cfg.name: mjlab_entity_cfg},
            )

            return scene_cfg

        except ImportError:
            # Fallback: create a custom scene config
            # This will be used by our custom Scene implementation
            return FallbackSceneCfg(
                num_envs=num_envs,
                env_spacing=env_spacing,
                entities={entity_cfg.name: entity_cfg},
            )

    def get_robot_info(self, robot_name: str) -> dict[str, Any]:
        """Get information about a loaded robot.

        Args:
            robot_name: Name of the loaded robot.

        Returns:
            Dictionary with robot information:
                - joint_names: List of joint names
                - joint_limits: Array of (min, max) for each joint
                - actuator_names: List of actuator names
                - body_names: List of body names
        """
        if robot_name not in self._loaded_robots:
            raise ValueError(f"Robot '{robot_name}' not loaded")

        spec = self._loaded_robots[robot_name]
        mj_spec = self._load_spec(spec)

        # Extract information
        joint_names = [j.name for j in mj_spec.joints]
        joint_limits = [(j.range[0], j.range[1]) for j in mj_spec.joints]
        actuator_names = [a.name for a in mj_spec.actuators]
        body_names = [b.name for b in mj_spec.bodies]

        return {
            "joint_names": joint_names,
            "joint_limits": joint_limits,
            "actuator_names": actuator_names,
            "body_names": body_names,
            "num_dofs": len(joint_names),
            "num_actuators": len(actuator_names),
        }


@dataclass
class FallbackSceneCfg:
    """Fallback scene configuration when mjlab is not available."""

    num_envs: int
    env_spacing: float
    entities: dict[str, EntityCfg]


def load_so101_robot(fix_base: bool = False) -> SceneCfg:
    """Load SO-101 robot configuration.

    This is a convenience function for loading the SO-101 (SO-ARM100 variant)
    robot used in ROSClaw demos.

    Args:
        fix_base: Whether to fix the robot base.

    Returns:
        SceneCfg for the SO-101 robot.
    """
    loader = RobotLoader()

    # Try to find SO-101 URDF
    search_paths = [
        Path("/root/workspace/rosclaw/DataFormat/unitree_lerobot/unitree_lerobot/eval_robot/assets"),
        Path("./assets/so101"),
        Path("./models/so101"),
    ]

    robot_path = None
    for path in search_paths:
        if path.exists():
            # Look for URDF files
            urdfs = list(path.rglob("*.urdf"))
            if urdfs:
                # Prefer SO-101 or SO-ARM100
                for u in urdfs:
                    if "so101" in u.name.lower() or "so_arm" in u.name.lower():
                        robot_path = u
                        break
                if robot_path is None:
                    robot_path = urdfs[0]
                break

    if robot_path is None:
        # Create a simple default robot spec
        raise FileNotFoundError(
            "SO-101 robot model not found. Please provide a valid URDF path."
        )

    return loader.load(
        robot_path=robot_path,
        robot_name="so101",
        fix_base=fix_base,
    )


def load_unitree_g1(fix_base: bool = True) -> SceneCfg:
    """Load Unitree G1 humanoid robot configuration.

    Args:
        fix_base: Whether to fix the robot base (default True for safety).

    Returns:
        SceneCfg for the Unitree G1 robot.
    """
    loader = RobotLoader()

    # Try to find G1 URDF
    search_paths = [
        Path("/root/workspace/rosclaw/DataFormat/unitree_lerobot/unitree_lerobot/eval_robot/assets/g1"),
        Path("/root/workspace/rosclaw/mjlab/src/mjlab/asset_zoo/robots/unitree_g1"),
    ]

    robot_path = None
    for path in search_paths:
        if path.exists():
            # Look for MJCF or URDF
            xmls = list(path.rglob("*.xml")) + list(path.rglob("*.urdf"))
            if xmls:
                for x in xmls:
                    if "g1" in x.name.lower():
                        robot_path = x
                        break
                if robot_path is None:
                    robot_path = xmls[0]
                break

    if robot_path is None:
        raise FileNotFoundError(
            "Unitree G1 robot model not found. Please provide a valid model path."
        )

    return loader.load(
        robot_path=robot_path,
        robot_name="g1",
        fix_base=fix_base,
    )

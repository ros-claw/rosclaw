"""
e-URDF Parser - Physical DNA Registry

Parses extended URDF files that include semantic annotations
for LLM grounding. The e-URDF format adds to standard URDF:

1. Semantic annotations:
   - Grasp points and affordances
   - Visual features for VLM grounding
   - Functional regions (gripper, arm, base)

2. Physical properties:
   - Detailed mass distribution
   - Surface friction coefficients
   - Contact dynamics

3. Control metadata:
   - Default PID parameters
   - Velocity/acceleration limits
   - Safety envelopes

4. Sensor configurations:
   - Camera intrinsics/extrinsics
   - Force sensor placements
   - Proprioception mapping
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(init=False)
class JointSpec:
    """Specification for a robot joint."""
    name: str
    joint_type: str  # revolute, prismatic, fixed, etc.
    parent: str
    child: str
    axis: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    limits: dict = field(default_factory=dict)
    dynamics: dict = field(default_factory=dict)
    safety: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        joint_type: str = "",
        parent: str = "",
        child: str = "",
        axis: np.ndarray = None,
        origin: np.ndarray = None,
        limits: dict = None,
        dynamics: dict = None,
        safety: dict = None,
        control: dict = None,
        type: str = "",  # URDF-compatible alias for joint_type
        lower_limit: float = None,
        upper_limit: float = None,
    ):
        self.name = name
        self.joint_type = joint_type or type or "fixed"
        self.parent = parent
        self.child = child
        self.axis = axis if axis is not None else np.array([0, 0, 1])
        self.origin = origin if origin is not None else np.eye(4)
        self.limits = limits if limits is not None else {}
        if lower_limit is not None:
            self.limits["lower"] = float(lower_limit)
        if upper_limit is not None:
            self.limits["upper"] = float(upper_limit)
        self.dynamics = dynamics if dynamics is not None else {}
        self.safety = safety if safety is not None else {}
        self.control = control if control is not None else {}


@dataclass
class LinkSpec:
    """Specification for a robot link."""
    name: str
    visual_mesh: Optional[str] = None
    collision_mesh: Optional[str] = None
    mass: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.eye(3))
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    semantic_tags: list[str] = field(default_factory=list)
    grasp_points: list[dict] = field(default_factory=list)
    material: Optional[str] = None


@dataclass
class SensorSpec:
    """Specification for a robot sensor."""
    name: str
    sensor_type: str
    parent_link: str
    origin: np.ndarray = field(default_factory=lambda: np.eye(4))
    parameters: dict = field(default_factory=dict)


@dataclass
class RobotModel:
    """
    Complete robot model parsed from e-URDF.

    This is the Physical DNA - the complete description of
    the robot's form, function, and capabilities.
    """
    name: str
    joints: dict[str, JointSpec] = field(default_factory=dict)
    links: dict[str, LinkSpec] = field(default_factory=dict)
    sensors: dict[str, SensorSpec] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def get_joint_names(self) -> list[str]:
        """Get all joint names in order."""
        return list(self.joints.keys())

    def get_actuated_joints(self) -> list[str]:
        """Get joints that can be actuated (not fixed)."""
        return [
            name for name, joint in self.joints.items()
            if joint.joint_type != "fixed"
        ]

    def get_joint_limits(self) -> dict[str, tuple[float, float]]:
        """Get position limits for all actuated joints."""
        limits = {}
        for name, joint in self.joints.items():
            if "lower" in joint.limits and "upper" in joint.limits:
                limits[name] = (joint.limits["lower"], joint.limits["upper"])
        return limits

    def get_end_effector_link(self) -> Optional[str]:
        """Get the end effector link (last link in chain)."""
        # Find link with no children
        child_links = {j.child for j in self.joints.values()}
        parent_links = {j.parent for j in self.joints.values()}
        end_effectors = parent_links - child_links
        return next(iter(end_effectors), None)

    def to_llm_context(self) -> str:
        """
        Generate natural language description for LLM grounding.

        This solves the Symbol Grounding Problem by translating
        the robot's physical form into language the LLM can understand.
        """
        lines = [
            f"Robot: {self.name}",
            f"Degrees of Freedom: {len(self.get_actuated_joints())}",
            f"Joints: {', '.join(self.get_actuated_joints())}",
            "",
            "Joint Limits:",
        ]
        for name, (lower, upper) in self.get_joint_limits().items():
            lines.append(f"  {name}: [{lower:.3f}, {upper:.3f}] rad")

        lines.extend(["", "Links:"])
        for name, link in self.links.items():
            tags = f" ({', '.join(link.semantic_tags)})" if link.semantic_tags else ""
            lines.append(f"  {name}: mass={link.mass:.3f}kg{tags}")

        if self.sensors:
            lines.extend(["", "Sensors:"])
            for name, sensor in self.sensors.items():
                lines.append(f"  {name}: {sensor.sensor_type} on {sensor.parent_link}")

        return "\n".join(lines)


class EURDFParser:
    """
    Parser for extended URDF (e-URDF) files.

    Loads robot models with semantic annotations for LLM grounding.
    Falls back to standard URDF if extensions are not present.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model: Optional[RobotModel] = None
        self._parse()

    def _parse(self) -> None:
        """Parse the e-URDF file."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        tree = ET.parse(self.model_path)
        root = tree.getroot()

        if root.tag != "robot":
            raise ValueError(f"Expected root element 'robot', got '{root.tag}'")

        self.model = RobotModel(name=root.get("name", "unknown"))

        # Parse links
        for link_elem in root.findall("link"):
            self._parse_link(link_elem)

        # Parse joints
        for joint_elem in root.findall("joint"):
            self._parse_joint(joint_elem)

        # Parse sensors (e-URDF extension)
        for sensor_elem in root.findall("sensor"):
            self._parse_sensor(sensor_elem)

        # Parse metadata (e-URDF extension)
        metadata_elem = root.find("metadata")
        if metadata_elem is not None:
            self._parse_metadata(metadata_elem)

    def _parse_link(self, elem: ET.Element) -> None:
        """Parse a link element."""
        name = elem.get("name", "")
        link = LinkSpec(name=name)

        # Visual mesh
        visual = elem.find("visual")
        if visual is not None:
            mesh = visual.find("mesh")
            if mesh is not None:
                link.visual_mesh = mesh.get("filename")
            material = visual.find("material")
            if material is not None:
                link.material = material.get("name")

        # Collision mesh
        collision = elem.find("collision")
        if collision is not None:
            mesh = collision.find("mesh")
            if mesh is not None:
                link.collision_mesh = mesh.get("filename")

        # Inertial properties
        inertial = elem.find("inertial")
        if inertial is not None:
            mass = inertial.find("mass")
            if mass is not None:
                link.mass = float(mass.get("value", 0))
            inertia = inertial.find("inertia")
            if inertia is not None:
                link.inertia = np.array([
                    [float(inertia.get("ixx", 1)), float(inertia.get("ixy", 0)), float(inertia.get("ixz", 0))],
                    [float(inertia.get("ixy", 0)), float(inertia.get("iyy", 1)), float(inertia.get("iyz", 0))],
                    [float(inertia.get("ixz", 0)), float(inertia.get("iyz", 0)), float(inertia.get("izz", 1))],
                ])

        # e-URDF extensions: semantic annotations
        semantic = elem.find("semantic")
        if semantic is not None:
            for tag in semantic.findall("tag"):
                link.semantic_tags.append(tag.get("name", ""))
            for grasp in semantic.findall("grasp_point"):
                link.grasp_points.append({
                    "name": grasp.get("name", ""),
                    "position": [
                        float(grasp.get("x", 0)),
                        float(grasp.get("y", 0)),
                        float(grasp.get("z", 0)),
                    ],
                    "approach": grasp.get("approach", "top"),
                })

        self.model.links[name] = link

    def _parse_joint(self, elem: ET.Element) -> None:
        """Parse a joint element."""
        name = elem.get("name", "")
        joint = JointSpec(
            name=name,
            joint_type=elem.get("type", "fixed"),
            parent=elem.find("parent").get("link") if elem.find("parent") is not None else "",
            child=elem.find("child").get("link") if elem.find("child") is not None else "",
        )

        # Origin
        origin = elem.find("origin")
        if origin is not None:
            joint.origin = self._parse_origin(origin)

        # Axis
        axis = elem.find("axis")
        if axis is not None:
            xyz = axis.get("xyz", "0 0 1").split()
            joint.axis = np.array([float(x) for x in xyz])

        # Limits
        limit = elem.find("limit")
        if limit is not None:
            joint.limits = {
                "lower": float(limit.get("lower", 0)),
                "upper": float(limit.get("upper", 0)),
                "velocity": float(limit.get("velocity", 0)),
                "effort": float(limit.get("effort", 0)),
            }

        # Dynamics
        dynamics = elem.find("dynamics")
        if dynamics is not None:
            joint.dynamics = {
                "damping": float(dynamics.get("damping", 0)),
                "friction": float(dynamics.get("friction", 0)),
            }

        # Safety limits
        safety = elem.find("safety_controller")
        if safety is not None:
            joint.safety = {
                "lower": float(safety.get("soft_lower_limit", joint.limits.get("lower", 0))),
                "upper": float(safety.get("soft_upper_limit", joint.limits.get("upper", 0))),
                "k_position": float(safety.get("k_position", 0)),
                "k_velocity": float(safety.get("k_velocity", 0)),
            }

        # e-URDF extension: control parameters
        control = elem.find("control")
        if control is not None:
            joint.control = {
                "p": float(control.get("p", 100)),
                "i": float(control.get("i", 10)),
                "d": float(control.get("d", 10)),
                "imax": float(control.get("imax", 1)),
            }

        self.model.joints[name] = joint

    def _parse_sensor(self, elem: ET.Element) -> None:
        """Parse a sensor element (e-URDF extension)."""
        name = elem.get("name", "")
        sensor = SensorSpec(
            name=name,
            sensor_type=elem.get("type", "unknown"),
            parent_link=elem.get("parent_link", ""),
        )

        origin = elem.find("origin")
        if origin is not None:
            sensor.origin = self._parse_origin(origin)

        for param in elem.findall("parameter"):
            sensor.parameters[param.get("name", "")] = param.get("value", "")

        self.model.sensors[name] = sensor

    def _parse_metadata(self, elem: ET.Element) -> None:
        """Parse metadata element (e-URDF extension)."""
        for child in elem:
            self.model.metadata[child.tag] = child.text or ""

    def _parse_origin(self, elem: ET.Element) -> np.ndarray:
        """Parse origin element to 4x4 homogeneous matrix."""
        xyz = elem.get("xyz", "0 0 0").split()
        rpy = elem.get("rpy", "0 0 0").split()

        x, y, z = [float(v) for v in xyz]
        roll, pitch, yaw = [float(v) for v in rpy]

        # Build rotation matrix from RPY
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([  # noqa: E226
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],  # noqa: E226
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],  # noqa: E226
            [-sp, cp*sr, cp*cr],  # noqa: E226
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    def get_model(self) -> RobotModel:
        """Get the parsed robot model."""
        if self.model is None:
            raise RuntimeError("Model not parsed")
        return self.model

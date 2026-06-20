"""e-URDF Loader — Physical DNA Registry

Loads extended URDF directories and generates internal profiles:
- RobotEmbodimentProfile
- RobotSafetyProfile
- RobotCapabilityProfile
- RobotSimulationProfile
- RobotSemanticProfile
"""

from __future__ import annotations

import importlib.resources as resources
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ── Default zoo resolution ──

def _packaged_zoo_path() -> Path | None:
    """Return the path to e-urdf-zoo data shipped inside the rosclaw package, if any."""
    try:
        ref = resources.files("rosclaw") / "eurdf_zoo_data"
        if ref.is_dir():
            return Path(str(ref))
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        pass
    return None


def _default_zoo_path() -> Path:
    """Default to project-root zoo; fall back to packaged data when installed."""
    project_root_zoo = Path(__file__).parent.parent.parent.parent / "e-urdf-zoo"
    if project_root_zoo.is_dir():
        return project_root_zoo
    packaged = _packaged_zoo_path()
    if packaged is not None:
        return packaged
    # No packaged data available; keep the project-root path for error messages.
    return project_root_zoo

# ── Profile Data Classes ──

@dataclass
class RobotEmbodimentProfile:
    """Physical form: links, joints, sensors, actuators, DOF."""
    robot_id: str
    name: str
    vendor: str
    version: str
    description: str
    dof: int
    links: list[dict] = field(default_factory=list)
    joints: list[dict] = field(default_factory=list)
    sensors: list[dict] = field(default_factory=list)
    actuators: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "vendor": self.vendor,
            "version": self.version,
            "description": self.description,
            "dof": self.dof,
            "links": self.links,
            "joints": self.joints,
            "sensors": self.sensors,
            "actuators": self.actuators,
            "metadata": self.metadata,
        }


@dataclass
class RobotSafetyProfile:
    """Safety constraints: limits, PFL, collision, e-stop."""
    robot_id: str
    safety_level: str
    safety_limits: dict = field(default_factory=dict)
    joint_soft_limits: dict = field(default_factory=dict)
    pfl: dict = field(default_factory=dict)
    collision_detection: dict = field(default_factory=dict)
    emergency_stop: dict = field(default_factory=dict)
    workspace_boundaries: dict = field(default_factory=dict)
    interaction: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "safety_level": self.safety_level,
            "safety_limits": self.safety_limits,
            "joint_soft_limits": self.joint_soft_limits,
            "pfl": self.pfl,
            "collision_detection": self.collision_detection,
            "emergency_stop": self.emergency_stop,
            "workspace_boundaries": self.workspace_boundaries,
            "interaction": self.interaction,
            "environment": self.environment,
        }


@dataclass
class RobotCapabilityProfile:
    """What the robot can do: skills, preconditions, metrics."""
    robot_id: str
    capabilities: list[dict] = field(default_factory=list)
    skill_registry: dict = field(default_factory=dict)
    precondition_checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "capabilities": self.capabilities,
            "skill_registry": self.skill_registry,
            "precondition_checks": self.precondition_checks,
        }


@dataclass
class RobotSimulationProfile:
    """Simulation backend configurations."""
    robot_id: str
    backends: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"robot_id": self.robot_id, "backends": self.backends}


@dataclass
class RobotSemanticProfile:
    """Semantic annotations for LLM/VLM grounding."""
    robot_id: str
    semantic_version: str = "1.0"
    functional_regions: list[dict] = field(default_factory=list)
    grasp_points: list[dict] = field(default_factory=list)
    visual_features: list[dict] = field(default_factory=list)
    task_descriptions: dict = field(default_factory=dict)
    semantic_tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "semantic_version": self.semantic_version,
            "functional_regions": self.functional_regions,
            "grasp_points": self.grasp_points,
            "visual_features": self.visual_features,
            "task_descriptions": self.task_descriptions,
            "semantic_tags": self.semantic_tags,
        }


@dataclass
class RobotBenchmarkProfile:
    """Benchmark configurations for regression testing."""
    robot_id: str
    kinematic_benchmarks: list[dict] = field(default_factory=list)
    dynamic_benchmarks: list[dict] = field(default_factory=list)
    simulation_benchmarks: dict = field(default_factory=dict)
    task_benchmarks: list[dict] = field(default_factory=list)
    safety_benchmarks: list[dict] = field(default_factory=list)
    baseline_hardware: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "kinematic_benchmarks": self.kinematic_benchmarks,
            "dynamic_benchmarks": self.dynamic_benchmarks,
            "simulation_benchmarks": self.simulation_benchmarks,
            "task_benchmarks": self.task_benchmarks,
            "safety_benchmarks": self.safety_benchmarks,
            "baseline_hardware": self.baseline_hardware,
        }


@dataclass
class RobotCompleteProfile:
    """Aggregated complete profile of a robot."""
    robot_id: str
    name: str
    vendor: str
    version: str
    description: str
    embodiment: RobotEmbodimentProfile
    safety: RobotSafetyProfile
    capability: RobotCapabilityProfile
    simulation: RobotSimulationProfile
    semantic: RobotSemanticProfile
    benchmark: RobotBenchmarkProfile

    def to_dict(self) -> dict:
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "vendor": self.vendor,
            "version": self.version,
            "description": self.description,
            "embodiment": self.embodiment.to_dict(),
            "safety": self.safety.to_dict(),
            "capability": self.capability.to_dict(),
            "simulation": self.simulation.to_dict(),
            "semantic": self.semantic.to_dict(),
            "benchmark": self.benchmark.to_dict(),
        }


# ── Loader ──

class EURDFLoader:
    """Load e-URDF directory and generate all profiles."""

    REQUIRED_FILES = [
        "robot.eurdf.yaml",
        "safety.yaml",
        "semantic.yaml",
        "capabilities.yaml",
        "benchmark.yaml",
    ]

    def __init__(self, zoo_path: str | Path | None = None):
        if zoo_path is None:
            # Default: project-root zoo when developing, packaged data when installed.
            self.zoo_path = _default_zoo_path()
        else:
            self.zoo_path = Path(zoo_path)

    def list_robots(self) -> list[str]:
        """List all robot directories in the zoo."""
        if not self.zoo_path.exists():
            return []
        return sorted([
            d.name for d in self.zoo_path.iterdir()
            if d.is_dir() and (d / "robot.eurdf.yaml").exists()
        ])

    def load(self, robot_id: str) -> RobotCompleteProfile:
        """Load a complete robot profile from the zoo."""
        robot_dir = self.zoo_path / robot_id
        if not robot_dir.exists():
            raise FileNotFoundError(f"Robot '{robot_id}' not found in {self.zoo_path}")

        # Load core e-URDF
        eurdf_path = robot_dir / "robot.eurdf.yaml"
        with open(eurdf_path, encoding="utf-8") as f:
            eurdf = yaml.safe_load(f)

        # Load sub-profiles
        safety = self._load_yaml(robot_dir / "safety.yaml") or {}
        semantic = self._load_yaml(robot_dir / "semantic.yaml") or {}
        capabilities = self._load_yaml(robot_dir / "capabilities.yaml") or {}
        benchmark = self._load_yaml(robot_dir / "benchmark.yaml") or {}

        # Build Embodiment Profile
        embodiment = RobotEmbodimentProfile(
            robot_id=eurdf.get("robot_id", robot_id),
            name=eurdf.get("name", robot_id),
            vendor=eurdf.get("vendor", "unknown"),
            version=str(eurdf.get("version", "1.0")),
            description=eurdf.get("description", ""),
            dof=eurdf.get("dof", 0),
            links=eurdf.get("links", []),
            joints=eurdf.get("joints", []),
            sensors=eurdf.get("sensors", []),
            actuators=eurdf.get("actuators", []),
            metadata=eurdf.get("metadata", {}),
        )

        # Build Safety Profile
        safety_profile = RobotSafetyProfile(
            robot_id=robot_id,
            safety_level=safety.get("safety_level", "MODERATE"),
            safety_limits=safety.get("safety_limits", {}),
            joint_soft_limits=safety.get("joint_soft_limits", {}),
            pfl=safety.get("pfl", {}),
            collision_detection=safety.get("collision_detection", {}),
            emergency_stop=safety.get("emergency_stop", {}),
            workspace_boundaries=safety.get("workspace_boundaries", {}),
            interaction=safety.get("interaction", {}),
            environment=safety.get("environment", {}),
        )

        # Build Capability Profile
        capability_profile = RobotCapabilityProfile(
            robot_id=robot_id,
            capabilities=capabilities.get("capabilities", []),
            skill_registry=capabilities.get("skill_registry", {}),
            precondition_checks=capabilities.get("precondition_checks", {}),
        )

        # Build Simulation Profile
        simulation_profile = RobotSimulationProfile(
            robot_id=robot_id,
            backends=eurdf.get("simulation_backends", {}),
        )

        # Build Semantic Profile
        semantic_profile = RobotSemanticProfile(
            robot_id=robot_id,
            semantic_version=semantic.get("semantic_version", "1.0"),
            functional_regions=semantic.get("functional_regions", []),
            grasp_points=semantic.get("grasp_points", []),
            visual_features=semantic.get("visual_features", []),
            task_descriptions=semantic.get("task_descriptions", {}),
            semantic_tags=eurdf.get("semantic_tags", []),
        )

        # Build Benchmark Profile
        benchmark_profile = RobotBenchmarkProfile(
            robot_id=embodiment.robot_id,
            kinematic_benchmarks=benchmark.get("kinematic_benchmarks", []),
            dynamic_benchmarks=benchmark.get("dynamic_benchmarks", []),
            simulation_benchmarks=benchmark.get("simulation_benchmarks", {}),
            task_benchmarks=benchmark.get("task_benchmarks", []),
            safety_benchmarks=benchmark.get("safety_benchmarks", []),
            baseline_hardware=benchmark.get("baseline_hardware", {}),
        )

        return RobotCompleteProfile(
            robot_id=embodiment.robot_id,
            name=embodiment.name,
            vendor=embodiment.vendor,
            version=embodiment.version,
            description=embodiment.description,
            embodiment=embodiment,
            safety=safety_profile,
            capability=capability_profile,
            simulation=simulation_profile,
            semantic=semantic_profile,
            benchmark=benchmark_profile,
        )

    def validate(self, robot_id: str) -> dict:
        """Validate e-URDF completeness for a robot."""
        robot_dir = self.zoo_path / robot_id
        result = {
            "robot_id": robot_id,
            "valid": True,
            "errors": [],
            "warnings": [],
            "files_found": [],
            "files_missing": [],
        }

        if not robot_dir.exists():
            result["valid"] = False
            result["errors"].append(f"Robot directory not found: {robot_dir}")
            return result

        for req_file in self.REQUIRED_FILES:
            path = robot_dir / req_file
            if path.exists():
                result["files_found"].append(req_file)
            else:
                result["files_missing"].append(req_file)
                result["valid"] = False
                result["errors"].append(f"Missing required file: {req_file}")

        # Optional files
        for opt_file in ["robot.urdf", "robot.mjcf.xml", "assets"]:
            path = robot_dir / opt_file
            exists = path.exists()
            result["files_found" if exists else "files_missing"].append(opt_file)
            if not exists:
                result["warnings"].append(f"Optional file/directory missing: {opt_file}")

        # Deep validation of robot.eurdf.yaml
        eurdf_path = robot_dir / "robot.eurdf.yaml"
        if eurdf_path.exists():
            try:
                with open(eurdf_path, encoding="utf-8") as f:
                    eurdf = yaml.safe_load(f)

                required_fields = ["robot_id", "name", "vendor", "dof", "links", "joints"]
                for fld in required_fields:
                    if fld not in eurdf or eurdf[fld] is None:
                        result["valid"] = False
                        result["errors"].append(f"robot.eurdf.yaml missing field: {fld}")

                if "sensors" not in eurdf:
                    result["warnings"].append("No sensors defined in robot.eurdf.yaml")
                if "actuators" not in eurdf:
                    result["warnings"].append("No actuators defined in robot.eurdf.yaml")
                if "capabilities" not in eurdf:
                    result["warnings"].append("No capabilities defined in robot.eurdf.yaml")
                if "simulation_backends" not in eurdf:
                    result["warnings"].append("No simulation_backends defined")

            except Exception as exc:
                result["valid"] = False
                result["errors"].append(f"Failed to parse robot.eurdf.yaml: {exc}")

        return result

    @staticmethod
    def _load_yaml(path: Path) -> dict | None:
        """Load a YAML file if it exists, otherwise return None.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed dict or None if the file does not exist.
        """
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)


# ── Registry ──

class RobotRegistry:
    """In-memory registry of loaded robot profiles."""

    def __init__(self, loader: EURDFLoader | None = None):
        self.loader = loader or EURDFLoader()
        self._profiles: dict[str, RobotCompleteProfile] = {}

    def install(self, robot_id: str) -> RobotCompleteProfile:
        """Load and register a robot profile."""
        profile = self.loader.load(robot_id)
        # Store under both the directory name and the canonical robot_id
        self._profiles[robot_id] = profile
        self._profiles[profile.robot_id] = profile
        return profile

    def get(self, robot_id: str) -> RobotCompleteProfile | None:
        """Get a registered profile (auto-install if not cached)."""
        if robot_id not in self._profiles:
            try:
                self.install(robot_id)
            except FileNotFoundError:
                return None
        return self._profiles.get(robot_id)

    def list(self) -> list[str]:
        """List all registered robot IDs."""
        return list(self._profiles.keys())

    def list_available(self) -> list[str]:
        """List all available robots in the zoo."""
        return self.loader.list_robots()

    def validate(self, robot_id: str) -> dict:
        """Validate a robot's e-URDF."""
        return self.loader.validate(robot_id)

    def inspect(self, robot_id: str) -> dict:
        """Return complete profile as a dictionary."""
        profile = self.get(robot_id)
        if profile is None:
            raise FileNotFoundError(f"Robot '{robot_id}' not found")
        return profile.to_dict()

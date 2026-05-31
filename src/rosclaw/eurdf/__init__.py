"""ROSClaw e-URDF — Extended URDF Robot Profile System.

Unified package for robot physical-DNA registry:
- models: Data classes for embodiment, safety, capability, simulation, semantic, benchmark
- loader: EURDFLoader — load robot profiles from zoo directories
- registry: RobotRegistry — in-memory registry with auto-install
- zoo: Built-in robot presets (ur5e, panda, go2, fetch, ...)

Usage:
    from rosclaw.eurdf import RobotRegistry, EURDFLoader
    registry = RobotRegistry()
    profile = registry.install("ur5e")
"""

from rosclaw.eurdf.loader import EURDFLoader
from rosclaw.eurdf.models import (
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)
from rosclaw.eurdf.registry import RobotRegistry

__all__ = [
    "EURDFLoader",
    "RobotRegistry",
    "RobotEmbodimentProfile",
    "RobotSafetyProfile",
    "RobotCapabilityProfile",
    "RobotSimulationProfile",
    "RobotSemanticProfile",
    "RobotBenchmarkProfile",
    "RobotCompleteProfile",
]

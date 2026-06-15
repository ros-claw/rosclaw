"""e-URDF-Zoo — Extended URDF Robot Profile Registry."""

from rosclaw.runtime.eurdf_loader import (
    EURDFLoader,
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotRegistry,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)

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

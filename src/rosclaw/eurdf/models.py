"""e-URDF Data Models — Robot Physical DNA.

Re-exports canonical profile dataclasses from runtime.eurdf_loader
so consumers can import from the unified `rosclaw.eurdf` namespace.
"""

from rosclaw.runtime.eurdf_loader import (
    RobotBenchmarkProfile,
    RobotCapabilityProfile,
    RobotCompleteProfile,
    RobotEmbodimentProfile,
    RobotSafetyProfile,
    RobotSemanticProfile,
    RobotSimulationProfile,
)

__all__ = [
    "RobotEmbodimentProfile",
    "RobotSafetyProfile",
    "RobotCapabilityProfile",
    "RobotSimulationProfile",
    "RobotSemanticProfile",
    "RobotBenchmarkProfile",
    "RobotCompleteProfile",
]

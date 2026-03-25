"""ROSClaw Simulation Layer - Digital Twin powered by mjlab (MuJoCo Warp)."""

from __future__ import annotations

from rosclaw_sim.digital_twin import DigitalTwin, DigitalTwinConfig
from rosclaw_sim.mjlab_env import MjlabEnv, MjlabEnvConfig
from rosclaw_sim.robot_loader import RobotLoader, RobotSpec
from rosclaw_sim.domain_randomization import DomainRandomization, RandomizationConfig
from rosclaw_sim.verify import TrajectoryVerifier, VerificationResult

__version__ = "0.1.0"

__all__ = [
    # Core Digital Twin
    "DigitalTwin",
    "DigitalTwinConfig",
    # Environment wrapper
    "MjlabEnv",
    "MjlabEnvConfig",
    # Robot loading
    "RobotLoader",
    "RobotSpec",
    # Domain randomization
    "DomainRandomization",
    "RandomizationConfig",
    # Trajectory verification
    "TrajectoryVerifier",
    "VerificationResult",
]

"""ROSClaw Core Definitions.

Robot and assembly manifest schemas for hardware specification.
"""

from rosclaw_core.definitions.robot import (
    RobotManifest,
    RobotCapabilities,
    JointConfig,
    JointType,
    JointLimits,
    SensorConfig,
    SensorType,
)

from rosclaw_core.definitions.assembly import (
    AssemblyManifest,
    AgentBinding,
    AgentRole,
    CoordinationMode,
    SynchronizationConfig,
)

__all__ = [
    # Robot
    "RobotManifest",
    "RobotCapabilities",
    "JointConfig",
    "JointType",
    "JointLimits",
    "SensorConfig",
    "SensorType",
    # Assembly
    "AssemblyManifest",
    "AgentBinding",
    "AgentRole",
    "CoordinationMode",
    "SynchronizationConfig",
]

"""ROSClaw Core - Embodied Intelligence Operating System.

This package provides the core integration layer between RoboClaw's embodied
intelligence system and ROS/ROS2-based robotics platforms.

Key Components:
    - definitions: Robot/Assembly/Sensor manifest schemas
    - runtime: Session management and procedure execution
    - adapters: Hardware abstraction layer for ROS/ROS2 integration
    - builtins: Pre-configured robot definitions (SO101, etc.)
"""

from __future__ import annotations

from rosclaw_core.definitions.robot import RobotManifest, RobotCapabilities
from rosclaw_core.definitions.assembly import AssemblyManifest, AgentRole
from rosclaw_core.runtime.manager import RuntimeManager
from rosclaw_core.runtime.session import RuntimeSession
from rosclaw_core.adapters.base import RobotAdapter, AdapterProtocol

__version__ = "4.0.0"

__all__ = [
    # Definitions
    "RobotManifest",
    "RobotCapabilities",
    "AssemblyManifest",
    "AgentRole",
    # Runtime
    "RuntimeManager",
    "RuntimeSession",
    # Adapters
    "RobotAdapter",
    "AdapterProtocol",
]

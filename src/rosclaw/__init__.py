"""
ROSClaw - The Universal Operating System for Software-Defined Embodied AI.

This package provides production-ready middleware for connecting LLMs to physical robots.

Architecture:
    Agent Runtime (LLM/MCP)
         |
         v
    ROSClaw Runtime (core.runtime)
    |-- EventBus (core.event_bus)
    |-- Firewall (firewall) - Action Grounding
    |-- Memory (memory) - Experience Grounding
    |-- Practice (practice) - Timeline Grounding
    |-- Swarm (swarm) - Collaboration Grounding
    |-- SkillManager (skill_manager) - Skill Grounding
    |-- e-URDF (e_urdf) - Physical Grounding
    |-- MCP Drivers (mcp_drivers) - Hardware abstraction
         |
         v
    Physical World (Robot)
"""

__version__ = "1.0.0"
__author__ = "ROSClaw Team"

# Enable namespace extension so rosclaw-sandbox and other
# sibling packages can add sub-packages under rosclaw.*
from pkgutil import extend_path as _extend_path
__path__ = _extend_path(__path__, __name__)

# Core OS (always available)
from rosclaw.core import (
    EventBus,
    Event,
    EventPriority,
    Runtime,
    RuntimeConfig,
    LifecycleState,
    LifecycleMixin,
)

# Grounding Engines (optional imports for environments without all deps)
try:
    from rosclaw.firewall.decorator import (
        DigitalTwinFirewall,
        SafetyLevel,
        SafetyViolationError,
        mujoco_firewall,
    )
except ImportError:
    DigitalTwinFirewall = None  # type: ignore
    SafetyLevel = None  # type: ignore
    SafetyViolationError = None  # type: ignore
    mujoco_firewall = None  # type: ignore

from rosclaw.e_urdf import EURDFParser, RobotModel
from rosclaw.agent_runtime import (
    MCPHub,
    AgentContext,
    LLMProvider,
    LLMConfig,
    DeepSeekProvider,
    OpenAIProvider,
    QwenProvider,
    get_provider,
    list_providers,
    register_provider,
    # Backward-compatible aliases
    DeepSeekClient,
    DeepSeekConfig,
)
from rosclaw.memory import MemoryInterface
from rosclaw.practice import PracticeRecorder
from rosclaw.swarm import SwarmRuntimeManager
from rosclaw.skill_manager import SkillRegistry, SkillEntry, SkillExecutor, SkillLoader
from rosclaw.mcp_drivers import BaseDriver, DriverState, TrajectoryCommand

# Provider Layer (optional - available when provider skeleton is installed)
try:
    from rosclaw.provider.core.provider import Provider
    from rosclaw.provider.core.registry import ProviderRegistry
    from rosclaw.provider.core.router import CapabilityRouter
    from rosclaw.provider.core.manifest import ProviderManifest
    from rosclaw.provider.core.request import ProviderRequest
    from rosclaw.provider.core.response import ProviderResponse
    from rosclaw.provider.guard.pipeline import GuardPipeline
    from rosclaw.provider.client import CapabilityClient
    from rosclaw.provider.loader import ProviderLoader
    from rosclaw.provider.adapters.generic import GenericProvider
except ImportError:
    Provider = None  # type: ignore
    ProviderRegistry = None  # type: ignore
    CapabilityRouter = None  # type: ignore
    ProviderManifest = None  # type: ignore
    ProviderRequest = None  # type: ignore
    ProviderResponse = None  # type: ignore
    GuardPipeline = None  # type: ignore
    CapabilityClient = None  # type: ignore
    ProviderLoader = None  # type: ignore
    GenericProvider = None  # type: ignore

__all__ = [
    # Core
    "EventBus",
    "Event",
    "EventPriority",
    "Runtime",
    "RuntimeConfig",
    "LifecycleState",
    "LifecycleMixin",
    # Grounding Engines
    "DigitalTwinFirewall",
    "SafetyLevel",
    "SafetyViolationError",
    "mujoco_firewall",
    "EURDFParser",
    "RobotModel",
    "MCPHub",
    "AgentContext",
    # LLM Providers
    "LLMProvider",
    "LLMConfig",
    "DeepSeekProvider",
    "OpenAIProvider",
    "QwenProvider",
    "get_provider",
    "list_providers",
    "register_provider",
    # Backward-compatible aliases
    "DeepSeekClient",
    "DeepSeekConfig",
    "MemoryInterface",
    "PracticeRecorder",
    "SwarmRuntimeManager",
    "SkillRegistry",
    "SkillEntry",
    "SkillExecutor",
    "SkillLoader",
    # MCP Drivers
    "BaseDriver",
    "DriverState",
    "TrajectoryCommand",
    # Provider Layer
    "Provider",
    "ProviderRegistry",
    "CapabilityRouter",
    "ProviderManifest",
    "ProviderRequest",
    "ProviderResponse",
    "GuardPipeline",
    "CapabilityClient",
    "ProviderLoader",
    "GenericProvider",
]

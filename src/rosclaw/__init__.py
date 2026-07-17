"""ROSClaw trustworthy physical execution runtime.

Public compatibility exports are loaded lazily.  This keeps lightweight
subprocesses, including isolated policy workers, independent from unrelated
robotics and control-plane dependencies.
"""

from __future__ import annotations

import importlib
from pkgutil import extend_path as _extend_path
from typing import Any

__version__ = "1.0.1"
__author__ = "ROSClaw Team"

# Allow sibling distributions such as rosclaw-sandbox to extend rosclaw.*.
__path__ = _extend_path(__path__, __name__)

_EXPORTS = {
    "Event": ("core", "Event"),
    "EventBus": ("core", "EventBus"),
    "EventPriority": ("core", "EventPriority"),
    "LifecycleMixin": ("core", "LifecycleMixin"),
    "LifecycleState": ("core", "LifecycleState"),
    "Runtime": ("core", "Runtime"),
    "RuntimeConfig": ("core", "RuntimeConfig"),
    "DigitalTwinFirewall": ("firewall.decorator", "DigitalTwinFirewall"),
    "SafetyLevel": ("firewall.decorator", "SafetyLevel"),
    "SafetyViolationError": ("firewall.decorator", "SafetyViolationError"),
    "mujoco_firewall": ("firewall.decorator", "mujoco_firewall"),
    "AgentContext": ("agent_runtime", "AgentContext"),
    "DeepSeekClient": ("agent_runtime", "DeepSeekClient"),
    "DeepSeekConfig": ("agent_runtime", "DeepSeekConfig"),
    "DeepSeekProvider": ("agent_runtime", "DeepSeekProvider"),
    "LLMConfig": ("agent_runtime", "LLMConfig"),
    "LLMProvider": ("agent_runtime", "LLMProvider"),
    "MCPHub": ("agent_runtime", "MCPHub"),
    "OpenAIProvider": ("agent_runtime", "OpenAIProvider"),
    "QwenProvider": ("agent_runtime", "QwenProvider"),
    "get_provider": ("agent_runtime", "get_provider"),
    "list_providers": ("agent_runtime", "list_providers"),
    "register_provider": ("agent_runtime", "register_provider"),
    "EURDFParser": ("e_urdf", "EURDFParser"),
    "RobotModel": ("e_urdf", "RobotModel"),
    "BaseDriver": ("mcp_drivers", "BaseDriver"),
    "DriverState": ("mcp_drivers", "DriverState"),
    "TrajectoryCommand": ("mcp_drivers", "TrajectoryCommand"),
    "MemoryInterface": ("memory", "MemoryInterface"),
    "PracticeRecorder": ("practice", "PracticeRecorder"),
    "SkillEntry": ("skill_manager", "SkillEntry"),
    "SkillExecutor": ("skill_manager", "SkillExecutor"),
    "SkillLoader": ("skill_manager", "SkillLoader"),
    "SkillRegistry": ("skill_manager", "SkillRegistry"),
    "SwarmRuntimeManager": ("swarm", "SwarmRuntimeManager"),
    "GenericProvider": ("provider.adapters.generic", "GenericProvider"),
    "CapabilityClient": ("provider.client", "CapabilityClient"),
    "ProviderManifest": ("provider.core.manifest", "ProviderManifest"),
    "Provider": ("provider.core.provider", "Provider"),
    "ProviderRegistry": ("provider.core.registry", "ProviderRegistry"),
    "ProviderRequest": ("provider.core.request", "ProviderRequest"),
    "ProviderResponse": ("provider.core.response", "ProviderResponse"),
    "CapabilityRouter": ("provider.core.router", "CapabilityRouter"),
    "GuardPipeline": ("provider.guard.pipeline", "GuardPipeline"),
    "ProviderLoader": ("provider.loader", "ProviderLoader"),
}

_OPTIONAL_EXPORTS = {
    "DigitalTwinFirewall",
    "SafetyLevel",
    "SafetyViolationError",
    "mujoco_firewall",
    "GenericProvider",
    "CapabilityClient",
    "ProviderManifest",
    "Provider",
    "ProviderRegistry",
    "ProviderRequest",
    "ProviderResponse",
    "CapabilityRouter",
    "GuardPipeline",
    "ProviderLoader",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    try:
        module = importlib.import_module(f"{__name__}.{module_name}")
        value = getattr(module, attribute)
    except ImportError:
        if name not in _OPTIONAL_EXPORTS:
            raise
        value = None
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

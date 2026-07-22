"""rosclaw.sandbox namespace for v1.0 integration.

Exports:
    SandboxSession, Sandbox          — MuJoCo-backed physics sandbox
    SandboxRuntimeAdapter            — Runtime lifecycle integration
    Decision, StaticActionGate       — Static policy validation (no physics)
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .firewall.gate import Decision, FirewallGate, StaticActionGate
from .runtime_adapter import SandboxRuntimeAdapter
from .sandbox_api import Sandbox, SandboxSession

__all__ = [
    "SandboxSession",
    "Sandbox",
    "SandboxRuntimeAdapter",
    "Decision",
    "FirewallGate",
    "StaticActionGate",
]

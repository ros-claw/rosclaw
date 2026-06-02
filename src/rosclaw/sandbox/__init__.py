"""rosclaw.sandbox namespace for v1.0 integration.

Exports:
    SandboxSession, Sandbox          — MuJoCo-backed physics sandbox
    SandboxRuntimeAdapter            — Runtime lifecycle integration
    Decision, FirewallGate           — Dynamic trajectory safety validation
"""
from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)

from .sandbox_api import SandboxSession, Sandbox
from .runtime_adapter import SandboxRuntimeAdapter
from .firewall.gate import Decision, FirewallGate

__all__ = [
    "SandboxSession",
    "Sandbox",
    "SandboxRuntimeAdapter",
    "Decision",
    "FirewallGate",
]

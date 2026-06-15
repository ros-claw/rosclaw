"""rosclaw.sandbox namespace for v1.0 integration.

Exports:
    SandboxSession, Sandbox          — MuJoCo-backed physics sandbox
    SandboxRuntimeAdapter            — Runtime lifecycle integration
    Decision, FirewallGate           — Dynamic trajectory safety validation
"""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .firewall.gate import Decision, FirewallGate
from .runtime_adapter import SandboxRuntimeAdapter
from .sandbox_api import Sandbox, SandboxSession

__all__ = [
    "SandboxSession",
    "Sandbox",
    "SandboxRuntimeAdapter",
    "Decision",
    "FirewallGate",
]

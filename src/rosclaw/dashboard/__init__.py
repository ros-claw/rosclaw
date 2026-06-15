"""ROSClaw Dashboard — Real-time monitoring and control plane.

Provides WebSocket streaming and HTTP API for:
- Runtime health and module status
- e-URDF robot registry
- Provider call metrics
- Sandbox validation results
- Practice episode timeline
- EventBus live event stream
"""

from .metrics import DashboardMetrics
from .server import DashboardServer

__all__ = ["DashboardServer", "DashboardMetrics"]

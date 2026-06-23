"""ROSClaw Dashboard — Real-time monitoring and control plane.

Provides WebSocket streaming and HTTP API for:
- Runtime health and module status
- e-URDF robot registry
- Provider call metrics
- Sandbox validation results
- Practice episode timeline
- EventBus live event stream
"""

from .launcher import get_dashboard_app, serve_dashboard
from .metrics import DashboardMetrics
from .server import DashboardServer

__all__ = ["DashboardServer", "DashboardMetrics", "serve_dashboard", "get_dashboard_app"]

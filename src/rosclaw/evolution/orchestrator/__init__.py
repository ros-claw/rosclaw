"""rosclaw.auto — ROSClaw Self-Evolution Control Plane."""

from .config import AutoConfig
from .engine.auto_engine import AutoEngine
from .plugin import AutoPlugin

__version__ = "1.0.0"
__all__ = ["AutoConfig", "AutoEngine", "AutoPlugin", "DashboardExporter", "ReportGenerator"]

from .dashboard import DashboardExporter
from .reports import ReportGenerator

"""rosclaw.darwin — Evaluation Pressure Module.

Darwin applies multi-seed benchmark stress to skills and detects
regressions before they reach real robots.
"""
from .engine import DarwinEngine
from .events import DarwinBenchmarkCompletedEvent
from .plugin import DarwinPlugin

__all__ = ["DarwinPlugin", "DarwinEngine", "DarwinBenchmarkCompletedEvent"]

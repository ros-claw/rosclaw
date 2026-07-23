"""Operating regime modeling (数据库优化v4 §4, PR-MEM-6 commit 3/5).

Regime-aware memory: memories only apply inside the working conditions they
were observed/validated in.
"""

from .builder import CurrentRegimeBuilder
from .features import TelemetrySample, WindowStats, compute_window, compute_windows
from .models import (
    OperatingRegime,
    RegimeLabel,
    RegimeThresholds,
    empty_regime,
    load_thresholds,
)

__all__ = [
    "CurrentRegimeBuilder",
    "OperatingRegime",
    "RegimeLabel",
    "RegimeThresholds",
    "TelemetrySample",
    "WindowStats",
    "compute_window",
    "compute_windows",
    "empty_regime",
    "load_thresholds",
]

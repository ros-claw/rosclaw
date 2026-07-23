"""Operating regime modeling (数据库优化v4 §4, PR-MEM-6 commits 3-4/5).

Regime-aware memory: memories only apply inside the working conditions they
were observed/validated in — and are hard-blocked where they are
contraindicated.
"""

from .builder import CurrentRegimeBuilder
from .detector import RegimeChangeDetector, RegimeTransition
from .features import TelemetrySample, WindowStats, compute_window, compute_windows
from .models import (
    OperatingRegime,
    RegimeLabel,
    RegimeThresholds,
    empty_regime,
    load_thresholds,
)
from .persistence import REGIME_HISTORY_TABLE, ApplicabilityStore, RegimeHistoryStore

__all__ = [
    "REGIME_HISTORY_TABLE",
    "ApplicabilityStore",
    "CurrentRegimeBuilder",
    "OperatingRegime",
    "RegimeChangeDetector",
    "RegimeHistoryStore",
    "RegimeLabel",
    "RegimeThresholds",
    "RegimeTransition",
    "TelemetrySample",
    "WindowStats",
    "compute_window",
    "compute_windows",
    "empty_regime",
    "load_thresholds",
]

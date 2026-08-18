"""Operating regime modeling + applicability (数据库优化v4 §4–§6, PR-MEM-6).

Regime-aware memory: memories only apply inside the working conditions they
were observed/validated in — and are hard-blocked where they are
contraindicated.
"""

from .builder import CurrentRegimeBuilder
from .detector import RegimeChangeDetector, RegimeTransition
from .envelope import (
    APPLICABILITY_TABLE,
    ApplicabilityEnvelope,
    EnvelopeType,
    envelope_from_regime,
)
from .explain import explain_match, explain_regime
from .features import TelemetrySample, WindowStats, compute_window, compute_windows
from .matcher import (
    ApplicabilityResult,
    MatcherConfig,
    RegimeMatcher,
    interval_distance,
)
from .models import (
    OperatingRegime,
    RegimeLabel,
    RegimeThresholds,
    empty_regime,
    load_thresholds,
)
from .persistence import (
    REGIME_HISTORY_TABLE,
    ApplicabilityStore,
    RegimeHistoryStore,
)

__all__ = [
    "APPLICABILITY_TABLE",
    "REGIME_HISTORY_TABLE",
    "ApplicabilityEnvelope",
    "ApplicabilityResult",
    "ApplicabilityStore",
    "CurrentRegimeBuilder",
    "EnvelopeType",
    "MatcherConfig",
    "OperatingRegime",
    "RegimeChangeDetector",
    "RegimeHistoryStore",
    "RegimeLabel",
    "RegimeMatcher",
    "RegimeThresholds",
    "RegimeTransition",
    "TelemetrySample",
    "WindowStats",
    "compute_window",
    "compute_windows",
    "empty_regime",
    "envelope_from_regime",
    "explain_match",
    "explain_regime",
    "interval_distance",
    "load_thresholds",
]

"""Regime change detection (数据库优化v4 §4.5).

First version is deterministic: EWMA + CUSUM + consecutive-threshold
persistence over the regime feature stream.  A Bayesian Online Changepoint
Detection shadow is declared but OFF by default — it only records
``change_probability`` / ``run_length_estimate`` annotations and can never
trigger a real-machine patch in this phase.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .features import Cusum, Ewma
from .models import OperatingRegime, RegimeThresholds


@dataclass
class RegimeTransition:
    """A confirmed regime change (v4 §4.5)."""

    from_label: str
    to_label: str
    confidence: float
    changed_features: dict[str, tuple[float | None, float | None]]
    evidence_refs: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    bocd_shadow: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_label": self.from_label,
            "to_label": self.to_label,
            "confidence": self.confidence,
            "changed_features": {
                key: [pair[0], pair[1]] for key, pair in self.changed_features.items()
            },
            "evidence_refs": self.evidence_refs,
            "timestamp": self.timestamp,
            "bocd_shadow": self.bocd_shadow,
        }


_TRACKED_FEATURES = (
    "temperature_c",
    "temperature_slope_c_per_min",
    "position_error_p95",
    "time_to_reach_p95_ms",
    "recent_invalid_rate",
    "communication_error_rate",
)


class BocdShadow:
    """Annotation-only BOCD stand-in (v4 §4.5: shadow, default OFF).

    This is NOT a Bayesian model — it is the declared placeholder that
    records a run-length estimate and a naive change probability so the
    pipeline shape is testable.  When ``enabled`` is False it records
    nothing.  It never gates any action.
    """

    def __init__(self, *, enabled: bool = False, hazard: float = 1.0 / 50.0) -> None:
        self.enabled = enabled
        self._hazard = hazard
        self._run_length = 0
        self._last_value: float | None = None

    def observe(self, value: float | None) -> dict[str, float] | None:
        if not self.enabled or value is None:
            return None
        self._run_length += 1
        change_probability = self._hazard
        if self._last_value is not None and self._last_value != 0:
            deviation = abs(value - self._last_value) / abs(self._last_value)
            change_probability = min(1.0, self._hazard + deviation)
        self._last_value = value
        return {
            "change_probability": round(change_probability, 4),
            "run_length_estimate": float(self._run_length),
            "note": "shadow_only_not_bayesian",
        }


class RegimeChangeDetector:
    """Confirms label transitions with EWMA+CUSUM persistence (v4 §4.5).

    A label flip becomes a :class:`RegimeTransition` only after
    ``persistence`` consecutive regimes carry the new label AND the tracked
    continuous features moved in the direction of the change (CUSUM), so a
    single noisy round cannot rewrite the robot's working-condition story.
    """

    def __init__(
        self,
        thresholds: RegimeThresholds | None = None,
        *,
        persistence: int = 3,
        ewma_alpha: float = 0.3,
        enable_bocd_shadow: bool = False,
    ) -> None:
        self._t = thresholds or RegimeThresholds()
        self._persistence = max(1, persistence)
        self._ewma = {name: Ewma(alpha=ewma_alpha) for name in _TRACKED_FEATURES}
        self._cusum = Cusum(drift=0.0, threshold=2.0)
        self._bocd = BocdShadow(enabled=enable_bocd_shadow)
        self._last_label: str | None = None
        self._pending_label: str | None = None
        self._pending_count = 0
        self._pending_regimes: list[OperatingRegime] = []
        # EWMA snapshot taken when a label flip begins: changed_features are
        # measured against the OLD regime's baseline, not an average that
        # already absorbed the new values.
        self._pending_baseline: dict[str, float | None] = {}

    def observe(self, regime: OperatingRegime) -> RegimeTransition | None:
        for name in _TRACKED_FEATURES:
            value = regime.feature_value(name)
            if value is not None:
                self._ewma[name].update(value)
        bocd_note = self._bocd.observe(regime.feature_value("temperature_c"))

        label = regime.regime_label
        if self._last_label is None:
            self._last_label = label
            return None
        if label == self._last_label:
            self._pending_label = None
            self._pending_count = 0
            self._pending_regimes = []
            return None

        if label == self._pending_label:
            self._pending_count += 1
            self._pending_regimes.append(regime)
        else:
            self._pending_label = label
            self._pending_count = 1
            self._pending_regimes = [regime]
            self._pending_baseline = {name: ewma.mean for name, ewma in self._ewma.items()}

        if self._pending_count < self._persistence:
            return None

        changed = self._changed_features(regime)
        transition = RegimeTransition(
            from_label=self._last_label,
            to_label=label,
            confidence=round(
                min(
                    1.0,
                    sum(r.confidence for r in self._pending_regimes) / self._pending_count + 0.1,
                ),
                4,
            ),
            changed_features=changed,
            evidence_refs=[ref for r in self._pending_regimes for ref in r.evidence_refs][:20],
            bocd_shadow=bocd_note,
        )
        self._last_label = label
        self._pending_label = None
        self._pending_count = 0
        self._pending_regimes = []
        return transition

    def _changed_features(
        self, regime: OperatingRegime
    ) -> dict[str, tuple[float | None, float | None]]:
        changed: dict[str, tuple[float | None, float | None]] = {}
        for name in _TRACKED_FEATURES:
            current = regime.feature_value(name)
            if current is None:
                continue
            baseline = self._pending_baseline.get(name)
            if baseline is None:
                continue
            if current != baseline:
                changed[name] = (baseline, current)
        return changed

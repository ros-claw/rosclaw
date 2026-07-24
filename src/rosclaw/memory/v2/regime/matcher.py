"""RegimeMatcher — applicability of memories to the current regime (v4 §6).

Two layers:

1. Hard constraints (v4 §6.1) — any single one rejects outright; no soft
   score can ever override:

       robot/body · hardware revision · firmware · calibration profile ·
       control profile · task · joint · failure type ·
       CONTRAINDICATED envelope hit · missing required features

2. Continuous feature distance (v4 §6.2) — interval distances over
   temperature, slope, elapsed time, action count, position error, and
   recent failure rate, combined as ``exp(-Σ w_i × d_i)`` and weighted by
   envelope confidence, evidence, and exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .envelope import ApplicabilityEnvelope, EnvelopeType
from .models import OperatingRegime


@dataclass(frozen=True)
class MatcherConfig:
    """Weights/scales/thresholds for the matcher (configs/regimes/*.yaml)."""

    feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "temperature_c": 1.0,
            "temperature_slope_c_per_min": 0.8,
            "session_elapsed_sec": 0.3,
            "cumulative_action_count": 0.2,
            "position_error_p95": 0.6,
            "recent_failure_rate": 0.8,
        }
    )
    feature_scales: dict[str, float] = field(
        default_factory=lambda: {
            "temperature_c": 5.0,
            "temperature_slope_c_per_min": 0.2,
            "session_elapsed_sec": 1800.0,
            "cumulative_action_count": 500.0,
            "position_error_p95": 10.0,
            "recent_failure_rate": 0.2,
        }
    )
    abstain_below: float = 0.70
    suggest_below: float = 0.85

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MatcherConfig:
        return cls(
            feature_weights=dict(raw.get("feature_weights") or {}) or cls().feature_weights,
            feature_scales=dict(raw.get("feature_scales") or {}) or cls().feature_scales,
            abstain_below=float(raw.get("abstain_below", 0.70)),
            suggest_below=float(raw.get("suggest_below", 0.85)),
        )


@dataclass
class ApplicabilityResult:
    """One memory's applicability verdict (v4 §6.3)."""

    memory_id: str
    applicable: bool
    score: float

    hard_rejections: list[str] = field(default_factory=list)
    missing_required_features: list[str] = field(default_factory=list)

    matched_envelope_id: str | None = None
    envelope_type: str | None = None

    feature_scores: dict[str, float] = field(default_factory=dict)
    evidence_count: int = 0
    confidence: float = 0.0

    explanation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "applicable": self.applicable,
            "score": round(self.score, 4),
            "hard_rejections": self.hard_rejections,
            "missing_required_features": self.missing_required_features,
            "matched_envelope_id": self.matched_envelope_id,
            "envelope_type": self.envelope_type,
            "feature_scores": {k: round(v, 4) for k, v in self.feature_scores.items()},
            "evidence_count": self.evidence_count,
            "confidence": self.confidence,
            "explanation": self.explanation,
        }


# Memory field → (envelope list, current regime field, rejection code)
_IDENTITY_CHECKS = (
    ("body_ids", "body_id", "body_mismatch"),
    ("hardware_revisions", "hardware_revision", "hardware_revision_mismatch"),
    ("firmware_versions", "firmware_version", "firmware_mismatch"),
    ("calibration_hashes", "calibration_hash", "calibration_profile_mismatch"),
    ("control_profile_hashes", "control_profile_hash", "control_profile_mismatch"),
    ("task_ids", "task_id", "task_mismatch"),
    ("joints", "joint_name", "joint_mismatch"),
)

# Continuous feature → (envelope (min,max) fields, regime feature)
_RANGE_CHECKS = (
    ("temperature_c", "temperature_min", "temperature_max"),
    ("temperature_slope_c_per_min", "temperature_slope_min", "temperature_slope_max"),
    ("session_elapsed_sec", "elapsed_sec_min", "elapsed_sec_max"),
    ("cumulative_action_count", "action_count_min", "action_count_max"),
    ("position_error_p95", "position_error_p95_min", "position_error_p95_max"),
    ("recent_failure_rate", "recent_failure_rate_min", "recent_failure_rate_max"),
)


def interval_distance(
    value: float | None, low: float | None, high: float | None, *, scale: float
) -> float | None:
    """Normalized distance to an interval (v4 §6.2).

    ``None`` value → ``None`` (unknown is not a match, never a zero).
    Unbounded interval → 0.  Inside → 0.  Outside → distance to the
    nearest boundary, normalized by ``scale``.
    """
    if value is None:
        return None
    if low is None and high is None:
        return 0.0
    if low is not None and value < low:
        return (low - value) / scale
    if high is not None and value > high:
        return (value - high) / scale
    return 0.0


class RegimeMatcher:
    """Matches memories (via their envelopes) against the current regime."""

    def __init__(self, config: MatcherConfig | None = None) -> None:
        self._config = config or MatcherConfig()

    @property
    def config(self) -> MatcherConfig:
        return self._config

    def match(
        self,
        memory_id: str,
        envelopes: list[ApplicabilityEnvelope],
        regime: OperatingRegime,
        *,
        robot_id: str | None = None,
    ) -> ApplicabilityResult:
        """Evaluate one memory's envelopes against the current regime.

        CONTRAINDICATED envelopes veto first; VALIDATED beats OBSERVED when
        both match; a memory with no envelopes at all is NOT applicable to
        the intervention path (no evidence it ever worked anywhere).
        """
        # 0) Contraindicated veto (v4 §6.1: hitting one is a hard reject).
        for envelope in envelopes:
            if envelope.envelope_type != EnvelopeType.CONTRAINDICATED.value:
                continue
            if self._envelope_matches(envelope, regime, robot_id=robot_id)[0]:
                return ApplicabilityResult(
                    memory_id=memory_id,
                    applicable=False,
                    score=0.0,
                    hard_rejections=["contraindicated_envelope_hit"],
                    matched_envelope_id=envelope.envelope_id,
                    envelope_type=envelope.envelope_type,
                    evidence_count=envelope.evidence_count,
                    confidence=envelope.confidence,
                    explanation={
                        "reason": envelope.reason,
                        "regime_label": regime.regime_label,
                    },
                )

        if not envelopes:
            return ApplicabilityResult(
                memory_id=memory_id,
                applicable=False,
                score=0.0,
                hard_rejections=["no_applicability_envelope"],
                explanation={"regime_label": regime.regime_label},
            )

        best: ApplicabilityResult | None = None
        # VALIDATED envelopes first (stronger semantics), then OBSERVED.
        ordered = sorted(
            envelopes,
            key=lambda e: 0 if e.envelope_type == EnvelopeType.VALIDATED.value else 1,
        )
        for envelope in ordered:
            result = self._evaluate(memory_id, envelope, regime, robot_id=robot_id)
            if result.applicable:
                return result
            if best is None or result.score > best.score:
                best = result
        assert best is not None
        return best

    # ------------------------------------------------------------------

    def _evaluate(
        self,
        memory_id: str,
        envelope: ApplicabilityEnvelope,
        regime: OperatingRegime,
        *,
        robot_id: str | None,
    ) -> ApplicabilityResult:
        matched, rejections, missing, unverified = self._envelope_matches(
            envelope, regime, robot_id=robot_id
        )
        if not matched:
            # Even a hard-rejected candidate reports its continuous feature
            # distances — the explanation must show WHY, not just THAT.
            distances = {
                name: value
                for name, value in self._feature_distances(envelope, regime).items()
                if value is not None
            }
            return ApplicabilityResult(
                memory_id=memory_id,
                applicable=False,
                score=0.0,
                hard_rejections=rejections,
                missing_required_features=missing,
                matched_envelope_id=envelope.envelope_id,
                envelope_type=envelope.envelope_type,
                feature_scores=distances,
                evidence_count=envelope.evidence_count,
                confidence=envelope.confidence,
                explanation={
                    "regime_label": regime.regime_label,
                    "unverified_identity": unverified,
                },
            )

        feature_scores = self._feature_distances(envelope, regime)
        missing_features = [name for name, d in feature_scores.items() if d is None]
        weighted = 0.0
        for name, distance in feature_scores.items():
            if distance is None:
                continue
            weighted += self._config.feature_weights.get(name, 0.5) * distance
        similarity = math.exp(-weighted)

        evidence_factor = (
            min(1.0, envelope.evidence_count / 3.0) if envelope.evidence_count else 0.0
        )
        # VALIDATED carries full weight; OBSERVED can reach the SUGGEST band
        # but never the APPLY band on its own (that requires validation).
        type_factor = 1.0 if envelope.envelope_type == EnvelopeType.VALIDATED.value else 0.8
        score = (
            similarity
            * (0.5 + 0.5 * max(envelope.confidence, 0.0))
            * (0.5 + 0.5 * evidence_factor)
            * type_factor
        )

        applicable = score >= self._config.abstain_below and not missing_features
        return ApplicabilityResult(
            memory_id=memory_id,
            applicable=applicable,
            score=score,
            hard_rejections=[] if applicable else ["score_below_threshold"],
            missing_required_features=missing_features,
            matched_envelope_id=envelope.envelope_id,
            envelope_type=envelope.envelope_type,
            feature_scores={k: v for k, v in feature_scores.items() if v is not None},
            evidence_count=envelope.evidence_count,
            confidence=envelope.confidence,
            explanation={
                "regime_similarity": round(similarity, 4),
                "evidence_factor": round(evidence_factor, 4),
                "envelope_type_factor": type_factor,
                "regime_label": regime.regime_label,
                "unverified_identity": unverified,
            },
        )

    def _envelope_matches(
        self,
        envelope: ApplicabilityEnvelope,
        regime: OperatingRegime,
        *,
        robot_id: str | None,
    ) -> tuple[bool, list[str], list[str], list[str]]:
        """Hard-constraint evaluation (v4 §6.1).

        Returns (matched, rejections, missing_required, unverified_identity)."""
        rejections: list[str] = []
        unverified: list[str] = []
        for envelope_field, regime_field, code in _IDENTITY_CHECKS:
            allowed = getattr(envelope, envelope_field)
            if not allowed:
                continue  # unconstrained on this dimension
            current = getattr(regime, regime_field, None)
            if current is None:
                # The regime carries no context on this dimension — NOT an
                # explicit mismatch (v4 §6.1 rejects only 明确 mismatch).
                # Recorded as unverified; require it via required_features
                # when it must be proven.
                unverified.append(regime_field)
                continue
            if current not in allowed:
                rejections.append(code)

        if envelope.regime_labels and regime.regime_label not in envelope.regime_labels:
            rejections.append(f"regime_label_mismatch:{regime.regime_label}")

        missing: list[str] = []
        for feature in envelope.required_features:
            if regime.feature_value(feature) is None:
                missing.append(feature)
        if missing:
            rejections.append("missing_required_features")

        return (not rejections, rejections, missing, unverified)

    def _feature_distances(
        self, envelope: ApplicabilityEnvelope, regime: OperatingRegime
    ) -> dict[str, float | None]:
        distances: dict[str, float | None] = {}
        for feature, min_field, max_field in _RANGE_CHECKS:
            value = regime.feature_value(feature)
            scale = self._config.feature_scales.get(feature, 1.0)
            distances[feature] = interval_distance(
                value,
                getattr(envelope, min_field),
                getattr(envelope, max_field),
                scale=scale,
            )
        return distances

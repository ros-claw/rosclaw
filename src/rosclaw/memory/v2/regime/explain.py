"""Human-readable regime and match explanations (v4 §14/§15)."""

from __future__ import annotations

from typing import Any

from .matcher import ApplicabilityResult
from .models import OperatingRegime


def explain_regime(regime: OperatingRegime) -> dict[str, Any]:
    """Why the current label was assigned (rosclaw regime explain)."""
    return {
        "regime_id": regime.regime_id,
        "regime_label": regime.regime_label,
        "confidence": regime.confidence,
        "missing_features": regime.missing_features,
        "key_features": {
            "temperature_c": regime.temperature_c,
            "temperature_slope_c_per_min": regime.temperature_slope_c_per_min,
            "position_error_p95": regime.position_error_p95,
            "time_to_reach_p95_ms": regime.time_to_reach_p95_ms,
            "recent_invalid_rate": regime.recent_invalid_rate,
            "communication_error_rate": regime.communication_error_rate,
            "session_elapsed_sec": regime.session_elapsed_sec,
        },
        "evidence_refs": regime.evidence_refs,
        "note": (
            "deterministic labeling from configs/regimes thresholds; "
            "missing features are unknown, never wildcard matches"
        ),
    }


def explain_match(result: ApplicabilityResult) -> dict[str, Any]:
    """Why a memory was (not) applicable (rosclaw how decide output)."""
    verdict = "applicable" if result.applicable else "not_applicable"
    return {
        "memory_id": result.memory_id,
        "verdict": verdict,
        "score": round(result.score, 4),
        "hard_rejections": result.hard_rejections,
        "missing_required_features": result.missing_required_features,
        "matched_envelope_id": result.matched_envelope_id,
        "envelope_type": result.envelope_type,
        "feature_scores": {k: round(v, 4) for k, v in result.feature_scores.items()},
        "evidence_count": result.evidence_count,
        "confidence": result.confidence,
        "explanation": result.explanation,
    }

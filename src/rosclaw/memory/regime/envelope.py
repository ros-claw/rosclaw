"""Memory applicability envelopes (数据库优化v4 §5).

An intervention memory is not globally valid: it was OBSERVED in some
regimes, VALIDATED (executed + critic-passed) in others, and may be
CONTRAINDICATED in still others (PR #98 run1: slow_down_and_delay in a
healthy regime broke reveal timing — 21–52% invalid).  One memory can carry
all three kinds; the matcher evaluates them against the current
:class:`OperatingRegime`.

Envelopes live in the ``memory_applicability`` table of the knowledge store
— separate from the memory itself, because validity is per-regime, not
per-memory.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

APPLICABILITY_TABLE = "memory_applicability"


class EnvelopeType(StrEnum):
    """v4 §5.2 — the three envelope semantics."""

    OBSERVED = "observed"
    VALIDATED = "validated"
    CONTRAINDICATED = "contraindicated"


# Continuous feature ranges carried by an envelope: field → (min, max).
RANGE_FEATURES = (
    "temperature",
    "temperature_slope",
    "elapsed_sec",
    "action_count",
    "position_error_p95",
    "recent_failure_rate",
)


def new_envelope_id() -> str:
    return f"env_{uuid.uuid4().hex[:16]}"


@dataclass
class ApplicabilityEnvelope:
    """The set of regimes in which one memory has one semantics (v4 §5.1).

    Empty lists mean "not constrained on this dimension" for identity
    fields; ``None`` range bounds mean "unbounded on this side".  An empty
    ``required_features`` + fully unconstrained envelope is an explicit
    "valid everywhere observed so far" statement — NOT a missing-data
    wildcard.
    """

    memory_id: str

    envelope_id: str = field(default_factory=new_envelope_id)

    body_ids: list[str] = field(default_factory=list)
    hardware_revisions: list[str] = field(default_factory=list)
    firmware_versions: list[str] = field(default_factory=list)
    calibration_hashes: list[str] = field(default_factory=list)
    control_profile_hashes: list[str] = field(default_factory=list)

    task_ids: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)
    gestures: list[str] = field(default_factory=list)
    joints: list[str] = field(default_factory=list)
    failure_types: list[str] = field(default_factory=list)

    temperature_min: float | None = None
    temperature_max: float | None = None
    temperature_slope_min: float | None = None
    temperature_slope_max: float | None = None

    elapsed_sec_min: float | None = None
    elapsed_sec_max: float | None = None
    action_count_min: int | None = None
    action_count_max: int | None = None

    position_error_p95_min: float | None = None
    position_error_p95_max: float | None = None
    recent_failure_rate_min: float | None = None
    recent_failure_rate_max: float | None = None

    regime_labels: list[str] = field(default_factory=list)

    envelope_type: str = EnvelopeType.OBSERVED.value

    evidence_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.0

    required_features: list[str] = field(default_factory=list)
    reason: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_record(self) -> dict[str, Any]:
        def _j(values: list[str]) -> str:
            return json.dumps(values, ensure_ascii=False)

        return {
            "id": self.envelope_id,
            "memory_id": self.memory_id,
            "body_ids": _j(self.body_ids),
            "hardware_revisions": _j(self.hardware_revisions),
            "firmware_versions": _j(self.firmware_versions),
            "calibration_hashes": _j(self.calibration_hashes),
            "control_profile_hashes": _j(self.control_profile_hashes),
            "task_ids": _j(self.task_ids),
            "skill_ids": _j(self.skill_ids),
            "gestures": _j(self.gestures),
            "joints": _j(self.joints),
            "failure_types": _j(self.failure_types),
            "temperature_min": self.temperature_min,
            "temperature_max": self.temperature_max,
            "temperature_slope_min": self.temperature_slope_min,
            "temperature_slope_max": self.temperature_slope_max,
            "elapsed_sec_min": self.elapsed_sec_min,
            "elapsed_sec_max": self.elapsed_sec_max,
            "action_count_min": self.action_count_min,
            "action_count_max": self.action_count_max,
            "position_error_p95_min": self.position_error_p95_min,
            "position_error_p95_max": self.position_error_p95_max,
            "recent_failure_rate_min": self.recent_failure_rate_min,
            "recent_failure_rate_max": self.recent_failure_rate_max,
            "regime_labels": _j(self.regime_labels),
            "envelope_type": self.envelope_type,
            "evidence_count": self.evidence_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "confidence": self.confidence,
            "required_features": _j(self.required_features),
            "reason": self.reason,
            "evidence_refs": _j(self.evidence_refs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> ApplicabilityEnvelope:
        def _l(value: Any) -> list[str]:
            if isinstance(value, list):
                return [str(v) for v in value]
            if isinstance(value, str) and value:
                try:
                    parsed = json.loads(value)
                    return [str(v) for v in parsed] if isinstance(parsed, list) else []
                except json.JSONDecodeError:
                    return []
            return []

        def _f(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _i(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        return cls(
            envelope_id=str(record.get("id") or record.get("envelope_id") or new_envelope_id()),
            memory_id=str(record.get("memory_id") or ""),
            body_ids=_l(record.get("body_ids")),
            hardware_revisions=_l(record.get("hardware_revisions")),
            firmware_versions=_l(record.get("firmware_versions")),
            calibration_hashes=_l(record.get("calibration_hashes")),
            control_profile_hashes=_l(record.get("control_profile_hashes")),
            task_ids=_l(record.get("task_ids")),
            skill_ids=_l(record.get("skill_ids")),
            gestures=_l(record.get("gestures")),
            joints=_l(record.get("joints")),
            failure_types=_l(record.get("failure_types")),
            temperature_min=_f(record.get("temperature_min")),
            temperature_max=_f(record.get("temperature_max")),
            temperature_slope_min=_f(record.get("temperature_slope_min")),
            temperature_slope_max=_f(record.get("temperature_slope_max")),
            elapsed_sec_min=_f(record.get("elapsed_sec_min")),
            elapsed_sec_max=_f(record.get("elapsed_sec_max")),
            action_count_min=_i(record.get("action_count_min")),
            action_count_max=_i(record.get("action_count_max")),
            position_error_p95_min=_f(record.get("position_error_p95_min")),
            position_error_p95_max=_f(record.get("position_error_p95_max")),
            recent_failure_rate_min=_f(record.get("recent_failure_rate_min")),
            recent_failure_rate_max=_f(record.get("recent_failure_rate_max")),
            regime_labels=_l(record.get("regime_labels")),
            envelope_type=str(record.get("envelope_type") or EnvelopeType.OBSERVED.value),
            evidence_count=int(record.get("evidence_count") or 0),
            success_count=int(record.get("success_count") or 0),
            failure_count=int(record.get("failure_count") or 0),
            confidence=float(record.get("confidence") or 0.0),
            required_features=_l(record.get("required_features")),
            reason=record.get("reason"),
            evidence_refs=_l(record.get("evidence_refs")),
            created_at=float(record.get("created_at") or time.time()),
            updated_at=float(record.get("updated_at") or time.time()),
        )


def envelope_from_regime(
    memory_id: str,
    regime: Any,
    *,
    envelope_type: str = EnvelopeType.OBSERVED.value,
    reason: str | None = None,
    evidence_refs: list[str] | None = None,
) -> ApplicabilityEnvelope:
    """Snapshot a regime into an envelope (used when a memory is created
    or when an intervention outcome is observed)."""
    return ApplicabilityEnvelope(
        memory_id=memory_id,
        body_ids=[regime.body_id] if regime.body_id else [],
        hardware_revisions=[regime.hardware_revision] if regime.hardware_revision else [],
        firmware_versions=[regime.firmware_version] if regime.firmware_version else [],
        calibration_hashes=[regime.calibration_hash] if regime.calibration_hash else [],
        control_profile_hashes=(
            [regime.control_profile_hash] if regime.control_profile_hash else []
        ),
        task_ids=[regime.task_id] if regime.task_id else [],
        skill_ids=[regime.skill_id] if regime.skill_id else [],
        gestures=[regime.gesture_name] if regime.gesture_name else [],
        joints=[regime.joint_name] if regime.joint_name else [],
        temperature_min=regime.temperature_c,
        temperature_max=regime.temperature_c,
        temperature_slope_min=regime.temperature_slope_c_per_min,
        temperature_slope_max=regime.temperature_slope_c_per_min,
        elapsed_sec_min=regime.session_elapsed_sec or None,
        elapsed_sec_max=regime.session_elapsed_sec or None,
        action_count_min=regime.cumulative_action_count or None,
        action_count_max=regime.cumulative_action_count or None,
        position_error_p95_min=regime.position_error_p95,
        position_error_p95_max=regime.position_error_p95,
        recent_failure_rate_min=regime.recent_failure_rate,
        recent_failure_rate_max=regime.recent_failure_rate,
        regime_labels=[regime.regime_label] if regime.regime_label else [],
        envelope_type=envelope_type,
        reason=reason,
        evidence_refs=list(evidence_refs or []),
    )

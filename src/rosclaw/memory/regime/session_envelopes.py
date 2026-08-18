"""Session-aggregated OBSERVED applicability envelopes (v4 §5 backfill).

The distill pipeline stores memories carrying regime metadata
(``metadata.temperature_c`` …), but until this module nothing ever wrote
the ``memory_applicability`` rows the :class:`RegimeMatcher` matches
against — so on real corpora every intervention candidate had zero
envelopes and the matcher answered "not applicable" for all of them.  The
self-loop was inert: the system recorded its thermal regime everywhere
except the one table that gates regime-aware intervention.

This module closes that link honestly:

* per session, extract only the regime features the hardware ACTUALLY
  measured (temperature from ``health_check`` events via
  :mod:`.session_samples`, invalid rate + action count from resolved
  rounds, elapsed time from round timestamps).  Features the session
  never recorded stay unbounded and never enter ``required_features`` —
  unknown is not wildcard, and a zero-width invented interval would be
  worse than no constraint;
* per memory distilled from that session, merge the session into ONE
  aggregated OBSERVED envelope: ranges widen monotonically,
  ``evidence_count`` grows once per DISTINCT session (``evidence_refs``
  carry the practice_id, so re-running the same session is a no-op),
  ``required_features`` is the intersection of what every merged session
  measured.

OBSERVED envelopes can never reach the APPLY rung on their own — the
matcher's ``type_factor`` (0.8) caps them below the validated band.
They make memories *addressable* by the regime matcher without
self-certifying them.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .envelope import ApplicabilityEnvelope, EnvelopeType
from .persistence import ApplicabilityStore
from .session_samples import extract_samples, load_session_events

BUILDER_TAG = "session_envelope_backfill"

# Matcher regime-feature names this builder can supply (v4 §6.2), mapped
# to the envelope bounds they feed.
_MEASURED_FEATURES = (
    "temperature_c",
    "temperature_slope_c_per_min",
    "session_elapsed_sec",
    "cumulative_action_count",
    "recent_failure_rate",
)


@dataclass(frozen=True)
class SessionEnvelopeFeatures:
    """The regime features one session actually measured."""

    temperature_min: float | None = None
    temperature_max: float | None = None
    temperature_slope: float | None = None  # °C per minute, first→last
    elapsed_sec: float | None = None
    action_count: int | None = None
    invalid_rate: float | None = None
    rounds: int = 0
    measured: tuple[str, ...] = field(default_factory=tuple)


def features_from_samples(samples: list[Any]) -> SessionEnvelopeFeatures:
    """Aggregate round-level :class:`TelemetrySample` rows into session
    envelope features.  Missing data stays ``None`` — never invented."""
    if not samples:
        return SessionEnvelopeFeatures()
    temps = [s.temperature_c for s in samples if s.temperature_c is not None]
    started = samples[0].timestamp
    ended = samples[-1].timestamp
    elapsed = max(0.0, ended - started) if ended >= started else None
    rounds = len(samples)
    invalid_rate = sum(1 for s in samples if s.invalid) / rounds if rounds else None
    slope: float | None = None
    if len(temps) >= 2 and elapsed and elapsed >= 60.0:
        slope = (temps[-1] - temps[0]) / (elapsed / 60.0)

    measured: list[str] = []
    if temps:
        measured.append("temperature_c")
    if slope is not None:
        measured.append("temperature_slope_c_per_min")
    if elapsed is not None:
        measured.append("session_elapsed_sec")
    if rounds:
        measured.append("cumulative_action_count")
    if invalid_rate is not None:
        measured.append("recent_failure_rate")

    return SessionEnvelopeFeatures(
        temperature_min=min(temps) if temps else None,
        temperature_max=max(temps) if temps else None,
        temperature_slope=slope,
        elapsed_sec=elapsed,
        action_count=rounds or None,
        invalid_rate=invalid_rate,
        rounds=rounds,
        measured=tuple(measured),
    )


def _widen(
    lo: float | None, hi: float | None, value: float | None
) -> tuple[float | None, float | None]:
    """Widen a [lo, hi] interval to include ``value`` (None = no data)."""
    if value is None:
        return lo, hi
    return (
        value if lo is None else min(lo, value),
        value if hi is None else max(hi, value),
    )


def envelope_from_session(
    memory: dict[str, Any],
    features: SessionEnvelopeFeatures,
    *,
    practice_id: str,
    now: float | None = None,
) -> ApplicabilityEnvelope:
    """Create the first aggregated OBSERVED envelope for one memory from
    one session's measured features."""
    measured = list(features.measured)
    return ApplicabilityEnvelope(
        memory_id=str(memory["id"]),
        body_ids=[str(memory["body_id"])] if memory.get("body_id") else [],
        task_ids=[str(memory["task_id"])] if memory.get("task_id") else [],
        skill_ids=[str(memory["skill_id"])] if memory.get("skill_id") else [],
        gestures=[str(memory["gesture_name"])] if memory.get("gesture_name") else [],
        joints=[str(memory["joint_name"])] if memory.get("joint_name") else [],
        failure_types=[str(memory["failure_type"])] if memory.get("failure_type") else [],
        temperature_min=features.temperature_min,
        temperature_max=features.temperature_max,
        temperature_slope_min=features.temperature_slope,
        temperature_slope_max=features.temperature_slope,
        elapsed_sec_min=0.0 if features.elapsed_sec is not None else None,
        elapsed_sec_max=features.elapsed_sec,
        action_count_min=0 if features.action_count is not None else None,
        action_count_max=features.action_count,
        recent_failure_rate_min=features.invalid_rate,
        recent_failure_rate_max=features.invalid_rate,
        envelope_type=EnvelopeType.OBSERVED.value,
        evidence_count=1,
        success_count=0 if memory.get("outcome") == "failure" else 1,
        failure_count=1 if memory.get("outcome") == "failure" else 0,
        confidence=_confidence(1),
        required_features=measured,
        reason=BUILDER_TAG,
        evidence_refs=[practice_id],
        created_at=now if now is not None else time.time(),
        updated_at=now if now is not None else time.time(),
    )


def merge_session_into_envelope(
    envelope: ApplicabilityEnvelope,
    features: SessionEnvelopeFeatures,
    *,
    practice_id: str,
    outcome: str | None = None,
) -> bool:
    """Merge one more session into an aggregated envelope in place.

    Returns False (no-op) when this session is already counted —
    idempotency is by practice_id in ``evidence_refs``.
    """
    if practice_id in envelope.evidence_refs:
        return False
    lo, hi = _widen(envelope.temperature_min, envelope.temperature_max, features.temperature_min)
    lo, hi = _widen(lo, hi, features.temperature_max)
    envelope.temperature_min, envelope.temperature_max = lo, hi
    envelope.temperature_slope_min, envelope.temperature_slope_max = _widen(
        envelope.temperature_slope_min, envelope.temperature_slope_max, features.temperature_slope
    )
    if features.elapsed_sec is not None:
        envelope.elapsed_sec_min, envelope.elapsed_sec_max = _widen(
            envelope.elapsed_sec_min, envelope.elapsed_sec_max, features.elapsed_sec
        )
    if features.action_count is not None:
        lo, hi = _widen(
            float(envelope.action_count_min) if envelope.action_count_min is not None else None,
            float(envelope.action_count_max) if envelope.action_count_max is not None else None,
            float(features.action_count),
        )
        envelope.action_count_min = int(lo) if lo is not None else None
        envelope.action_count_max = int(hi) if hi is not None else None
    envelope.recent_failure_rate_min, envelope.recent_failure_rate_max = _widen(
        envelope.recent_failure_rate_min, envelope.recent_failure_rate_max, features.invalid_rate
    )
    # required_features = what EVERY merged session measured.
    envelope.required_features = [
        name for name in envelope.required_features if name in features.measured
    ]
    envelope.evidence_count += 1
    if outcome == "failure":
        envelope.failure_count += 1
    else:
        envelope.success_count += 1
    envelope.confidence = _confidence(envelope.evidence_count)
    envelope.evidence_refs = [*envelope.evidence_refs, practice_id]
    envelope.updated_at = time.time()
    return True


def _confidence(evidence_count: int) -> float:
    """Single-observation envelopes are weak (0.4); confidence grows with
    distinct-session evidence and caps below any VALIDATED prior."""
    return min(0.8, 0.3 + 0.1 * max(1, evidence_count))


def build_session_envelopes(
    session_dir: str | Path,
    store: Any,
    *,
    hand: str = "right",
    memory_table: str = "memory_items",
) -> dict[str, Any]:
    """Build/merge OBSERVED envelopes for every memory distilled from one
    practice session.

    ``store`` is a knowledge-store client (the same backend the memories
    live in).  Returns a JSON-able summary; idempotent per session.
    """
    events = load_session_events(session_dir)
    if not events:
        return {"ok": False, "reason": "no events", "session_dir": str(session_dir)}
    practice_id = events[0].get("practice_id")
    if not practice_id:
        return {"ok": False, "reason": "no practice_id in events", "session_dir": str(session_dir)}

    samples = extract_samples(events, hand=hand)
    features = features_from_samples(samples)

    # Memories observed in this session: (a) rows whose practice_id IS
    # this session (first distillation), plus (b) rows re-confirmed here
    # via the distiller's MERGE path — those keep the originating
    # session's practice_id, so we discover them through memory_evidence
    # source_event_ids present in THIS session's event stream.
    rows = {
        str(row["id"]): dict(row)
        for row in store.query(memory_table, filters={"practice_id": practice_id}, limit=1000)
    }
    session_event_ids = {e.get("event_id") for e in events if e.get("event_id")}
    try:
        evidence_rows = store.query("memory_evidence", limit=50000)
    except Exception:  # backend without memory_evidence — (a) still works
        evidence_rows = []
    reconfirmed_ids = {
        str(ev["memory_id"])
        for ev in evidence_rows
        if ev.get("source_event_id") in session_event_ids and ev.get("memory_id")
    }
    missing = [mid for mid in reconfirmed_ids if mid not in rows]
    for mid in missing:
        row = store.query(memory_table, filters={"id": mid}, limit=1)
        if row:
            rows[mid] = dict(row[0])

    applicability = ApplicabilityStore(store)
    created = merged = already = 0
    details: list[dict[str, Any]] = []
    for row in rows.values():
        memory = row
        meta = memory.get("metadata")
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except json.JSONDecodeError:
                meta = {}
        if not memory.get("gesture_name"):
            memory["gesture_name"] = (meta or {}).get("gesture")
        existing = [
            env
            for env in applicability.for_memory(str(memory["id"]))
            if env.reason == BUILDER_TAG and env.envelope_type == EnvelopeType.OBSERVED.value
        ]
        if existing:
            envelope = existing[0]
            if merge_session_into_envelope(
                envelope, features, practice_id=practice_id, outcome=memory.get("outcome")
            ):
                applicability.upsert(envelope)
                merged += 1
                details.append({"memory_id": memory["id"], "action": "merged"})
            else:
                already += 1
                details.append({"memory_id": memory["id"], "action": "already_counted"})
        else:
            envelope = envelope_from_session(memory, features, practice_id=practice_id)
            applicability.upsert(envelope)
            created += 1
            details.append({"memory_id": memory["id"], "action": "created"})

    return {
        "ok": True,
        "session_dir": str(session_dir),
        "practice_id": practice_id,
        "hand": hand,
        "rounds": features.rounds,
        "measured": list(features.measured),
        "temperature_range": [features.temperature_min, features.temperature_max],
        "invalid_rate": features.invalid_rate,
        "memories": len(rows),
        "envelopes_created": created,
        "envelopes_merged": merged,
        "envelopes_already_counted": already,
        "details": details,
    }

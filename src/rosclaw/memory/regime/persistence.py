"""Persistence for applicability envelopes and regime records (v4 §4/§5).

Two tables on the knowledge store:

* ``memory_applicability`` — envelopes, one row per (memory, regime-set,
  semantics).  Evidence counters update via read-modify-write; confidence
  is the Wilson lower bound of the success rate (same estimator as HOW
  rule efficacy — one statistics convention across the codebase).
* ``regime_history`` — point-in-time OperatingRegime snapshots + confirmed
  transitions, so `rosclaw regime replay` and later calibration can audit
  what the robot believed its working condition was.
"""

from __future__ import annotations

import time
from typing import Any

from rosclaw.how.rule_efficacy import wilson_lower_bound

from .envelope import APPLICABILITY_TABLE, ApplicabilityEnvelope

REGIME_HISTORY_TABLE = "regime_history"


class ApplicabilityStore:
    """CRUD + evidence accumulation for memory_applicability rows."""

    def __init__(self, client: Any, *, table: str = APPLICABILITY_TABLE) -> None:
        self._client = client
        self._table = table

    def upsert(self, envelope: ApplicabilityEnvelope) -> str:
        envelope.updated_at = time.time()
        self._client.insert(self._table, envelope.to_record())
        return envelope.envelope_id

    def for_memory(self, memory_id: str) -> list[ApplicabilityEnvelope]:
        rows = self._client.query(self._table, filters={"memory_id": memory_id}, limit=1000)
        return [ApplicabilityEnvelope.from_record(row) for row in rows]

    def query(
        self, filters: dict[str, Any] | None = None, *, limit: int = 1000
    ) -> list[ApplicabilityEnvelope]:
        rows = self._client.query(self._table, filters=filters, limit=limit)
        return [ApplicabilityEnvelope.from_record(row) for row in rows]

    def delete(self, envelope_id: str) -> bool:
        return bool(self._client.delete(self._table, envelope_id))

    def record_outcome(
        self,
        envelope_id: str,
        *,
        success: bool,
        evidence_ref: str | None = None,
    ) -> ApplicabilityEnvelope | None:
        """Fold one real execution outcome into an envelope's evidence.

        Only outcomes whose patch was actually applied and judged by the
        critic should reach this method (v4 §9) — the caller owns that gate.
        """
        rows = self._client.query(self._table, filters={"id": envelope_id}, limit=1)
        if not rows:
            return None
        envelope = ApplicabilityEnvelope.from_record(rows[0])
        envelope.evidence_count += 1
        if success:
            envelope.success_count += 1
        else:
            envelope.failure_count += 1
        envelope.confidence = round(
            wilson_lower_bound(envelope.success_count, envelope.evidence_count), 4
        )
        if evidence_ref:
            envelope.evidence_refs = sorted(set(envelope.evidence_refs) | {evidence_ref})
        self.upsert(envelope)
        return envelope


class RegimeHistoryStore:
    """Append-only regime snapshots and transitions."""

    def __init__(self, client: Any, *, table: str = REGIME_HISTORY_TABLE) -> None:
        self._client = client
        self._table = table

    def append_snapshot(self, regime: Any) -> str:
        record = {
            "id": regime.regime_id,
            "row_kind": "snapshot",
            **regime.to_dict(),
        }
        self._client.insert(self._table, _flatten(record))
        return regime.regime_id

    def append_transition(self, transition: Any, *, robot_id: str, body_id: str) -> str:
        import uuid

        record = {
            "id": f"trans_{uuid.uuid4().hex[:16]}",
            "row_kind": "transition",
            "robot_id": robot_id,
            "body_id": body_id,
            **transition.to_dict(),
        }
        self._client.insert(self._table, _flatten(record))
        return str(record["id"])

    def transitions(self, *, robot_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"row_kind": "transition"}
        if robot_id:
            filters["robot_id"] = robot_id
        return self._client.query(self._table, filters=filters, limit=limit)

    def snapshots(self, *, body_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        filters: dict[str, Any] = {"row_kind": "snapshot"}
        if body_id:
            filters["body_id"] = body_id
        return self._client.query(self._table, filters=filters, limit=limit)


def _flatten(record: dict[str, Any]) -> dict[str, Any]:
    """Knowledge-store cells must be JSON primitives."""
    import json

    flat: dict[str, Any] = {}
    for key, value in record.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            flat[key] = value
        else:
            flat[key] = json.dumps(value, ensure_ascii=False)
    return flat

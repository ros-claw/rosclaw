"""KnowledgeUsageLedger (PR-DF-21 / phase-II §32).

The durable observation layer for knowledge utility: every
ReferencePack presentation, Advice usage, and conservative outcome
verdict lands as a ``knowledge_usage_events`` row so per-Knowledge-Unit
utility can be measured BEFORE any ranking formula exists (§32.3 —
observe first; semantic relevance × evidence quality × body
applicability × freshness × historical utility comes later, with data).

Verdict vocabulary (§32.1): presented / used / useful / unknown /
stale / incompatible / misleading.  Automatic tracking only ever emits
presented / used / useful / unknown — the negative attributions require
verifier or human evidence (same discipline as PR-DF-10).

All writes are best-effort: the ledger NEVER breaks the execution path.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

logger = logging.getLogger("rosclaw.knowledge.usage_ledger")

TABLE = "knowledge_usage_events"

VERDICT_PRESENTED = "presented"
VERDICT_USED = "used"
VERDICT_USEFUL = "useful"
VERDICT_UNKNOWN = "unknown"
VERDICT_STALE = "stale"
VERDICT_INCOMPATIBLE = "incompatible"
VERDICT_MISLEADING = "misleading"

_AGGREGATE_KEYS = (
    "presented_count",
    "used_count",
    "useful_count",
    "unknown_count",
    "incompatible_count",
    "stale_count",
    "misleading_count",
    "verified_success_count",
    "verified_failure_count",
)


class KnowledgeUsageLedger:
    """Append-only usage event log + per-unit aggregation (§32.2)."""

    def __init__(self, structured_store: Any) -> None:
        self._store = structured_store

    # -- write -------------------------------------------------------------

    def record(
        self,
        verdict: str,
        *,
        reference_pack_id: str = "",
        knowledge_unit_id: str = "",
        advice_id: str | None = None,
        trace_id: str = "",
        practice_id: str | None = None,
        episode_id: str | None = None,
        receipt_id: str | None = None,
        robot_id: str = "",
        body_id: str = "",
        task_id: str = "",
        skill_id: str = "",
        confidence: float | None = None,
    ) -> str | None:
        """Append one usage event; returns the event id, None on failure."""
        try:
            event_id = f"use_{uuid.uuid4().hex[:16]}"
            self._store.insert(
                TABLE,
                {
                    "id": event_id,
                    "usage_id": f"{reference_pack_id}:{knowledge_unit_id}",
                    "reference_pack_id": reference_pack_id,
                    "knowledge_unit_id": knowledge_unit_id,
                    "advice_id": advice_id,
                    "trace_id": trace_id,
                    "practice_id": practice_id,
                    "episode_id": episode_id,
                    "receipt_id": receipt_id,
                    "robot_id": robot_id,
                    "body_id": body_id,
                    "task_id": task_id,
                    "skill_id": skill_id,
                    "verdict": verdict,
                    "confidence": confidence,
                    "created_at": time.time(),
                },
            )
            return event_id
        except Exception as exc:  # noqa: BLE001 — observation never breaks execution
            logger.info("knowledge usage ledger write failed (non-fatal): %s", exc)
            return None

    # -- read (§32.2 aggregation) -------------------------------------------

    def aggregate(self, knowledge_unit_id: str | None = None) -> dict[str, dict[str, Any]]:
        """Per-unit counters (§32.2).

        presented/used come from the pack/advice stages; useful/unknown
        from outcome verdicts; verified_success/failure count outcomes by
        task result (useful ⇒ verified success, unknown ⇒ unverified).
        """
        filters = {"knowledge_unit_id": knowledge_unit_id} if knowledge_unit_id else {}
        try:
            rows = self._store.query(TABLE, filters, limit=500_000)
        except Exception as exc:  # noqa: BLE001
            logger.info("knowledge usage aggregate failed: %s", exc)
            return {}
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            unit = row.get("knowledge_unit_id") or "unknown_unit"
            entry = out.setdefault(unit, dict.fromkeys(_AGGREGATE_KEYS, 0))
            verdict = row.get("verdict")
            if verdict == VERDICT_PRESENTED:
                entry["presented_count"] += 1
            elif verdict == VERDICT_USED:
                entry["used_count"] += 1
            elif verdict == VERDICT_USEFUL:
                entry["useful_count"] += 1
                entry["verified_success_count"] += 1
            elif verdict == VERDICT_UNKNOWN:
                entry["unknown_count"] += 1
            elif verdict == VERDICT_INCOMPATIBLE:
                entry["incompatible_count"] += 1
                entry["verified_failure_count"] += 1
            elif verdict == VERDICT_STALE:
                entry["stale_count"] += 1
            elif verdict == VERDICT_MISLEADING:
                entry["misleading_count"] += 1
                entry["verified_failure_count"] += 1
        return out

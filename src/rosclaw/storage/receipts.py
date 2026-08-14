"""ReceiptProjector (PR-DF-14 / flywheel §21-22).

Async projection of kernel ExecutionReceipts into the data plane's
``execution_receipts`` table, plus the first lineage edges
(``action --generated_from--> receipt``).

Safety red line (§22): this table is an *index for learning and audit*.
rosclawd's permit/action ledgers stay the only authorization authority;
nothing in the dispatch or E-stop path may ever wait on this projection.
Writes are idempotent on receipt identity and failures are logged, never
raised into the execution path.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from rosclaw.core.event_topics import EventTopics

from .lineage import LineageRepository

logger = logging.getLogger("rosclaw.storage.receipts")

TABLE = "execution_receipts"


class ReceiptProjector:
    """Project action receipts into the structured store (+ lineage)."""

    def __init__(
        self,
        event_bus: Any,
        structured_store: Any,
        lineage: LineageRepository | None = None,
    ) -> None:
        self._bus = event_bus
        self._store = structured_store
        self._lineage = lineage
        self._subscribed = False

    def subscribe(self) -> None:
        if self._bus is None or self._subscribed:
            return
        self._bus.subscribe(EventTopics.ACTION_RECEIPT, self._on_receipt)
        self._subscribed = True
        logger.info("ReceiptProjector subscribed")

    def unsubscribe(self) -> None:
        if self._bus is None or not self._subscribed:
            return
        self._bus.unsubscribe(EventTopics.ACTION_RECEIPT, self._on_receipt)
        self._subscribed = False

    def _on_receipt(self, event: Any) -> None:
        payload = getattr(event, "payload", None)
        if not isinstance(payload, dict):
            return
        try:
            self.project(payload, trace_id=getattr(event, "trace_id", "") or "")
        except Exception as exc:  # noqa: BLE001 — projection must never break execution
            logger.info("receipt projection failed (non-fatal): %s", exc)

    def project(self, receipt: dict[str, Any], *, trace_id: str = "") -> str | None:
        """Upsert one receipt record; returns the row id (idempotent)."""
        action_id = receipt.get("action_id", "")
        row_id = f"rcpt_{action_id}" if action_id else None
        if row_id is None:
            logger.warning("receipt without action_id — projection skipped")
            return None
        record = {
            "id": row_id,
            "receipt_id": row_id,
            "action_id": action_id,
            "trace_id": receipt.get("trace_id") or trace_id,
            "session_id": receipt.get("session_id"),
            "practice_id": receipt.get("practice_id"),
            "episode_id": receipt.get("episode_id"),
            "robot_id": receipt.get("robot_id"),
            "body_id": receipt.get("body_id"),
            "capability_id": receipt.get("capability_id"),
            "execution_mode": _enum_value(receipt.get("mode") or receipt.get("execution_mode")),
            "final_state": _enum_value(receipt.get("final_state")),
            "acknowledgement_stage": _enum_value(receipt.get("acknowledgement_stage")),
            "evidence_level": _enum_value(receipt.get("evidence_level")),
            "evidence_domain": _enum_value(receipt.get("evidence_domain")),
            "dispatched_at": _ts(receipt.get("started_at")),
            "observed_at": _ts(receipt.get("finished_at")),
            "verified_at": _ts((receipt.get("verification_result") or {}).get("verified_at")),
            "artifact_refs": json.dumps(receipt.get("artifacts") or [], default=str),
            "verification_refs": json.dumps(
                (receipt.get("verification_result") or {}).get("evidence_refs") or [], default=str
            ),
            "schema_version": receipt.get("schema_version", ""),
            "created_at": time.time(),
        }
        self._store.insert(TABLE, record)
        if self._lineage is not None and action_id:
            self._lineage.link(
                "receipt",
                row_id,
                "generated_from",
                "action",
                action_id,
                trace_id=record["trace_id"] or "",
            )
        return row_id


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def _ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        from datetime import datetime

        if isinstance(value, datetime):
            return value.timestamp()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None

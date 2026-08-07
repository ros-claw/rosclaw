"""Convert receipts into governance feedback without copying execution data."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Literal

from .contracts import KnowledgeUsageFeedbackV1


def build_usage_feedback(
    *,
    reference_pack_id: str,
    knowledge_unit_id: str,
    context_hash: str,
    verdict: Literal["useful", "irrelevant", "stale", "incompatible", "misleading", "unknown"],
    advice_id: str | None = None,
    presented: bool = True,
    opened: bool = False,
    used_by_agent: bool = False,
    reason: str | None = None,
    receipt_ref: str | None = None,
    practice_ref: str | None = None,
    origin: Literal["user", "agent", "verifier"] = "agent",
) -> KnowledgeUsageFeedbackV1:
    stable = ":".join(
        [
            reference_pack_id,
            knowledge_unit_id,
            advice_id or "",
            verdict,
            receipt_ref or "",
            practice_ref or "",
        ]
    )
    return KnowledgeUsageFeedbackV1(
        feedback_id=f"feedback_{hashlib.sha256(stable.encode()).hexdigest()[:24]}",
        reference_pack_id=reference_pack_id,
        advice_id=advice_id,
        knowledge_unit_id=knowledge_unit_id,
        presented=presented,
        opened=opened,
        used_by_agent=used_by_agent,
        verdict=verdict,
        reason=reason[:1000] if reason else None,
        context_hash=context_hash,
        receipt_ref=receipt_ref,
        practice_ref=practice_ref,
        origin=origin,
        created_at=datetime.now(UTC),
    )

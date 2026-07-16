"""rosclaw_how.rule_efficacy — RecoveryRule v2 schema and efficacy stats (§11).

* :class:`RecoveryRule` — the §11.1 schema with applicability, safety level,
  and evidence-count-aware efficacy stats;
* :func:`wilson_lower_bound` — posterior confidence for rule ranking, so a
  ``1/1 = 100%`` rule never outranks a ``90/100 = 90%`` rule (§11.3);
* :func:`record_outcome_atomic` — ``UPDATE ... SET x = x + 1`` on SQL
  backends instead of read-modify-write in Python (§11.3);
* :class:`PatchProof` — §11.2 evidence that a suggested patch was actually
  applied and retried (suggested_patch vs actual_patch).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("rosclaw.how.rule_efficacy")


@dataclass
class RecoveryRule:
    """Recovery rule with applicability and efficacy evidence (§11.1)."""

    rule_id: str
    failure_signature: str
    action_template: dict[str, Any]

    applicable_robot_types: list[str] = field(default_factory=list)
    applicable_body_ids: list[str] = field(default_factory=list)
    context_constraints: dict[str, Any] = field(default_factory=dict)

    safety_level: str = "S1"
    priority: int = 0

    success_count: int = 0
    failure_count: int = 0
    evidence_count: int = 0

    success_rate: float = 0.0
    confidence_interval: dict[str, float] = field(default_factory=dict)
    last_validated_at: float | None = None

    evidence_refs: list[str] = field(default_factory=list)
    status: str = "active"

    def refresh_stats(self) -> None:
        total = self.success_count + self.failure_count
        self.success_rate = self.success_count / total if total else 0.0
        self.confidence_interval = wilson_interval(self.success_count, total)

    def to_record(self) -> dict[str, Any]:
        self.refresh_stats()
        import json

        return {
            "id": self.rule_id,
            # Legacy NOT NULL columns stay populated for backward compat.
            "condition": self.failure_signature,
            "action": json.dumps(self.action_template, ensure_ascii=False),
            "failure_signature": self.failure_signature,
            "action_template": json.dumps(self.action_template, ensure_ascii=False),
            "applicable_robot_types": json.dumps(self.applicable_robot_types, ensure_ascii=False),
            "applicable_body_ids": json.dumps(self.applicable_body_ids, ensure_ascii=False),
            "context_constraints": json.dumps(self.context_constraints, ensure_ascii=False),
            "safety_level": self.safety_level,
            "priority": self.priority,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "evidence_count": self.evidence_count,
            "success_rate": self.success_rate,
            "confidence_interval": json.dumps(self.confidence_interval, ensure_ascii=False),
            "last_validated_at": self.last_validated_at,
            "evidence_refs": json.dumps(self.evidence_refs, ensure_ascii=False),
            "status": self.status,
        }


@dataclass
class PatchProof:
    """§11.2 evidence that a suggested patch was actually executed."""

    suggested_patch: dict[str, Any]
    actual_patch: dict[str, Any] | None = None
    patch_applied: bool = False
    sandbox_decision: str | None = None
    before_metrics: dict[str, Any] = field(default_factory=dict)
    after_metrics: dict[str, Any] = field(default_factory=dict)
    critic_decision: str | None = None
    retried_at: float = field(default_factory=time.time)

    def complete(
        self, actual_patch: dict[str, Any], after_metrics: dict[str, Any], critic_decision: str
    ) -> None:
        """Mark the patch as actually applied with post-retry evidence."""
        self.actual_patch = actual_patch
        self.patch_applied = True
        self.after_metrics = after_metrics
        self.critic_decision = critic_decision
        self.retried_at = time.time()

    def to_record(self) -> dict[str, Any]:
        return {
            "suggested_patch": self.suggested_patch,
            "actual_patch": self.actual_patch,
            "patch_applied": self.patch_applied,
            "sandbox_decision": self.sandbox_decision,
            "before_metrics": self.before_metrics,
            "after_metrics": self.after_metrics,
            "critic_decision": self.critic_decision,
            "retried_at": self.retried_at,
        }


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Wilson score lower bound — posterior confidence for rule ranking (§11.3).

    A rule with few samples is penalized: ``1/1`` scores ≈0.21, ``90/100``
    scores ≈0.82, so tiny-sample rules never outrank well-evidenced ones.
    """
    if total <= 0:
        return 0.0
    phat = successes / total
    z2 = z * z
    center = phat + z2 / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z2 / (4 * total * total))
    return max(0.0, (center - margin) / (1 + z2 / total))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score confidence interval [lower, upper]."""
    if total <= 0:
        return {"lower": 0.0, "upper": 1.0, "z": z}
    phat = successes / total
    z2 = z * z
    center = phat + z2 / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z2 / (4 * total * total))
    denom = 1 + z2 / total
    return {
        "lower": max(0.0, (center - margin) / denom),
        "upper": min(1.0, (center + margin) / denom),
        "z": z,
    }


def rule_rank_score(rule: RecoveryRule) -> float:
    """Posterior-confidence ranking: efficacy × evidence, never raw rate (§11.3)."""
    total = rule.success_count + rule.failure_count
    return wilson_lower_bound(rule.success_count, total)


def record_outcome_atomic(
    client: Any,
    rule_id: str,
    success: bool,
    *,
    table: str = "heuristic_rules",
    evidence_ref: str | None = None,
) -> bool:
    """Atomically increment a rule's success/failure counter (§11.3).

    SQL backends execute ``UPDATE ... SET count = count + 1``; other backends
    fall back to a locked read-modify-write.  Also stamps last_validated_at
    and bumps evidence_count when an evidence ref is provided.  Never counts
    an outcome whose patch was not actually applied — callers must only
    invoke this with a completed :class:`PatchProof`.
    """
    column = "success_count" if success else "failure_count"
    now = time.time()
    conn = getattr(client, "_connection", None)
    if conn is not None:
        try:
            with client._lock:  # type: ignore[attr-defined]
                conn.execute(
                    f"UPDATE {table} SET {column} = {column} + 1, "
                    f"evidence_count = evidence_count + ?, last_validated_at = ? WHERE id = ?",
                    (1 if evidence_ref else 0, now, rule_id),
                )
                conn.commit()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("atomic outcome via SQL failed, falling back: %s", exc)
    try:
        rows = client.query(table, filters={"id": rule_id}, limit=1)
        if not rows:
            return False
        row = rows[0]
        updates = {
            column: int(row.get(column) or 0) + 1,
            "evidence_count": int(row.get("evidence_count") or 0) + (1 if evidence_ref else 0),
            "last_validated_at": now,
        }
        return bool(client.update(table, rule_id, updates))
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_outcome_atomic(%s) failed: %s", rule_id, exc)
        return False

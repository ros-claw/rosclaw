"""Learning candidate pipeline (PR-EV-081, 总纲 §14).

Practice（发生了什么）→ 证据门 → Memory/Know/How/Auto 候选。铁律：

- 只有 measured / verified_receipt / curated 级事实可以形成候选；
  inferred / unverified 被显式拒绝并记录（记忆污染防线 §13）。
- 候选默认 CANDIDATE 状态；晋升必须通过评测引用 + 人类 principal
  （Darwin 晋升门），任何代码路径都不能自动晋升。
- AUTO 候选只是"自动化候选"，绝不等于已授权自动执行。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

from rosclaw.agentd.context.sources import EvidenceClass
from rosclaw.contracts.common import ValidationError, new_id

_ADMITTED = frozenset(
    {EvidenceClass.MEASURED, EvidenceClass.VERIFIED_RECEIPT, EvidenceClass.CURATED}
)
_KINDS = ("MEMORY", "KNOW", "HOW", "AUTO")


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


class EvidenceRejectedError(ValidationError):
    """Evidence below curated level may not form learning candidates."""


class PromotionGateError(ValidationError):
    """Promotion requires evaluation reference + human principal."""


class LearningPipeline:
    def __init__(self, conn: sqlite3.Connection, *, actor_id: str) -> None:
        self._conn = conn
        self._actor_id = actor_id

    # ------------------------------------------------------------------
    def extract_from_mission(self, mission_id: str) -> list[str]:
        """Form candidates from a mission's verified evidence.

        Sources of facts (all evidence-classed):
        - ACCEPTED work orders → verified_receipt (worker output passed
          deterministic verifiers)
        - validated decisions → curated (structured, validated, but
          model-generated)
        Rejected candidates are recorded for audit, not silently dropped.
        """
        created: list[str] = []
        orders = self._conn.execute(
            "SELECT order_json, status FROM work_orders WHERE mission_id = ?",
            (mission_id,),
        ).fetchall()
        for row in orders:
            order = json.loads(row["order_json"])
            if row["status"] != "ACCEPTED":
                self._reject(
                    kind="MEMORY",
                    title=f"work order {order['work_order_id']} not accepted",
                    evidence_class=EvidenceClass.UNVERIFIED,
                    evidence_refs=[f"wo://{order['work_order_id']}"],
                    mission_id=mission_id,
                    reason="not a verified receipt",
                )
                continue
            created.append(
                self.propose(
                    kind="MEMORY",
                    title=f"verified worker outcome for {order['capability']}",
                    content={
                        "goal": order["goal"],
                        "worker": order["assigned_to"],
                        "capability": order["capability"],
                    },
                    evidence_class=EvidenceClass.VERIFIED_RECEIPT,
                    evidence_refs=[f"wo://{order['work_order_id']}"],
                    mission_id=mission_id,
                )
            )
        decisions = self._conn.execute(
            "SELECT decision_json, validated FROM decisions WHERE mission_id = ?",
            (mission_id,),
        ).fetchall()
        for row in decisions:
            if not row["validated"]:
                continue
            decision = json.loads(row["decision_json"])
            created.append(
                self.propose(
                    kind="KNOW",
                    title=f"validated decision pattern: {decision['next_intent']}",
                    content={
                        "intent": decision["next_intent"],
                        "summary": decision.get("summary", ""),
                        "public_rationale": decision.get("public_rationale", ""),
                    },
                    evidence_class=EvidenceClass.CURATED,
                    evidence_refs=[f"dec://{decision['decision_id']}"],
                    mission_id=mission_id,
                )
            )
        return created

    # ------------------------------------------------------------------
    def propose(
        self,
        *,
        kind: str,
        title: str,
        content: dict,
        evidence_class: EvidenceClass,
        evidence_refs: list[str],
        mission_id: str | None = None,
        body_scope: str | None = None,
        prompt_hash: str | None = None,
    ) -> str:
        if kind not in _KINDS:
            raise ValidationError(f"unknown candidate kind {kind!r}")
        if evidence_class not in _ADMITTED:
            raise EvidenceRejectedError(
                f"evidence class {evidence_class.value} below curated — "
                "unverified facts never enter the fact layer"
            )
        candidate_id = new_id("lc")
        self._conn.execute(
            "INSERT INTO learning_candidates (candidate_id, kind, title, "
            "content_json, evidence_class, evidence_refs_json, body_scope, "
            "source_mission_id, prompt_hash, status, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'CANDIDATE', ?, ?)",
            (
                candidate_id,
                kind,
                title,
                json.dumps(content, sort_keys=True, ensure_ascii=False),
                evidence_class.value,
                json.dumps(evidence_refs, ensure_ascii=False),
                body_scope,
                mission_id,
                prompt_hash,
                self._actor_id,
                _utcnow(),
            ),
        )
        return candidate_id

    def _reject(
        self,
        *,
        kind: str,
        title: str,
        evidence_class: EvidenceClass,
        evidence_refs: list[str],
        mission_id: str,
        reason: str,
    ) -> None:
        self._conn.execute(
            "INSERT INTO learning_candidates (candidate_id, kind, title, "
            "content_json, evidence_class, evidence_refs_json, "
            "source_mission_id, status, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'REJECTED', ?, ?)",
            (
                new_id("lc"),
                kind,
                title,
                json.dumps({"rejection_reason": reason}, ensure_ascii=False),
                evidence_class.value,
                json.dumps(evidence_refs, ensure_ascii=False),
                mission_id,
                self._actor_id,
                _utcnow(),
            ),
        )

    # ------------------------------------------------------------------
    def list(self, *, status: str | None = None) -> list[dict]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM learning_candidates WHERE status = ? ORDER BY created_at",
                (status,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM learning_candidates ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def promote(
        self,
        candidate_id: str,
        *,
        principal: str,
        evaluation_ref: str,
    ) -> None:
        """Darwin 晋升门：评测引用 + 人类 principal 缺一不可（§14.1）。"""
        if not principal.startswith("user:"):
            raise PromotionGateError(f"promotion requires a human principal, got {principal!r}")
        if not evaluation_ref:
            raise PromotionGateError("promotion requires an evaluation reference")
        row = self._conn.execute(
            "SELECT status FROM learning_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            raise ValidationError(f"unknown candidate {candidate_id!r}")
        if row["status"] not in ("CANDIDATE", "EVALUATING"):
            raise PromotionGateError(f"candidate already {row['status']}")
        self._conn.execute(
            "UPDATE learning_candidates SET status = 'PROMOTED', "
            "evaluation_ref = ?, promoted_by = ?, promoted_at = ? "
            "WHERE candidate_id = ?",
            (evaluation_ref, principal, _utcnow(), candidate_id),
        )

    def reject(self, candidate_id: str, *, principal: str, reason: str) -> None:
        self._conn.execute(
            "UPDATE learning_candidates SET status = 'REJECTED', "
            "promoted_by = ?, promoted_at = ? WHERE candidate_id = ?",
            (principal, _utcnow(), candidate_id),
        )

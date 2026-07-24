"""Selective risk accounting (数据库优化v4 §7.5).

The goal is NOT maximum coverage — it is maximum *effective* coverage under
safety constraints:

    coverage                  = APPLY / eligible
    selective harm risk       = harmful APPLY / APPLY   (upper CI ≤ 5% gate)
    abstention rate           = ABSTAIN / eligible
    helpful apply precision   = helpful APPLY / APPLY
    wrong body/joint/regime APPLY — always 0, tracked explicitly

Decisions land in the ``selective_intervention_ledger`` table; outcomes are
recorded against the decision id.  Only real execution outcomes (patch
applied + critic judgment) may mark helpful/harmful — the caller owns that
gate, same contract as PatchProof.
"""

from __future__ import annotations

import time
from typing import Any

from rosclaw.how.rule_efficacy import wilson_interval

LEDGER_TABLE = "selective_intervention_ledger"

OUTCOME_PENDING = "pending"
OUTCOME_HELPFUL = "helpful"
OUTCOME_HARMFUL = "harmful"
OUTCOME_NEUTRAL = "neutral"

HARM_RISK_GATE = 0.05


class SelectiveRiskLedger:
    """Decision + outcome ledger with selective risk metrics."""

    def __init__(self, client: Any, *, table: str = LEDGER_TABLE) -> None:
        self._client = client
        self._table = table

    def record_decision(self, decision: Any, *, body_id: str | None = None) -> str:
        self._client.insert(
            self._table,
            {
                "id": decision.decision_id,
                "action": decision.action.value,
                "failure_signature": decision.failure_signature,
                "memory_id": decision.selected_memory_id,
                "body_id": body_id,
                "regime_label": decision.regime_label,
                "applicability_score": decision.applicability_score,
                "reason_codes": ",".join(decision.reason_codes),
                "outcome": OUTCOME_PENDING,
                "wrong_body_apply": 0,
                "wrong_joint_apply": 0,
                "wrong_regime_apply": 0,
                "created_at": time.time(),
            },
        )
        return decision.decision_id

    def record_outcome(
        self,
        decision_id: str,
        outcome: str,
        *,
        wrong_body: bool = False,
        wrong_joint: bool = False,
        wrong_regime: bool = False,
    ) -> bool:
        if outcome not in {OUTCOME_HELPFUL, OUTCOME_HARMFUL, OUTCOME_NEUTRAL}:
            raise ValueError(f"invalid outcome {outcome!r}")
        rows = self._client.query(self._table, filters={"id": decision_id}, limit=1)
        if not rows:
            return False
        row = dict(rows[0])
        row["outcome"] = outcome
        row["wrong_body_apply"] = 1 if wrong_body else 0
        row["wrong_joint_apply"] = 1 if wrong_joint else 0
        row["wrong_regime_apply"] = 1 if wrong_regime else 0
        self._client.insert(self._table, row)
        return True

    def metrics(self, *, limit: int = 100_000) -> dict[str, Any]:
        rows = self._client.query(self._table, limit=limit)
        eligible = len(rows)
        applies = [row for row in rows if row.get("action") == "APPLY"]
        abstains = [row for row in rows if row.get("action") == "ABSTAIN"]
        helpful = [row for row in applies if row.get("outcome") == OUTCOME_HELPFUL]
        harmful = [row for row in applies if row.get("outcome") == OUTCOME_HARMFUL]
        judged = helpful + harmful

        apply_count = len(applies)
        harm_upper = wilson_interval(len(harmful), len(judged))["upper"]
        return {
            "eligible": eligible,
            "apply_count": apply_count,
            "suggest_count": sum(1 for row in rows if row.get("action") == "SUGGEST"),
            "escalate_count": sum(1 for row in rows if row.get("action") == "ESCALATE"),
            "coverage": (apply_count / eligible) if eligible else 0.0,
            "abstention_rate": (len(abstains) / eligible) if eligible else 0.0,
            "helpful_apply_precision": (len(helpful) / apply_count) if apply_count else None,
            "selective_harm_risk": (len(harmful) / len(judged)) if judged else None,
            "selective_harm_risk_upper_ci": harm_upper if judged else None,
            "wrong_body_apply": sum(int(row.get("wrong_body_apply") or 0) for row in applies),
            "wrong_joint_apply": sum(int(row.get("wrong_joint_apply") or 0) for row in applies),
            "wrong_regime_apply": sum(int(row.get("wrong_regime_apply") or 0) for row in applies),
            "judged_applies": len(judged),
        }

    def gate_report(self, *, limit: int = 100_000) -> dict[str, Any]:
        """v4 §7.5 initial gate: harm upper CI ≤ 5%, wrong-* APPLY == 0."""
        metrics = self.metrics(limit=limit)
        harm_ok = (
            metrics["selective_harm_risk_upper_ci"] is not None
            and metrics["selective_harm_risk_upper_ci"] <= HARM_RISK_GATE
        )
        wrong_ok = (
            metrics["wrong_body_apply"] == 0
            and metrics["wrong_joint_apply"] == 0
            and metrics["wrong_regime_apply"] == 0
        )
        return {
            "passed": harm_ok and wrong_ok,
            "harm_risk_upper_ci_ok": harm_ok,
            "wrong_apply_zero": wrong_ok,
            "metrics": metrics,
        }

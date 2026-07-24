"""Regime benchmark evaluation metrics (数据库优化v4 §10.3).

Per-query labels (ground truth, independent of the matcher under test):

    relevant                            — semantically about this failure
    applicable                          — AND valid in the query's regime
    semantically_related_but_inapplicable — same failure, WRONG regime
    contraindicated                     — harmful in the query's regime

Metrics: Retrieval Recall@K, Applicable Recall@K, Inapplicable Top-1,
Contraindicated Top-1, Abstention Accuracy, Apply Precision, Regime
Confusion Rate.
"""

from __future__ import annotations

from typing import Any


def retrieval_recall_at_k(ranked_ids: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 1.0 if not ranked_ids else 0.0
    top = set(ranked_ids[:k])
    return 1.0 if any(rid in top for rid in relevant) else 0.0


def applicable_recall_at_k(
    ranked_ids: list[str], applicable: list[str], judged_applicable: set[str], k: int
) -> float:
    """A truly-applicable memory in top-K that the gate ALSO judged applicable.

    Ground truth (``applicable``) and the system judgment
    (``judged_applicable``) must agree on the same memory for the point to
    count — surfacing a memory the gate then rejects is not a win.
    """
    if not applicable:
        return 1.0
    top = ranked_ids[:k]
    hits = [rid for rid in top if rid in applicable and rid in judged_applicable]
    return 1.0 if hits else 0.0


def inapplicable_top1(ranked_ids: list[str], inapplicable: list[str]) -> float:
    """Top-1 is semantically related but regime-inapplicable (the PR #98 harm)."""
    if not ranked_ids:
        return 0.0
    return 1.0 if ranked_ids[0] in inapplicable else 0.0


def contraindicated_top1(ranked_ids: list[str], contraindicated: list[str]) -> float:
    """Top-1 is contraindicated in the query's regime — must always be 0."""
    if not ranked_ids:
        return 0.0
    return 1.0 if ranked_ids[0] in contraindicated else 0.0


def abstention_correct(action: str, has_validated_applicable_gt: bool) -> bool:
    """ABSTAIN is correct exactly when no VALIDATED applicable memory
    exists — thin/observed-only evidence legitimately yields abstention
    (the healthy-regime no-harm policy); APPLY is correct only when a
    validated applicable memory exists.  SUGGEST/ESCALATE are always
    safe (no autonomous motion)."""
    if action == "ABSTAIN":
        return not has_validated_applicable_gt
    if action == "APPLY":
        return has_validated_applicable_gt
    return True  # SUGGEST / ESCALATE are always safe choices


def apply_correct(action: str, applied_memory: str | None, applicable: list[str]) -> bool | None:
    """APPLY precision: an APPLY whose memory is truly applicable."""
    if action != "APPLY":
        return None
    return applied_memory in applicable


def regime_confusion(judged_applicable: set[str], inapplicable: list[str]) -> float:
    """A regime-inapplicable memory judged applicable — always counts."""
    return 1.0 if any(rid in judged_applicable for rid in inapplicable) else 0.0


def aggregate(records: list[dict[str, Any]], *, k: int = 5) -> dict[str, Any]:
    """Aggregate per-query metric records (mean over queries)."""
    n = len(records)
    if n == 0:
        return {"queries": 0}

    def _mean(key: str) -> float:
        return round(sum(r[key] for r in records) / n, 4)

    applies = [r for r in records if r["apply_correct"] is not None]
    apply_precision = (
        round(sum(1 for r in applies if r["apply_correct"]) / len(applies), 4) if applies else None
    )
    return {
        "queries": n,
        "k": k,
        "retrieval_recall_at_k": _mean("retrieval_recall"),
        "applicable_recall_at_k": _mean("applicable_recall"),
        "inapplicable_top1_rate": _mean("inapplicable_top1"),
        "contraindicated_top1_rate": _mean("contraindicated_top1"),
        "abstention_accuracy": _mean("abstention_correct"),
        "apply_precision": apply_precision,
        "apply_decisions": len(applies),
        "regime_confusion_rate": _mean("regime_confusion"),
    }

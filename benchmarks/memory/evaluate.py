"""Evaluation metrics for the memory retrieval benchmark (§6.7)."""

from __future__ import annotations

import math
from typing import Any


def recall_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    """1.0 when any relevant (grade >= 1) item appears in the top-k."""
    if not relevant:
        return 1.0  # decoy query: vacuously correct to return nothing
    top = set(ranked_ids[:k])
    return 1.0 if any(rid in top for rid in relevant) else 0.0


def reciprocal_rank(ranked_ids: list[str], relevant: dict[str, int]) -> float:
    if not relevant:
        return 1.0 if not ranked_ids else 0.0
    for rank, rid in enumerate(ranked_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    score = 0.0
    for rank, rid in enumerate(ranked_ids[:k], start=1):
        grade = relevant.get(rid, 0)
        if grade:
            score += (2**grade - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    if not relevant:
        return 1.0 if not ranked_ids else 0.0
    ideal = sorted(relevant.values(), reverse=True)[:k]
    ideal_dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevant, k) / ideal_dcg


def cross_robot_leakage(ranked: list[dict[str, Any]], expected_robot: str | None) -> float:
    """Fraction of results leaking a different robot into a robot-scoped query."""
    if expected_robot is None or not ranked:
        return 0.0
    leaked = sum(1 for item in ranked if item.get("robot_id") not in (None, expected_robot))
    return leaked / len(ranked)


def stale_memory_rate(ranked: list[dict[str, int]], relevant: dict[str, int]) -> float:
    """Fraction of top results that are stale (non-relevant, superseded-style)
    when a fresh relevant answer exists.  Caller passes age-in-days via the
    ranked dicts (``age_days``) and marks stale via ``stale=True``."""
    if not ranked or not relevant:
        return 0.0
    stale = sum(1 for item in ranked if item.get("stale") and item["memory_id"] not in relevant)
    return stale / len(ranked)


def precision_at_k(ranked_ids: list[str], relevant: dict[str, int], k: int) -> float:
    if not ranked_ids[:k]:
        return 1.0 if not relevant else 0.0
    hits = sum(1 for rid in ranked_ids[:k] if rid in relevant)
    return hits / len(ranked_ids[:k])


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[rank]


def aggregate(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-query metric dicts into the benchmark summary."""
    n = len(per_query) or 1
    latencies = [q["latency_ms"] for q in per_query]
    leakages = [q["leakage"] for q in per_query if q["expected_robot"]]
    return {
        "queries": len(per_query),
        "recall@1": round(sum(q["recall@1"] for q in per_query) / n, 4),
        "recall@5": round(sum(q["recall@5"] for q in per_query) / n, 4),
        "recall@10": round(sum(q["recall@10"] for q in per_query) / n, 4),
        "mrr": round(sum(q["rr"] for q in per_query) / n, 4),
        "ndcg@5": round(sum(q["ndcg@5"] for q in per_query) / n, 4),
        "context_precision@5": round(sum(q["precision@5"] for q in per_query) / n, 4),
        "cross_robot_leakage": round(sum(leakages) / max(len(leakages), 1), 4),
        "stale_memory_rate": round(sum(q["stale"] for q in per_query) / n, 4),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
        },
    }

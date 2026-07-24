"""Regime benchmark harness tests (PR-BENCH-4, v4 §10/§13)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "memory" / "regime"))

import evaluate_regime as ev  # noqa: E402
from fixture_corpus import TEST_SESSIONS, corpus, queries  # noqa: E402
from run_regime_benchmark import run_queries  # noqa: E402

aggregate = ev.aggregate

HUMAN_QUERIES = REPO_ROOT / "benchmarks" / "memory" / "regime" / "human_queries.jsonl"
REQUIRED_LABEL_FIELDS = {
    "query_id",
    "text",
    "regime",
    "relevant",
    "applicable",
    "applicable_validated",
    "semantically_related_but_inapplicable",
    "contraindicated",
}


def _all_queries() -> list[dict]:
    human = [
        json.loads(line)
        for line in HUMAN_QUERIES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return queries() + human


def test_human_query_lane_shape() -> None:
    """v4 §10.2: ≥100 hand-authored free queries with the 4-label schema."""
    rows = [
        json.loads(line)
        for line in HUMAN_QUERIES.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) >= 100
    regimes = set()
    sessions = set()
    for row in rows:
        assert set(row) >= REQUIRED_LABEL_FIELDS, row.get("query_id")
        regimes.add(row["regime"])
        sessions.add(row.get("session"))
    # Cold/warm/hot contexts and both splits are exercised.
    assert {"cold", "warm", "hot"} <= regimes
    assert set(TEST_SESSIONS) & sessions


def test_session_holdout_split() -> None:
    """v4 §10.2: test queries come from holdout sessions only."""
    rows = _all_queries()
    test_rows = [q for q in rows if q.get("session") in TEST_SESSIONS]
    dev_rows = [q for q in rows if q.get("session") not in TEST_SESSIONS]
    assert test_rows, "expected holdout queries"
    assert dev_rows, "expected dev queries"
    hot = [q for q in test_rows if q["regime"] == "hot"]
    assert hot, "holdout must include the new-temperature regime"


def test_benchmark_gates_hold() -> None:
    """v4 Gate C: abstention correct, no wrong-regime apply, no confusion."""
    records = run_queries(_all_queries(), k=5)
    summary = aggregate(records, k=5)
    assert summary["abstention_accuracy"] == 1.0
    assert summary["apply_precision"] == 1.0
    assert summary["regime_confusion_rate"] == 0.0
    assert summary["apply_decisions"] > 0
    # Retrieval surfaces wrong-regime memories (that's its job); the gate
    # blocks them — the two rates must both be non-trivial to prove the
    # negative lanes actually bite.
    assert summary["inapplicable_top1_rate"] > 0.0
    assert summary["contraindicated_top1_rate"] > 0.0


def test_benchmark_deterministic_across_runs() -> None:
    first = aggregate(run_queries(_all_queries(), k=5), k=5)
    second = aggregate(run_queries(_all_queries(), k=5), k=5)
    assert first == second


def test_counter_regime_negative_lane() -> None:
    """v4 §11.3: hot memory in a healthy regime → retrieved, never applied."""
    cold_queries = [q for q in _all_queries() if q["regime"] == "cold" and q["contraindicated"]]
    assert cold_queries, "expected counter-regime queries"
    records = run_queries(cold_queries, k=5)
    for record in records:
        assert record["decision"] == "ABSTAIN", record["query_id"]
        assert record["regime_confusion"] == 0.0


def test_evaluate_metric_units() -> None:
    assert ev.retrieval_recall_at_k(["a", "b"], ["b"], 1) == 0.0
    assert ev.retrieval_recall_at_k(["a", "b"], ["b"], 2) == 1.0
    assert ev.inapplicable_top1(["x"], ["x"]) == 1.0
    assert ev.contraindicated_top1([], ["x"]) == 0.0
    assert ev.abstention_correct("ABSTAIN", False) is True
    assert ev.abstention_correct("ABSTAIN", True) is False
    assert ev.abstention_correct("APPLY", True) is True
    assert ev.abstention_correct("SUGGEST", False) is True
    assert ev.apply_correct("APPLY", "m1", ["m1"]) is True
    assert ev.apply_correct("APPLY", "m2", ["m1"]) is False
    assert ev.apply_correct("ABSTAIN", None, ["m1"]) is None
    assert ev.regime_confusion({"a"}, ["a"]) == 1.0
    assert ev.regime_confusion({"b"}, ["a"]) == 0.0


def test_corpus_ground_truth_is_not_matcher_derived() -> None:
    """Guard against the circular-labels failure mode: hot memories must be
    inapplicable to cold queries BY CONSTRUCTION (session + envelope)."""
    memories, envelopes = corpus()
    hot_ids = {m["memory_id"] for m in memories if m["session_id"] == "sess_test_hot_01"}
    cold_contra = [
        e
        for e in envelopes
        if e["envelope_type"] == "contraindicated" and "COLD_HEALTHY" in (e["regime_labels"] or [])
    ]
    contra_ids = {e["memory_id"] for e in cold_contra}
    # The run1 death-spiral memory is contraindicated in healthy regimes.
    assert "mem_hot_middle_jnr" in contra_ids
    assert "mem_hot_middle_jnr" in hot_ids

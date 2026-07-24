"""Regime applicability benchmark runner (数据库优化v4 §10, PR-BENCH-4).

Pipeline per query:

    session-holdout corpus (fixture or real-session-derived)
    → facade.retrieve (HUMAN_SEARCH — retrieval quality measured first)
    → RegimeMatcher on candidates (applicability gate)
    → SelectiveInterventionPipeline.decide (ABSTAIN behavior)
    → per-query metrics + aggregate (v4 §10.3)

Splits: dev queries (historical sessions) vs test queries (new sessions,
new temperature processes) are reported separately — no same-session
memory/query pair crosses the split.

Usage::

    python benchmarks/memory/regime/run_regime_benchmark.py \
        [--queries benchmarks/memory/regime/queries_regime.jsonl] \
        [--human benchmarks/memory/regime/human_queries.jsonl] \
        [--out /tmp/regime_bench]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import evaluate_regime as ev
from fixture_corpus import REGIME_CONTEXTS, TEST_SESSIONS, corpus, queries

from rosclaw.how.selective import SelectiveInterventionPipeline
from rosclaw.memory.seekdb_client import InMemoryKnowledgeStore
from rosclaw.memory.v2.models import MemoryItem
from rosclaw.memory.v2.regime import (
    ApplicabilityEnvelope,
    ApplicabilityStore,
    RegimeMatcher,
    empty_regime,
)
from rosclaw.memory.v2.runtime_retrieval import build_retrieval_facade

REPO_ROOT = Path(__file__).resolve().parents[3]


def _regime_for(context: dict[str, Any], *, body_id: str | None, joint: str | None):
    regime = empty_regime(
        robot_id="rh56_rps_robot",
        body_id=body_id or "rh56_right_01",
        task_id="rh56_rps",
    )
    for key, value in context.items():
        setattr(regime, key, value)
    regime.joint_name = joint
    regime.confidence = 0.85
    return regime


def build_stack() -> tuple[Any, ApplicabilityStore, list[dict[str, Any]]]:
    """In-memory fixture store with corpus memories + envelopes.

    Timestamps are FIXED (session-anchored) so ranking is deterministic —
    recency jitter between runs must never flip a benchmark result.
    """
    client = InMemoryKnowledgeStore()
    client.connect()
    memories, envelopes = corpus()
    # One shared timestamp for every item: recency then contributes the
    # SAME score to all candidates and can never flip an order between
    # runs (the retriever's clock is wall time).  The benchmark measures
    # lexical/metadata/fusion + gates, not recency.
    stamp = 1_784_000_000.0
    for index, record in enumerate(memories):
        item = MemoryItem(
            memory_id=record["memory_id"],
            memory_type=record["memory_type"],
            robot_id=record["robot_id"],
            session_id=record["session_id"],
            body_id=record["body_id"],
            joint_name=record["joint_name"],
            failure_type=record["failure_type"],
            title=record["title"],
            document=record["document"],
            outcome=record["outcome"],
            evidence_refs=record["evidence_refs"],
            metadata=record["metadata"],
            event_time=stamp,
            created_at=stamp,
            updated_at=stamp,
        )
        client.insert("memory_items", item.to_record())
    store = ApplicabilityStore(client)
    for raw in envelopes:
        store.upsert(ApplicabilityEnvelope.from_record(raw))
    return client, store, memories


def run_queries(
    query_rows: list[dict[str, Any]],
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    client, applicability_store, _ = build_stack()
    facade = build_retrieval_facade(sqlite_store=client)
    matcher = RegimeMatcher()
    choreography = None
    contract_path = REPO_ROOT / "configs" / "choreography" / "rh56_rps_v1.yaml"
    if contract_path.is_file():
        from rosclaw.how.choreography import ChoreographyValidator, load_contract

        choreography = ChoreographyValidator(load_contract(str(contract_path)))
    pipeline = SelectiveInterventionPipeline(
        facade, applicability_store, choreography_validator=choreography
    )
    from rosclaw.memory.v2.retrieval import MemoryQuery
    from rosclaw.memory.v2.runtime_retrieval import RetrievalPurpose

    records: list[dict[str, Any]] = []
    for row in query_rows:
        context = REGIME_CONTEXTS[row["regime"]]
        regime = _regime_for(context, body_id=row.get("body_id"), joint=row.get("joint_name"))
        response = facade.retrieve(
            MemoryQuery(text=row["text"], outcome="failure", limit=k),
            purpose=RetrievalPurpose.HUMAN_SEARCH,
        )
        ranked_ids = [c.memory_id for c in response.candidates]

        judged_applicable: set[str] = set()
        for candidate in response.candidates:
            envelopes = applicability_store.for_memory(candidate.memory_id)
            result = matcher.match(candidate.memory_id, envelopes, regime)
            if result.applicable:
                judged_applicable.add(candidate.memory_id)

        decision = pipeline.decide(
            row["text"],
            regime,
            body_id=row.get("body_id"),
            joint_name=row.get("joint_name"),
        )

        records.append(
            {
                "query_id": row["query_id"],
                "regime": row["regime"],
                "ranked_ids": ranked_ids,
                "decision": decision.action.value,
                "selected_memory_id": decision.selected_memory_id,
                "retrieval_recall": ev.retrieval_recall_at_k(ranked_ids, row["relevant"], k),
                "applicable_recall": ev.applicable_recall_at_k(
                    ranked_ids, row["applicable"], judged_applicable, k
                ),
                "inapplicable_top1": ev.inapplicable_top1(
                    ranked_ids, row["semantically_related_but_inapplicable"]
                ),
                "contraindicated_top1": ev.contraindicated_top1(ranked_ids, row["contraindicated"]),
                "abstention_correct": 1.0
                if ev.abstention_correct(
                    decision.action.value, bool(row.get("applicable_validated"))
                )
                else 0.0,
                "apply_correct": ev.apply_correct(
                    decision.action.value,
                    decision.selected_memory_id,
                    row.get("applicable_validated", row["applicable"]),
                ),
                "regime_confusion": ev.regime_confusion(
                    judged_applicable, row["semantically_related_but_inapplicable"]
                ),
            }
        )
    return records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--queries",
        default=str(Path(__file__).resolve().parent / "queries_regime.jsonl"),
    )
    parser.add_argument(
        "--human",
        default=str(Path(__file__).resolve().parent / "human_queries.jsonl"),
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out or f"/tmp/regime_bench_{int(time.time())}")
    out_dir.mkdir(parents=True, exist_ok=True)

    machine_queries = queries()
    human_queries = _load_jsonl(Path(args.human)) if Path(args.human).is_file() else []
    all_queries = machine_queries + human_queries

    dev_rows = [q for q in all_queries if q.get("session") not in TEST_SESSIONS]
    test_rows = [q for q in all_queries if q.get("session") in TEST_SESSIONS]
    # Queries without an explicit session tag are development by default
    # (they were authored against the dev corpus).

    dev_records = run_queries(dev_rows, k=args.k)
    test_records = run_queries(test_rows, k=args.k)

    summary = {
        "benchmark": "regime_applicability_v1",
        "k": args.k,
        "dev": ev.aggregate(dev_records, k=args.k),
        "test_holdout": ev.aggregate(test_records, k=args.k),
        "all": ev.aggregate(dev_records + test_records, k=args.k),
        "session_split": {
            "dev_sessions": "historical (fixture sess_dev_*)",
            "test_sessions": list(TEST_SESSIONS),
        },
    }
    (out_dir / "per_query_dev.json").write_text(
        json.dumps(dev_records, indent=2, ensure_ascii=False)
    )
    (out_dir / "per_query_test.json").write_text(
        json.dumps(test_records, indent=2, ensure_ascii=False)
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

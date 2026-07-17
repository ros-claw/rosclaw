#!/usr/bin/env python3
"""Run the memory retrieval benchmark (§6.7).

Compares three retriever configurations over the same dataset/queries:

* ``keyword`` — metadata filter + lexical token overlap only;
* ``vector`` — TF-IDF vector similarity only (SQLite fallback);
* ``hybrid`` — full fusion pipeline (lexical + vector + metadata + recency
  + evidence + confidence).

Writes ``benchmark_results.json`` next to this script.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from evaluate import (  # noqa: E402  # noqa: E402
    aggregate,  # noqa: E402
    cross_robot_leakage,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore  # noqa: E402
from rosclaw.memory.v2.index import EmbeddingIndexManager  # noqa: E402
from rosclaw.memory.v2.models import MemoryItem  # noqa: E402
from rosclaw.memory.v2.repository import MemoryRepository  # noqa: E402
from rosclaw.memory.v2.retrieval import MemoryQuery, MemoryRetriever  # noqa: E402
from rosclaw.storage.vector import SQLiteVectorStore, TfidfEmbedder  # noqa: E402


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _item_from_bench(row: dict) -> MemoryItem:
    return MemoryItem(
        memory_id=row["memory_id"],
        memory_type=row["memory_type"],
        robot_id=row["robot_id"],
        body_id=row.get("body_id"),
        task_id=row.get("task_id"),
        title=row["title"],
        document=row["document"],
        outcome=row.get("outcome"),
        confidence=row.get("confidence", 0.85),
        importance=row.get("importance", 0.6),
        event_time=row["event_time"],
        evidence_refs=row.get("evidence_refs", []),
        tags=row.get("tags", []),
        content_hash=row.get("content_hash", ""),
        status=row.get("status", "active"),
    )


def _build_stack(dataset: list[dict], tmpdir: str, *, with_vector: bool):
    client = SQLiteKnowledgeStore(str(Path(tmpdir) / "knowledge.sqlite"))
    client.connect()
    repo = MemoryRepository(client)
    items = [_item_from_bench(row) for row in dataset]
    for item in items:
        repo.store(item)
    vector = SQLiteVectorStore(client) if with_vector else None
    embedder = TfidfEmbedder() if with_vector else None
    if vector is not None and embedder is not None:
        embedder.fit([f"{i.title}\n{i.document}" for i in items])
        manager = EmbeddingIndexManager(client, vector)
        manager.build(items, embedder)
    return client, repo, vector, embedder


def _run_queries(
    retriever: MemoryRetriever,
    queries: list[dict],
    labels: dict[str, dict[str, int]],
    dataset_by_id: dict[str, dict],
) -> list[dict]:
    per_query: list[dict] = []
    for query in queries:
        memory_query = MemoryQuery(
            text=query["text"],
            memory_types=query.get("memory_types") or [],
            robot_id=query.get("robot_id"),
            body_id=query.get("body_id"),
            task_id=query.get("task_id"),
            limit=10,
        )
        start = time.perf_counter()
        results = retriever.retrieve(memory_query)
        latency_ms = (time.perf_counter() - start) * 1000.0
        relevant = labels.get(query["query_id"], {})
        ranked_ids = [r.memory_id for r in results]
        ranked_dicts = [
            {
                "memory_id": r.memory_id,
                "robot_id": r.memory.robot_id,
                "age_days": (1784208000.0 - r.memory.event_time) / 86400.0,
                "stale": "已被新证据取代" in r.memory.document,
            }
            for r in results
        ]
        per_query.append(
            {
                "query_id": query["query_id"],
                "expected_robot": query.get("robot_id"),
                "recall@1": recall_at_k(ranked_ids, relevant, 1),
                "recall@5": recall_at_k(ranked_ids, relevant, 5),
                "recall@10": recall_at_k(ranked_ids, relevant, 10),
                "rr": reciprocal_rank(ranked_ids, relevant),
                "ndcg@5": ndcg_at_k(ranked_ids, relevant, 5),
                "precision@5": precision_at_k(ranked_ids, relevant, 5),
                "leakage": cross_robot_leakage(ranked_dicts, query.get("robot_id")),
                "stale": (
                    sum(1 for d in ranked_dicts if d["stale"] and d["memory_id"] not in relevant)
                    / max(len(ranked_dicts), 1)
                ),
                "latency_ms": latency_ms,
            }
        )
    return per_query


def main() -> int:
    dataset = _load_jsonl(HERE / "dataset.jsonl")
    queries = _load_jsonl(HERE / "queries.jsonl")
    labels = {
        row["query_id"]: row["relevance"] for row in _load_jsonl(HERE / "relevance_labels.jsonl")
    }
    dataset_by_id = {row["memory_id"]: row for row in dataset}

    results: dict[str, object] = {"dataset_size": len(dataset), "query_count": len(queries)}

    # keyword (lexical + metadata only)
    with tempfile.TemporaryDirectory() as tmpdir:
        client, repo, _, _ = _build_stack(dataset, tmpdir, with_vector=False)
        retriever = MemoryRetriever(repo)
        results["keyword"] = aggregate(_run_queries(retriever, queries, labels, dataset_by_id))
        client.disconnect()

    # vector only (TF-IDF vectors, lexical weight 0)
    with tempfile.TemporaryDirectory() as tmpdir:
        client, repo, vector, embedder = _build_stack(dataset, tmpdir, with_vector=True)
        vector_only_weights = {
            "vector": 0.70,
            "lexical": 0.0,
            "metadata": 0.10,
            "recency": 0.05,
            "evidence": 0.05,
            "confidence": 0.10,
        }
        retriever = MemoryRetriever(
            repo, vector_store=vector, embedder=embedder, fusion_weights=vector_only_weights
        )
        results["vector_tfidf"] = aggregate(_run_queries(retriever, queries, labels, dataset_by_id))
        client.disconnect()

    # hybrid (full fusion pipeline)
    with tempfile.TemporaryDirectory() as tmpdir:
        client, repo, vector, embedder = _build_stack(dataset, tmpdir, with_vector=True)
        retriever = MemoryRetriever(repo, vector_store=vector, embedder=embedder)
        results["hybrid"] = aggregate(_run_queries(retriever, queries, labels, dataset_by_id))
        client.disconnect()

    out_path = HERE / "benchmark_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

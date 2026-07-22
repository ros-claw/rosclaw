#!/usr/bin/env python3
"""Multilingual embedding bake-off (数据库优化v3 §10).

For every candidate profile: build a versioned collection from the SAME
corpus on the real SeekDB server, run the labeled query set, compute
Recall@1/5/10, MRR, nDCG@5, per-kind CJK/cross-lingual/error-code
recall, joint confusion (hard-negative top1 in forbidden), cross-body
leakage, and query latency.  Optionally add a reranker lane for the
high-risk kinds.

Writes reports/embedding_bakeoff/<timestamp>/ per §16.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "src"))

from evaluate import ndcg_at_k, recall_at_k  # noqa: E402

from rosclaw.embedding.registry import get_provider  # noqa: E402
from rosclaw.storage.seekdb_native import SeekDBServerStore  # noqa: E402
from rosclaw.storage.versioned_collections import (  # noqa: E402
    VersionedCollectionManager,
)

PROFILES = [
    "qwen3_06b_1024_v1",
    "qwen3_06b_768_v1",
    "qwen3_06b_512_v1",
    "gte_multi_768_v1",
    "gte_multi_512_v1",
]
LOGICAL = "memory_bench"
RERANK_KINDS = {"hard_negative_body", "error_code", "same_symptom_diff_cause"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * q))], 1)


def evaluate_ranking(ranked_ids: list[str], labels: dict[str, int]) -> dict:
    return {
        "recall@1": recall_at_k(ranked_ids, labels, 1),
        "recall@5": recall_at_k(ranked_ids, labels, 5),
        "recall@10": recall_at_k(ranked_ids, labels, 10),
        "mrr": _mrr(ranked_ids, labels),
        "ndcg@5": ndcg_at_k(ranked_ids, labels, 5),
    }


def _mrr(ranked_ids: list[str], labels: dict[str, int]) -> float:
    for rank, mid in enumerate(ranked_ids, start=1):
        if labels.get(mid, 0) > 0:
            return 1.0 / rank
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(HERE / "v3"))
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--include-minilm", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--database", default="rosclaw")
    parser.add_argument("--cache", default="/tmp/mem3_scratch/embedding_cache.sqlite")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    corpus = load_jsonl(data_dir / "dataset.jsonl")
    queries = load_jsonl(data_dir / "queries.jsonl")
    print(f"corpus={len(corpus)} queries={len(queries)}")

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir = Path(args.out or f"reports/embedding_bakeoff/{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    store = SeekDBServerStore(host=args.host, port=args.port, database=args.database)
    store.connect()
    records = corpus

    results: dict[str, dict] = {}
    profiles_run = [p for p in args.profiles.split(",") if p]
    for profile_id in profiles_run:
        provider = get_provider(profile_id, cache_path=args.cache)
        mgr = VersionedCollectionManager(store, provider)
        print(f"--- building {profile_id} ({provider.profile.dimension}d)")
        t0 = time.monotonic()
        mgr.build(LOGICAL, records, analyzer="ngram")
        build_s = time.monotonic() - t0
        per_query = []
        lat: list[float] = []
        for query in queries:
            t1 = time.monotonic()
            rows = mgr.shadow_query(
                LOGICAL, query["text"], analyzer="ngram", limit=10, candidate_window=20
            )
            lat.append((time.monotonic() - t1) * 1000.0)
            ranked = [r["id"] for r in rows]
            entry = {"query_id": query["id"], "kind": query["kind"], "ranked": ranked}
            entry.update(evaluate_ranking(ranked, query["labels"]))
            entry["top1_forbidden"] = bool(
                ranked and ranked[0] in set(query.get("forbidden") or [])
            )
            per_query.append(entry)
        results[profile_id] = {
            "build_s": round(build_s, 1),
            "per_query": per_query,
            "latency": {
                "p50": percentile(lat, 0.50),
                "p95": percentile(lat, 0.95),
                "p99": percentile(lat, 0.99),
            },
        }
        print(f"    build {build_s:.1f}s, {len(queries)} queries, p50 {percentile(lat, 0.50)}ms")

    # MiniLM 384 built-in baseline on the same corpus (server-side embedder)
    if args.include_minilm:
        lane = "minilm_384_builtin"
        name = f"{LOGICAL}__minilm384_baseline"
        print(f"--- building {lane}")
        import contextlib

        with contextlib.suppress(Exception):
            store._client.delete_collection(name)
        t0 = time.monotonic()
        for record in records:
            row = dict(record)
            row.setdefault("id", record["id"])
            store.insert(name, row)
        store.refresh_index(name)
        build_s = time.monotonic() - t0
        per_query = []
        lat = []
        for query in queries:
            t1 = time.monotonic()
            rows = store.hybrid_search(name, query["text"], limit=10, candidate_window=20)
            lat.append((time.monotonic() - t1) * 1000.0)
            ranked = [r["id"] for r in rows]
            entry = {"query_id": query["id"], "kind": query["kind"], "ranked": ranked}
            entry.update(evaluate_ranking(ranked, query["labels"]))
            entry["top1_forbidden"] = bool(
                ranked and ranked[0] in set(query.get("forbidden") or [])
            )
            per_query.append(entry)
        results[lane] = {
            "build_s": round(build_s, 1),
            "per_query": per_query,
            "latency": {
                "p50": percentile(lat, 0.50),
                "p95": percentile(lat, 0.95),
                "p99": percentile(lat, 0.99),
            },
        }
        print(f"    build {build_s:.1f}s, p50 {percentile(lat, 0.50)}ms")

    # Reranker lane on the first Qwen profile (high-risk kinds only)
    if args.reranker and any(p.startswith("qwen3") for p in profiles_run):
        from rosclaw.embedding.reranker import Qwen3RerankerProvider

        base_profile = next(p for p in profiles_run if p.startswith("qwen3"))
        provider = get_provider(base_profile, cache_path=args.cache)
        mgr = VersionedCollectionManager(store, provider)
        reranker = Qwen3RerankerProvider()
        print(f"--- reranker lane on {base_profile}")
        per_query = []
        lat = []
        high_risk = [q for q in queries if q["kind"] in RERANK_KINDS]
        for query in high_risk:
            t1 = time.monotonic()
            rows = mgr.shadow_query(
                LOGICAL, query["text"], analyzer="ngram", limit=20, candidate_window=20
            )
            reranked = reranker.rerank(query["text"], rows, top_k=5)
            lat.append((time.monotonic() - t1) * 1000.0)
            ranked = [r["id"] for r in reranked]
            entry = {"query_id": query["id"], "kind": query["kind"], "ranked": ranked}
            entry.update(evaluate_ranking(ranked, query["labels"]))
            entry["top1_forbidden"] = bool(
                ranked and ranked[0] in set(query.get("forbidden") or [])
            )
            per_query.append(entry)
        results[f"{base_profile}+reranker"] = {
            "subset": sorted(RERANK_KINDS),
            "per_query": per_query,
            "latency": {
                "p50": percentile(lat, 0.50),
                "p95": percentile(lat, 0.95),
                "p99": percentile(lat, 0.99),
            },
        }

    # Aggregate metrics per lane
    summary: dict[str, dict] = {}
    for lane, data in results.items():
        pq = data["per_query"]
        agg: dict[str, float] = {}
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5"):
            agg[metric] = round(statistics.fmean(e[metric] for e in pq), 4)
        kinds: dict[str, dict] = {}
        for kind in sorted({e["kind"] for e in pq}):
            subset = [e for e in pq if e["kind"] == kind]
            kinds[kind] = {
                "recall@1": round(statistics.fmean(e["recall@1"] for e in subset), 4),
                "n": len(subset),
            }
        hard_neg = [e for e in pq if e["kind"] == "hard_negative_body"]
        agg["joint_body_confusion_top1"] = (
            round(statistics.fmean(1.0 if e["top1_forbidden"] else 0.0 for e in hard_neg), 4)
            if hard_neg
            else None
        )
        agg["kinds"] = kinds
        agg["latency"] = data.get("latency")
        agg["build_s"] = data.get("build_s")
        summary[lane] = agg

    (out_dir / "retrieval_metrics.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "per_query.json").write_text(
        json.dumps({lane: data["per_query"] for lane, data in results.items()}, indent=1)
    )
    (out_dir / "environment.json").write_text(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "database": args.database,
                "profiles": profiles_run,
                "minilm_baseline": args.include_minilm,
                "reranker": args.reranker,
            },
            indent=1,
        )
    )
    lines = ["# Embedding bake-off", f"- corpus: {len(corpus)}, queries: {len(queries)}", ""]
    for lane, agg in summary.items():
        lines.append(
            f"- **{lane}**: R@1={agg['recall@1']} R@5={agg['recall@5']} "
            f"MRR={agg['mrr']} nDCG@5={agg['ndcg@5']} "
            f"confusion={agg['joint_body_confusion_top1']} p50={agg['latency']['p50']}ms"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"report -> {out_dir}")
    store.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

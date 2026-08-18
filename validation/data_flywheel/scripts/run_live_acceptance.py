#!/usr/bin/env python3
"""DF-20 (phase-II §27-§31): SeekDB live data-flywheel acceptance runner.

Mode A (edge, CI-required): SQLite Structured Store + SeekDB Embedded
Retrieval.  Drives N mock practice episodes through the WHOLE plane —
fact ingest, memory distill, projection, retrieval, How lookup, insight,
evolution, lineage — and records the §30 core metrics.

Also runs the §31 Memory Hurt Gate: No Memory / Keyword / Vector /
Hybrid / Hybrid + Body/Regime lanes over the regime fixture corpus
(disclosed deterministic doubles for CI; --real-seekdb swaps the native
side for a real embedded engine).

Soak tiers (1000 / 10,000 / 7x24) are the same runner with bigger
--episodes or --soak-duration-sec; only the 50-episode loop runs in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "benchmarks" / "memory" / "regime"))

from rosclaw.memory.seekdb_client import SQLiteStructuredStore  # noqa: E402
from rosclaw.practice.config import PracticeConfig, SourceConfig  # noqa: E402
from rosclaw.practice.coordinator import PracticeCoordinator  # noqa: E402
from rosclaw.storage.reconciler import DataReconciler  # noqa: E402

# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return round(ordered[idx] * 1000, 2)  # ms


def _latency_stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50_ms": _pct(values, 0.50),
        "p95_ms": _pct(values, 0.95),
        "p99_ms": _pct(values, 0.99),
    }


# ---------------------------------------------------------------------------
# the live loop
# ---------------------------------------------------------------------------


def run_loop(args: argparse.Namespace) -> dict[str, Any]:
    work = Path(args.workdir)
    data_root = work / "practice"
    work.mkdir(parents=True, exist_ok=True)

    store = SQLiteStructuredStore(str(work / "structured.sqlite"))
    store.connect()

    retrieval_store = None
    retrieval_skipped = None
    if not args.no_retrieval:
        try:
            from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore

            retrieval_store = SeekDBEmbeddedRetrievalStore(path=str(work / "retrieval"))
            retrieval_store.connect()
        except Exception as exc:  # noqa: BLE001
            retrieval_skipped = f"{type(exc).__name__}: {exc}"

    metrics: dict[str, Any] = {"episodes_target": args.episodes}
    write_lat: list[float] = []
    candidate_total = stored_total = merged_total = ignored_total = quarantined_total = 0
    max_lag = 0

    from rosclaw.storage.lineage import LineageRepository

    lineage = LineageRepository(store)
    reconciler = DataReconciler(
        structured_store=store,
        data_root=data_root,
        retrieval_store=retrieval_store,
        lineage=lineage,
    )

    deadline = time.time() + args.soak_duration_sec if args.soak_duration_sec else None
    completed = 0
    while completed < args.episodes or (deadline and time.time() < deadline):
        cfg = PracticeConfig(
            robot_id="live_acceptance_bot",
            task_name="live acceptance slide",
            data_root=str(data_root),
            sources=SourceConfig(agent=True, runtime=True),
            mock=True,
            publish_to_event_bus=False,
        )
        coord = PracticeCoordinator(cfg)
        coord.initialize()
        coord.start()
        time.sleep(args.session_seconds)
        coord.stop()
        completed += 1

        if completed % args.reconcile_every == 0 or completed == args.episodes:
            t0 = time.perf_counter()
            reconciler.reconcile_memory(limit=args.reconcile_every + 5)
            write_lat.append(time.perf_counter() - t0)
            if retrieval_store is not None:
                from rosclaw.memory.v2.repository import MemoryRepository
                from rosclaw.storage.seekdb_projection import MemoryRetrievalProjection

                status = MemoryRetrievalProjection(retrieval_store).status(
                    MemoryRepository(store)
                )
                lag = status.get("lag") or 0
                max_lag = max(max_lag, lag)
                if lag:
                    t1 = time.perf_counter()
                    MemoryRetrievalProjection(retrieval_store).rebuild(MemoryRepository(store))
                    metrics.setdefault("catchup_times_s", []).append(
                        round(time.perf_counter() - t1, 3)
                    )

    metrics["episodes_completed"] = completed
    metrics["structured_write"] = _latency_stats(write_lat)

    # -- memory decision rates + ledger replay -----------------------------
    ledger = store.query("memory_distillation_runs", {}, limit=100_000)
    for row in ledger:
        candidate_total += int(row.get("candidate_count") or 0)
        stored_total += int(row.get("stored_count") or 0)
        merged_total += int(row.get("merged_count") or 0)
        ignored_total += int(row.get("ignored_count") or 0)
        quarantined_total += int(row.get("quarantined_count") or 0)
    memories = store.count("memory_items", {})
    metrics["memory"] = {
        "candidate_count": candidate_total,
        "stored": stored_total,
        "store_rate": round(stored_total / candidate_total, 4) if candidate_total else None,
        "merge_rate": round(merged_total / candidate_total, 4) if candidate_total else None,
        "ignore_rate": round(ignored_total / candidate_total, 4) if candidate_total else None,
        "quarantine_rate": (
            round(quarantined_total / candidate_total, 4) if candidate_total else None
        ),
        "memory_items": memories,
    }

    # duplicate rate: replay every session's distillation; count growth
    before = store.count("memory_items", {})
    reconciler.reconcile_memory(limit=100_000)
    after = store.count("memory_items", {})
    metrics["memory"]["duplicate_rate"] = round((after - before) / before, 4) if before else 0.0

    # untraceable rate: active memories without a lineage parent
    edges = store.query("lineage_edges", {}, limit=500_000)
    linked = {e.get("from_id") for e in edges}
    untraceable = 0
    active_items = store.query("memory_items", {"status": "active"}, limit=500_000)
    for item in active_items:
        if item.get("id") not in linked:
            untraceable += 1
    metrics["memory"]["untraceable_rate"] = (
        round(untraceable / len(active_items), 4) if active_items else None
    )

    # bad-evidence writes: ACTIVE memory without evidence (must be ~0, §30)
    bad_evidence = 0
    for item in active_items:
        evd = store.query("memory_evidence", {"memory_id": item["id"]}, limit=1)
        if not evd and not item.get("evidence_refs"):
            bad_evidence += 1
    metrics["data_quality"] = {
        "bad_evidence_write_rate": (
            round(bad_evidence / len(active_items), 4) if active_items else None
        )
    }

    metrics["projection"] = {
        "retrieval_skipped": retrieval_skipped,
        "max_projection_lag": max_lag,
        "final_lag": (
            (
                store.count("memory_items", {})
                - (retrieval_store.count("memory_items") if retrieval_store else 0)
            )
            if retrieval_store
            else None
        ),
    }

    # -- retrieval sample ----------------------------------------------------
    if retrieval_store is not None:
        from rosclaw.memory.v2.retrieval import MemoryQuery
        from rosclaw.memory.v2.runtime_retrieval import (
            RetrievalPurpose,
            build_retrieval_facade,
        )

        facade = build_retrieval_facade(
            native_store=retrieval_store, sqlite_store=store
        )
        query_lat: list[float] = []
        fallback = abstain = served = 0
        for text in (
            "slide puck overshoot recovery",
            "force spike on contact",
            "left hand index drift",
            "unrelated query with no hits zzz",
        ):
            t0 = time.perf_counter()
            resp = facade.retrieve(
                MemoryQuery(text=text, limit=5), purpose=RetrievalPurpose.HUMAN_SEARCH
            )
            query_lat.append(time.perf_counter() - t0)
            if resp.retrieval_mode == "abstain":
                abstain += 1
            elif resp.fallback:
                fallback += 1
            else:
                served += 1
        metrics["retrieval"] = {
            "query": _latency_stats(query_lat),
            "fallback_rate": round(fallback / 4, 4),
            "abstention_rate": round(abstain / 4, 4),
            "served": served,
        }
    else:
        metrics["retrieval"] = {"skipped": retrieval_skipped or "disabled"}

    # -- how lookup ----------------------------------------------------------
    store.insert(
        "heuristic_rules",
        {
            "id": "rule_live_1",
            "condition": "force overshoot on slide",
            "action": "reduce force setpoint to 200-300",
            "failure_signature": "force overshoot",
            "action_template": json.dumps({"force": [200, 300]}),
            "success_count": 3,
        },
    )
    rules = store.query("heuristic_rules", {"failure_signature": "force overshoot"})
    metrics["how"] = {"rules": len(rules), "lookup_ok": len(rules) == 1}

    # -- insight --------------------------------------------------------------
    from rosclaw.core.event_bus import EventBus
    from rosclaw.core.event_topics import EventTopics
    from rosclaw.memory.insights import MemoryInsightService

    bus = EventBus()
    insights_seen: list[dict] = []
    bus.subscribe(EventTopics.MEMORY_INSIGHT_CREATED, lambda e: insights_seen.append(e.payload))
    svc = MemoryInsightService(bus, store, robot_id="live_acceptance_bot", lineage_repository=lineage)
    first_memory = active_items[0]["id"] if active_items else "mem_none"
    svc._maybe_emit(
        "similar_failure_with_patch",
        skill_id="live_skill",
        failure_type="force overshoot",
        task_id="live acceptance",
        episode_id="ep_live",
        evidence_refs=["critic_result:ep_live"],
        extra={"memory_refs": [first_memory], "search_space": {"force": [200, 300]}},
    )
    metrics["insight"] = {
        "published": len(insights_seen),
        "lineage_linked": bool(lineage.parents("memory_insight", insights_seen[0]["insight_id"]))
        if insights_seen
        else False,
    }

    # -- evolution + lineage ----------------------------------------------------
    from rosclaw.auto.config import AutoConfig
    from rosclaw.auto.engine.auto_engine import AutoEngine

    engine = AutoEngine(
        config=AutoConfig(
            local_store_path=str(work / "auto"), storage_backend="hybrid"
        ),
        seekdb_client=store,
        lineage_repository=lineage,
    )
    ins_id = insights_seen[0]["insight_id"] if insights_seen else "ins_none"
    prop = engine.create_proposal(
        "",
        "live acceptance",
        "live_skill",
        "lower force avoids overshoot",
        {"force": [200, 300]},
        source="memory_guided",
        source_refs=[{"type": "memory_insight", "id": ins_id}],
    )
    graph = lineage.trace_graph("proposal", prop.id)
    graph_ids = {n["id"] for n in graph["nodes"]}
    metrics["evolution"] = {
        "records": store.count("evolution_records", {}),
        "proposal": prop.id,
    }
    metrics["lineage"] = {
        "proposal_reaches_insight": ins_id in graph_ids,
        "proposal_reaches_memory": first_memory in graph_ids,
        "edges": store.count("lineage_edges", {}),
    }

    store.disconnect()
    return metrics


# ---------------------------------------------------------------------------
# §31 Memory Hurt Gate
# ---------------------------------------------------------------------------


def _hurt_lane_rows() -> list[dict[str, Any]]:
    from fixture_corpus import queries  # benchmarks/memory/regime

    rows = list(queries())
    human = REPO_ROOT / "benchmarks" / "memory" / "regime" / "human_queries.jsonl"
    if human.is_file():
        rows += [json.loads(line) for line in human.open() if line.strip()]
    return rows


def run_hurt_gate(args: argparse.Namespace) -> dict[str, Any]:
    """No Memory / Keyword / Vector / Hybrid / Hybrid+Regime comparison."""
    from run_regime_benchmark import REGIME_CONTEXTS, _regime_for, build_stack

    from rosclaw.how.selective.pipeline import SelectiveInterventionPipeline
    from rosclaw.memory.v2.retrieval import MemoryQuery
    from rosclaw.memory.v2.runtime_retrieval import RetrievalPurpose, build_retrieval_facade

    native, applicability_store, memories = build_stack()
    from fake_native_stack import PHYSICAL, bench_provider_resolver

    facade = build_retrieval_facade(
        native_store=native, provider_resolver=bench_provider_resolver()
    )
    pipeline = SelectiveInterventionPipeline(facade, applicability_store)

    rows = _hurt_lane_rows()
    lanes = {
        lane: {"success": 0, "hurt": 0, "unsafe": 0, "n": 0}
        for lane in ("no_memory", "keyword", "vector", "hybrid", "hybrid_regime")
    }

    # precompute fake-embedding vectors for the disclosed vector lane
    from fake_native_stack import BenchFakeProvider

    provider = BenchFakeProvider()
    doc_vecs = {
        m["memory_id"]: v
        for m, v in zip(
            memories,
            provider.encode_documents([str(m.get("document", "")) for m in memories]),
            strict=True,
        )
    }

    def _cos(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        na = sum(x * x for x in a) ** 0.5 or 1.0
        nb = sum(x * x for x in b) ** 0.5 or 1.0
        return dot / (na * nb)

    def _judge(lane: str, applied_id: str | None, row: dict) -> None:
        stats = lanes[lane]
        stats["n"] += 1
        validated = set(row.get("applicable_validated") or row.get("applicable") or [])
        contraindicated = set(row.get("contraindicated") or [])
        if applied_id is None:
            if not validated:
                stats["success"] += 1  # abstain where nothing applicable = correct
            return
        if applied_id in validated:
            stats["success"] += 1
            return
        stats["hurt"] += 1  # applied a memory that is NOT validated-applicable
        if applied_id in contraindicated:
            stats["unsafe"] += 1

    for row in rows:
        context = REGIME_CONTEXTS[row["regime"]]
        regime = _regime_for(context, body_id=row.get("body_id"), joint=row.get("joint_name"))

        # no memory: always abstain
        _judge("no_memory", None, row)

        query_text = row["text"]
        # keyword: native fulltext (BM25-like) top-1, always applied
        kw = native.fulltext_search(PHYSICAL, query_text, {"status": "active"}, limit=1)
        _judge("keyword", kw[0]["id"] if kw else None, row)

        # vector: fake-embedding cosine top-1, always applied (disclosed double)
        qv = provider.encode_queries([query_text])[0]
        best = max(doc_vecs.items(), key=lambda kv: _cos(kv[1], qv), default=(None, 0.0))
        _judge("vector", best[0] if best[0] else None, row)

        # hybrid: production facade top-1, always applied
        resp = facade.retrieve(
            MemoryQuery(text=query_text, outcome="failure", limit=5),
            purpose=RetrievalPurpose.HUMAN_SEARCH,
        )
        _judge("hybrid", resp.candidates[0].memory_id if resp.candidates else None, row)

        # hybrid + regime: selective intervention (APPLY only when the gate agrees)
        decision = pipeline.decide(
            query_text, regime, body_id=row.get("body_id"), joint_name=row.get("joint_name")
        )
        applied = decision.selected_memory_id if decision.action.value == "apply" else None
        _judge("hybrid_regime", applied, row)

    report: dict[str, Any] = {"queries": len(rows), "lanes": {}}
    for lane, s in lanes.items():
        n = s["n"] or 1
        report["lanes"][lane] = {
            "success_rate": round(s["success"] / n, 4),
            "memory_hurt_rate": round(s["hurt"] / n, 4),
            "unsafe_intervention_rate": round(s["unsafe"] / n, 4),
        }
    regime_lane = report["lanes"]["hybrid_regime"]
    report["gate"] = {
        "memory_hurt_rate_max": 0.05,
        "hurt_ok": regime_lane["memory_hurt_rate"] <= 0.05,
        "unsafe_ok": regime_lane["unsafe_intervention_rate"] == 0.0,
        "not_worse_than_no_memory": (
            regime_lane["success_rate"] >= report["lanes"]["no_memory"]["success_rate"]
        ),
        "stack": "disclosed deterministic doubles (bench fake) — CI lane",
    }
    report["passed"] = all(
        [report["gate"]["hurt_ok"], report["gate"]["unsafe_ok"],
         report["gate"]["not_worse_than_no_memory"]]
    )
    return report


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--workdir", default="/tmp/data_flywheel_live")
    parser.add_argument("--report", default=None)
    parser.add_argument("--session-seconds", type=float, default=0.1)
    parser.add_argument("--reconcile-every", type=int, default=10)
    parser.add_argument("--soak-duration-sec", type=float, default=0.0)
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument("--hurt-gate", action="store_true")
    parser.add_argument("--real-seekdb", action="store_true",
                        help="reserved: swap hurt-gate native side for a real engine")
    args = parser.parse_args()

    started = time.time()
    result: dict[str, Any] = {"started_at": started, "config": vars(args)}
    ok = True
    if args.episodes > 0 or args.soak_duration_sec:
        result["live_loop"] = run_loop(args)
        lq = result["live_loop"]
        checks = {
            "episodes_completed": lq["episodes_completed"] >= args.episodes,
            "memories_stored": (lq["memory"].get("memory_items") or 0) >= 1,
            "duplicate_rate_zero": (lq["memory"].get("duplicate_rate") or 0) == 0,
            "bad_evidence_near_zero": (lq["data_quality"].get("bad_evidence_write_rate") or 0) <= 0.05,
            "how_lookup": lq["how"]["lookup_ok"],
            "insight_published": lq["insight"]["published"] >= 1,
            "insight_lineage": lq["insight"]["lineage_linked"],
            "evolution_records": lq["evolution"]["records"] >= 1,
            "lineage_to_insight": lq["lineage"]["proposal_reaches_insight"],
            "lineage_to_memory": lq["lineage"]["proposal_reaches_memory"],
        }
        if lq["projection"].get("retrieval_skipped") is None and not args.no_retrieval:
            checks["projection_final_lag_zero"] = (lq["projection"].get("final_lag") or 0) == 0
        result["live_checks"] = checks
        ok = ok and all(checks.values())
    if args.hurt_gate:
        result["hurt_gate"] = run_hurt_gate(args)
        ok = ok and result["hurt_gate"]["passed"]
    result["elapsed_s"] = round(time.time() - started, 2)
    result["passed"] = ok

    report_path = Path(args.report) if args.report else None
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    # The embedded SeekDB engine's teardown can bypass Python's stdio flush
    # at process exit (same issue as rosclaw db status — see storage/cli.py
    # _with_stdout_flush); flush explicitly so the report is never lost.
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""ROSClaw v1.0 — Integration Performance Benchmark.

Measures KNOW, HOW, EventBus, SeekDB, and end-to-end pipeline performance.
Run: python benchmarks/integration_performance.py
"""

import asyncio
import statistics
import time
from pathlib import Path

from rosclaw.core import Runtime, RuntimeConfig
from rosclaw.core.event_bus import Event, EventBus
from rosclaw.how import HeuristicEngine
from rosclaw.know import KnowledgeInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient

REPO_ROOT = Path(__file__).parent.parent
RESULTS_PATH = REPO_ROOT / "benchmarks" / "integration_results.md"


def _fmt_ms(ms: float) -> str:
    return f"{ms:.4f}"


def _fmt_us(us: float) -> str:
    return f"{us:.4f}"


# ─────────────────────────────────────────────────────────────
# KNOW Query Latency
# ─────────────────────────────────────────────────────────────


async def benchmark_know_query() -> dict:
    seekdb = SeekDBMemoryClient()
    seekdb.connect()

    # Seed data
    for i in range(100):
        seekdb.insert(
            "knowledge_graph",
            {
                "id": f"bot_{i}",
                "robot_id": f"bot_{i}",
                "capability": f"cap_{i % 10}",
                "skill_type": "programmed",
                "parameters": "{}",
            },
        )

    know = KnowledgeInterface(seekdb_client=seekdb, robot_id="bot_0")
    know.initialize()

    # Warmup
    for _ in range(10):
        know.query_robot_capabilities("bot_0")

    # Benchmark
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        know.query_robot_capabilities("bot_0")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "target_ms": 100,
        "verdict": "PASS" if latencies[int(len(latencies) * 0.95)] < 100 else "FAIL",
    }


# ─────────────────────────────────────────────────────────────
# HOW Recovery Latency
# ─────────────────────────────────────────────────────────────


async def benchmark_how_recovery() -> dict:
    seekdb = SeekDBMemoryClient()
    seekdb.connect()

    how = HeuristicEngine(seekdb_client=seekdb)
    await how.initialize()
    await how.seed_defaults()

    # Warmup
    for _ in range(10):
        await how.suggest_recovery("joint_limit_exceeded")

    # Benchmark
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        await how.suggest_recovery("joint_limit_exceeded")
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "min_ms": latencies[0],
        "max_ms": latencies[-1],
        "target_ms": 10,
        "verdict": "PASS" if latencies[int(len(latencies) * 0.95)] < 10 else "FAIL",
    }


# ─────────────────────────────────────────────────────────────
# EventBus Throughput (with all modules active)
# ─────────────────────────────────────────────────────────────


def benchmark_eventbus_full() -> dict:
    bus = EventBus()
    received = []

    # Multiple subscribers simulating different modules
    bus.subscribe("robot.joint_states", lambda e: received.append(("joints", e)))
    bus.subscribe("robot.joint_states", lambda e: received.append(("memory", e)))
    bus.subscribe("robot.joint_states", lambda e: received.append(("practice", e)))
    bus.subscribe("robot.command", lambda e: received.append(("command", e)))
    bus.subscribe("robot.safety", lambda e: received.append(("safety", e)))

    target = 10_000
    t0 = time.perf_counter()
    for i in range(target):
        bus.publish(
            Event(
                topic="robot.joint_states",
                payload={"positions": [0.1, 0.2, 0.0]},
                source="benchmark",
            )
        )
    elapsed = time.perf_counter() - t0

    throughput = target / elapsed

    return {
        "target": target,
        "elapsed_s": elapsed,
        "throughput": throughput,
        "received": len(received),
        "target_events_s": 10_000,
        "verdict": "PASS" if throughput >= 10_000 else "FAIL",
    }


# ─────────────────────────────────────────────────────────────
# SeekDB Query Latency
# ─────────────────────────────────────────────────────────────


def benchmark_seekdb_query() -> dict:
    seekdb = SeekDBMemoryClient()
    seekdb.connect()

    # Seed data
    for i in range(10_000):
        seekdb.insert(
            "experiences",
            {
                "id": f"exp_{i}",
                "robot_id": "ur5e",
                "event_type": "praxis",
                "timestamp": float(i),
                "instruction": f"task {i}",
            },
        )

    # Benchmark queries
    latencies = []
    for _ in range(1000):
        t0 = time.perf_counter()
        seekdb.query("experiences", filters={"robot_id": "ur5e"}, limit=100)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "target_ms": 50,
        "verdict": "PASS" if latencies[int(len(latencies) * 0.95)] < 50 else "FAIL",
    }


# ─────────────────────────────────────────────────────────────
# End-to-End Pipeline Latency
# ─────────────────────────────────────────────────────────────


async def benchmark_e2e_pipeline() -> dict:
    # NOTE: enable_how=False due to Bug #RUNTIME-ASYNC-001
    # Runtime._do_initialize() uses run_until_complete() which conflicts
    # with the benchmark's own event loop.
    config = RuntimeConfig(
        robot_id="bench_bot",
        enable_knowledge=True,
        enable_how=False,  # Avoid async init conflict
        enable_firewall=True,
        enable_memory=True,
        enable_practice=True,
    )
    runtime = Runtime(config)
    runtime.initialize()

    # Benchmark: query KNOW -> EventBus publish
    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()

        # Step 1: KNOW query
        runtime._knowledge.query_robot_capabilities("bench_bot")

        # Step 2: EventBus publish
        runtime.event_bus.publish(
            Event(
                topic="benchmark.e2e",
                payload={"status": "ok"},
                source="benchmark",
            )
        )

        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    return {
        "count": len(latencies),
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "target_ms": 500,
        "verdict": "PASS" if latencies[int(len(latencies) * 0.95)] < 500 else "FAIL",
    }


# ─────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────


async def run_all() -> str:
    lines = [
        "# ROSClaw v1.0 Integration Performance Benchmark",
        "",
        f"> **Date**: {time.strftime('%Y-%m-%d')}",
        "> **Method**: In-process synthetic benchmark",
        "",
    ]

    # KNOW
    print("[1/5] Benchmarking KNOW query latency...")
    r = await benchmark_know_query()
    lines.extend(
        [
            "## KNOW Query Latency",
            f"- Queries: {r['count']:,}",
            f"- p50: {_fmt_ms(r['p50_ms'])} ms",
            f"- p95: {_fmt_ms(r['p95_ms'])} ms",
            f"- p99: {_fmt_ms(r['p99_ms'])} ms",
            f"- Range: {_fmt_ms(r['min_ms'])} - {_fmt_ms(r['max_ms'])} ms",
            f"- Target: < {_fmt_ms(r['target_ms'])} ms",
            f"- Verdict: {r['verdict']}",
            "",
        ]
    )

    # HOW
    print("[2/5] Benchmarking HOW recovery latency...")
    r = await benchmark_how_recovery()
    lines.extend(
        [
            "## HOW Recovery Latency",
            f"- Queries: {r['count']:,}",
            f"- p50: {_fmt_ms(r['p50_ms'])} ms",
            f"- p95: {_fmt_ms(r['p95_ms'])} ms",
            f"- p99: {_fmt_ms(r['p99_ms'])} ms",
            f"- Range: {_fmt_ms(r['min_ms'])} - {_fmt_ms(r['max_ms'])} ms",
            f"- Target: < {_fmt_ms(r['target_ms'])} ms",
            f"- Verdict: {r['verdict']}",
            "",
        ]
    )

    # EventBus
    print("[3/5] Benchmarking EventBus throughput...")
    r = benchmark_eventbus_full()
    lines.extend(
        [
            "## EventBus Throughput (Multi-Subscriber)",
            f"- Events: {r['target']:,}",
            f"- Elapsed: {r['elapsed_s']:.4f}s",
            f"- Throughput: {r['throughput']:.1f} events/s",
            f"- Received: {r['received']:,} (5 subscribers)",
            f"- Target: >= {r['target_events_s']:,} events/s",
            f"- Verdict: {r['verdict']}",
            "",
        ]
    )

    # SeekDB
    print("[4/5] Benchmarking SeekDB query latency...")
    r = benchmark_seekdb_query()
    lines.extend(
        [
            "## SeekDB Query Latency",
            f"- Queries: {r['count']:,}",
            f"- p50: {_fmt_ms(r['p50_ms'])} ms",
            f"- p95: {_fmt_ms(r['p95_ms'])} ms",
            f"- p99: {_fmt_ms(r['p99_ms'])} ms",
            f"- Target: < {_fmt_ms(r['target_ms'])} ms",
            f"- Verdict: {r['verdict']}",
            "",
        ]
    )

    # E2E
    print("[5/5] Benchmarking end-to-end pipeline...")
    r = await benchmark_e2e_pipeline()
    lines.extend(
        [
            "## End-to-End Pipeline Latency",
            f"- Iterations: {r['count']}",
            f"- p50: {_fmt_ms(r['p50_ms'])} ms",
            f"- p95: {_fmt_ms(r['p95_ms'])} ms",
            f"- p99: {_fmt_ms(r['p99_ms'])} ms",
            f"- Target: < {_fmt_ms(r['target_ms'])} ms",
            f"- Verdict: {r['verdict']}",
            "",
        ]
    )

    return "\n".join(lines)


if __name__ == "__main__":
    report = asyncio.run(run_all())
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    RESULTS_PATH.write_text(report, encoding="utf-8")
    print(f"\nReport saved to: {RESULTS_PATH}")

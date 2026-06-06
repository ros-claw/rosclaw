"""ROSClaw-Memory Stress & Scale Benchmark Suite.

Priority order:
  1. Concurrency stress (multi-thread write + query)
  2. Backend comparison (MemoryClient vs SQLiteClient)
  3. Scale test (10K → 1M records)
  4. Memory leak detection (long-running tracemalloc)
  5. RecoveryLoop stress (rapid failure/success events)
  6. EventBus throughput (publish/subscribe latency)
  7. Index efficiency (indexed vs full-scan)
  8. Capacity boundary (extreme configurations)

Usage:
    PYTHONPATH=src python3 benchmarks/corebench_memory_stress.py
"""

from __future__ import annotations

import gc
import random
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, "src")

from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient, SeekDBSQLiteClient
from rosclaw.memory.types import PraxisEvent, FailureMemory
from rosclaw.core.event_bus import EventBus, Event
from rosclaw.how.engine import HeuristicEngine
from rosclaw.how.recovery_loop import RecoveryLoop


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

@dataclass
class BenchMetrics:
    name: str
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0

    @property
    def p50(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.99)]

    @property
    def throughput(self) -> float:
        t = sum(self.latencies_ms) / 1000.0
        return len(self.latencies_ms) / t if t > 0 else 0.0


def timing() -> float:
    return time.perf_counter() * 1000.0


def fmt_ms(v: float) -> str:
    return f"{v:.3f} ms"


def random_instruction() -> str:
    skills = ["pick_up", "place_on", "grasp", "pour", "navigate", "scan"]
    objs = ["cup", "block", "bottle", "plate", "jar"]
    locs = ["table", "shelf", "counter", "floor"]
    return f"{random.choice(skills)} the {random.choice(objs)} on {random.choice(locs)}"


# ---------------------------------------------------------------------------
# Phase 1 — Concurrency Stress
# ---------------------------------------------------------------------------

def bench_concurrency() -> dict[str, Any]:
    print("\n[Phase 1] Concurrency Stress — 4 writers + 4 readers, 5s burst")
    mem = MemoryInterface("stress_bot")
    mem.initialize()

    write_metrics = BenchMetrics("concurrent_write")
    read_metrics = BenchMetrics("concurrent_read")
    stop_event = threading.Event()

    def writer():
        i = 0
        while not stop_event.is_set():
            t0 = timing()
            try:
                mem.store_experience(
                    event_id=f"cw_{threading.current_thread().name}_{i}",
                    event_type="praxis",
                    instruction=random_instruction(),
                    outcome=random.choice(["success", "failure"]),
                )
            except Exception:
                write_metrics.errors += 1
            write_metrics.latencies_ms.append(timing() - t0)
            i += 1

    def reader():
        while not stop_event.is_set():
            t0 = timing()
            try:
                mem.find_similar_experiences(random_instruction(), limit=3)
            except Exception:
                read_metrics.errors += 1
            read_metrics.latencies_ms.append(timing() - t0)
            time.sleep(0.001)

    threads = []
    for i in range(4):
        t = threading.Thread(target=writer, name=f"W{i}")
        t.start()
        threads.append(t)
    for i in range(4):
        t = threading.Thread(target=reader, name=f"R{i}")
        t.start()
        threads.append(t)

    time.sleep(5.0)
    stop_event.set()
    for t in threads:
        t.join(timeout=2.0)

    mem.stop()
    return {
        "write": write_metrics,
        "read": read_metrics,
    }


# ---------------------------------------------------------------------------
# Phase 2 — Backend Comparison
# ---------------------------------------------------------------------------

def bench_backend(n: int = 5000) -> dict[str, Any]:
    print(f"\n[Phase 2] Backend Comparison — {n} records")
    results = {}

    for backend_name, client in [("Memory", SeekDBMemoryClient()), ("SQLite", SeekDBSQLiteClient("/tmp/bench_seekdb.sqlite"))]:
        mem = MemoryInterface("back_bot", seekdb_client=client)
        mem.initialize()

        # Write
        write_lat = []
        for i in range(n):
            t0 = timing()
            mem.store_experience(f"b_{i}", "praxis", random_instruction(), outcome="success")
            write_lat.append(timing() - t0)

        # Query
        query_lat = []
        for _ in range(100):
            t0 = timing()
            mem.find_similar_experiences("pick up cup", limit=5)
            query_lat.append(timing() - t0)

        # Count
        t0 = timing()
        mem._client.count("experience_graph")
        count_lat = timing() - t0

        results[backend_name] = {
            "write_p50": statistics.median(write_lat),
            "write_p99": sorted(write_lat)[int(len(write_lat) * 0.99)],
            "query_p50": statistics.median(query_lat),
            "query_p99": sorted(query_lat)[int(len(query_lat) * 0.99)],
            "count_lat_ms": count_lat,
        }
        mem.stop()

    return results


# ---------------------------------------------------------------------------
# Phase 3 — Scale Test
# ---------------------------------------------------------------------------

def bench_scale() -> dict[str, Any]:
    print("\n[Phase 3] Scale Test — 10K → 50K → 100K → 500K")
    mem = MemoryInterface("scale_bot")
    mem.initialize()

    scales = [10_000, 50_000, 100_000, 500_000]
    results = {}

    for target in scales:
        current = mem._client.count("experience_graph")
        to_insert = target - current
        if to_insert <= 0:
            continue

        t0 = timing()
        for i in range(to_insert):
            mem.store_experience(
                f"sc_{target}_{i}", "praxis", random_instruction(), outcome="success"
            )
        ingest_ms = timing() - t0

        t0 = timing()
        mem.find_similar_experiences("pick up cup", limit=5)
        query_ms = timing() - t0

        t0 = timing()
        mem._client.count("experience_graph")
        count_ms = timing() - t0

        results[target] = {
            "ingest_total_sec": ingest_ms / 1000.0,
            "query_ms": query_ms,
            "count_ms": count_ms,
            "throughput_hz": to_insert / (ingest_ms / 1000.0),
        }

    mem.stop()
    return results


# ---------------------------------------------------------------------------
# Phase 4 — Memory Leak Detection
# ---------------------------------------------------------------------------

def bench_memory_leak(duration_sec: int = 30) -> dict[str, Any]:
    print(f"\n[Phase 4] Memory Leak Detection — {duration_sec}s continuous load")
    mem = MemoryInterface("leak_bot")
    mem.initialize()

    tracemalloc.start()
    samples = []
    start = time.time()
    iteration = 0

    while time.time() - start < duration_sec:
        # Write 100 records
        for i in range(100):
            mem.store_experience(f"lk_{iteration}_{i}", "praxis", random_instruction())
        # Query 50 times
        for _ in range(50):
            mem.find_similar_experiences("pick up cup", limit=5)
        # Force capacity check
        mem.enforce_capacity(max_experiences=10_000)

        # Sample memory every 5s
        if iteration % 5 == 0:
            gc.collect()
            current, peak = tracemalloc.get_traced_memory()
            samples.append((iteration, current / (1024 * 1024), peak / (1024 * 1024)))

        iteration += 1
        time.sleep(1.0)

    tracemalloc.stop()
    mem.stop()

    # Linear regression on memory growth
    if len(samples) >= 2:
        x = [s[0] for s in samples]
        y = [s[1] for s in samples]
        n = len(x)
        slope = (n * sum(xi * yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / (n * sum(xi * xi for xi in x) - sum(x) ** 2)
    else:
        slope = 0.0

    return {
        "samples": samples,
        "slope_mb_per_iter": slope,
        "leak_detected": slope > 0.5,  # >0.5 MB per iteration = suspicious
    }


# ---------------------------------------------------------------------------
# Phase 5 — RecoveryLoop Stress
# ---------------------------------------------------------------------------

def bench_recovery_loop_stress() -> dict[str, Any]:
    print("\n[Phase 5] RecoveryLoop Stress — 500 rapid failure/success events")
    bus = EventBus()
    mem = MemoryInterface("rl_bot", event_bus=bus)
    mem.initialize()
    he = HeuristicEngine(mem.seekdb_client)
    rl = RecoveryLoop(bus, mem, he)
    rl.subscribe()

    # Seed rules
    for i in range(10):
        he._seekdb.insert("heuristic_rules", {
            "id": f"rule_{i}", "condition": f"fail_{i}",
            "action": "retry", "priority": 1,
            "success_count": 0, "failure_count": 0,
        })

    latencies = []
    for i in range(500):
        t0 = timing()
        # Alternate failure and success
        bus.publish(Event(
            topic="rosclaw.how.recovery_hint.generated",
            payload={
                "request_id": f"req_{i}",
                "failure_type": f"fail_{i % 10}",
                "retry_plan": {"rule_id": f"rule_{i % 10}", "max_retries": 3},
            },
            source="stress",
        ))
        if i % 2 == 0:
            bus.publish(Event(
                topic="rosclaw.sandbox.episode.succeeded",
                payload={"request_id": f"req_{i}", "episode_id": f"ep_{i}"},
                source="stress",
            ))
        else:
            bus.publish(Event(
                topic="rosclaw.sandbox.episode.failed",
                payload={"request_id": f"req_{i}", "episode_id": f"ep_{i}"},
                source="stress",
            ))
        latencies.append(timing() - t0)

    # Verify bookkeeping
    retries = mem.seekdb_client.count("retries")
    success_patterns = mem.seekdb_client.count("success_patterns")

    rl.unsubscribe()
    mem.stop()

    return {
        "p50_ms": statistics.median(latencies),
        "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "retries_recorded": retries,
        "success_patterns": success_patterns,
    }


# ---------------------------------------------------------------------------
# Phase 6 — EventBus Throughput
# ---------------------------------------------------------------------------

def bench_eventbus() -> dict[str, Any]:
    print("\n[Phase 6] EventBus Throughput — 10K events")
    bus = EventBus()
    received = 0
    latencies = []

    def handler(event):
        nonlocal received
        received += 1

    bus.subscribe("stress.topic", handler)

    t0 = timing()
    for i in range(10_000):
        evt_t0 = timing()
        bus.publish(Event(topic="stress.topic", payload={"i": i}, source="bench"))
        latencies.append(timing() - evt_t0)
    total_ms = timing() - t0

    return {
        "published": 10_000,
        "received": received,
        "total_sec": total_ms / 1000.0,
        "throughput_hz": 10_000 / (total_ms / 1000.0),
        "publish_p50_ms": statistics.median(latencies),
    }


# ---------------------------------------------------------------------------
# Phase 7 — Index Efficiency
# ---------------------------------------------------------------------------

def bench_index_efficiency() -> dict[str, Any]:
    print("\n[Phase 7] Index Efficiency — indexed vs full-scan")
    mem = MemoryInterface("idx_bot")
    mem.initialize()

    # Populate
    for i in range(5000):
        mem.store_experience(
            f"ie_{i}", "praxis", random_instruction(),
            outcome=random.choice(["success", "failure"]),
            tags=[random.choice(["grasp", "navigate", "scan"])],
        )

    # Indexed query (robot_id + outcome)
    t0 = timing()
    for _ in range(100):
        mem._client.query("experience_graph", filters={"robot_id": "idx_bot", "outcome": "success"}, limit=50)
    indexed_ms = timing() - t0

    # Non-indexed query (instruction — not indexed)
    t0 = timing()
    for _ in range(100):
        mem._client.query("experience_graph", filters={"robot_id": "idx_bot"}, limit=50)
    non_indexed_ms = timing() - t0

    mem.stop()
    return {
        "indexed_100queries_ms": indexed_ms,
        "non_indexed_100queries_ms": non_indexed_ms,
        "speedup": non_indexed_ms / max(indexed_ms, 0.001),
    }


# ---------------------------------------------------------------------------
# Phase 8 — Capacity Boundary
# ---------------------------------------------------------------------------

def bench_capacity_boundary() -> dict[str, Any]:
    print("\n[Phase 8] Capacity Boundary — extreme configurations")
    results = {}

    # 8a — max_experiences=0 (should evict all)
    mem = MemoryInterface("cap0")
    mem.initialize()
    for i in range(100):
        mem.store_experience(f"c0_{i}", "praxis", "task")
    evicted = mem.enforce_capacity(max_experiences=0)
    remaining = mem._client.count("experience_graph")
    results["evict_to_zero"] = {"evicted": evicted, "remaining": remaining}
    mem.stop()

    # 8b — max_age_days=0 (should delete all)
    mem = MemoryInterface("cap1")
    mem.initialize()
    for i in range(100):
        mem.store_experience(f"c1_{i}", "praxis", "task")
    deleted = mem.forget_old_experiences(max_age_days=0)
    remaining = mem._client.count("experience_graph")
    results["forget_age_zero"] = {"deleted": deleted, "remaining": remaining}
    mem.stop()

    # 8c — Rapid insert/delete cycle
    mem = MemoryInterface("cap2")
    mem.initialize()
    cycles = 100
    t0 = timing()
    for i in range(cycles):
        mem.store_experience(f"c2_{i}", "praxis", "task")
        mem.delete_experience(f"c2_{i}")
    cycle_ms = timing() - t0
    results["insert_delete_cycle"] = {"cycles": cycles, "total_ms": cycle_ms, "per_op_ms": cycle_ms / (cycles * 2)}
    mem.stop()

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ROSClaw-Memory Stress & Scale Benchmark Suite")
    print("=" * 70)

    p1 = bench_concurrency()
    p2 = bench_backend(n=5000)
    p3 = bench_scale()
    p4 = bench_memory_leak(duration_sec=30)
    p5 = bench_recovery_loop_stress()
    p6 = bench_eventbus()
    p7 = bench_index_efficiency()
    p8 = bench_capacity_boundary()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n[Phase 1] Concurrency Stress (4W + 4R, 5s)")
    for name, m in [("Write", p1["write"]), ("Read", p1["read"])]:
        print(f"  {name:6s}  p50={fmt_ms(m.p50)}  p99={fmt_ms(m.p99)}  "
              f"throughput={m.throughput:.1f} hz  errors={m.errors}")

    print("\n[Phase 2] Backend Comparison (5K records)")
    for name, m in p2.items():
        print(f"  {name:8s}  write p50={fmt_ms(m['write_p50'])}  query p50={fmt_ms(m['query_p50'])}  "
              f"count={fmt_ms(m['count_lat_ms'])}")

    print("\n[Phase 3] Scale Test")
    for scale, m in p3.items():
        print(f"  {scale:8d}  ingest={m['ingest_total_sec']:.1f}s  query={fmt_ms(m['query_ms'])}  "
              f"throughput={m['throughput_hz']:.0f} hz")

    print("\n[Phase 4] Memory Leak Detection (30s)")
    print(f"  Slope: {p4['slope_mb_per_iter']:.3f} MB/iter")
    print(f"  Leak detected: {'YES ⚠️' if p4['leak_detected'] else 'NO ✅'}")
    if p4['samples']:
        print(f"  First sample: {p4['samples'][0][1]:.2f} MB")
        print(f"  Last sample:  {p4['samples'][-1][1]:.2f} MB")

    print("\n[Phase 5] RecoveryLoop Stress (500 events)")
    print(f"  p50={fmt_ms(p5['p50_ms'])}  p99={fmt_ms(p5['p99_ms'])}")
    print(f"  retries={p5['retries_recorded']}  success_patterns={p5['success_patterns']}")

    print("\n[Phase 6] EventBus Throughput (10K events)")
    print(f"  throughput={p6['throughput_hz']:.0f} hz  publish_p50={fmt_ms(p6['publish_p50_ms'])}")
    print(f"  received={p6['received']}/{p6['published']}")

    print("\n[Phase 7] Index Efficiency (100 queries each)")
    print(f"  Indexed:     {fmt_ms(p7['indexed_100queries_ms'])}")
    print(f"  Non-indexed: {fmt_ms(p7['non_indexed_100queries_ms'])}")
    print(f"  Speedup:     {p7['speedup']:.1f}x")

    print("\n[Phase 8] Capacity Boundary")
    print(f"  Evict to 0:     evicted={p8['evict_to_zero']['evicted']}  remaining={p8['evict_to_zero']['remaining']}")
    print(f"  Forget age 0:   deleted={p8['forget_age_zero']['deleted']}  remaining={p8['forget_age_zero']['remaining']}")
    print(f"  Insert/delete:  {p8['insert_delete_cycle']['per_op_ms']:.3f} ms/op")

    print("\n" + "=" * 70)
    print("Stress Benchmark COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

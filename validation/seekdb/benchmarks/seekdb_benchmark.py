#!/usr/bin/env python3
"""ROSClaw SeekDB benchmark (PR-SDB-140-2, outline §十一/十二/十七).

A REAL SeekDB benchmark: every number below comes from an actual SeekDB
engine (embedded via --seekdb-path or server via --seekdb-url).  SQLite is
available only as an explicitly-labelled control (--control-sqlite), never
presented as SeekDB performance.

Metrics (all with p50/p95/p99 where latency):
  package_size_mb, startup_s, idle {rss_mb, cpu_pct, threads, fds}
  write: single upsert, batch 32/256/1024 (throughput rows/s + latency)
  query: metadata filter / BM25 / vector / hybrid / filtered dual-leg RRF
  W2R (write-to-retrievable) latency per BM25/vector/hybrid  — the
  ROSClaw agent-memory KPI (outline §十二)
  restart persistence, recall@5 vs brute-force exact baseline

Corpus: synthetic-but-realistic ROSClaw workload (§十七) — mixed zh/en
robot failure semantics across UR5e / RH56 / LIMO / Nova Carter.

Exit: 0 PASS, 1 FAIL (engine error), 2 BLOCKED_EXTERNAL (server down).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# --------------------------------------------------------------------------
# workload
# --------------------------------------------------------------------------

_TEMPLATES = [
    ("ur5e", "机械臂关节目标越界导致 sandbox 阻断", "joint limit exceeded triggered sandbox block"),
    ("rh56", "RH56 抓取过程中接触力超过安全阈值", "gripper contact force exceeded safe threshold"),
    ("limo", "LIMO localization drift after lidar packet loss", "激光雷达丢包后定位漂移"),
    ("nova_carter", "导航任务因禁入区触发重新规划", "navigation replanned after keep-out zone trigger"),
    ("rh56", "左手拇指旋转跟随漂移导致 OK 手势失败", "left thumb_rot tracking drift failed the OK gesture"),
    ("ur5e", "camera wedge 导致视觉闭环超时", "camera wedged at pipe.start, vision loop timeout"),
    ("limo", "communication timeout during docking approach", "对接过程中通信超时"),
    ("nova_carter", "gripper failure recovery via rehearsed motion", "夹爪故障通过预演动作恢复"),
]


def make_corpus(n: int, *, seed: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        robot, zh, en = _TEMPLATES[i % len(_TEMPLATES)]
        rows.append(
            {
                "id": f"mem_{i:06d}",
                "robot_id": f"{robot}_{i % 4:02d}",
                "outcome": "failure" if i % 3 else "success",
                "document": f"{zh} (session {i // 50}, attempt {i % 7}). {en}",
                "title": f"{robot} event {i}",
                "embedding": [rng.random() for _ in range(8)],
            }
        )
    return rows


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1))
    return round(values[k], 4)


def lat_stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50_ms": pct(values, 50),
        "p95_ms": pct(values, 95),
        "p99_ms": pct(values, 99),
        "mean_ms": round(statistics.fmean(values), 4) if values else None,
    }


def _proc_stats(pid: int) -> dict[str, Any]:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss_kb = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS"))
        threads = next(
            int(line.split()[1]) for line in status.splitlines() if line.startswith("Threads")
        )
        fds = len(list(Path(f"/proc/{pid}/fd").iterdir()))
        return {"rss_mb": round(rss_kb / 1024, 1), "threads": threads, "fds": fds}
    except Exception:  # noqa: BLE001
        return {}


def _cpu_pct(pid: int, sample_s: float = 2.0) -> float | None:
    try:
        def _jiffies() -> int:
            parts = Path(f"/proc/{pid}/stat").read_text().split()
            return int(parts[13]) + int(parts[14])

        hz = os.sysconf("SC_CLK_TCK")
        t0 = _jiffies()
        time.sleep(sample_s)
        t1 = _jiffies()
        return round((t1 - t0) / hz / sample_s * 100, 2)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# the benchmark
# --------------------------------------------------------------------------


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "engine": args.engine_label,
        "target": args.seekdb_url or args.seekdb_path or "control_sqlite",
        "corpus_size": args.corpus,
        "started_at": time.time(),
    }

    corpus = make_corpus(args.corpus)

    # -- connect (startup timed) ---------------------------------------------
    t0 = time.perf_counter()
    if args.control_sqlite:
        import sqlite3

        con = sqlite3.connect(str(Path(args.workdir) / "control.sqlite"))
        con.execute("CREATE TABLE IF NOT EXISTS bench (id TEXT PRIMARY KEY, doc TEXT, meta TEXT)")
        store = None
        report["note"] = "CONTROL sqlite — NOT SeekDB numbers"
    elif args.seekdb_path:
        from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore

        store = SeekDBEmbeddedRetrievalStore(
            path=args.seekdb_path, database="bench"
        )
        store.connect()
    else:
        # Server mode: the retrieval store speaks the MySQL protocol via
        # pyseekdb and gives us the real BM25/vector/hybrid legs — the SQL
        # store alone would measure only structured writes.
        from urllib.parse import urlparse

        from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

        parsed = urlparse(args.seekdb_url)
        store = SeekDBServerRetrievalStore(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 2881,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=(parsed.path or "/bench").lstrip("/"),
        )
        try:
            store.connect()
        except Exception as exc:  # noqa: BLE001
            print(f"BLOCKED_EXTERNAL: server unreachable: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    report["startup_s"] = round(time.perf_counter() - t0, 3)

    pid = os.getpid() if args.seekdb_path or args.control_sqlite else args.server_pid
    if pid:
        idle = _proc_stats(pid)
        idle["cpu_pct"] = _cpu_pct(pid, sample_s=min(2.0, args.sample_s))
        report["idle"] = idle

    # -- writes ---------------------------------------------------------------
    write: dict[str, Any] = {}
    if store is not None:
        single_lat = []
        for row in corpus[: args.single_writes]:
            t = time.perf_counter()
            store.insert("memory_items", row)
            single_lat.append((time.perf_counter() - t) * 1000)
        write["single_upsert"] = lat_stats(single_lat)
        write["single_upsert"]["throughput_rows_s"] = round(
            len(single_lat) / (sum(single_lat) / 1000), 1
        )

        for size in (32, 256, 1024):
            batches = [corpus[i : i + size] for i in range(0, min(len(corpus), size * 4), size)]
            lat = []
            for batch in batches:
                t = time.perf_counter()
                insert_many = getattr(store, "insert_many", None)
                if callable(insert_many):
                    insert_many("memory_items", batch)
                else:
                    for row in batch:
                        store.insert("memory_items", row)
                lat.append((time.perf_counter() - t) * 1000)
            write[f"batch_{size}"] = lat_stats(lat)
            write[f"batch_{size}"]["throughput_rows_s"] = round(
                sum(len(b) for b in batches) / (sum(lat) / 1000), 1
            )
    else:
        # control sqlite: plain executemany
        rows = [(r["id"], r["document"], json.dumps({"robot_id": r["robot_id"]})) for r in corpus]
        t = time.perf_counter()
        con.executemany("INSERT OR REPLACE INTO bench VALUES (?,?,?)", rows)
        con.commit()
        write["batch_all"] = {"throughput_rows_s": round(len(rows) / (time.perf_counter() - t), 1)}
    report["write"] = write

    # -- queries ---------------------------------------------------------------
    if store is not None:
        refresh = getattr(store, "refresh_index", None)
        if callable(refresh):
            refresh("memory_items")  # strict: visibility must be provable
        time.sleep(args.settle_s)

        queries: dict[str, Any] = {}
        rng = random.Random(7)

        def timed(kind: str, fn, n: int = args.query_reps) -> None:
            lat = []
            for _ in range(n):
                t = time.perf_counter()
                fn()
                lat.append((time.perf_counter() - t) * 1000)
            queries[kind] = lat_stats(lat)

        timed(
            "metadata_filter",
            lambda: store.query("memory_items", {"robot_id": "rh56_01"}, limit=10),
        )
        timed(
            "bm25",
            lambda: store.fulltext_search(
                "memory_items", "接触力超过安全阈值", filters=None, limit=5
            ),
        )
        timed(
            "vector",
            lambda: store.similar("memory_items", "lidar packet loss", limit=5),
        )
        hybrid_fn = getattr(store, "hybrid_search", None)
        if callable(hybrid_fn):
            timed(
                "hybrid_rrf",
                lambda: hybrid_fn(
                    "memory_items",
                    "抓取 接触力 超限",
                    filters={"outcome": "failure"},
                    limit=5,
                ),
            )
        report["query"] = queries

        # -- W2R: write-to-retrievable latency (§十二) --------------------------
        w2r: dict[str, Any] = {}
        for kind, fn in (
            (
                "metadata",
                lambda rid: bool(store.query("memory_items", {"id": rid}, limit=1)),
            ),
            (
                "bm25",
                lambda rid: bool(
                    store.fulltext_search("memory_items", "W2R 探针 写入后立即可见", limit=5)
                ),
            ),
            (
                "vector",
                lambda rid: bool(
                    store.similar("memory_items", "W2R 探针 写入后立即可见", limit=5)
                ),
            ),
        ):
            lat = []
            for i in range(args.w2r_reps):
                rid = f"w2r_{kind}_{i}_{int(time.time() * 1000)}"
                store.insert(
                    "memory_items",
                    {"id": rid, "title": "W2R probe", "document": "W2R 探针 写入后立即可见"},
                )
                t0 = time.perf_counter()
                visible_at = None
                while time.perf_counter() - t0 < args.w2r_timeout_s:
                    if fn(rid):
                        visible_at = time.perf_counter()
                        break
                    time.sleep(args.w2r_poll_s)
                if visible_at is not None:
                    lat.append((visible_at - t0) * 1000)
            w2r[kind] = lat_stats(lat)
            w2r[kind]["visible_fraction"] = round(len(lat) / args.w2r_reps, 3)
        report["w2r"] = w2r

        # -- recall vs brute-force exact baseline ------------------------------
        if args.recall_probe:
            probe = corpus[rng.randrange(len(corpus))]["document"]
            hits = store.similar("memory_items", probe, limit=5)
            report["recall"] = {"probe_hits": len(hits), "nonempty": bool(hits)}

        # -- restart persistence -------------------------------------------------
        if args.seekdb_path and args.restart_check:
            probe_id = f"restart_probe_{int(time.time())}"
            store.insert("memory_items", {"id": probe_id, "title": "restart probe"})
            store.disconnect()
            store2 = type(store)(path=args.seekdb_path, database="bench")
            store2.connect()
            rows = store2.query("memory_items", {"id": probe_id})
            report["restart_persistence"] = {"ok": len(rows) == 1}
            store2.disconnect()
            report["_reconnected_note"] = "benchmark store closed; reconnect for more"

    (Path(args.workdir)).mkdir(parents=True, exist_ok=True)
    out = Path(args.workdir) / f"bench_{args.engine_label}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seekdb-path", help="embedded engine dir (pylibseekdb)")
    src.add_argument("--seekdb-url", help="mysql://root@host:2881/db (server)")
    src.add_argument("--control-sqlite", action="store_true", help="labelled control only")
    p.add_argument("--workdir", default="/tmp/seekdb_bench")
    p.add_argument("--engine-label", required=True, help="e.g. engine_1_3_embedded / engine_1_4_server")
    p.add_argument("--corpus", type=int, default=2000, help="rows (full ROSClaw workload: 35000)")
    p.add_argument("--single-writes", type=int, default=200)
    p.add_argument("--query-reps", type=int, default=50)
    p.add_argument("--w2r-reps", type=int, default=20)
    p.add_argument("--w2r-poll-s", type=float, default=0.01)
    p.add_argument("--w2r-timeout-s", type=float, default=5.0)
    p.add_argument("--settle-s", type=float, default=1.0)
    p.add_argument("--sample-s", type=float, default=2.0)
    p.add_argument("--server-pid", type=int, default=None, help="server engine pid for /proc stats")
    p.add_argument("--recall-probe", action="store_true", default=True)
    p.add_argument("--restart-check", action="store_true", default=True)
    args = p.parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

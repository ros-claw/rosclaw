#!/usr/bin/env python3
"""ROSClaw Memory Workload Medium (PR-SDB-140-5, P1-4, outline §十七).

Release-scale (not CI smoke): 10,000 episodes + 5,000 failures + 20,000
memory_nodes = 35,000 records with real ROSClaw semantics (UR5e / RH56 /
LIMO / Nova Carter; sandbox block / joint limit / contact force / lidar
dropout / localization drift / replanning / network failure / recovery
hints; zh/en mixed).

Measures: insert throughput (batched), per-leg query p50/p95/p99
(metadata/BM25/vector/hybrid), W2R, and --restart-check persistence.

Release-only: the 400-row benchmark stays the CI smoke.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

ROBOTS = ["ur5e", "rh56", "limo", "nova_carter"]
FAILURES = [
    ("机械臂 %s 关节目标越界,sandbox 在派发前阻断。joint target out of range", "joint_limit"),
    ("%s 抓取时接触力超过安全阈值,force_set 降额恢复。contact force exceeded", "contact_force"),
    ("%s 激光雷达丢包后定位漂移,重定位恢复。lidar packet loss, relocalized", "lidar_dropout"),
    ("%s 导航进入禁入区触发重新规划。keep-out zone, replanned", "keepout"),
    ("%s 对接通信超时,serial guardian 重连。communication timeout during docking", "network"),
    ("%s thumb_rot 跟随漂移导致手势失败,校准后通过。tracking drift, recalibrated", "drift"),
    ("%s 相机 pipe.start 卡死 UVC -110,hardware_reset 恢复。camera wedge", "camera_wedge"),
    ("%s 预演动作恢复执行。recovered via rehearsed motion", "recovery"),
]


def _record(rng: random.Random, kind: str, i: int) -> dict:
    robot = ROBOTS[i % len(ROBOTS)]
    text, ftype = FAILURES[i % len(FAILURES)]
    return {
        "id": f"{kind}_{i:06d}",
        "memory_type": kind,
        "robot_id": f"{robot}_{i % 3:02d}",
        "failure_type": ftype,
        "document": (text % robot.upper()) + f" occurrence {i}",
        "event_time": 1700000000 + i,
    }


def _pct(values: list[float], p: float) -> float:
    values = sorted(values)
    k = max(0, min(len(values) - 1, int(p / 100 * len(values) + 0.999) - 1))
    return round(values[k], 3)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seekdb-url", required=True)
    p.add_argument("--episodes", type=int, default=10_000)
    p.add_argument("--failures", type=int, default=5_000)
    p.add_argument("--memory-nodes", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--query-reps", type=int, default=50)
    p.add_argument("--w2r-reps", type=int, default=20)
    p.add_argument("--report", default=None)
    args = p.parse_args()

    from urllib.parse import urlparse

    from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

    parsed = urlparse(args.seekdb_url)
    store = SeekDBServerRetrievalStore(
        host=parsed.hostname or "127.0.0.1",
        port=parsed.port or 2881,
        user=parsed.username or "root",
        password=parsed.password or "",
        database=(parsed.path or "/rosclaw").lstrip("/"),
    )
    store.connect()

    rng = random.Random(42)
    report: dict = {"target": args.seekdb_url.rsplit("/", 1)[-1], "started_at": time.time()}

    # ---- load phase -------------------------------------------------------
    counts = {"episode": args.episodes, "failure": args.failures, "memory_node": args.memory_nodes}
    t0 = time.perf_counter()
    total = 0
    for kind, n in counts.items():
        batch = []
        for i in range(n):
            batch.append(_record(rng, kind, i))
            if len(batch) >= args.batch:
                store.insert_many("memory_items", batch)
                total += len(batch)
                batch = []
        if batch:
            store.insert_many("memory_items", batch)
            total += len(batch)
        print(f"loaded {kind}: {n}", flush=True)
    load_s = time.perf_counter() - t0
    report["load"] = {
        "records": total,
        "seconds": round(load_s, 1),
        "rows_per_s": round(total / load_s, 1),
    }
    store.refresh_index("memory_items", strict=False)
    time.sleep(2.0)

    # ---- workload phase ----------------------------------------------------
    queries = [
        ("metadata", lambda: store.query("memory_items", {"robot_id": "rh56_00"}, limit=10)),
        ("bm25", lambda: store.fulltext_search("memory_items", "接触力 阈值", limit=10)),
        ("vector", lambda: store.similar("memory_items", "lidar packet loss recovery", limit=10)),
        ("hybrid", lambda: store.hybrid_search("memory_items", "对接超时 恢复", limit=10)),
    ]
    legs: dict[str, list[float]] = {}
    for _ in range(args.query_reps):
        for name, fn in queries:
            t = time.perf_counter()
            fn()
            legs.setdefault(name, []).append((time.perf_counter() - t) * 1000)
    report["query"] = {
        name: {
            "n": len(lat),
            "p50_ms": _pct(lat, 50),
            "p95_ms": _pct(lat, 95),
            "p99_ms": _pct(lat, 99),
        }
        for name, lat in legs.items()
    }

    # ---- W2R ---------------------------------------------------------------
    w2r: list[float] = []
    for i in range(args.w2r_reps):
        rid = f"w2r_probe_{i}"
        t = time.perf_counter()
        store.insert(
            "memory_items",
            {"id": rid, "memory_type": "episode", "robot_id": "w2r", "document": f"w2r probe {i}"},
        )
        visible = False
        while time.perf_counter() - t < 5.0:
            if store.query("memory_items", {"id": rid}):
                visible = True
                break
            time.sleep(0.002)
        if visible:
            w2r.append((time.perf_counter() - t) * 1000)
    report["w2r"] = {
        "n": len(w2r),
        "p50_ms": _pct(w2r, 50),
        "p95_ms": _pct(w2r, 95),
        "p99_ms": _pct(w2r, 99),
        "visible_fraction": len(w2r) / args.w2r_reps,
    }

    # ---- persistence marker (restart check by caller) ----------------------
    store.insert(
        "memory_items",
        {
            "id": "medium_persist_marker",
            "memory_type": "episode",
            "robot_id": "medium",
            "document": "restart persistence marker",
        },
    )
    store.disconnect()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text)
    ok = report["w2r"]["visible_fraction"] == 1.0
    print("MEDIUM WORKLOAD:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

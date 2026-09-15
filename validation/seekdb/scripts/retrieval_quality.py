#!/usr/bin/env python3
"""Retrieval quality on the ROSClaw golden corpus (PR-SDB-140-3, §十一/十七).

A zh/en mixed golden corpus of robot-failure semantics across UR5e / RH56 /
LIMO / Nova Carter, evaluated against a REAL SeekDB store (embedded via
--seekdb-path or server via --seekdb-url).  Per leg (metadata / BM25 /
vector / hybrid): Recall@1, Recall@5, MRR, latency p50/p95/p99.

Exit 0 always (quality report, not a gate); --fail-under sets a Recall@5
floor per leg when used in CI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Golden corpus: (doc_id, document, relevant query ids it must answer)
# Semantics deliberately cross languages — a retrieval stack that only
# handles English fails the RH56/LIMO zh entries.

_DOCS: list[tuple[str, str, str]] = [
    # (id, robot, document)
    ("d_joint", "ur5e", "机械臂 UR5e 关节目标越界,sandbox 在派发前阻断。joint target out of range, blocked before dispatch."),
    ("d_force", "rh56", "RH56 抓取时接触力超过安全阈值,force_set 从 300 降到 50 后恢复。gripper contact force exceeded the safe threshold; recovered by lowering force_set."),
    ("d_lidar", "limo", "LIMO 激光雷达丢包后定位漂移,重定位恢复。localization drift after lidar packet loss; relocalized."),
    ("d_keepout", "nova_carter", "Nova Carter 导航进入禁入区触发重新规划。navigation entered keep-out zone, replanned."),
    ("d_thumb", "rh56", "左手 thumb_rot 跟随漂移导致 OK 手势失败,校准后通过。left thumb_rot tracking drift failed the OK gesture; passed after calibration."),
    ("d_wedge", "ur5e", "RealSense 相机在 pipe.start 卡死,UVC GET_CUR -110。camera wedged at pipe.start with UVC error -110."),
    ("d_dock", "limo", "LIMO 对接过程中通信超时,串口 guardian 重连恢复。communication timeout during docking; serial guardian reconnected."),
    ("d_griprec", "nova_carter", "夹爪故障通过预演动作恢复。gripper failure recovered via a rehearsed motion."),
]

# (query_id, query, relevant doc ids, body filter or None)
_QUERIES: list[tuple[str, str, list[str], str | None]] = [
    ("q_force_zh", "RH56 抓取过程中接触力过大的历史失败有哪些?", ["d_force"], None),
    ("q_block", "之前 UR5e 在 sandbox 为什么被阻断?", ["d_joint"], None),
    ("q_lidar", "类似这次 lidar 丢帧的恢复办法是什么?", ["d_lidar"], None),
    ("q_thumb", "OK 手势失败的已知原因", ["d_thumb"], "rh56_00"),
    ("q_wedge_en", "camera wedge at pipeline start", ["d_wedge"], None),
    ("q_dock", "对接超时 恢复", ["d_dock"], None),
]


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = max(0, min(len(values) - 1, math.ceil(p / 100 * len(values)) - 1))
    return round(values[k], 3)


def evaluate(store: Any, *, reps: int = 5) -> dict[str, Any]:
    report: dict[str, Any] = {"queries": len(_QUERIES), "legs": {}}

    def bench_leg(name: str, fn) -> None:
        recalls1, recalls5, mrrs, lat = [], [], [], []
        for _ in range(reps):
            for _qid, qtext, relevant, robot in _QUERIES:
                t0 = time.perf_counter()
                rows = fn(qtext, robot)
                lat.append((time.perf_counter() - t0) * 1000)
                ids = [r.get("id") for r in rows]
                recalls1.append(1.0 if ids and ids[0] in relevant else 0.0)
                recalls5.append(1.0 if any(i in relevant for i in ids[:5]) else 0.0)
                rr = 0.0
                for rank, i in enumerate(ids):
                    if i in relevant:
                        rr = 1.0 / (rank + 1)
                        break
                mrrs.append(rr)
        report["legs"][name] = {
            "recall@1": round(statistics.fmean(recalls1), 4),
            "recall@5": round(statistics.fmean(recalls5), 4),
            "mrr": round(statistics.fmean(mrrs), 4),
            "p50_ms": _pct(lat, 50),
            "p95_ms": _pct(lat, 95),
            "p99_ms": _pct(lat, 99),
        }

    bench_leg(
        "metadata",
        lambda q, robot: store.query(
            "memory_items", {"robot_id": robot} if robot else {}, limit=5
        ),
    )
    bench_leg(
        "bm25",
        lambda q, robot: store.fulltext_search(
            "memory_items", q, filters={"robot_id": robot} if robot else None, limit=5
        ),
    )
    bench_leg(
        "vector",
        lambda q, robot: store.similar(
            "memory_items", q, filters={"robot_id": robot} if robot else None, limit=5
        ),
    )
    hybrid = getattr(store, "hybrid_search", None)
    if callable(hybrid):
        bench_leg(
            "hybrid",
            lambda q, robot: hybrid(
                "memory_items", q, filters={"robot_id": robot} if robot else None, limit=5
            ),
        )
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--seekdb-path")
    src.add_argument("--seekdb-url")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--report", default=None)
    p.add_argument("--fail-under", type=float, default=None, help="Recall@5 floor per leg")
    args = p.parse_args()

    if args.seekdb_path:
        from rosclaw.storage.seekdb_native import SeekDBEmbeddedRetrievalStore

        store = SeekDBEmbeddedRetrievalStore(path=args.seekdb_path, database="golden_rq")
    else:
        from urllib.parse import urlparse

        from rosclaw.storage.seekdb_native import SeekDBServerRetrievalStore

        parsed = urlparse(args.seekdb_url)
        store = SeekDBServerRetrievalStore(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 2881,
            user=parsed.username or "root",
            password=parsed.password or "",
            database=(parsed.path or "/golden_rq").lstrip("/"),
        )
    store.connect()

    rng = random.Random(11)
    for doc_id, robot, document in _DOCS:
        store.insert(
            "memory_items",
            {
                "id": doc_id,
                "robot_id": f"{robot}_00",
                "document": document,
                "title": doc_id,
                # deterministic filler vectors — the engine's built-in
                # embedder indexes the document text; this key only feeds
                # stores that take manual vectors.
                "embedding": [rng.random() for _ in range(8)],
            },
        )
    store.refresh_index("memory_items")
    time.sleep(1.0)

    report = evaluate(store, reps=args.reps)
    store.disconnect()

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text)

    if args.fail_under is not None:
        for leg, stats in report["legs"].items():
            if leg == "metadata":
                continue  # metadata leg answers filter-only questions
            if (stats.get("recall@5") or 0.0) < args.fail_under:
                print(f"FAIL: {leg} recall@5 {stats.get('recall@5')} < {args.fail_under}")
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

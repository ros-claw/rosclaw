#!/usr/bin/env python3
"""Build + validate versioned multilingual memory index (PR-SDB-2, 数据库优化v3 §8).

Real-machine validation on the SeekDB server:

1. backfill memory_items from a SQLite knowledge store into
   memory_items__qwen3_06b_1024_v1__{ik,ngram} (manual Qwen3 embeddings);
2. verify record counts + dimension;
3. shadow-query both analyzers (CJK / EN / error-code probes) and the
   MiniLM 384 baseline collection;
4. activate the winner, verify registry switch, rollback;
5. write a JSON report (truth fields per §13).

Usage:
    build_qwen3_index.py --sqlite /tmp/mem3_scratch/knowledge.sqlite \
        --out /tmp/qwen3_index_report.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from rosclaw.embedding.registry import get_provider
from rosclaw.storage.seekdb_native import SeekDBServerStore
from rosclaw.storage.versioned_collections import (
    VersionedCollectionManager,
)

LOGICAL = "memory_items"
PROFILE_ID = "qwen3_06b_1024_v1"
BASELINE_COLLECTION = "memory_items"  # MiniLM 384 built-in embedder

PROBES = [
    ("cjk_rps", "石头剪刀布 猜拳 机器人"),
    ("cjk_middle", "中指未到位"),
    ("cjk_left_scissors", "左手 剪刀 失败 原因"),
    ("en_middle", "right rock joint not reached"),
    ("error_code", "EIO -110 serial"),
    ("mixed", "RH56 右手 rock joint_not_reached"),
]


def load_memories(sqlite_path: str) -> list[dict]:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, memory_type, robot_id, tenant_id, project_id, site_id,"
        " body_id, practice_id, session_id, episode_id, task_id, task_name,"
        " skill_id, policy_id, failure_type, joint_name, gesture_name, title,"
        " document, summary, outcome, confidence, importance, evidence_refs,"
        " artifact_refs, tags, metadata, event_time, status"
        " FROM memory_items WHERE status='active'"
    ).fetchall()
    return [dict(r) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2881)
    parser.add_argument("--database", default="rosclaw")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    records = load_memories(args.sqlite)
    print(f"loaded {len(records)} memories from {args.sqlite}")

    provider = get_provider(PROFILE_ID, cache_path="/tmp/mem3_scratch/embedding_cache.sqlite")
    store = SeekDBServerStore(host=args.host, port=args.port, database=args.database)
    store.connect()
    mgr = VersionedCollectionManager(store, provider)

    report: dict = {
        "profile": mgr.describe(LOGICAL)["embedding"],
        "records": len(records),
        "analyzers": {},
        "probes": [],
    }

    # --- health + provider truth
    report["provider_health"] = provider.health()

    # --- build both analyzers
    for analyzer in ("ik", "ngram"):
        t0 = time.monotonic()
        row = mgr.build(LOGICAL, records, analyzer=analyzer)
        build_s = time.monotonic() - t0
        verify = mgr.verify(LOGICAL, analyzer=analyzer)
        report["analyzers"][analyzer] = {
            "build_s": round(build_s, 1),
            "registry_status": row["status"],
            "verify": verify,
        }
        print(f"built {analyzer}: {build_s:.1f}s verify={verify['ok']}")

    # --- probe queries on both analyzers + baseline
    for probe_id, text in PROBES:
        probe: dict = {"id": probe_id, "query": text}
        for analyzer in ("ik", "ngram"):
            t0 = time.monotonic()
            rows = mgr.shadow_query(LOGICAL, text, analyzer=analyzer, limit=3)
            probe[analyzer] = {
                "ms": round((time.monotonic() - t0) * 1000.0, 1),
                "top": [
                    {
                        "id": r.get("id"),
                        "title": str(r.get("title") or "")[:60],
                        "body_id": r.get("body_id"),
                        "gesture_name": r.get("gesture_name"),
                    }
                    for r in rows
                ],
            }
        t0 = time.monotonic()
        baseline = store.hybrid_search(BASELINE_COLLECTION, text, limit=3)
        probe["minilm_384"] = {
            "ms": round((time.monotonic() - t0) * 1000.0, 1),
            "top": [{"id": r.get("id"), "title": str(r.get("title") or "")[:60]} for r in baseline],
        }
        report["probes"].append(probe)
        print(
            f"probe {probe_id}: ik={len(probe['ik']['top'])} ngram={len(probe['ngram']['top'])} baseline={len(probe['minilm_384']['top'])}"
        )

    # --- pick the analyzer with more non-empty CJK probes (honest heuristic)
    cjk_probes = [p for p in report["probes"] if p["id"].startswith("cjk")]
    ik_hits = sum(len(p["ik"]["top"]) for p in cjk_probes)
    ngram_hits = sum(len(p["ngram"]["top"]) for p in cjk_probes)
    winner = "ik" if ik_hits >= ngram_hits else "ngram"
    report["analyzer_decision"] = {
        "winner": winner,
        "ik_cjk_hits": ik_hits,
        "ngram_cjk_hits": ngram_hits,
    }

    # --- switch exercise: activate winner -> activate other -> rollback
    # -> winner ACTIVE again (first rollback before any OLD generation
    # must fail HONESTLY, proven below).
    activated = mgr.activate(LOGICAL, analyzer=winner)
    active = mgr.active(LOGICAL)
    loser = "ngram" if winner == "ik" else "ik"
    first_rollback_error = None
    try:
        mgr.rollback(LOGICAL)
    except RuntimeError as exc:
        first_rollback_error = str(exc)
    mgr.activate(LOGICAL, analyzer=loser)  # winner -> OLD
    rollback = mgr.rollback(LOGICAL)  # loser -> OLD, winner -> ACTIVE
    report["switch"] = {
        "activated": activated["physical_collection"],
        "active_after_switch": (active or {}).get("physical_collection"),
        "first_rollback_error": first_rollback_error,
        "second_active": loser,
        "rollback_to": rollback["physical_collection"],
        "final_active": (mgr.active(LOGICAL) or {}).get("physical_collection"),
    }
    report["final"] = mgr.describe(LOGICAL)

    Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(f"winner={winner} active={report['switch']['final_active']}")
    print(f"report -> {args.out}")
    store.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

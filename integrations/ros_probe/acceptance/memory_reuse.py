"""Replay an actual accepted mission into a private copy of its Memory store.

This measures retrieval and persistence, not a new physical execution or a
causal benefit from Memory. The source SQLite database and verifier are read-only.
"""

import argparse
import hashlib
import json
import sqlite3
import statistics
import time
from pathlib import Path

from rosclaw.connectors.ros.practice import RosPracticeAdapter
from rosclaw.core.event_bus import Event, EventBus
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SQLiteStructuredStore


def run(source: Path, verification_path: Path, output: Path):
    verification_bytes = verification_path.read_bytes()
    verification = json.loads(verification_bytes)
    if verification["verification_status"] != "PASS" or verification["success"] is not True:
        raise ValueError("an actual successful independent verification is required")
    receipts = verification["execution_receipts"]
    if not receipts or any(r["final_state"] != "COMPLETED" for r in receipts):
        raise ValueError("completed canonical execution receipts are required")
    output.mkdir(parents=True, exist_ok=False)
    database = output / "memory.sqlite"
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as original,
        sqlite3.connect(database) as copy,
    ):
        original.backup(copy)
    queries = (
        "complete room cleaning",
        "cleaning coverage",
        "区域清扫",
        "完成整个房间清扫。",
        "clean the entire room",
    )
    mission_id = verification["mission_id"]
    bus = EventBus()
    memory = MemoryInterface(verification["body_id"], bus, SQLiteStructuredStore(str(database)))
    memory.initialize()
    baseline = {q: [r["id"] for r in memory.find_similar_experiences(q)] for q in queries}
    original_record = memory.get_experience(mission_id)
    if original_record is None:
        raise ValueError("the source Memory must contain the actual accepted mission")
    count_before = memory.seekdb_client.count("experience_graph")
    adapter = RosPracticeAdapter(bus)
    adapter.initialize()
    payload = {
        "mission_id": mission_id,
        "robot_id": verification["body_id"],
        "verification": verification,
    }
    for _ in range(2):
        bus.publish(
            Event(
                topic="rosclaw.ros.verification.completed",
                payload=payload,
                source="historical_acceptance_replay",
            )
        )
    count_after = memory.seekdb_client.count("experience_graph")
    adapter.stop()
    memory.stop()
    reopened = MemoryInterface(
        verification["body_id"], seekdb_client=SQLiteStructuredStore(str(database))
    )
    reopened.initialize()
    try:
        result = {
            "status": "FAIL",
            "evidence_domain": "HISTORICAL_REPLAY",
            "new_physical_mission": False,
            "causal_memory_benefit_verified": False,
            "mission_id": mission_id,
            "verification_sha256": hashlib.sha256(verification_bytes).hexdigest(),
            "source_instruction": original_record["instruction"],
            "source_duration_sec": original_record["duration_sec"],
            "baseline_retrieval": baseline,
            "retrieval_after_reopen": {},
            "record_count_before": count_before,
            "record_count_after_double_replay": count_after,
        }
        if count_after != count_before:
            raise RuntimeError("historical replay duplicated the original mission")
        for query in queries:
            start = time.perf_counter()
            rows = reopened.find_similar_experiences(query, outcome_filter="success")
            first_ms = (time.perf_counter() - start) * 1000
            if not rows or rows[0]["id"] != mission_id:
                raise RuntimeError(f"accepted mission was not retrieved for {query}")
            row = rows[0]
            if row["metadata"]["raw"]["verification"] != verification or row["duration_sec"] <= 0:
                raise RuntimeError("original verification or measured duration was lost")
            timings = []
            for _ in range(50):
                start = time.perf_counter()
                warm = reopened.find_similar_experiences(query, outcome_filter="success")
                timings.append((time.perf_counter() - start) * 1000)
                if warm[0]["id"] != mission_id:
                    raise RuntimeError("cached retrieval changed the mission")
            result["retrieval_after_reopen"][query] = {
                "mission_id": row["id"],
                "duration_sec": row["duration_sec"],
                "first_query_ms": first_ms,
                "warm_samples": len(timings),
                "warm_median_ms": statistics.median(timings),
                "warm_p95_ms": sorted(timings)[47],
            }
        result["status"] = "PASS"
        (output / "acceptance.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n"
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        reopened.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-memory", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_memory, args.verification, args.output)

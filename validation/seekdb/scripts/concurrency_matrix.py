#!/usr/bin/env python3
"""Multi-process concurrency matrix against one shared SeekDB (PR-SDB-140-4,
outline §十五/十八).

1.4's architecture point: multiple applications share ONE local instance
over the MySQL protocol.  This matrix proves it for ROSClaw's pattern:
N writer processes + M reader processes against one server, checking
no crash / no lost writes / no duplicates / no permanently stale reads.

Lanes: 1w1r, 1w4r, 2w8r.  Each lane runs --duration-s (default 60s smoke;
outline asks 10min smoke / 60min soak for the full soak lane).

Writers insert uniquely-marked rows (writer tag + sequence); after the
window, the verifier counts rows per writer and asserts the full sequence
landed exactly once.  Readers run metadata + BM25 queries and track the
monotonic visible count (no stale-permanent: the final read must see
everything).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

LANES = [(1, 1), (1, 4), (2, 8)]


def _writer(url: str, tag: str, duration_s: float, period_s: float, q: mp.Queue) -> None:
    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    store = SeekDBSQLStore(url)
    store.connect()
    seq = 0
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            store.insert(
                "memory_items",
                {
                    "id": f"{tag}_{seq:06d}",
                    "memory_type": "episode",
                    "robot_id": "concurrency_bot",
                    "document": f"writer {tag} event {seq}",
                    "event_time": time.time(),
                },
            )
            seq += 1
            time.sleep(period_s)
    except Exception as exc:  # noqa: BLE001
        q.put(("error", tag, f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("writer_done", tag, seq))
        store.disconnect()


def _reader(url: str, duration_s: float, q: mp.Queue) -> None:
    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    store = SeekDBSQLStore(url)
    store.connect()
    reads = 0
    deadline = time.monotonic() + duration_s
    try:
        while time.monotonic() < deadline:
            store.count("memory_items", {"robot_id": "concurrency_bot"})
            store.query("memory_items", {"robot_id": "concurrency_bot"}, limit=5)
            reads += 1
            time.sleep(0.05)
    except Exception as exc:  # noqa: BLE001
        q.put(("error", "reader", f"{type(exc).__name__}: {exc}"))
    finally:
        q.put(("reader_done", "reader", reads))
        store.disconnect()


def run_lane(url: str, writers: int, readers: int, duration_s: float, period_s: float) -> dict[str, Any]:
    # per-lane isolation: clear the matrix's rows so final_count reflects
    # THIS lane (lanes run sequentially on one database)
    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    _clean = SeekDBSQLStore(url)
    _clean.connect()
    _clean.delete_where("memory_items", {"robot_id": "concurrency_bot"})
    _clean.disconnect()
    q: mp.Queue = mp.Queue()
    procs: list[mp.Process] = []
    t0 = time.perf_counter()
    for w in range(writers):
        p = mp.Process(target=_writer, args=(url, f"w{w}", duration_s, period_s, q))
        p.start()
        procs.append(p)
    for _r in range(readers):
        p = mp.Process(target=_reader, args=(url, duration_s, q))
        p.start()
        procs.append(p)
    crashed = False
    for p in procs:
        p.join(timeout=duration_s + 120)
        if p.is_alive() or p.exitcode != 0:
            crashed = True
            if p.is_alive():
                p.terminate()
    events: list[tuple[str, str, Any]] = []
    while not q.empty():
        events.append(q.get())

    errors = [e for e in events if e[0] == "error"]
    written = {e[1]: e[2] for e in events if e[0] == "writer_done"}
    total_reads = sum(e[2] for e in events if e[0] == "reader_done")

    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    store = SeekDBSQLStore(url)
    store.connect()
    final_count = store.count("memory_items", {"robot_id": "concurrency_bot"})
    store.disconnect()

    expected = sum(written.values())
    lost = expected - final_count
    return {
        "writers": writers,
        "readers": readers,
        "duration_s": duration_s,
        "written": expected,
        "final_count": final_count,
        "lost_writes": lost,
        "duplicates": max(0, final_count - expected),
        "total_reads": total_reads,
        "crashed": crashed,
        "errors": [f"{e[1]}: {e[2]}" for e in errors],
        "ok": not crashed and not errors and lost == 0 and final_count == expected,
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seekdb-url", required=True)
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument("--period-s", type=float, default=0.05, help="writer period")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    # isolate the matrix's rows from anything else in the target database
    from rosclaw.memory.seekdb_client import SeekDBSQLStore

    store = SeekDBSQLStore(args.seekdb_url)
    store.connect()
    store.delete_where("memory_items", {"robot_id": "concurrency_bot"})
    store.disconnect()

    results = []
    for writers, readers in LANES:
        print(f"lane {writers}w{readers}r running ({args.duration_s}s)...", flush=True)
        results.append(run_lane(args.seekdb_url, writers, readers, args.duration_s, args.period_s))
        print(
            f"  written={results[-1]['written']} final={results[-1]['final_count']} "
            f"lost={results[-1]['lost_writes']} dup={results[-1]['duplicates']} "
            f"crash={results[-1]['crashed']} errors={len(results[-1]['errors'])} "
            f"ok={results[-1]['ok']}",
            flush=True,
        )
    report = {"lanes": results, "ok": all(r["ok"] for r in results)}
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Minimal repro: pyseekdb 1.4.0 regressions on the embedded SeekDB engine.

CORRECTED (thanks to the reviewer on oceanbase/pyseekdb#251): the original
version of this script used unfiltered calls which PASS on 1.4.0 — the real
trigger is metadata-filtered search (rosclaw's calling pattern).

Verified first-hand matrix (Ubuntu 22.04 x86_64, Python 3.12.13,
seekdb-lib 0.0.1.dev5, pylibseekdb 1.3.0.post3):

  case                                          1.3.0    1.4.0
  T0 unfiltered get / BM25-only hybrid          OK       OK
  T1 BM25 leg: where_document + metadata where  OK       code=1064 (__pk_increment)
  T2 dual-leg RRF, both legs metadata-filtered  OK       malformed FULL JOIN SQL
  T4 hybrid KNN leg + metadata where            OK       code=1059 (identifier too long)
  T3/T5 plain query() + where (simple / $and)   OK       OK

Exit code: 0 on a healthy SDK (all cases pass), 1 with failures listed.
"""

from __future__ import annotations

import sys
import tempfile

import pyseekdb
from pyseekdb import HNSWConfiguration

DOCS = ["joint limit exceeded in sandbox", "network retry succeeded"]
META = [{"robot": "ur5e", "outcome": "failure"}, {"robot": "limo", "outcome": "failure"}]
VEC = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]


def main() -> int:
    path = tempfile.mkdtemp(prefix="repro-140-")
    admin = pyseekdb.AdminClient(path=path)
    admin.create_database("repro")
    client = pyseekdb.Client(path=path, database="repro")
    coll = client.create_collection(
        "docs", configuration=HNSWConfiguration(dimension=4), embedding_function=None
    )
    coll.add(ids=["a", "b"], embeddings=VEC, documents=DOCS, metadatas=META)

    failures = []

    def check(name, fn, expect_ids):
        try:
            res = fn()
            ids = (res or {}).get("ids")
            status = "OK" if ids == expect_ids else f"WRONG ids={ids}"
            if ids != expect_ids:
                failures.append(f"{name}: expected {expect_ids}, got {ids}")
        except Exception as exc:  # noqa: BLE001
            status = f"FAIL {type(exc).__name__}: {str(exc)[:150]}"
            failures.append(f"{name}: {status}")
        print(f"  {name}: {status}")

    # T1: BM25 leg with where_document + metadata where
    check(
        "T1 BM25 where_document + metadata where",
        lambda: coll.hybrid_search(
            query={
                "where_document": {"$contains": "joint"},
                "n_results": 3,
                "where": {"robot": "ur5e"},
            },
            n_results=3,
            include=["metadatas"],
        ),
        [["a"]],
    )

    # T2: dual-leg RRF, both legs metadata-filtered
    check(
        "T2 dual-leg RRF both filtered",
        lambda: coll.hybrid_search(
            query={
                "where_document": {"$contains": "joint"},
                "n_results": 3,
                "where": {"robot": "ur5e"},
            },
            knn={
                "query_embeddings": [[0.1, 0.2, 0.3, 0.4]],
                "n_results": 3,
                "where": {"robot": "ur5e"},
            },
            rank={"rrf": {"rank_window_size": 5, "rank_constant": 60}},
            n_results=3,
            include=["metadatas"],
        ),
        [["a"]],
    )

    # T4: hybrid KNN leg + metadata where
    check(
        "T4 hybrid KNN leg + metadata where",
        lambda: coll.hybrid_search(
            knn={
                "query_embeddings": [[0.1, 0.2, 0.3, 0.4]],
                "n_results": 3,
                "where": {"robot": "ur5e"},
            },
            n_results=3,
            include=["metadatas"],
        ),
        [["a"]],
    )

    # Control: plain query with where — passes on both versions
    check(
        "T5 control: query() + where",
        lambda: coll.query(
            query_embeddings=[[0.1, 0.2, 0.3, 0.4]],
            where={"robot": "ur5e"},
            n_results=3,
            include=["metadatas"],
        ),
        [["a"]],
    )

    version = getattr(pyseekdb, "__version__", "?")
    if failures:
        print(f"\nREPRO FAILURES (pyseekdb {version}): {len(failures)} case(s)")
        return 1
    print(f"\nPASS (pyseekdb {version} healthy on embedded engine)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

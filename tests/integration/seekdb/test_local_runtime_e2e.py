"""Gate H2 (P0-10): SeekDB 1.4 local-runtime REAL E2E — no mocks.

Runs the actual `seekdb==1.4.0.dev2` bindings: a real background engine
process, its real connection options (unix socket on Unix), the four query
legs, multi-process attach, close, reopen, persistence.

x86_64-only today (upstream ships no aarch64 wheel); skipped elsewhere.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.uname().machine != "x86_64", reason="seekdb bindings are x86_64-only"),
    pytest.mark.skipif(
        __import__("importlib.util").util.find_spec("seekdb") is None,
        reason="seekdb bindings not installed",
    ),
]


def _attach_probe(db_dir: str, q: mp.Queue) -> None:
    """Process B/C: attach to the live runtime, read, report — never own."""
    try:
        from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

        store = LocalRuntimeStructuredStore(db_dir)
        store.connect()
        role = store.runtime.role
        n = store.count("memory_items", {"robot_id": "h2_bot"})
        store.disconnect()
        q.put(("ok", role, n))
    except Exception as exc:  # noqa: BLE001
        q.put(("error", f"{type(exc).__name__}: {exc}", -1))


def test_local_runtime_real_e2e(tmp_path):
    from rosclaw.storage.seekdb_runtime import LocalRuntimeStructuredStore

    db_dir = str(tmp_path / "rt")
    store = LocalRuntimeStructuredStore(db_dir, database="h2_e2e")
    store.connect()
    assert store.runtime.role == "owner"
    options = store.runtime.connection_options()
    assert options, "no connection options from the real runtime"
    # unix-socket or host/port — whichever the bindings chose, it must be
    # what the store is actually using (identity, P0-7)
    if options.get("unix_socket"):
        assert store._inner._unix_socket == options["unix_socket"]  # noqa: SLF001

    # four legs on real data
    docs = [
        {
            "id": "h2_a",
            "robot_id": "h2_bot",
            "memory_type": "failure",
            "document": "RH56 抓取接触力超过阈值,gripper contact force exceeded threshold",
        },
        {
            "id": "h2_b",
            "robot_id": "h2_bot",
            "memory_type": "failure",
            "document": "LIMO 激光雷达丢帧导致定位漂移,lidar frame loss localization drift",
        },
    ]
    for d in docs:
        store.insert("memory_items", d)
    store.refresh_index("memory_items", strict=False)
    time.sleep(1.0)
    assert store.count("memory_items", {"robot_id": "h2_bot"}) == 2  # metadata leg
    assert store.fulltext_search("memory_items", "接触力", limit=5)  # BM25 leg
    assert store.similar("memory_items", "gripper force failure", limit=5)  # vector leg
    assert store.hybrid_search("memory_items", "lidar drift 恢复", limit=5)  # hybrid leg

    # process B + C attach (no second engine, no ownership)
    q: mp.Queue = mp.Queue()
    procs = [mp.Process(target=_attach_probe, args=(db_dir, q)) for _ in range(2)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = [q.get() for _ in procs]
    assert all(r[0] == "ok" for r in results), results
    assert all(r[1] == "attacher" for r in results), results
    assert all(r[2] == 2 for r in results), results

    store.disconnect()  # owner close

    # reopen — persistence
    store2 = LocalRuntimeStructuredStore(db_dir, database="h2_e2e")
    store2.connect()
    assert store2.count("memory_items", {"robot_id": "h2_bot"}) == 2, (
        "data lost across close/reopen"
    )
    store2.disconnect()

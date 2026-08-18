"""PR-DF-07: retrieval projection wiring — status watermark/lag, runtime
assembly, target-filtered outbox drain."""

import time

import pytest


def test_projection_status_reports_lag():
    from rosclaw.memory.repository import MemoryRepository
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore
    from rosclaw.storage.seekdb_projection import MemoryRetrievalProjection

    source = InMemoryStructuredStore()
    source.connect()
    native = InMemoryStructuredStore()  # stand-in for the native store (count API)
    native.connect()
    repo = MemoryRepository(source)

    from rosclaw.memory.models import MemoryItem

    for i in range(3):
        repo.store(
            MemoryItem(
                memory_type="episodic",
                robot_id="sim",
                title=f"mem {i}",
                document=f"doc {i}",
                evidence_refs=[f"evd_{i}"],
            )
        )
    proj = MemoryRetrievalProjection(native)
    status = proj.status(repo)
    assert status["source_count"] == 3
    assert status["projection_count"] == 0
    assert status["lag"] == 3

    # project one record; lag closes
    proj.project(source.query("memory_items", {})[0])
    status = proj.status(repo)
    assert status["lag"] == 2


def test_projection_outbox_enqueue():
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore
    from rosclaw.storage.outbox import OutboxStore
    from rosclaw.storage.seekdb_projection import MemoryRetrievalProjection

    outbox = OutboxStore(db_path=":memory:")
    outbox.connect()
    native = InMemoryStructuredStore()
    proj = MemoryRetrievalProjection(native, outbox=outbox)
    proj.project({"id": "mem_1", "title": "t", "document": "d"})
    stats = outbox.stats()
    assert stats["pending"] == 1
    # nothing written to the native store until a worker drains
    assert native.count("memory_items") == 0


def test_target_filtered_worker_drains_only_its_target():
    """Two pipelines share one outbox; each worker claims only its target."""
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore
    from rosclaw.storage.outbox import OutboxStore, OutboxWorker
    from rosclaw.storage.seekdb_projection import MemoryRetrievalProjectionCommitter

    outbox = OutboxStore(db_path=":memory:")
    outbox.connect()
    outbox.enqueue("seekdb_projection", {"id": "mem_a", "title": "a"}, entity_id="mem_a")
    outbox.enqueue("practice_events", {"id": "evt_1"}, entity_id="evt_1")

    native = InMemoryStructuredStore()
    worker = OutboxWorker(
        outbox,
        MemoryRetrievalProjectionCommitter(native),
        target="seekdb_projection",
        interval_sec=0.05,
    )
    delivered = worker.flush(timeout=5.0)
    assert delivered == 1
    assert native.count("memory_items") == 1
    # the practice record is untouched for its own worker
    assert outbox.stats()["pending"] == 1


def test_runtime_builds_projection_when_retrieval_enabled(tmp_path):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    cfg = RuntimeConfig(
        robot_id="sim_test",
        seekdb_backend="sqlite",
        seekdb_path=str(tmp_path / "knowledge.sqlite"),
        enable_memory=True,
        enable_practice=False,
        enable_how=False,
        enable_auto=False,
        enable_knowledge=False,
        enable_skill_manager=False,
        enable_provider=False,
        enable_sense=False,
        enable_firewall=False,
        enable_event_persistence=False,
        enable_tracing=False,
        storage={"retrieval": {"enabled": True, "path": str(tmp_path / "seekdb")}},
    )
    rt = Runtime(cfg)
    try:
        rt.initialize()
    except Exception as exc:  # native store unavailable on this host
        pytest.skip(f"native SeekDB unavailable: {exc}")
    assert rt._data_plane is not None
    assert rt._data_plane.retrieval_store is not None
    assert rt._data_plane.memory_projection is not None
    # repository got the projection injected (DF-05 x DF-07 link)
    assert rt._memory_repository is not None
    assert rt._memory_repository._projection is rt._data_plane.memory_projection
    rt.stop()
    time.sleep(0.1)


def test_runtime_projection_disabled_by_default(tmp_path):
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    cfg = RuntimeConfig(
        robot_id="sim_test",
        seekdb_backend="sqlite",
        seekdb_path=str(tmp_path / "knowledge.sqlite"),
        enable_memory=True,
        enable_practice=False,
        enable_how=False,
        enable_auto=False,
        enable_knowledge=False,
        enable_skill_manager=False,
        enable_provider=False,
        enable_sense=False,
        enable_firewall=False,
        enable_event_persistence=False,
        enable_tracing=False,
    )
    rt = Runtime(cfg)
    rt.initialize()
    assert rt._data_plane.memory_projection is None
    rt.stop()

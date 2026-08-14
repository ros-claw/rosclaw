"""PR-DF-03: DataPlaneContext — single assembly, per-piece fault isolation."""

from unittest.mock import MagicMock, patch

from rosclaw.storage.context import DataPlaneContext


def test_context_defaults_all_none():
    ctx = DataPlaneContext()
    assert ctx.structured_store is None
    assert ctx.retrieval_store is None
    assert ctx.outbox is None
    assert ctx.memory_projection is None
    assert ctx.memory_retrieval is None
    assert ctx.practice_sink is None


def test_capabilities_snapshot_names_backends():
    from rosclaw.memory.seekdb_client import InMemoryStructuredStore

    ctx = DataPlaneContext(structured_store=InMemoryStructuredStore())
    caps = ctx.capabilities()
    assert caps["structured_backend"] == "InMemoryStructuredStore"
    assert caps["outbox_enabled"] is False
    assert caps["memory_retrieval"] is False
    assert caps["practice_sink"] is None


def test_runtime_assembles_data_plane_once():
    """Runtime.initialize populates _data_plane and memory uses its store."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    cfg = RuntimeConfig(
        robot_id="sim_test",
        seekdb_backend="memory",
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
    store = MagicMock()
    with patch.object(Runtime, "_create_seekdb_client", return_value=store) as create:
        rt.initialize()
    assert create.call_count <= 1  # assembled once, not per-module
    assert rt._data_plane is not None
    assert rt._data_plane.structured_store is store


def test_runtime_survives_structured_store_failure():
    """ADR-0009 §4: a data-plane failure must never take the Runtime down."""
    from rosclaw.core.runtime import Runtime, RuntimeConfig

    cfg = RuntimeConfig(
        robot_id="sim_test",
        seekdb_backend="memory",
        enable_memory=False,
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
    with patch.object(Runtime, "_create_seekdb_client", side_effect=RuntimeError("db dead")):
        rt.initialize()  # must not raise
    assert rt._data_plane is not None
    assert rt._data_plane.structured_store is None

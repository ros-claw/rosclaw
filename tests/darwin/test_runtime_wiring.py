"""PR-DF-13: Darwin enters the Runtime lifecycle (flywheel §39-40)."""

from rosclaw.core.runtime import Runtime, RuntimeConfig


def _cfg(**over):
    base = dict(  # noqa: C408 — kwargs spread pattern
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
    base.update(over)
    return RuntimeConfig(**base)


def test_darwin_off_by_default():
    rt = Runtime(_cfg())
    rt.initialize()
    try:
        assert rt._darwin is None
    finally:
        rt.stop()


def test_darwin_wired_when_enabled_with_data_plane():
    rt = Runtime(_cfg(enable_darwin=True, darwin={"seeds": [0], "episodes": 5}))
    rt.initialize()
    try:
        assert rt._darwin is not None
        assert rt._darwin.engine is not None
        # shares the runtime event bus and the data plane structured store
        assert rt._darwin.engine._bus is rt.event_bus
        assert rt._darwin.engine._seekdb is rt._data_plane.structured_store
    finally:
        rt.stop()
    assert rt._darwin is None or rt._darwin.engine is not None  # stop is graceful

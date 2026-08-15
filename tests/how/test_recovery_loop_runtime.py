"""PR-DF-08: RecoveryLoop wired into the Runtime lifecycle (flywheel §23-24)."""

from rosclaw.core.runtime import Runtime, RuntimeConfig


def _cfg(**over):
    base = dict(  # noqa: C408 — kwargs spread pattern
        robot_id="sim_test",
        seekdb_backend="memory",
        enable_memory=True,
        enable_practice=False,
        enable_how=True,
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


def test_recovery_loop_subscribed_when_memory_and_how_live():
    rt = Runtime(_cfg())
    rt.initialize()
    try:
        assert rt._how is not None, "test precondition: how engine built"
        assert rt._recovery_loop is not None
        # subscribed to the three loop topics
        subs = getattr(rt.event_bus, "_subscribers", {})
        for topic in (
            "rosclaw.how.recovery_hint.generated",
            "rosclaw.sandbox.episode.succeeded",
            "rosclaw.sandbox.episode.failed",
        ):
            assert any(
                getattr(cb, "__self__", None) is rt._recovery_loop
                for cb in subs.get(topic, [])
            ), topic
    finally:
        rt.stop()
    # after stop the loop unsubscribes
    assert rt._recovery_loop is None


def test_recovery_loop_absent_without_how():
    rt = Runtime(_cfg(enable_how=False))
    rt.initialize()
    try:
        assert rt._recovery_loop is None
    finally:
        rt.stop()


def test_recovery_loop_disabled_by_config():
    rt = Runtime(_cfg(enable_recovery_loop=False))
    rt.initialize()
    try:
        assert rt._how is not None
        assert rt._recovery_loop is None
    finally:
        rt.stop()

"""PR-DF-09: Knowledge package consolidation — canonical rosclaw.knowledge,
legacy shim, and federation coordinates actually passed by the Runtime."""

from rosclaw.core.runtime import Runtime, RuntimeConfig


def test_legacy_import_surface():
    from rosclaw.knowledge.legacy import (
        EmbodimentCard,
        KnowledgeInterface,
        LegacyKnowledgeRuntime,
        TaskCard,
        VerifierCard,
    )

    assert LegacyKnowledgeRuntime is KnowledgeInterface
    for cls in (KnowledgeInterface, TaskCard, EmbodimentCard, VerifierCard):
        assert cls.__name__.startswith(("Knowledge", "Task", "Embodiment", "Verifier"))


def test_know_package_still_importable():
    import rosclaw.know  # noqa: F401 — compat shim must keep working

    assert "DEPRECATED" in rosclaw.knowledge.legacy.__doc__


def _cfg(**over):
    base = dict(  # noqa: C408 — kwargs spread pattern
        robot_id="sim_test",
        seekdb_backend="memory",
        enable_memory=False,
        enable_practice=False,
        enable_how=False,
        enable_auto=False,
        enable_knowledge=True,
        enable_skill_manager=False,
        enable_provider=False,
        enable_sense=False,
        enable_firewall=False,
        enable_event_persistence=False,
        enable_tracing=False,
        knowledge_v2_mode="inprocess",
    )
    base.update(over)
    return RuntimeConfig(**base)


def test_runtime_passes_federation_coordinates(tmp_path):
    rt = Runtime(_cfg(seekdb_path=str(tmp_path / "knowledge.sqlite")))
    rt.initialize()
    try:
        mgr = rt._knowledge_v2_manager
        assert mgr is not None, "v2 manager should initialize in inprocess mode"
        cfg = mgr.config
        assert cfg.memory_path == str(tmp_path / "knowledge.sqlite")
        assert cfg.practice_path.endswith("data/practice")
    finally:
        rt.stop()


def test_federation_database_from_dsn(tmp_path):
    rt = Runtime(
        _cfg(
            seekdb_url="mysql://root@127.0.0.1:2881/rosclaw_prod",
            seekdb_backend="sqlite",
            seekdb_path=str(tmp_path / "knowledge.sqlite"),
        )
    )
    rt.initialize()
    try:
        cfg = rt._knowledge_v2_manager.config
        assert cfg.memory_database == "rosclaw_prod"
    finally:
        rt.stop()

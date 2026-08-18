"""PR-DF-12: EvolutionRepository — structured store canonical, LocalStore spool."""

import json

from rosclaw.evolution.orchestrator.storage.local_store import LocalStore
from rosclaw.evolution.repository import EvolutionRepository
from rosclaw.memory.seekdb_client import InMemoryStructuredStore, SQLiteStructuredStore


def _repo(tmp_path):
    store = InMemoryStructuredStore()
    store.connect()
    cache = LocalStore(str(tmp_path / "auto"))
    return EvolutionRepository(store, cache), store, cache


def test_save_goes_to_store_and_cache(tmp_path):
    repo, store, cache = _repo(tmp_path)
    repo.save("proposals", "prop_1", {"task_id": "t1", "hypothesis": "h"})
    rows = store.query("evolution_records", {"namespace": "proposals"})
    assert len(rows) == 1
    assert rows[0]["id"] == "proposals:prop_1"
    assert json.loads(rows[0]["data"])["hypothesis"] == "h"
    assert cache.load("proposals", "prop_1")["hypothesis"] == "h"


def test_load_reads_store_first(tmp_path):
    repo, store, cache = _repo(tmp_path)
    repo.save("experiments", "exp_1", {"result": 0.87})
    cache.save("experiments", "exp_1", {"result": 0.0})  # stale cache
    assert repo.load("experiments", "exp_1")["result"] == 0.87


def test_store_down_falls_back_to_spool(tmp_path):
    repo, store, cache = _repo(tmp_path)
    repo.save("champions", "champ_1", {"skill": "pick", "version": "1.7"})

    class _DeadStore:
        def query(self, *a, **k):
            raise ConnectionError("seekdb down")

        def insert(self, *a, **k):
            raise ConnectionError("seekdb down")

        def delete(self, *a, **k):
            raise ConnectionError("seekdb down")

    repo._store = _DeadStore()
    # save spools instead of raising; load/list still work from the spool
    repo.save("champions", "champ_2", {"skill": "place", "version": "1.0"})
    assert repo.load("champions", "champ_1")["version"] == "1.7"
    assert set(repo.list_keys("champions")) == {"champ_1", "champ_2"}


def test_sqlite_roundtrip_with_schema(tmp_path):
    """The evolution_records table exists in ROSCLAW_STRUCTURED_SCHEMAS."""
    store = SQLiteStructuredStore(str(tmp_path / "knowledge.sqlite"))
    store.connect()
    cache = LocalStore(str(tmp_path / "auto"))
    repo = EvolutionRepository(store, cache)
    repo.save("patches", "patch_1", {"changes": {"force": 300}})
    assert repo.load("patches", "patch_1")["changes"] == {"force": 300}
    assert repo.list_keys("patches") == ["patch_1"]
    store.disconnect()


def test_namespaces_isolated_by_composite_id(tmp_path):
    repo, store, _ = _repo(tmp_path)
    repo.save("proposals", "x", {"kind": "proposal"})
    repo.save("deadends", "x", {"kind": "deadend"})
    assert repo.load("proposals", "x")["kind"] == "proposal"
    assert repo.load("deadends", "x")["kind"] == "deadend"
    assert repo.stats()["namespaces"] == {"proposals": 1, "deadends": 1}


def test_auto_engine_uses_repository_when_store_injected(tmp_path):
    from rosclaw.evolution.orchestrator.config import AutoConfig
    from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine

    store = InMemoryStructuredStore()
    store.connect()
    cfg = AutoConfig(local_store_path=str(tmp_path / "auto"), storage_backend="hybrid")
    engine = AutoEngine(config=cfg, event_bus=None, seekdb_client=store)
    assert isinstance(engine.store, EvolutionRepository)
    engine.store.save("tasks", "task_1", {"goal": "pick"})
    assert store.count("evolution_records", {"namespace": "tasks"}) == 1


def test_auto_engine_local_default_unchanged(tmp_path):
    from rosclaw.evolution.orchestrator.config import AutoConfig
    from rosclaw.evolution.orchestrator.engine.auto_engine import AutoEngine

    store = InMemoryStructuredStore()
    store.connect()
    cfg = AutoConfig(local_store_path=str(tmp_path / "auto"))  # storage_backend="local"
    engine = AutoEngine(config=cfg, event_bus=None, seekdb_client=store)
    assert isinstance(engine.store, LocalStore)

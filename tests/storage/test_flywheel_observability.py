"""PR-DF-15: observability — db doctor flywheel checks + dashboard endpoint."""


from rosclaw.memory.seekdb_client import SQLiteStructuredStore
from rosclaw.storage.cli import _flywheel_checks


def _make_cfg(tmp_path, **over):
    cfg = {
        "backend": "sqlite",
        "path": str(tmp_path / "knowledge.sqlite"),
        "retrieval_enabled": False,
        "outbox_enabled": False,
        "outbox_path": str(tmp_path / "outbox.sqlite"),
        "retrieval_path": str(tmp_path / "seekdb"),
    }
    cfg.update(over)
    return cfg


def test_doctor_flags_active_memory_without_evidence(tmp_path):
    store = SQLiteStructuredStore(str(tmp_path / "knowledge.sqlite"))
    store.connect()
    store.insert(
        "memory_items",
        {
            "id": "mem_orphan",
            "memory_type": "episodic",
            "robot_id": "sim",
            "title": "t",
            "document": "d",
            "status": "active",
            "content_hash": "x",
        },
    )
    checks, issues = [], []
    _flywheel_checks(_make_cfg(tmp_path), store, checks, issues)
    names = [c[0] for c in checks]
    assert "memory evidence" in names
    assert any("no evidence" in i for i in issues)
    store.disconnect()


def test_doctor_clean_when_evidence_present(tmp_path):
    store = SQLiteStructuredStore(str(tmp_path / "knowledge.sqlite"))
    store.connect()
    store.insert(
        "memory_items",
        {
            "id": "mem_ok",
            "memory_type": "episodic",
            "robot_id": "sim",
            "title": "t",
            "document": "d",
            "status": "active",
            "content_hash": "y",
        },
    )
    store.insert(
        "memory_evidence",
        {"id": "evd_1", "memory_id": "mem_ok", "evidence_type": "practice_event"},
    )
    checks, issues = [], []
    _flywheel_checks(_make_cfg(tmp_path), store, checks, issues)
    mem = next(c for c in checks if c[0] == "memory evidence")
    assert mem[2] is True
    assert not issues
    store.disconnect()


def test_doctor_flags_unlinked_receipts(tmp_path):
    store = SQLiteStructuredStore(str(tmp_path / "knowledge.sqlite"))
    store.connect()
    store.insert("execution_receipts", {"id": "rcpt_1", "action_id": "a1"})
    checks, issues = [], []
    _flywheel_checks(_make_cfg(tmp_path), store, checks, issues)
    lin = next(c for c in checks if c[0] == "lineage edges")
    assert lin[2] is False
    assert any("lineage" in i for i in issues)
    store.disconnect()


def test_doctor_survives_dead_store(tmp_path):
    class _Dead:
        def query(self, *a, **k):
            raise ConnectionError("down")

        def count(self, *a, **k):
            raise ConnectionError("down")

    checks, issues = [], []
    _flywheel_checks(_make_cfg(tmp_path), _Dead(), checks, issues)  # must not raise


def test_dashboard_flywheel_endpoint(tmp_path, monkeypatch):
    """The aggregate endpoint reports store counts without a runtime."""
    monkeypatch.setenv("ROSCLAW_HOME", str(tmp_path / ".rosclaw"))
    store = SQLiteStructuredStore(str(tmp_path / ".rosclaw" / "data" / "memory" / "knowledge.sqlite"))
    store.connect()
    store.insert(
        "memory_items",
        {"id": "m1", "memory_type": "episodic", "robot_id": "sim", "title": "t",
         "document": "d", "status": "active", "content_hash": "z"},
    )
    store.disconnect()

    import asyncio

    from rosclaw.dashboard.web_server import DashboardWebServer

    server = DashboardWebServer()
    routes = {r.path: r.endpoint for r in server.app.routes}
    assert "/api/data-flywheel" in routes
    payload = asyncio.run(routes["/api/data-flywheel"]())
    assert payload["structured_store_connected"] is True
    assert payload["memory"]["items"] == 1

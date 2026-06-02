"""Tests for know/storage.py seed and ingest helpers."""


from rosclaw.know.storage import seed_knowledge_graph, ingest_e_urdf_capabilities


class FakeSeekDB:
    def __init__(self, fail_on=None):
        self._records = []
        self.fail_on = fail_on or set()

    def insert(self, table, record):
        if record.get("id") in self.fail_on:
            raise RuntimeError("insert fail")
        self._records.append(record)


def test_seed_knowledge_graph_none_client():
    result = seed_knowledge_graph(None)
    assert result == {"capabilities": 0, "symptoms": 0, "total": 0}


def test_seed_knowledge_graph_success():
    db = FakeSeekDB()
    result = seed_knowledge_graph(db)
    assert result["total"] > 0
    assert result["capabilities"] > 0
    assert result["symptoms"] > 0


def test_seed_knowledge_graph_partial_failure():
    db = FakeSeekDB(fail_on={"ur5e_has_capability_6dof_arm"})
    result = seed_knowledge_graph(db)
    assert result["total"] > 0  # Some succeeded
    assert result["total"] < 50  # At least one failed


def test_ingest_e_urdf_capabilities_none_client():
    assert ingest_e_urdf_capabilities("ur5e", ["tag"], None) == 0


def test_ingest_e_urdf_capabilities_empty_tags():
    class FakeDB:
        def insert(self, t, r): pass  # noqa: E704
    assert ingest_e_urdf_capabilities("ur5e", [], FakeDB()) == 0
    assert ingest_e_urdf_capabilities("ur5e", [""], FakeDB()) == 0


def test_ingest_e_urdf_capabilities_success():
    db = FakeSeekDB()
    count = ingest_e_urdf_capabilities("ur5e", ["grasp", "place"], db)
    assert count == 2
    assert len(db._records) == 2
    assert db._records[0]["subject"] == "ur5e"


def test_ingest_e_urdf_capabilities_partial_failure():
    db = FakeSeekDB(fail_on={"ur5e_has_capability_grasp"})
    count = ingest_e_urdf_capabilities("ur5e", ["grasp", "place"], db)
    assert count == 1

"""Tests for know/graph.py helpers."""

from rosclaw.know.graph import (
    count_knowledge_facts,
    get_related_robots,
    get_robot_capabilities,
    get_robot_symptoms,
)


class FakeSeekDB:
    def __init__(self, records=None, counts=None):
        self._records = records or []
        self._counts = counts or {}

    def query(self, table, filters=None, order_by=None, limit=None):
        return [r for r in self._records if self._matches(r, filters)]

    def count(self, table, filters=None):
        key = str(filters)
        return self._counts.get(key, 0)

    @staticmethod
    def _matches(record, filters):
        if not filters:
            return True
        return all(record.get(k) == v for k, v in filters.items())


def test_get_robot_capabilities_none_client():
    assert get_robot_capabilities(None, "ur5e") == []


def test_get_robot_capabilities_with_data():
    db = FakeSeekDB(records=[
        {"subject": "ur5e", "predicate": "has_capability", "object": "pick", "confidence": 0.9, "source": "test"},
        {"subject": "ur5e", "predicate": "has_capability", "object": "place", "confidence": 0.8, "source": "test"},
    ])
    caps = get_robot_capabilities(db, "ur5e")
    assert len(caps) == 2
    assert caps[0]["capability"] == "pick"


def test_get_robot_symptoms_none_client():
    assert get_robot_symptoms(None, "ur5e") == []


def test_get_robot_symptoms_with_data():
    db = FakeSeekDB(records=[
        {"subject": "ur5e", "predicate": "has_symptom", "object": "drift", "confidence": 0.7, "source": "test"},
    ])
    syms = get_robot_symptoms(db, "ur5e")
    assert len(syms) == 1
    assert syms[0]["symptom"] == "drift"


def test_get_related_robots_none_client():
    assert get_related_robots(None, "pick") == []


def test_get_related_robots_with_data():
    db = FakeSeekDB(records=[
        {"subject": "ur5e", "predicate": "has_capability", "object": "pick"},
        {"subject": "panda", "predicate": "has_capability", "object": "pick"},
    ])
    robots = get_related_robots(db, "pick")
    assert robots == ["panda", "ur5e"]


def test_count_knowledge_facts_none_client():
    assert count_knowledge_facts(None) == {"total": 0, "capabilities": 0, "symptoms": 0}


def test_count_knowledge_facts_with_data():
    db = FakeSeekDB(counts={
        "None": 100,
        "{'predicate': 'has_capability'}": 40,
        "{'predicate': 'has_symptom'}": 20,
    })
    result = count_knowledge_facts(db)
    assert result == {"total": 100, "capabilities": 40, "symptoms": 20}


def test_count_knowledge_facts_exception():
    class BadDB:
        def count(self, table, filters=None):
            raise RuntimeError("db down")

    result = count_knowledge_facts(BadDB())
    assert result == {"total": 0, "capabilities": 0, "symptoms": 0}

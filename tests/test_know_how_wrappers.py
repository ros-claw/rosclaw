"""Tests for KNOW/HOW wrapper classes and MemoryInterface methods.

Covers:
- MemoryInterface.query_knowledge_graph()
- MemoryInterface.get_heuristic_rules()
- KnowledgeGraphWrapper
- HeuristicRuleWrapper
"""

import pytest

from rosclaw.memory.interface import HeuristicRuleWrapper, KnowledgeGraphWrapper, MemoryInterface
from rosclaw.memory.seekdb_client import SeekDBMemoryClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mem():
    m = MemoryInterface("test_bot")
    m.initialize()
    yield m
    m.stop()


@pytest.fixture
def kg_wrapper():
    client = SeekDBMemoryClient()
    client.connect()
    w = KnowledgeGraphWrapper(client)
    return w


@pytest.fixture
def hr_wrapper():
    client = SeekDBMemoryClient()
    client.connect()
    w = HeuristicRuleWrapper(client)
    return w


# ---------------------------------------------------------------------------
# MemoryInterface.query_knowledge_graph
# ---------------------------------------------------------------------------


class TestQueryKnowledgeGraph:
    def test_empty_returns_empty(self, mem):
        assert mem.query_knowledge_graph(entity_id="r1") == []

    def test_filter_by_entity_id(self, mem):
        mem._client.insert("knowledge_graph", {
            "id": "k1", "subject": "ur5e", "predicate": "has_capability",
            "object": "grasp", "confidence": 0.9, "source": "seed", "timestamp": 1.0,
        })
        mem._client.insert("knowledge_graph", {
            "id": "k2", "subject": "panda", "predicate": "has_capability",
            "object": "place", "confidence": 0.8, "source": "seed", "timestamp": 1.0,
        })
        results = mem.query_knowledge_graph(entity_id="ur5e")
        assert len(results) == 1
        assert results[0]["object"] == "grasp"

    def test_filter_by_predicate(self, mem):
        mem._client.insert("knowledge_graph", {
            "id": "k1", "subject": "ur5e", "predicate": "has_capability",
            "object": "grasp", "confidence": 0.9, "source": "seed", "timestamp": 1.0,
        })
        mem._client.insert("knowledge_graph", {
            "id": "k2", "subject": "ur5e", "predicate": "has_symptom",
            "object": "wrist_drift", "confidence": 0.7, "source": "seed", "timestamp": 1.0,
        })
        results = mem.query_knowledge_graph(predicate="has_symptom")
        assert len(results) == 1
        assert results[0]["object"] == "wrist_drift"

    def test_filter_by_object_value(self, mem):
        mem._client.insert("knowledge_graph", {
            "id": "k1", "subject": "ur5e", "predicate": "has_capability",
            "object": "grasp", "confidence": 0.9, "source": "seed", "timestamp": 1.0,
        })
        results = mem.query_knowledge_graph(object_value="grasp")
        assert len(results) == 1

    def test_combined_filters(self, mem):
        mem._client.insert("knowledge_graph", {
            "id": "k1", "subject": "ur5e", "predicate": "has_capability",
            "object": "grasp", "confidence": 0.9, "source": "seed", "timestamp": 1.0,
        })
        results = mem.query_knowledge_graph(
            entity_id="ur5e", predicate="has_capability", object_value="grasp"
        )
        assert len(results) == 1

    def test_limit(self, mem):
        for i in range(5):
            mem._client.insert("knowledge_graph", {
                "id": f"k{i}", "subject": "ur5e", "predicate": "has_capability",
                "object": f"cap{i}", "confidence": 0.5, "source": "seed", "timestamp": 1.0,
            })
        results = mem.query_knowledge_graph(entity_id="ur5e", limit=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# MemoryInterface.get_heuristic_rules
# ---------------------------------------------------------------------------


class TestGetHeuristicRules:
    def test_empty_returns_empty(self, mem):
        assert mem.get_heuristic_rules() == []

    def test_filter_by_condition(self, mem):
        mem._client.insert("heuristic_rules", {
            "id": "h1", "condition": "gripper_force_low", "action": "increase_force",
            "priority": 5, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        mem._client.insert("heuristic_rules", {
            "id": "h2", "condition": "collision_risk", "action": "slow_down",
            "priority": 3, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        results = mem.get_heuristic_rules(condition="gripper_force_low")
        assert len(results) == 1
        assert results[0]["id"] == "h1"

    def test_min_priority_filter(self, mem):
        mem._client.insert("heuristic_rules", {
            "id": "h1", "condition": "a", "action": "act1",
            "priority": 5, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        mem._client.insert("heuristic_rules", {
            "id": "h2", "condition": "b", "action": "act2",
            "priority": 1, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        results = mem.get_heuristic_rules(min_priority=3)
        assert len(results) == 1
        assert results[0]["id"] == "h1"

    def test_order_by_priority_desc(self, mem):
        mem._client.insert("heuristic_rules", {
            "id": "h1", "condition": "a", "action": "act1",
            "priority": 3, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        mem._client.insert("heuristic_rules", {
            "id": "h2", "condition": "b", "action": "act2",
            "priority": 10, "success_count": 0, "failure_count": 0, "last_triggered": None,
        })
        results = mem.get_heuristic_rules()
        assert results[0]["id"] == "h2"
        assert results[1]["id"] == "h1"


# ---------------------------------------------------------------------------
# KnowledgeGraphWrapper
# ---------------------------------------------------------------------------


class TestKnowledgeGraphWrapper:
    def test_get_triples_empty(self, kg_wrapper):
        assert kg_wrapper.get_triples() == []

    def test_add_and_get_triple(self, kg_wrapper):
        rid = kg_wrapper.add_triple("t1", "ur5e", "has_capability", "grasp", confidence=0.9)
        assert rid == "t1"
        results = kg_wrapper.get_triples(subject="ur5e")
        assert len(results) == 1
        assert results[0]["object"] == "grasp"

    def test_get_capabilities(self, kg_wrapper):
        kg_wrapper.add_triple("t1", "ur5e", "has_capability", "grasp", confidence=0.9)
        kg_wrapper.add_triple("t2", "ur5e", "has_capability", "place", confidence=0.8)
        kg_wrapper.add_triple("t3", "panda", "has_capability", "grasp", confidence=0.7)
        caps = kg_wrapper.get_capabilities("ur5e")
        assert len(caps) == 2
        assert caps[0]["capability"] == "grasp"  # higher confidence first
        assert caps[1]["capability"] == "place"

    def test_count(self, kg_wrapper):
        assert kg_wrapper.count() == 0
        kg_wrapper.add_triple("t1", "ur5e", "has_capability", "grasp")
        assert kg_wrapper.count() == 1


# ---------------------------------------------------------------------------
# HeuristicRuleWrapper
# ---------------------------------------------------------------------------


class TestHeuristicRuleWrapper:
    def test_list_rules_empty(self, hr_wrapper):
        assert hr_wrapper.list_rules() == []

    def test_add_and_get_rule(self, hr_wrapper):
        rid = hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        assert rid == "r1"
        rule = hr_wrapper.get_rule("r1")
        assert rule is not None
        assert rule["condition"] == "force_low"
        assert rule["action"] == "increase_force"

    def test_list_rules_with_filter(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        hr_wrapper.add_rule("r2", "collision", "slow_down", priority=3)
        results = hr_wrapper.list_rules(condition_filter="force_low")
        assert len(results) == 1
        assert results[0]["id"] == "r1"

    def test_min_priority_filter(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "a", "act1", priority=5)
        hr_wrapper.add_rule("r2", "b", "act2", priority=1)
        results = hr_wrapper.list_rules(min_priority=3)
        assert len(results) == 1
        assert results[0]["id"] == "r1"

    def test_update_rule(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        ok = hr_wrapper.update_rule("r1", priority=10)
        assert ok is True
        rule = hr_wrapper.get_rule("r1")
        assert rule["priority"] == 10

    def test_update_rule_no_allowed_fields(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        ok = hr_wrapper.update_rule("r1", success_count=99)
        assert ok is False

    def test_record_trigger_success(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        ok = hr_wrapper.record_trigger("r1", success=True)
        assert ok is True
        rule = hr_wrapper.get_rule("r1")
        assert rule["success_count"] == 1
        assert rule["last_triggered"] is not None

    def test_record_trigger_failure(self, hr_wrapper):
        hr_wrapper.add_rule("r1", "force_low", "increase_force", priority=5)
        ok = hr_wrapper.record_trigger("r1", success=False)
        assert ok is True
        rule = hr_wrapper.get_rule("r1")
        assert rule["failure_count"] == 1

    def test_record_trigger_missing_rule(self, hr_wrapper):
        assert hr_wrapper.record_trigger("missing", success=True) is False

    def test_count(self, hr_wrapper):
        assert hr_wrapper.count() == 0
        hr_wrapper.add_rule("r1", "a", "act1")
        assert hr_wrapper.count() == 1

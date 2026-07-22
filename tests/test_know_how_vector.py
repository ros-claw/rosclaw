"""Tests for vector search plumbing into KNOW and HOW interfaces."""

from __future__ import annotations

import pytest

from rosclaw.how.engine import HeuristicEngine
from rosclaw.know.interface import KnowledgeInterface
from rosclaw.memory.interface import MemoryInterface
from rosclaw.memory.seekdb_client import SQLiteKnowledgeStore


@pytest.fixture
def vector_client(tmp_path):
    db_path = tmp_path / "know_how_memory.sqlite"
    client = SQLiteKnowledgeStore(db_path=str(db_path), vector_enabled=True)
    client.connect()
    return client


@pytest.fixture
def memory_interface(vector_client):
    mem = MemoryInterface("test_robot", seekdb_client=vector_client)
    mem.initialize()
    try:
        yield mem
    finally:
        mem.stop()


def test_knowledge_interface_vector_match(memory_interface):
    memory_interface.store_experience(
        event_id="fail1",
        event_type="praxis",
        instruction="gripper slipped while lifting the red cup",
        outcome="failure",
        error_details="gripper slip caused the cup to fall",
        tags=["gripper", "slip", "cup"],
        metadata={"recovery_hint": "Increase grip force by 20%", "domain": "Manipulation"},
    )

    know = KnowledgeInterface(
        robot_id="test_robot",
        seekdb_client=memory_interface.seekdb_client,
        memory_interface=memory_interface,
        similarity_floor=0.02,
    )
    know.initialize()
    try:
        result = know.match_symptom("the gripper slipped and dropped the cup")
        assert result is not None
        assert result.get("source") == "vector_memory"
        assert result.get("experience_id") == "fail1"
        assert "gripper" in result.get("symptom", "").lower()
        assert result.get("fix") == "Increase grip force by 20%"
        assert result.get("similarity", 0) >= 0.02
    finally:
        know.stop()


def test_knowledge_interface_vector_match_returns_none_below_floor(memory_interface):
    memory_interface.store_experience(
        event_id="fail2",
        event_type="praxis",
        instruction="unrelated navigation problem",
        outcome="failure",
        error_details="odometry drift in hallway",
        tags=["navigation"],
    )

    know = KnowledgeInterface(
        robot_id="test_robot",
        seekdb_client=memory_interface.seekdb_client,
        memory_interface=memory_interface,
        similarity_floor=0.99,
    )
    know.initialize()
    try:
        result = know.match_symptom("gripper slipped")
        # With a very high floor the vector match should be rejected.
        assert result is None or result.get("source") != "vector_memory"
    finally:
        know.stop()


@pytest.mark.asyncio
async def test_how_engine_memory_analogy_fallback(memory_interface):
    memory_interface.store_experience(
        event_id="fail3",
        event_type="praxis",
        instruction="gripper slipped while lifting the red cup",
        outcome="failure",
        error_details="gripper slipped and the cup fell",
        tags=["gripper", "slip", "cup"],
        metadata={"recovery_hint": "Increase grip force by 20% and retry"},
    )

    engine = HeuristicEngine(
        seekdb_client=memory_interface.seekdb_client,
        memory_interface=memory_interface,
    )
    await engine.initialize()
    try:
        # Query uses wording unlikely to hit the safety taxonomy or seeded rules.
        result = await engine.suggest_recovery("the cup fell because the gripper slipped")
        assert result is not None
        assert result.get("source") == "memory_analogy"
        assert "Increase grip force" in result.get("action", "")
    finally:
        await engine.shutdown()


@pytest.mark.asyncio
async def test_how_engine_extracts_bounded_hint_from_nested_metadata(memory_interface):
    memory_interface.store_experience(
        event_id="nested-hint",
        event_type="praxis",
        instruction="quasar kinematic dimensionality mismatch",
        outcome="failure",
        error_details="quasar kinematic dimensionality mismatch",
        tags=["quasar", "dimensionality"],
        metadata={
            "recovery_hint": {
                "hint": "Use the matching mobile-base executor" + "x" * 10_000,
                "body_readiness": {"nested": "y" * 100_000},
            }
        },
    )

    analogy = memory_interface.find_analogy("quasar kinematic dimensionality mismatch", limit=1)
    assert analogy is not None
    assert analogy["action_suggestion"].startswith("Use the matching mobile-base executor")
    assert len(analogy["action_suggestion"]) == 4096

    engine = HeuristicEngine(
        seekdb_client=memory_interface.seekdb_client,
        memory_interface=memory_interface,
    )
    await engine.initialize()
    try:
        result = await engine.suggest_recovery("quasar kinematic dimensionality mismatch")
        assert result is not None
        assert result["source"] == "memory_analogy"
        assert len(result["action"]) == 4096
    finally:
        await engine.shutdown()


def test_knowledge_interface_falls_back_to_keyword_without_memory(vector_client):
    know = KnowledgeInterface(
        robot_id="test_robot",
        seekdb_client=vector_client,
        memory_interface=None,
    )
    know.initialize()
    try:
        result = know.match_symptom("torque overflow saturation")
        assert result is not None
        assert result.get("source") != "vector_memory"
    finally:
        know.stop()
        vector_client.disconnect()

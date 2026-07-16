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

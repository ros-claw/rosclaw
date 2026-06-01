"""Tests for SeekDB Client (Sprint 5)."""

import os

from rosclaw.memory.seekdb_client import SeekDBMemoryClient, SeekDBSQLiteClient
from rosclaw.memory.interface import MemoryInterface
from rosclaw.core.event_bus import EventBus, Event


def test_memory_client_crud():
    """Insert, query, update, count on in-memory backend."""
    client = SeekDBMemoryClient()
    client.connect()

    record_id = client.insert("experience_graph", {
        "id": "exp1",
        "event_type": "praxis",
        "robot_id": "r1",
        "timestamp": 1.0,
        "instruction": "pick up block",
    })
    assert record_id == "exp1"

    results = client.query("experience_graph", filters={"robot_id": "r1"})
    assert len(results) == 1
    assert results[0]["instruction"] == "pick up block"

    updated = client.update("experience_graph", "exp1", {"instruction": "place block"})
    assert updated is True

    results = client.query("experience_graph", filters={"id": "exp1"})
    assert results[0]["instruction"] == "place block"

    count = client.count("experience_graph", {"robot_id": "r1"})
    assert count == 1


def test_sqlite_client_crud():
    """Same operations on SQLite backend."""
    db_path = "/tmp/test_seekdb.sqlite"
    if os.path.exists(db_path):
        os.remove(db_path)

    client = SeekDBSQLiteClient(db_path)
    client.connect()

    record_id = client.insert("experience_graph", {
        "id": "exp2",
        "event_type": "praxis",
        "robot_id": "r2",
        "timestamp": 2.0,
        "instruction": "move arm",
        "cot_trace": ["step1", "step2"],
        "tags": ["grasp", "red"],
    })
    assert record_id is not None

    results = client.query("experience_graph", filters={"robot_id": "r2"})
    assert len(results) == 1
    assert results[0]["instruction"] == "move arm"
    # JSON fields deserialized
    assert results[0]["cot_trace"] == ["step1", "step2"]
    assert results[0]["tags"] == ["grasp", "red"]

    count = client.count("experience_graph")
    assert count == 1

    client.disconnect()
    os.remove(db_path)


def test_experience_storage():
    """store_experience() -> findable in experience_graph."""
    bus = EventBus()
    memory = MemoryInterface("test_robot", event_bus=bus)
    memory.initialize()

    exp_id = memory.store_experience(
        event_id="exp3",
        event_type="praxis",
        instruction="pick up red block",
        outcome="success",
        duration_sec=3.2,
        tags=["grasp", "red"],
    )
    assert exp_id == "exp3"

    exp = memory.get_experience("exp3")
    assert exp is not None
    assert exp["instruction"] == "pick up red block"
    assert exp["outcome"] == "success"
    memory.stop()


def test_similarity_search():
    """find_similar_experiences() returns keyword-matched results."""
    bus = EventBus()
    memory = MemoryInterface("test_robot", event_bus=bus)
    memory.initialize()

    memory.store_experience("e1", "praxis", "pick up red block", outcome="success", tags=["grasp"])
    memory.store_experience("e2", "praxis", "pick up blue block", outcome="failure", tags=["grasp"])
    memory.store_experience("e3", "praxis", "place block on table", outcome="success", tags=["place"])

    results = memory.find_similar_experiences("pick up block")
    assert len(results) >= 2
    # "pick up red block" and "pick up blue block" should match
    instructions = [r["instruction"] for r in results]
    assert "pick up red block" in instructions
    assert "pick up blue block" in instructions
    memory.stop()


def test_praxis_auto_ingestion():
    """praxis.recorded event -> experience auto-stored."""
    bus = EventBus()
    memory = MemoryInterface("test_robot", event_bus=bus)
    memory.initialize()

    stored_events = []
    bus.subscribe("memory.experience.stored", lambda e: stored_events.append(e.payload))

    bus.publish(Event(
        topic="praxis.recorded",
        payload={
            "event_id": "auto1",
            "event_type": "success",
            "instruction": "auto-ingested task",
            "duration_sec": 1.5,
        },
        source="test",
    ))

    assert len(stored_events) == 1
    assert stored_events[0]["experience_id"] == "auto1"

    exp = memory.get_experience("auto1")
    assert exp is not None
    assert exp["instruction"] == "auto-ingested task"
    memory.stop()


def test_memory_statistics():
    """get_statistics() returns correct counts."""
    bus = EventBus()
    memory = MemoryInterface("test_robot", event_bus=bus)
    memory.initialize()

    memory.store_experience("s1", "praxis", "task1", outcome="success")
    memory.store_experience("s2", "praxis", "task2", outcome="success")
    memory.store_experience("f1", "praxis", "task3", outcome="failure")

    stats = memory.get_statistics()
    assert stats["total_experiences"] == 3
    assert stats["success_count"] == 2
    assert stats["failure_count"] == 1
    assert stats["success_rate"] == 2 / 3
    memory.stop()

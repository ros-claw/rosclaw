"""Tests for Memory module."""

from rosclaw.memory.interface import MemoryInterface


def test_memory_store_and_query():
    mem = MemoryInterface("test_bot")
    mem.initialize()
    exp_id = mem.store_experience(
        event_id="exp1",
        event_type="praxis",
        instruction="pick up block",
        outcome="success",
        metadata={"task_type": "pick", "data": "test"},
    )
    assert exp_id == "exp1"

    exp = mem.get_experience("exp1")
    assert exp is not None
    assert exp["instruction"] == "pick up block"
    assert exp["outcome"] == "success"
    mem.stop()


def test_memory_get_experience():
    mem = MemoryInterface("test_bot")
    mem.initialize()
    mem.store_experience(
        event_id="exp2",
        event_type="skill",
        instruction="grasp object",
        outcome="success",
        tags=["grasp"],
    )
    exp = mem.get_experience("exp2")
    assert exp is not None
    assert exp["instruction"] == "grasp object"
    mem.stop()


def test_memory_get_experience_missing():
    mem = MemoryInterface("test_bot")
    mem.initialize()
    assert mem.get_experience("nonexistent") is None
    mem.stop()


def test_memory_statistics():
    mem = MemoryInterface("test_bot")
    mem.initialize()
    mem.store_experience("s1", "praxis", "task1", outcome="success")
    mem.store_experience("s2", "praxis", "task2", outcome="success")
    mem.store_experience("f1", "praxis", "task3", outcome="failure")

    stats = mem.get_statistics()
    assert stats["total_experiences"] == 3
    assert stats["success_count"] == 2
    assert stats["failure_count"] == 1
    mem.stop()

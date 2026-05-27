"""Tests for Skill Manager."""

import pytest

from rosclaw.core.event_bus import EventBus
from rosclaw.skill_manager.registry import SkillRegistry, SkillEntry
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.loader import SkillLoader


def test_skill_registry_register():
    reg = SkillRegistry()
    reg.initialize()
    entry = SkillEntry(name="pick", description="Pick object", skill_type="programmed")
    reg.register(entry)
    assert reg.count == 1
    assert reg.get("pick") is not None
    reg.stop()


def test_skill_registry_unregister():
    reg = SkillRegistry()
    reg.initialize()
    reg.register(SkillEntry(name="place", description="Place object", skill_type="programmed"))
    assert reg.unregister("place") is True
    assert reg.count == 0
    assert reg.unregister("missing") is False
    reg.stop()


def test_skill_registry_list():
    reg = SkillRegistry()
    reg.initialize()
    reg.register(SkillEntry(name="a", description="", skill_type="programmed"))
    reg.register(SkillEntry(name="b", description="", skill_type="learned"))
    assert reg.list_skills() == ["a", "b"]
    assert reg.list_skills("programmed") == ["a"]
    reg.stop()


def test_skill_registry_stats():
    reg = SkillRegistry()
    reg.initialize()
    reg.register(SkillEntry(name="s", description="", skill_type="programmed"))
    reg.update_stats("s", success=True)
    reg.update_stats("s", success=False)
    skill = reg.get("s")
    assert skill.execution_count == 2
    assert 0.0 < skill.success_rate < 1.0
    reg.stop()


def test_skill_registry_get_stats():
    reg = SkillRegistry()
    reg.initialize()
    reg.register(SkillEntry(name="s1", description="", skill_type="programmed"))
    reg.register(SkillEntry(name="s2", description="", skill_type="learned"))
    reg.update_stats("s1", success=True)
    reg.update_stats("s1", success=True)
    reg.update_stats("s2", success=False)

    stats = reg.get_stats()
    assert stats["total_skills"] == 2
    assert stats["total_executions"] == 3
    assert 0.0 <= stats["average_success_rate"] <= 1.0
    assert "programmed" in stats["by_type"]
    assert "learned" in stats["by_type"]
    reg.stop()


def test_skill_executor_success():
    bus = EventBus()
    reg = SkillRegistry()
    reg.initialize()
    executor = SkillExecutor(bus, reg)
    executor.initialize()

    def handler(params):
        return {"done": True}

    reg.register(SkillEntry(
        name="test_skill",
        description="Test",
        skill_type="programmed",
        handler=handler,
    ))

    result = executor.execute("test_skill")
    assert result["status"] == "success"
    assert not executor.is_executing()
    executor.stop()
    reg.stop()


def test_skill_executor_not_found():
    bus = EventBus()
    reg = SkillRegistry()
    reg.initialize()
    executor = SkillExecutor(bus, reg)
    executor.initialize()
    result = executor.execute("missing")
    assert result["status"] == "error"
    executor.stop()
    reg.stop()


def test_skill_loader_json(tmp_path):
    reg = SkillRegistry()
    reg.initialize()
    loader = SkillLoader(reg)

    skill_file = tmp_path / "test_skill.json"
    skill_file.write_text('{"name": "push", "description": "Push object", "skill_type": "programmed"}')

    entry = loader.load_from_json(skill_file)
    assert entry.name == "push"
    assert reg.get("push") is not None
    reg.stop()


def test_skill_loader_directory(tmp_path):
    reg = SkillRegistry()
    reg.initialize()
    loader = SkillLoader(reg)

    (tmp_path / "a.json").write_text('{"name": "a", "skill_type": "programmed"}')
    (tmp_path / "b.json").write_text('{"name": "b", "skill_type": "programmed"}')

    count = loader.load_from_directory(tmp_path)
    assert count == 2
    assert reg.count == 2
    reg.stop()


def test_skill_loader_programmed():
    reg = SkillRegistry()
    reg.initialize()
    loader = SkillLoader(reg)

    def my_handler(params):
        return params

    entry = loader.create_programmed_skill("grasp", "Grasp object", my_handler)
    assert entry.skill_type == "programmed"
    assert reg.get("grasp") is not None
    reg.stop()

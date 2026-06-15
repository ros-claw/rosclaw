"""SkillRegistry coverage tests — fills gaps not covered by test_skill_manager.py."""

import pytest

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry


class TestSkillRegistryEventHandlers:
    def test_on_praxis_completed_no_skill_name(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="", skill_type="programmed"))
        bus.publish(Event(topic="praxis.completed", payload={"other": "data"}))
        skill = reg.get("pick")
        assert skill.execution_count == 0
        reg.stop()

    def test_on_praxis_failed_no_skill_name(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="", skill_type="programmed"))
        bus.publish(Event(topic="praxis.failed", payload={"other": "data"}))
        skill = reg.get("pick")
        assert skill.execution_count == 0
        reg.stop()

    def test_on_skill_complete_failure(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="", skill_type="programmed"))
        bus.publish(Event(
            topic="skill.execution.complete",
            payload={"skill_name": "pick", "result": {"status": "failure"}},
        ))
        skill = reg.get("pick")
        assert skill.execution_count == 1
        assert skill.success_rate == 0.0
        reg.stop()

    def test_on_skill_complete_no_skill_name(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="", skill_type="programmed"))
        bus.publish(Event(topic="skill.execution.complete", payload={"result": {"status": "success"}}))
        skill = reg.get("pick")
        assert skill.execution_count == 0
        reg.stop()


class TestSkillRegistryRegister:
    def test_register_overwrite(self, caplog):
        import logging
        reg = SkillRegistry()
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="First", skill_type="programmed"))
        with caplog.at_level(logging.INFO, logger="rosclaw.skill_manager.registry"):
            reg.register(SkillEntry(name="pick", description="Second", skill_type="learned"))
        assert "Overwriting skill: pick" in caplog.text
        skill = reg.get("pick")
        assert skill.skill_type == "learned"
        reg.stop()

    def test_register_publishes_event(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        received = []
        bus.subscribe("skill.registered", lambda e: received.append(e.payload))
        reg.register(SkillEntry(name="place", description="", skill_type="programmed"))
        assert len(received) == 1
        assert received[0]["skill_name"] == "place"
        assert received[0]["skill_type"] == "programmed"
        reg.stop()

    def test_register_type_error(self):
        reg = SkillRegistry()
        reg.initialize()
        with pytest.raises(TypeError, match="Expected SkillEntry"):
            reg.register("not an entry")
        reg.stop()

    def test_register_invalid_name(self):
        reg = SkillRegistry()
        reg.initialize()
        with pytest.raises(ValueError, match="Skill name must be"):
            reg.register(SkillEntry(name="", description="", skill_type="programmed"))
        with pytest.raises(ValueError, match="Skill name must be"):
            reg.register(SkillEntry(name=123, description="", skill_type="programmed"))
        reg.stop()


class TestSkillRegistryListAndFind:
    def test_list_skills_filter_by_type(self):
        reg = SkillRegistry()
        reg.initialize()
        reg.register(SkillEntry(name="a", description="", skill_type="programmed"))
        reg.register(SkillEntry(name="b", description="", skill_type="learned"))
        reg.register(SkillEntry(name="c", description="", skill_type="hybrid"))
        assert reg.list_skills("programmed") == ["a"]
        assert reg.list_skills("learned") == ["b"]
        assert set(reg.list_skills("hybrid")) == {"c"}
        reg.stop()

    def test_list_skills_return_entries(self):
        reg = SkillRegistry()
        reg.initialize()
        reg.register(SkillEntry(name="a", description="", skill_type="programmed"))
        entries = reg.list_skills(return_entries=True)
        assert len(entries) == 1
        assert isinstance(entries[0], SkillEntry)
        assert entries[0].name == "a"
        reg.stop()

    def test_find_by_precondition(self):
        reg = SkillRegistry()
        reg.initialize()
        reg.register(SkillEntry(
            name="pick",
            description="",
            skill_type="programmed",
            preconditions=["object_visible", "gripper_empty"],
        ))
        reg.register(SkillEntry(
            name="place",
            description="",
            skill_type="programmed",
            preconditions=["gripper_full"],
        ))
        found = reg.find_by_precondition("gripper_empty")
        assert len(found) == 1
        assert found[0].name == "pick"
        found2 = reg.find_by_precondition("nonexistent")
        assert found2 == []
        reg.stop()


class TestSkillRegistryStats:
    def test_update_stats_not_found(self):
        reg = SkillRegistry()
        reg.initialize()
        reg.update_stats("missing", success=True)
        reg.stop()

    def test_get_stats_empty(self):
        reg = SkillRegistry()
        reg.initialize()
        stats = reg.get_stats()
        assert stats == {
            "total_skills": 0,
            "total_executions": 0,
            "average_success_rate": 0.0,
        }
        reg.stop()

    def test_get_stats_by_type(self):
        reg = SkillRegistry()
        reg.initialize()
        reg.register(SkillEntry(name="s1", description="", skill_type="programmed"))
        reg.register(SkillEntry(name="s2", description="", skill_type="programmed"))
        reg.register(SkillEntry(name="s3", description="", skill_type="learned"))
        reg.update_stats("s1", success=True)
        reg.update_stats("s1", success=True)
        reg.update_stats("s2", success=False)
        reg.update_stats("s3", success=True)
        stats = reg.get_stats()
        assert stats["by_type"]["programmed"]["count"] == 2
        assert stats["by_type"]["programmed"]["executions"] == 3
        assert stats["by_type"]["learned"]["count"] == 1
        assert stats["by_type"]["learned"]["executions"] == 1
        reg.stop()


class TestSkillEntry:
    def test_entry_to_dict(self):
        entry = SkillEntry(
            name="pick",
            description="Pick object",
            skill_type="programmed",
            parameters={"speed": 0.5},
            preconditions=["visible"],
            success_criteria=["grasped"],
            metadata={"author": "test"},
        )
        d = entry.to_dict()
        assert d["name"] == "pick"
        assert d["description"] == "Pick object"
        assert d["skill_type"] == "programmed"
        assert d["parameters"] == {"speed": 0.5}
        assert d["preconditions"] == ["visible"]
        assert d["success_criteria"] == ["grasped"]
        assert d["metadata"] == {"author": "test"}
        assert "handler" not in d

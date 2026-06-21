"""Tests for Skill Manager."""

from pathlib import Path

import pytest
from pytest import MonkeyPatch

from rosclaw.core.event_bus import Event, EventBus
from rosclaw.sense.interface import SenseInterface
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.loader import SkillLoader
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Isolate tests from the real ~/.rosclaw workspace."""
    monkeypatch.setenv("HOME", str(tmp_path))


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


class TestSkillRegistryEventHandlers:
    def test_on_praxis_completed_no_skill_name(self):
        bus = EventBus()
        reg = SkillRegistry(event_bus=bus)
        reg.initialize()
        reg.register(SkillEntry(name="pick", description="", skill_type="programmed"))
        # No skill_name in payload → stats should not change
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
        # Should not raise
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


class TestSkillRegistrySkillEntry:
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


class TestSkillExecutorBodyCheck:
    def test_body_check_fail_closed_on_resolver_error(self, monkeypatch):
        bus = EventBus()
        reg = SkillRegistry()
        reg.initialize()
        executor = SkillExecutor(bus, reg)
        executor.initialize()

        def handler(params):
            return {"done": True}

        reg.register(SkillEntry(
            name="body_test_skill",
            description="Test",
            skill_type="programmed",
            handler=handler,
        ))

        def _exploding_resolver(*args, **kwargs):
            raise RuntimeError("resolver exploded")

        monkeypatch.setattr(
            "rosclaw.body.resolver.BodyResolver",
            _exploding_resolver,
        )

        result = executor.execute("body_test_skill")
        assert result["status"] == "blocked"
        assert "resolver exploded" in result.get("message", "")
        executor.stop()
        reg.stop()

    def test_body_check_blocks_unknown_compatibility(self, monkeypatch):
        bus = EventBus()
        reg = SkillRegistry()
        reg.initialize()
        executor = SkillExecutor(bus, reg)
        executor.initialize()

        reg.register(SkillEntry(
            name="body_test_skill",
            description="Test",
            skill_type="programmed",
        ))

        class _LinkedResolver:
            def is_linked(self):
                return True
            def check_skill_compatibility(self, name, version=None):
                from rosclaw.body.schema import SkillCompatibilityResult
                return SkillCompatibilityResult(
                    skill_id=name,
                    skill_version=version or "",
                    status="unknown",
                    reason="missing manifest",
                )

        monkeypatch.setattr(
            "rosclaw.body.resolver.BodyResolver",
            _LinkedResolver,
        )

        result = executor.execute("body_test_skill")
        assert result["status"] == "blocked"
        assert "unknown" in result.get("message", "").lower()
        executor.stop()
        reg.stop()


class TestSkillExecutorBodySense:
    @pytest.fixture
    def sense_interface_kick_not_ready(self):
        iface = SenseInterface(
            robot_id="g1_lab_01",
            collector="mock",
            scenario="kick_not_ready",
        )
        iface.initialize()
        yield iface
        iface.stop()

    @pytest.fixture
    def sense_interface_normal(self):
        iface = SenseInterface(
            robot_id="g1_lab_01",
            collector="mock",
            scenario="normal",
        )
        iface.initialize()
        yield iface
        iface.stop()

    def test_skill_blocked_when_body_not_ready(self, sense_interface_kick_not_ready):
        bus = EventBus()
        reg = SkillRegistry()
        reg.initialize()
        executor = SkillExecutor(bus, reg, sense_interface=sense_interface_kick_not_ready)
        executor.initialize()

        blocked_events = []
        bus.subscribe("rosclaw.sense.capability.blocked", lambda e: blocked_events.append(e.payload))

        reg.register(SkillEntry(
            name="kick_ball",
            description="Kick the ball",
            skill_type="programmed",
            metadata={"requires_body_sense": {"battery_percent_min": 40.0}},
        ))

        result = executor.execute("kick_ball")
        assert result["status"] == "blocked"
        assert result["reason"] == "blocked_by_body_sense"
        assert "body_sense_check" in result
        assert result["body_sense_check"]["status"] == "not_ready"
        assert len(blocked_events) == 1
        assert blocked_events[0]["capability"] == "kick_ball"

        executor.stop()
        reg.stop()

    def test_skill_allowed_when_body_ready(self, sense_interface_normal):
        bus = EventBus()
        reg = SkillRegistry()
        reg.initialize()
        executor = SkillExecutor(bus, reg, sense_interface=sense_interface_normal)
        executor.initialize()

        reg.register(SkillEntry(
            name="observe_scene",
            description="Observe the scene",
            skill_type="programmed",
            metadata={"requires_body_sense": {"camera_fps_min": 10.0}},
        ))

        result = executor.execute("observe_scene")
        assert result["status"] in ("success", "dispatched")
        assert "body_sense_check" in result
        assert result["body_sense_check"]["status"] == "ready"

        executor.stop()
        reg.stop()

    def test_skill_requires_body_sense_fails_closed_without_interface(self):
        bus = EventBus()
        reg = SkillRegistry()
        reg.initialize()
        executor = SkillExecutor(bus, reg, sense_interface=None)
        executor.initialize()

        reg.register(SkillEntry(
            name="kick_ball",
            description="Kick the ball",
            skill_type="programmed",
            metadata={"requires_body_sense": {"battery_percent_min": 40.0}},
        ))

        result = executor.execute("kick_ball")
        assert result["status"] == "blocked"
        assert "Sense module is not available" in result["message"]
        assert "body_sense_check" not in result

        executor.stop()
        reg.stop()

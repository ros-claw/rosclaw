"""Tests for the skill runtime plugin dispatch path (Milestone 7)."""

from __future__ import annotations

from typing import Any

from rosclaw.core.event_bus import EventBus
from rosclaw.runtime.handlers import camera  # noqa: F401 - registers handlers
from rosclaw.runtime.plugin import get_runtime_plugin, runtime_handler
from rosclaw.skill_manager.executor import SkillExecutor
from rosclaw.skill_manager.registry import SkillEntry, SkillRegistry


class _NoBodyResolver:
    """Body resolver that reports no linked body for testing."""

    def is_linked(self) -> bool:
        return False


def _make_executor() -> tuple[SkillExecutor, SkillRegistry, list[dict[str, Any]]]:
    bus = EventBus()
    registry = SkillRegistry(event_bus=bus)
    captured: list[dict[str, Any]] = []
    bus.subscribe("skill.execution.start", lambda e: captured.append({"topic": e.topic, "payload": e.payload}))
    bus.subscribe("skill.execution.complete", lambda e: captured.append({"topic": e.topic, "payload": e.payload}))
    executor = SkillExecutor(event_bus=bus, registry=registry, body_resolver=_NoBodyResolver())
    return executor, registry, captured


def test_runtime_handler_takes_priority_over_legacy_handler() -> None:
    executor, registry, captured = _make_executor()

    @runtime_handler("runtime_only_skill")
    def _runtime_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "source": "runtime"}

    entry = SkillEntry(
        name="runtime_only_skill",
        description="A skill with no legacy handler",
        skill_type="programmed",
        handler=None,
    )
    registry.register(entry)

    result = executor.execute("runtime_only_skill")
    assert result["status"] == "success"
    assert result["handler_result"]["source"] == "runtime"
    assert any(e["topic"] == "skill.execution.start" for e in captured)
    assert any(e["topic"] == "skill.execution.complete" for e in captured)


def test_legacy_handler_falls_back_when_no_runtime_handler() -> None:
    executor, registry, _ = _make_executor()

    def legacy_handler(params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "success", "source": "legacy"}

    entry = SkillEntry(
        name="legacy_only_skill",
        description="A skill with a legacy handler",
        skill_type="programmed",
        handler=legacy_handler,
    )
    registry.register(entry)

    result = executor.execute("legacy_only_skill")
    assert result["status"] == "success"
    assert result["handler_result"]["source"] == "legacy"


def test_builtin_camera_handlers_are_registered() -> None:
    plugin = get_runtime_plugin()
    assert "realsense_capture_rgbd" in plugin.list_handlers()
    assert "scene_risk_scan" in plugin.list_handlers()
    handler = plugin.get_handler("realsense_capture_rgbd")
    assert handler is not None
    result = handler({})
    assert result["status"] == "success"
    assert "frames" in result


def test_runtime_handler_failure_is_recorded() -> None:
    executor, registry, _ = _make_executor()

    @runtime_handler("failing_runtime_skill")
    def _failing_handler(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry.register(SkillEntry(
        name="failing_runtime_skill",
        description="Fails at runtime",
        skill_type="programmed",
        handler=None,
    ))

    result = executor.execute("failing_runtime_skill")
    assert result["status"] == "error"
    assert "boom" in result["error"]
    entry = registry.get("failing_runtime_skill")
    assert entry is not None
    assert entry.execution_count == 1
    assert entry.success_rate == 0.0

"""Tests for the skill runtime plugin dispatch path (Milestone 7)."""

from __future__ import annotations

import json
from typing import Any

from rosclaw.core.event_bus import EventBus
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
    bus.subscribe(
        "skill.execution.start", lambda e: captured.append({"topic": e.topic, "payload": e.payload})
    )
    bus.subscribe(
        "skill.execution.complete",
        lambda e: captured.append({"topic": e.topic, "payload": e.payload}),
    )
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


def test_builtin_camera_handlers_are_registered(tmp_path) -> None:
    # The autouse fixture clears the global plugin before each test, so
    # re-import the camera handler module to re-register its decorators.
    import importlib

    from rosclaw.runtime.handlers import camera

    importlib.reload(camera)

    plugin = get_runtime_plugin()
    assert "realsense_capture_rgbd" in plugin.list_handlers()
    assert "scene_risk_scan" in plugin.list_handlers()
    handler = plugin.get_handler("realsense_capture_rgbd")
    assert handler is not None

    # The handler talks to a real ROS2/MCP server; for a unit test we mock the
    # MCP capture result so the test is hermetic and does not require live
    # RealSense hardware.
    color_path = tmp_path / "color.png"
    depth_path = tmp_path / "depth.png"
    color_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    depth_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    import rosclaw.mcp.onboarding.stdio_client as stdio_client

    original_call = stdio_client.call_server_tool
    original_discover = camera._discover_realsense_ros_mcp

    def _fake_call(server_name, tool_name, arguments, home=None, timeout=None):
        return {
            "structuredContent": {
                "result": json.dumps(
                    {
                        "success": True,
                        "color": {
                            "path": str(color_path),
                            "topic": "/camera/d435i/color/image_raw",
                            "width": 424,
                            "height": 240,
                            "encoding": "rgb8",
                        },
                        "depth": {
                            "path": str(depth_path),
                            "topic": "/camera/d435i/depth/image_rect_raw",
                            "width": 424,
                            "height": 240,
                            "encoding": "16UC1",
                        },
                    }
                )
            }
        }

    stdio_client.call_server_tool = _fake_call
    camera._discover_realsense_ros_mcp = lambda home: "realsense-ros-mcp"
    try:
        result = handler({"output_dir": str(tmp_path / "out")})
        assert result["status"] == "success"
        assert "frames" in result
        assert "artifacts" in result
    finally:
        stdio_client.call_server_tool = original_call
        camera._discover_realsense_ros_mcp = original_discover


def test_runtime_handler_failure_is_recorded() -> None:
    executor, registry, _ = _make_executor()

    @runtime_handler("failing_runtime_skill")
    def _failing_handler(params: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("boom")

    registry.register(
        SkillEntry(
            name="failing_runtime_skill",
            description="Fails at runtime",
            skill_type="programmed",
            handler=None,
        )
    )

    result = executor.execute("failing_runtime_skill")
    assert result["status"] == "error"
    assert "boom" in result["error"]
    entry = registry.get("failing_runtime_skill")
    assert entry is not None
    assert entry.execution_count == 1
    assert entry.success_rate == 0.0


def test_navigation_stub_direct_reports_missing_backend_and_preserves_target() -> None:
    plugin = get_runtime_plugin()
    saved = {name: plugin.get_handler(name) for name in plugin.list_handlers()}
    try:
        from rosclaw.runtime.handlers.navigation import _handle_navigate_to

        target = {"frame": "map", "position": [1.0, 2.0]}
        for params, expected in (
            ({}, "unknown"),
            ({"target": None}, None),
            ({"target": target}, target),
            ({"target": "dock"}, "dock"),
        ):
            original = dict(params)
            result = _handle_navigate_to(params)
            assert result["status"] == "blocked"
            assert result["reason"] == "NAVIGATION_BACKEND_UNAVAILABLE"
            assert "No navigation backend" in result["message"]
            assert "no command was dispatched" in result["message"]
            assert result["target"] == expected
            if expected is target:
                assert result["target"] is target
            assert result["skill"] == "navigate_to"
            assert result["source"] == "runtime_handler"
            assert params == original
    finally:
        plugin.clear()
        for name, handler in saved.items():
            plugin.register(name, handler)


def _navigation_truth_sum_handler(params: dict[str, Any]) -> dict[str, Any]:
    """Unrelated, genuinely completed in-memory work for the success control."""
    total = sum(params["values"])
    return {"status": "success", "total": total}


def test_navigation_blocked_propagates_through_executor_without_success_credit() -> None:
    from unittest.mock import patch

    from rosclaw.observability.tracer import Tracer

    plugin = get_runtime_plugin()
    saved = {name: plugin.get_handler(name) for name in plugin.list_handlers()}
    bus = EventBus()
    tracer = Tracer(event_bus=bus, exporters=[])
    completed = []

    def capture(event):
        completed.append(event.payload)

    bus.subscribe("skill.execution.complete", capture)
    try:
        from rosclaw.runtime.handlers.navigation import _handle_navigate_to

        # Use the real registration surface, even if the module was cached after
        # the suite's autouse fixture cleared its import-time registration.
        plugin.register("navigate_to", _handle_navigate_to)
        plugin.register("navigation_truth_sum_control", _navigation_truth_sum_handler)
        registry = SkillRegistry(event_bus=bus)
        registry.register(
            SkillEntry(
                name="navigate_to",
                description="Navigation stub regression",
                skill_type="programmed",
                parameters={"target": "default-dock"},
            )
        )
        registry.register(
            SkillEntry(
                name="navigation_truth_sum_control",
                description="In-memory sum control",
                skill_type="programmed",
            )
        )
        # Only unrelated entry-point discovery is isolated. Dispatch, status
        # mapping, handler execution and registry statistics remain production.
        with patch.object(plugin, "discover_handlers", return_value=None):
            executor = SkillExecutor(
                event_bus=bus,
                registry=registry,
                body_resolver=_NoBodyResolver(),
                tracer=tracer,
            )
        assert plugin.get_handler("navigate_to") is _handle_navigate_to
        entry = registry.get("navigate_to")
        assert entry is not None
        assert entry.execution_count == 0
        assert entry.success_rate == 0.0
        targets = ("default-dock", {"frame": "map", "xy": [0.2, 0.3]}, None)
        for count, target in enumerate(targets, 1):
            params = None if count == 1 else {"target": target}
            result = executor.execute("navigate_to", params)
            assert result["status"] == "blocked"
            nested = result["handler_result"]
            assert nested["status"] == "blocked"
            assert nested["reason"] == "NAVIGATION_BACKEND_UNAVAILABLE"
            assert "No navigation backend" in nested["message"]
            assert nested["target"] == target
            assert nested["skill"] == "navigate_to"
            assert nested["source"] == "runtime_handler"
            assert entry.execution_count == count
            assert entry.success_rate == 0.0
            assert entry.execution_count * entry.success_rate == 0.0
            assert completed[-1]["skill_name"] == "navigate_to"
            assert completed[-1]["result"]["status"] == "blocked"
            assert completed[-1]["result"]["handler_result"]["reason"] == nested["reason"]
            assert completed[-1]["result"]["handler_result"]["target"] == target
        assert registry.get_stats()["average_success_rate"] == 0.0
        control = executor.execute("navigation_truth_sum_control", {"values": [2, 3, 5]})
        assert control["status"] == "success"
        assert control["handler_result"]["total"] == 10
        control_entry = registry.get("navigation_truth_sum_control")
        assert control_entry is not None
        assert control_entry.execution_count == 1
        assert control_entry.success_rate == 1.0
        assert entry.execution_count == 3
        assert entry.success_rate == 0.0
        assert len(completed) == 4
    finally:
        try:
            tracer.close()
        finally:
            bus.unsubscribe("skill.execution.complete", capture)
            bus.clear_history()
            plugin.clear()
            for name, handler in saved.items():
                plugin.register(name, handler)

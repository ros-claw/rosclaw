"""Integration tests for Heuristic Recovery (rosclaw.how) module."""

import pytest

from rosclaw.how.engine import HeuristicEngine
from rosclaw.how.rules import RuleManager
from rosclaw.how.recovery import RecoveryEngine, RecoveryFormatter, format_recovery_suggestion
from rosclaw.memory.seekdb_client import SeekDBMemoryClient


class TestHeuristicEngine:
    """Tests for HeuristicEngine core functionality."""

    @pytest.fixture
    def engine(self):
        client = SeekDBMemoryClient()
        client.connect()
        return HeuristicEngine(seekdb_client=client)

    @pytest.mark.asyncio
    async def test_seed_defaults(self, engine):
        count = await engine.seed_defaults()
        assert count > 0
        assert engine._cache_valid
        assert len(engine._rule_cache) > 0

    @pytest.mark.asyncio
    async def test_suggest_recovery_exact_match(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("joint limit exceeded")
        assert recovery is not None
        assert recovery["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_suggest_recovery_substring_match(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("ERROR: joint limit exceeded on axis 3")
        assert recovery is not None
        assert recovery["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_suggest_recovery_no_match(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("completely unknown magical unicorn failure")
        assert recovery is None

    @pytest.mark.asyncio
    async def test_record_outcome(self, engine):
        await engine.seed_defaults()
        rule_id = list(engine._rule_cache.keys())[0]
        ok = await engine.record_outcome(rule_id, success=True)
        assert ok is True
        rule = engine._rule_cache[rule_id]
        assert rule["success_count"] >= 1

    @pytest.mark.asyncio
    async def test_get_stats(self, engine):
        await engine.seed_defaults()
        stats = engine.get_stats()
        assert stats["rule_count"] > 0
        assert stats["cache_valid"] is True

    @pytest.mark.asyncio
    async def test_suggest_recovery_joint_overload(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("joint overload detected on axis 2")
        assert recovery is not None
        assert "payload" in recovery["action"].lower()
        assert recovery["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_suggest_recovery_collision_avoidance(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("collision avoidance triggered near workspace boundary")
        assert recovery is not None
        assert "compliant" in recovery["action"].lower()
        assert recovery["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_suggest_recovery_communication_timeout(self, engine):
        await engine.seed_defaults()
        recovery = await engine.suggest_recovery("communication timeout to ROS master")
        assert recovery is not None
        assert "backoff" in recovery["action"].lower()
        assert recovery["source"] == "heuristic"

    @pytest.mark.asyncio
    async def test_get_retry_plan_grasp_slippage(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("grasp slippage")
        assert plan is not None
        assert plan["action"] == "retry_with_adjustments"
        patch = plan["parameter_patch"]
        assert "gripper_force_offset" in patch
        assert "approach_offset_z" in patch

    @pytest.mark.asyncio
    async def test_get_retry_plan_collision_predicted(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("collision predicted")
        assert plan is not None
        patch = plan["parameter_patch"]
        assert "safety_clearance" in patch
        assert "velocity_factor" in patch

    @pytest.mark.asyncio
    async def test_get_retry_plan_object_not_found(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("object not found")
        assert plan is not None
        patch = plan["parameter_patch"]
        assert "camera_angle_offset" in patch

    @pytest.mark.asyncio
    async def test_get_retry_plan_force_exceeded(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("force exceeded")
        assert plan is not None
        patch = plan["parameter_patch"]
        assert patch.get("control_mode") == "compliant"

    @pytest.mark.asyncio
    async def test_get_retry_plan_sensor_failure(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("sensor failure")
        assert plan is not None
        patch = plan["parameter_patch"]
        assert patch.get("sensor_fusion") is True

    @pytest.mark.asyncio
    async def test_get_retry_plan_communication_lost(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("communication lost")
        assert plan is not None
        patch = plan["parameter_patch"]
        assert "timeout_multiplier" in patch

    @pytest.mark.asyncio
    async def test_get_retry_plan_no_match(self, engine):
        await engine.seed_defaults()
        plan = await engine.get_retry_plan("totally unknown failure xyz")
        assert plan is None


class TestRuleManager:
    """Tests for RuleManager CRUD operations."""

    @pytest.fixture
    def manager(self):
        client = SeekDBMemoryClient()
        client.connect()
        return RuleManager(client)

    def test_add_and_get_rule(self, manager):
        rid = manager.add_rule("test_001", "test condition", "test action", priority=5)
        rule = manager.get_rule(rid)
        assert rule is not None
        assert rule["condition"] == "test condition"
        assert rule["priority"] == 5

    def test_update_rule(self, manager):
        rid = manager.add_rule("test_002", "old condition", "old action")
        ok = manager.update_rule(rid, action="new action", priority=3)
        assert ok is True
        rule = manager.get_rule(rid)
        assert rule["action"] == "new action"

    def test_list_rules_filters_negative(self, manager):
        rid = manager.add_rule("test_del", "to delete", "action")
        manager.delete_rule(rid)
        rules = manager.list_rules()
        ids = [r["id"] for r in rules]
        assert rid not in ids


class TestRecoveryFormatter:
    """Tests for RecoveryFormatter utilities."""

    def test_to_event_payload(self):
        rule = {
            "rule_id": "r1", "condition": "collision", "action": "replan",
            "priority": 2, "source": "heuristic", "success_count": 3, "failure_count": 1,
        }
        payload = RecoveryFormatter.to_event_payload(rule, request_id="req42")
        assert payload["request_id"] == "req42"
        assert payload["suggestion"] == "replan"

    def test_apply_trajectory_reduce_velocity(self):
        traj = [[1.0, 2.0], [3.0, 4.0]]
        result = RecoveryFormatter.apply_trajectory_adjustment(traj, "Reduce velocity by 50%")
        assert result == [[0.5, 1.0], [1.5, 2.0]]

    def test_format_recovery_suggestion(self):
        recovery = {"action": "clamp torque", "source": "heuristic"}
        text = format_recovery_suggestion(recovery)
        assert "clamp torque" in text

    def test_format_recovery_suggestion_none(self):
        text = format_recovery_suggestion(None)
        assert "No heuristic" in text


class TestRecoveryEngine:
    """Tests for RecoveryEngine RecoveryHint generation."""

    @pytest.fixture
    def recovery_engine(self):
        client = SeekDBMemoryClient()
        client.connect()
        how = HeuristicEngine(seekdb_client=client)
        return RecoveryEngine(how)

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_grasp_slippage(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("grasp slippage")
        assert hint is not None
        assert hint["failure_type"] == "grasp slippage"
        assert "gripper" in hint["hint"].lower()
        assert 0.0 <= hint["confidence"] <= 1.0
        assert "retry_plan" in hint
        assert hint["retry_plan"]["parameter_patch"]["gripper_force_offset"] == 0.15

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_collision_predicted(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("collision predicted")
        assert hint is not None
        assert "trajectory" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_unstable_grasp(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("unstable grasp")
        assert hint is not None
        assert "support" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_path_blocked(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("path blocked")
        assert hint is not None
        assert "obstacle" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_no_match(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("magical unicorn failure")
        assert hint is None

    @pytest.mark.asyncio
    async def test_format_for_eventbus(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = recovery_engine.generate_recovery_hint("grasp slippage")
        payload = recovery_engine.format_for_eventbus(hint, request_id="req99")
        assert payload["request_id"] == "req99"
        assert payload["failure_type"] == "grasp slippage"
        assert "confidence" in payload
        assert "retry_plan" in payload

    @pytest.mark.asyncio
    async def test_build_retry_plan_with_context(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        rule = await recovery_engine._how.suggest_recovery("grasp slippage")
        plan = recovery_engine.build_retry_plan("grasp slippage", rule, {"max_retries": 5})
        assert plan["max_retries"] == 5


class TestMCPHeuristicTool:
    """Tests for MCP get_recovery_strategy tool."""

    @pytest.mark.asyncio
    async def test_mcp_tool_registered(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._do_initialize()
        hub._register_get_recovery_strategy_tool()

        tool_names = [t["name"] for t in hub.tools]
        assert "get_recovery_strategy" in tool_names
        hub._do_stop()

    @pytest.mark.asyncio
    async def test_mcp_handle_get_recovery_strategy(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._register_get_recovery_strategy_tool()

        class MockRuntime:
            def __init__(self):
                client = SeekDBMemoryClient()
                client.connect()
                self.how = HeuristicEngine(seekdb_client=client)

        hub.runtime = MockRuntime()
        await hub.runtime.how.seed_defaults()

        result = await hub.handle_tool_call(
            "get_recovery_strategy", {"error_log": "joint limit exceeded"}
        )

        assert result["status"] == "ok"
        assert result["matched"] is True
        assert "action" in result

    @pytest.mark.asyncio
    async def test_mcp_handle_get_recovery_strategy_no_match(self):
        from rosclaw.agent_runtime.mcp_hub import MCPHub
        from rosclaw.core.event_bus import EventBus

        event_bus = EventBus()
        hub = MCPHub(event_bus=event_bus, robot_id="ur5e")
        hub._register_get_recovery_strategy_tool()

        class MockRuntime:
            def __init__(self):
                client = SeekDBMemoryClient()
                client.connect()
                self.how = HeuristicEngine(seekdb_client=client)

        hub.runtime = MockRuntime()
        await hub.runtime.how.seed_defaults()

        result = await hub.handle_tool_call(
            "get_recovery_strategy", {"error_log": "unicorn magic failure unknown"}
        )

        assert result["status"] == "ok"
        assert result["matched"] is False


class TestRuntimeRecoveryHandlers:
    """Tests for Runtime EventBus recovery handlers."""

    def test_sandbox_episode_failed_handler(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig
        from rosclaw.core.event_bus import Event, EventPriority

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        # Publish a sandbox episode failed event
        event = Event(
            topic="rosclaw.sandbox.episode.failed",
            payload={"failure_type": "grasp slippage", "request_id": "ep001"},
            source="sandbox",
            priority=EventPriority.HIGH,
        )
        runtime.event_bus.publish(event)

        # Check that recovery_hint.generated event was published
        history = runtime.event_bus.get_history("rosclaw.how.recovery_hint.generated")
        assert len(history) >= 1
        last = history[-1]
        assert last.payload["failure_type"] == "grasp slippage"
        assert "retry_plan" in last.payload

    def test_sandbox_action_blocked_handler(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig
        from rosclaw.core.event_bus import Event, EventPriority

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        event = Event(
            topic="rosclaw.sandbox.action.blocked",
            payload={"reason": "collision predicted", "request_id": "act042"},
            source="sandbox",
            priority=EventPriority.HIGH,
        )
        runtime.event_bus.publish(event)

        history = runtime.event_bus.get_history("rosclaw.how.recovery_hint.generated")
        assert len(history) >= 1
        last = history[-1]
        assert last.payload["failure_type"] == "collision predicted"

    def test_runtime_execution_failed_handler(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig
        from rosclaw.core.event_bus import Event, EventPriority

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        event = Event(
            topic="rosclaw.runtime.execution.failed",
            payload={"error_type": "sensor failure", "request_id": "exec123"},
            source="runtime",
            priority=EventPriority.HIGH,
        )
        runtime.event_bus.publish(event)

        history = runtime.event_bus.get_history("rosclaw.how.recovery_hint.generated")
        assert len(history) >= 1
        last = history[-1]
        assert last.payload["failure_type"] == "sensor failure"

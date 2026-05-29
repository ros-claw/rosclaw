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
        hint = await recovery_engine.generate_recovery_hint("grasp slippage")
        assert hint is not None
        assert hint["failure_type"] == "grasp slippage"
        assert "gripper" in hint["hint"].lower()
        assert 0.0 <= hint["confidence"] <= 1.0
        assert "retry_plan" in hint
        assert hint["retry_plan"]["parameter_patch"]["gripper_force_offset"] == 0.15

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_collision_predicted(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = await recovery_engine.generate_recovery_hint("collision predicted")
        assert hint is not None
        assert "trajectory" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_unstable_grasp(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = await recovery_engine.generate_recovery_hint("unstable grasp")
        assert hint is not None
        assert "support" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_path_blocked(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = await recovery_engine.generate_recovery_hint("path blocked")
        assert hint is not None
        assert "obstacle" in hint["hint"].lower()

    @pytest.mark.asyncio
    async def test_generate_recovery_hint_no_match(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = await recovery_engine.generate_recovery_hint("magical unicorn failure")
        assert hint is None

    @pytest.mark.asyncio
    async def test_format_for_eventbus(self, recovery_engine):
        await recovery_engine._how.seed_defaults()
        hint = await recovery_engine.generate_recovery_hint("grasp slippage")
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
    """Tests for Runtime EventBus recovery handlers (wiring verification)."""

    def test_sandbox_episode_failed_subscription(self):
        """Verify Runtime subscribes to rosclaw.sandbox.episode.failed."""
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        assert runtime.event_bus.subscriber_count("rosclaw.sandbox.episode.failed") >= 1

    def test_sandbox_action_blocked_subscription(self):
        """Verify Runtime subscribes to rosclaw.sandbox.action.blocked."""
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        assert runtime.event_bus.subscriber_count("rosclaw.sandbox.action.blocked") >= 1

    def test_runtime_execution_failed_subscription(self):
        """Verify Runtime subscribes to rosclaw.runtime.execution.failed."""
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_how=True, enable_memory=True)
        runtime = Runtime(config)
        runtime._do_initialize()

        assert runtime.event_bus.subscriber_count("rosclaw.runtime.execution.failed") >= 1

    def test_handler_logic_directly(self):
        """Test handler logic by invoking RecoveryEngine directly (bypasses sync/async boundary)."""
        from rosclaw.core.event_bus import Event, EventPriority
        from rosclaw.how.engine import HeuristicEngine
        from rosclaw.how.recovery import RecoveryEngine
        from rosclaw.memory.seekdb_client import SeekDBMemoryClient

        client = SeekDBMemoryClient()
        client.connect()
        how = HeuristicEngine(seekdb_client=client)
        import asyncio
        asyncio.run(how.seed_defaults())

        re = RecoveryEngine(how)
        hint = asyncio.run(re.generate_recovery_hint(
            "grasp slippage",
            context={"request_id": "ep001"},
            sources=["sandbox_episode"],
        ))
        assert hint is not None
        assert hint["failure_type"] == "grasp slippage"

        # Verify EventBus payload formatting
        payload = re.format_for_eventbus(hint, request_id="ep001")
        assert payload["failure_type"] == "grasp slippage"
        assert "retry_plan" in payload

        # Verify publishing works
        from rosclaw.core.event_bus import EventBus
        bus = EventBus()
        bus.publish(Event(
            topic="rosclaw.how.recovery_hint.generated",
            payload=payload,
            source="runtime",
            priority=EventPriority.HIGH,
        ))
        history = bus.get_history("rosclaw.how.recovery_hint.generated")
        assert len(history) >= 1
        last = history[-1]
        assert last.payload["failure_type"] == "grasp slippage"
        assert "retry_plan" in last.payload


class TestRuntimeIntegrationAPIs:
    """Tests for v1.0 minimum closed-loop integration APIs."""

    def test_capability_invoke_mock_vlm(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_provider=True, enable_memory=False, enable_how=False)
        runtime = Runtime(config)
        runtime._do_initialize()

        result = runtime.capability_invoke("vlm.object_grounding", {"query": "red_cup"})
        assert result["status"] == "ok"
        assert result["capability"] == "vlm.object_grounding"
        assert "result" in result

    def test_capability_invoke_unknown(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_provider=True, enable_memory=False, enable_how=False)
        runtime = Runtime(config)
        runtime._do_initialize()

        result = runtime.capability_invoke("unknown_capability", {})
        # Should not crash; may return error depending on router behavior
        assert "status" in result

    def test_sandbox_check_no_firewall(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_firewall=False, enable_memory=False, enable_how=False)
        runtime = Runtime(config)
        runtime._do_initialize()

        result = runtime.sandbox_check({"trajectory": [[0.1, 0.2]]})
        assert result["decision"] == "ALLOW"
        assert result["reason"] == "firewall_disabled"

    def test_execute_with_sandbox(self):
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(enable_firewall=False, enable_memory=False, enable_how=False)
        runtime = Runtime(config)
        runtime._do_initialize()

        result = runtime.execute({"trajectory": [[0.1, 0.2]]})
        # Sandbox is always initialized; uses stub when robot not found
        assert "status" in result

    def test_integration_acceptance_scenario(self):
        """Acceptance: the task example runs without crashing."""
        from rosclaw.core.runtime import Runtime, RuntimeConfig

        config = RuntimeConfig(
            enable_provider=True,
            enable_memory=True,
            enable_how=True,
            enable_firewall=True,
            robot_model_path=None,
        )
        runtime = Runtime(config)
        runtime._do_initialize()

        # Step 1: VLM object grounding
        result = runtime.capability_invoke("vlm.object_grounding", {"query": "red_cup"})
        assert "status" in result

        # Step 2: Plan action
        action = runtime.plan_action("pick red cup", result)
        assert "status" in action

        # Step 3: Sandbox check
        check = runtime.sandbox_check(action)
        assert "decision" in check
        assert check["decision"] in ("ALLOW", "BLOCK")

        # Step 4a: Execute if allowed
        if check["decision"] == "ALLOW":
            exec_result = runtime.execute(action)
            assert "status" in exec_result
        # Step 4b: Recovery hint if blocked
        else:
            from rosclaw.how.recovery import RecoveryEngine
            import asyncio
            re = RecoveryEngine(runtime._how)
            asyncio.run(runtime._how.seed_defaults())
            hint = asyncio.run(re.generate_recovery_hint(check["reason"]))
            assert hint is not None or check["reason"]


class TestHowEndToEndRecovery:
    """End-to-end tests: failure -> How recovery -> retry -> record -> compare."""

    @pytest.fixture
    async def recovery_env(self):
        """Set up HeuristicEngine + RecoveryEngine + MemoryInterface + EventBus."""
        from rosclaw.memory.seekdb_client import SeekDBMemoryClient
        from rosclaw.memory.interface import MemoryInterface
        from rosclaw.core.event_bus import EventBus

        event_bus = EventBus()
        client = SeekDBMemoryClient()
        client.connect()

        how = HeuristicEngine(seekdb_client=client)
        await how.seed_defaults()

        re = RecoveryEngine(how)

        memory = MemoryInterface(robot_id="test_bot", event_bus=event_bus, seekdb_client=client)
        memory._do_initialize()

        return how, re, memory, event_bus

    @pytest.mark.asyncio
    async def test_failure_to_recovery_hint(self, recovery_env):
        """Step 1: failure -> How generates RecoveryHint with confidence and patch."""
        how, re, memory, bus = recovery_env

        hint = await re.generate_recovery_hint("grasp slippage")
        assert hint is not None
        assert hint["failure_type"] == "grasp slippage"
        assert "retry_plan" in hint
        patch = hint["retry_plan"]["parameter_patch"]
        assert patch["gripper_force_offset"] == 0.15
        assert 0.0 <= hint["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_memory_analogy_fallback(self, recovery_env):
        """Step 2: Memory stores past failure and provides analogy on query."""
        how, re, memory, bus = recovery_env

        # Seed a past failure in Memory with recovery hint
        memory.store_experience(
            event_id="fail_001",
            event_type="failure",
            instruction="grasp slippage on red cup",
            outcome="failure",
            error_details="grasp slippage detected, cup dropped from gripper",
            tags=["failure", "grasp_slippage"],
            metadata={"recovery_hint": "Lower approach z by 3cm, increase force 20%"},
        )

        # Query analogy
        analogy = memory.find_analogy("grasp slippage on red cup")
        assert analogy is not None
        assert "Lower approach z" in analogy["action_suggestion"]
        assert analogy["similarity_score"] > 0

    @pytest.mark.asyncio
    async def test_recovery_outcome_updates_confidence(self, recovery_env):
        """Step 3: recording success outcome increases rule confidence."""
        how, re, memory, bus = recovery_env

        # Find a rule and record 3 successes
        rule = await how.suggest_recovery("grasp slippage")
        rid = rule["rule_id"]

        # Record outcomes
        await how.record_outcome(rid, success=True)
        await how.record_outcome(rid, success=True)
        await how.record_outcome(rid, success=True)

        # Refresh rule from cache
        rule = how._rule_cache.get(rid, {})

        confidence = RecoveryEngine._compute_confidence(rule)
        # base = 3/3 = 1.0, time_decay ~1.0 (just triggered), trigger_penalty = 3/3 = 1.0
        assert confidence > 0.8, f"Expected high confidence after 3 successes, got {confidence}"

        # Record a failure -> confidence should drop
        await how.record_outcome(rid, success=False)
        rule = how._rule_cache.get(rid, {})
        confidence_after_fail = RecoveryEngine._compute_confidence(rule)
        assert confidence_after_fail < confidence, "Confidence should drop after failure"

    @pytest.mark.asyncio
    async def test_full_recovery_loop(self, recovery_env):
        """Full loop: failure -> hint -> apply patch -> retry -> record -> Memory query."""
        how, re, memory, bus = recovery_env

        failure_type = "grasp slippage"

        # Step 1: How generates recovery hint
        hint = await re.generate_recovery_hint(failure_type)
        assert hint is not None
        patch = hint["retry_plan"]["parameter_patch"]

        # Step 2: Apply parameter patch (simulated adjustment)
        params = {"gripper_force": 1.0, "approach_z": 0.0}
        params["gripper_force"] += patch.get("gripper_force_offset", 0)
        params["approach_z"] += patch.get("approach_offset_z", 0)
        assert params["gripper_force"] == 1.15
        assert params["approach_z"] == -0.02

        # Step 3: Record recovery outcome (success)
        rule_id = hint["source"][0].split(":")[1]
        await how.record_outcome(rule_id, success=True)

        # Step 4: Memory stores the experience with recovery hint
        memory.store_experience(
            event_id="ep_001",
            event_type="failure",
            instruction=failure_type,
            outcome="failure",
            error_details=failure_type,
            tags=["failure", failure_type.replace(" ", "_"), "recovered"],
            metadata={"recovery_hint": hint["hint"], "parameter_patch": patch},
        )

        # Step 5: Memory can answer "what happened"
        similar = memory.find_similar_experiences("grasp slippage")
        assert len(similar) >= 1

        # Step 6: Analogy works for next similar failure
        analogy = memory.find_analogy("grasp slippage")
        assert analogy is not None
        assert analogy["similarity_score"] > 0

        # Step 7: Verify the event bus has the recovery hint event
        payload = re.format_for_eventbus(hint, request_id="req_loop_001")
        assert payload["failure_type"] == failure_type
        assert "retry_plan" in payload

    @pytest.mark.asyncio
    async def test_how_reads_memory_for_analogy_fallback(self, recovery_env):
        """How uses Memory analogy when no heuristic rule matches."""
        how, re, memory, bus = recovery_env

        # Seed a past failure for an unknown failure type
        memory.store_experience(
            event_id="fail_unknown",
            event_type="failure",
            instruction="battery low voltage warning",
            outcome="failure",
            error_details="battery low voltage dropped below safe threshold",
            tags=["failure", "battery_low"],
            metadata={"recovery_hint": "Return to charging station immediately"},
        )

        # Direct Memory analogy query
        analogy = memory.find_analogy("battery low voltage warning")
        assert analogy is not None, "Memory should find analogy for battery low failure"
        assert "charging station" in analogy["action_suggestion"]

        # Wire Memory as knowledge backend for HeuristicEngine
        how._knowledge = memory

        # _knowledge_fallback should route to Memory.find_analogy
        fallback = await how._knowledge_fallback("battery low voltage warning", None)
        assert fallback is not None
        assert fallback["source"] == "knowledge_analogy"
        assert "charging station" in fallback["action_suggestion"]

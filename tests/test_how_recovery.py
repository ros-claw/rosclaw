"""Tests for how.recovery — RecoveryEngine."""


import pytest
import time

from rosclaw.how.recovery import (
    RecoveryEngine,
    RecoveryFormatter,
    format_recovery_suggestion,
)


class FakeHeuristic:
    """Fake heuristic engine for testing."""

    def __init__(self, rules=None):
        self._rules = rules or []

    async def suggest_recovery(self, failure_type, context=None):
        for r in self._rules:
            if r.get("condition", "") in failure_type:
                return r
        return None


class FakeEventBus:
    """Fake EventBus that captures published events."""

    def __init__(self):
        self.events = []

    def subscribe(self, topic, handler):
        pass

    def unsubscribe(self, topic, handler):
        pass

    def publish(self, event):
        self.events.append(event)


@pytest.fixture
def engine():
    return RecoveryEngine(FakeHeuristic())


@pytest.fixture
def engine_with_rules():
    rules = [
        {
            "rule_id": "r1",
            "condition": "joint limit",
            "action": "Reduce velocity and retract 5cm",
            "success_count": 10,
            "failure_count": 2,
            "last_triggered": time.time(),
        },
        {
            "rule_id": "r2",
            "condition": "collision",
            "action": "Increase safety clearance by 5cm",
            "success_count": 5,
            "failure_count": 0,
        },
    ]
    return RecoveryEngine(FakeHeuristic(rules))


class TestRecoveryEngineInit:
    def test_init_no_event_bus(self):
        eng = RecoveryEngine(FakeHeuristic())
        assert eng._event_bus is None
        assert eng._subscribed is False

    def test_init_with_event_bus(self):
        bus = FakeEventBus()
        eng = RecoveryEngine(FakeHeuristic(), event_bus=bus)
        assert eng._event_bus is bus

    def test_initialize_passive_mode(self, engine):
        engine.initialize()
        assert engine._subscribed is False

    def test_initialize_active_mode(self, engine_with_rules):
        bus = FakeEventBus()
        eng = RecoveryEngine(engine_with_rules._how, event_bus=bus)
        eng.initialize()
        assert eng._subscribed is True

    def test_shutdown_no_subscription(self, engine):
        engine.shutdown()
        assert engine._subscribed is False

    def test_initialize_then_shutdown(self, engine_with_rules):
        bus = FakeEventBus()
        eng = RecoveryEngine(engine_with_rules._how, event_bus=bus)
        eng.initialize()
        assert eng._subscribed is True
        eng.shutdown()
        assert eng._subscribed is False


class TestGenerateRecoveryHint:
    @pytest.mark.asyncio
    async def test_empty_failure_type(self, engine):
        result = await engine.generate_recovery_hint("")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_matching_rule(self, engine):
        result = await engine.generate_recovery_hint("unknown_failure_xyz")
        assert result is None

    @pytest.mark.asyncio
    async def test_matching_rule(self, engine_with_rules):
        result = await engine_with_rules.generate_recovery_hint("joint limit exceeded")
        assert result is not None
        assert result["failure_type"] == "joint limit exceeded"
        assert "hint" in result
        assert "confidence" in result
        assert "retry_plan" in result
        assert "all_candidates" in result

    @pytest.mark.asyncio
    async def test_sources_included(self, engine_with_rules):
        result = await engine_with_rules.generate_recovery_hint("collision detected", sources=["sandbox"])
        assert "sandbox" in result["source"]

    @pytest.mark.asyncio
    async def test_publish_to_event_bus(self, engine_with_rules):
        bus = FakeEventBus()
        await engine_with_rules.generate_recovery_hint(
            "joint limit exceeded", event_bus=bus, request_id="req1"
        )
        assert len(bus.events) == 1
        assert bus.events[0]["topic"] == "rosclaw.how.recovery_hint.generated"


class TestComputeConfidence:
    def test_no_history(self):
        rule = {"success_count": 0, "failure_count": 0}
        conf = RecoveryEngine._compute_confidence(rule)
        assert conf == 0.5

    def test_perfect_record(self):
        rule = {"success_count": 10, "failure_count": 0, "last_triggered": time.time()}
        conf = RecoveryEngine._compute_confidence(rule)
        assert conf > 0.5
        assert conf <= 1.0

    def test_all_failures(self):
        rule = {"success_count": 0, "failure_count": 10, "last_triggered": time.time()}
        conf = RecoveryEngine._compute_confidence(rule)
        assert conf == 0.0

    def test_time_decay(self):
        old = time.time() - 86400 * 30  # 30 days ago
        rule = {"success_count": 10, "failure_count": 0, "last_triggered": old}
        conf_old = RecoveryEngine._compute_confidence(rule)

        rule["last_triggered"] = time.time()
        conf_new = RecoveryEngine._compute_confidence(rule)

        assert conf_new > conf_old

    def test_trigger_penalty(self):
        rule = {"success_count": 1, "failure_count": 0, "last_triggered": time.time()}
        conf_low = RecoveryEngine._compute_confidence(rule)

        rule["success_count"] = 100
        conf_high = RecoveryEngine._compute_confidence(rule)

        assert conf_high >= conf_low


class TestBuildRetryPlan:
    def test_grip_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("grasp", {"action": "grasp"})
        assert plan["action"] == "retry_with_adjustments"
        assert "gripper_force_offset" in plan["parameter_patch"]

    def test_collision_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("collision", {"action": "clearance"})
        patch = plan["parameter_patch"]
        assert "safety_clearance" in patch
        assert "velocity_factor" in patch

    def test_joint_limit_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("joint limit", {"action": "limit"})
        patch = plan["parameter_patch"]
        assert "velocity_factor" in patch
        assert "joint_range_reduction" in patch

    def test_timeout_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("timeout", {"action": "backoff"})
        patch = plan["parameter_patch"]
        assert "timeout_multiplier" in patch
        assert "max_retries" in patch

    def test_compliant_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("force", {"action": "compliant"})
        patch = plan["parameter_patch"]
        assert "control_mode" in patch
        assert patch["control_mode"] == "compliant"

    def test_sensor_related(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("sensor", {"action": "camera"})
        patch = plan["parameter_patch"]
        assert "sensor_fusion" in patch

    def test_robot_specific_ur5e(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("test", {"action": "x"}, {"robot_id": "ur5e_01"})
        patch = plan["parameter_patch"]
        assert patch.get("_robot_specific") == "ur5e_safe_mode"

    def test_default_patch(self, engine_with_rules):
        plan = engine_with_rules.build_retry_plan("unknown", {"action": "unknown_action"})
        patch = plan["parameter_patch"]
        assert "velocity_factor" in patch


class TestFormatForEventbus:
    def test_format(self, engine_with_rules):
        formatted = engine_with_rules.format_for_eventbus(
            {
                "failure_type": "test",
                "hint": "do this",
                "confidence": 0.8,
                "source": ["heuristic:r1"],
                "retry_plan": {"action": "retry"},
                "all_candidates": [],
            },
            request_id="req1",
        )
        assert formatted["request_id"] == "req1"
        assert formatted["failure_type"] == "test"
        assert formatted["hint"] == "do this"
        assert formatted["confidence"] == 0.8


class TestRecoveryFormatter:
    def test_to_event_payload(self):
        rule = {"rule_id": "r1", "condition": "collision", "action": "retry", "priority": 5}
        payload = RecoveryFormatter.to_event_payload(rule, request_id="req1")
        assert payload["request_id"] == "req1"
        assert payload["rule_id"] == "r1"
        assert payload["priority"] == 5

    def test_apply_trajectory_reduction(self):
        traj = [[1.0, 2.0], [3.0, 4.0]]
        result = RecoveryFormatter.apply_trajectory_adjustment(traj, "reduce velocity by 50%")
        assert result[0][0] == 0.5
        assert result[1][1] == 2.0

    def test_apply_trajectory_grip_force(self):
        traj = [[1.0, 2.0, 0.5], [3.0, 4.0, 0.6]]
        result = RecoveryFormatter.apply_trajectory_adjustment(traj, "increase grip force by 20%")
        assert result[0][2] == pytest.approx(0.7)
        assert result[1][2] == pytest.approx(0.8)

    def test_apply_trajectory_no_match(self):
        traj = [[1.0, 2.0], [3.0, 4.0]]
        result = RecoveryFormatter.apply_trajectory_adjustment(traj, "unknown suggestion")
        assert result == traj


class TestFormatRecoverySuggestion:
    def test_with_recovery(self):
        result = format_recovery_suggestion({"action": "retry with lower speed", "source": "heuristic"})
        assert "retry with lower speed" in result
        assert "heuristic" in result

    def test_without_recovery(self):
        result = format_recovery_suggestion(None)
        assert "No heuristic recovery" in result

    def test_empty_recovery(self):
        result = format_recovery_suggestion({})
        assert result  # Should not crash


class TestRecoveryEngineEventHandlers:
    def test_handle_failure_event_no_how(self):
        bus = FakeEventBus()
        eng = RecoveryEngine(None, event_bus=bus)
        from rosclaw.core.event_bus import Event
        eng._handle_failure_event(Event(topic="test", payload={"failure_type": "x"}, source="test"), "sandbox")
        assert len(bus.events) == 0

    def test_handle_failure_event_no_failure_type(self):
        bus = FakeEventBus()
        eng = RecoveryEngine(FakeHeuristic(), event_bus=bus)
        from rosclaw.core.event_bus import Event
        eng._handle_failure_event(Event(topic="test", payload={}, source="test"), "sandbox")
        assert len(bus.events) == 0

    def test_handle_failure_event_extracts_from_error_log(self):
        bus = FakeEventBus()
        eng = RecoveryEngine(FakeHeuristic([
            {"condition": "collision", "action": "clear", "rule_id": "r1"}
        ]), event_bus=bus)
        from rosclaw.core.event_bus import Event
        eng._handle_failure_event(
            Event(topic="test", payload={"error_log": "collision detected"}, source="test"),
            "sandbox"
        )
        assert len(bus.events) >= 1

    def test_handle_failure_event_extracts_from_violations(self):
        bus = FakeEventBus()
        eng = RecoveryEngine(FakeHeuristic([
            {"condition": "limit", "action": "retract", "rule_id": "r1"}
        ]), event_bus=bus)
        from rosclaw.core.event_bus import Event
        eng._handle_failure_event(
            Event(topic="test", payload={"violations": [{"description": "joint limit"}]}, source="test"),
            "sandbox"
        )
        assert len(bus.events) >= 1

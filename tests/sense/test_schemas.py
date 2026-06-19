"""Tests for rosclaw.sense.schemas."""



from rosclaw.sense.schemas import (
    BodyEvent,
    BodyReadiness,
    BodyRiskSummary,
    BodySense,
    BodyState,
    FailedRequirement,
    ReadinessItem,
    max_risk,
)


class TestBodyState:
    def test_create_minimal(self):
        state = BodyState(robot_id="g1", timestamp=1.0)
        assert state.robot_id == "g1"
        assert state.timestamp == 1.0
        assert state.energy.battery_percent is None

    def test_missing_fields_tolerated(self):
        state = BodyState(robot_id="g1", timestamp=1.0)
        assert state.joints == {}
        assert state.balance.support_margin is None

    def test_json_round_trip(self):
        state = BodyState(robot_id="g1", timestamp=1.0)
        data = state.to_dict()
        restored = BodyState.from_dict(data)
        assert restored.robot_id == "g1"
        assert restored.timestamp == 1.0


class TestBodyRiskSummary:
    def test_defaults(self):
        summary = BodyRiskSummary()
        assert summary.power_risk == "unknown"

    def test_max_risk(self):
        assert max_risk("low", "medium", "unknown") == "medium"
        assert max_risk("critical", "high") == "critical"
        assert max_risk() == "unknown"


class TestBodyReadiness:
    def test_create(self):
        req = FailedRequirement("battery", 20, ">=40")
        item = ReadinessItem("kick_ball", "not_ready", failed_requirements=[req])
        readiness = BodyReadiness("g1", 1.0, capabilities={"kick_ball": item})
        assert readiness.capabilities["kick_ball"].status == "not_ready"

    def test_to_dict(self):
        readiness = BodyReadiness("g1", 1.0)
        assert "robot_id" in readiness.to_dict()


class TestBodySense:
    def test_create(self):
        sense = BodySense(
            robot_id="g1",
            timestamp=1.0,
            overall_status="caution",
            blocked_capabilities=["kick_ball"],
        )
        assert sense.overall_status == "caution"
        assert "kick_ball" in sense.blocked_capabilities

    def test_to_dict(self):
        sense = BodySense(robot_id="g1", timestamp=1.0, overall_status="ready")
        assert sense.to_dict()["overall_status"] == "ready"


class TestBodyEvent:
    def test_create(self):
        event = BodyEvent(event_id="e1", robot_id="g1", timestamp=1.0, type="low_battery")
        assert event.type == "low_battery"

    def test_from_dict(self):
        event = BodyEvent.from_dict({"event_id": "e2"})
        assert event.event_id == "e2"

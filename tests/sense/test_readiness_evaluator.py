"""Tests for rosclaw.sense.estimators.readiness."""

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.readiness import ReadinessEvaluator
from rosclaw.sense.estimators.risk import RiskEstimator


class TestReadinessEvaluator:
    def _readiness(self, scenario, task):
        state = MockCollector(scenario=scenario).collect()
        risk = RiskEstimator().evaluate(state)[0]
        evaluator = ReadinessEvaluator()
        return evaluator.evaluate(state, risk, task=task)

    def test_observe_scene_normal_ready(self):
        readiness = self._readiness("normal", "observe_scene")
        assert readiness.overall_status == "ready"

    def test_kick_ball_hot_knee_not_ready(self):
        readiness = self._readiness("hot_knee", "kick_ball")
        item = readiness.capabilities["kick_ball"]
        assert item.status == "not_ready"
        assert any("right_knee_temperature" in req.name for req in item.failed_requirements)

    def test_kick_ball_low_confidence_not_ready(self):
        readiness = self._readiness("camera_degraded", "kick_ball")
        item = readiness.capabilities["kick_ball"]
        assert item.status == "not_ready"
        assert any(req.name == "target_detector_confidence" for req in item.failed_requirements)

    def test_kick_ball_unstable_not_ready(self):
        readiness = self._readiness("unstable_support", "kick_ball")
        item = readiness.capabilities["kick_ball"]
        assert item.status == "not_ready"
        assert any(req.name == "support_margin" for req in item.failed_requirements)

    def test_kick_not_ready_scenario(self):
        readiness = self._readiness("kick_not_ready", "kick_ball")
        assert readiness.overall_status == "not_ready"
        item = readiness.capabilities["kick_ball"]
        assert len(item.failed_requirements) >= 3

    def test_high_power_motion_low_battery_blocked(self):
        readiness = self._readiness("low_battery", "high_power_motion")
        item = readiness.capabilities["high_power_motion"]
        assert item.status == "not_ready"

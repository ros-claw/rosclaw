"""Tests for rosclaw.sense.estimators.health."""

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.health import HealthEstimator


class TestHealthEstimator:
    def test_normal_is_ok(self):
        state = MockCollector(scenario="normal").collect()
        health = HealthEstimator().estimate(state)
        assert health["energy"] == "ok"
        assert health["thermal"] == "ok"
        assert health["communication"] == "ok"

    def test_hot_knee_thermal_degraded(self):
        state = MockCollector(scenario="hot_knee").collect()
        health = HealthEstimator().estimate(state)
        assert health["thermal"] == "degraded"

    def test_critical_battery(self):
        state = MockCollector(scenario="critical_battery").collect()
        health = HealthEstimator().estimate(state)
        assert health["energy"] == "bad"

    def test_unknown_partial(self):
        state = MockCollector(scenario="unknown_partial").collect()
        health = HealthEstimator().estimate(state)
        assert health["energy"] == "unknown"

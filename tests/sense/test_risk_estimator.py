"""Tests for rosclaw.sense.estimators.risk."""

import pytest

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.risk import RiskEstimator


class TestRiskEstimator:
    @pytest.fixture
    def estimator(self):
        return RiskEstimator()

    def test_normal_low_risk(self, estimator):
        state = MockCollector(scenario="normal").collect()
        summary, events = estimator.evaluate(state)
        assert summary.overall_risk == "low"
        assert not events

    def test_critical_battery(self, estimator):
        state = MockCollector(scenario="critical_battery").collect()
        summary, events = estimator.evaluate(state)
        assert summary.power_risk == "critical"
        assert any(e.type == "critical_battery" for e in events)

    def test_hot_knee(self, estimator):
        state = MockCollector(scenario="hot_knee").collect()
        summary, events = estimator.evaluate(state)
        assert summary.thermal_risk == "high"
        assert any(e.type == "joint_hot" for e in events)

    def test_overheat_joint(self, estimator):
        state = MockCollector(scenario="overheat_joint").collect()
        summary, events = estimator.evaluate(state)
        assert summary.thermal_risk == "critical"

    def test_dds_latency_high(self, estimator):
        state = MockCollector(scenario="dds_latency_high").collect()
        summary, events = estimator.evaluate(state)
        assert summary.communication_risk == "high"

    def test_camera_degraded(self, estimator):
        state = MockCollector(scenario="camera_degraded").collect()
        summary, events = estimator.evaluate(state)
        assert summary.perception_risk == "medium"

    def test_compute_overload(self, estimator):
        state = MockCollector(scenario="compute_overload").collect()
        summary, events = estimator.evaluate(state)
        assert summary.compute_risk == "critical"

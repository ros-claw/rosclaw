"""Tests for rosclaw.sense.estimators.fatigue."""

import pytest

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.fatigue import FatigueEstimator
from rosclaw.sense.schemas import BodyState


class TestFatigueEstimator:
    @pytest.fixture
    def estimator(self):
        return FatigueEstimator()

    def test_normal_low_fatigue(self, estimator):
        state = MockCollector(scenario="normal").collect()
        result = estimator.estimate(state)
        assert 0.0 <= result["fatigue_score"] <= 1.0
        assert result["fatigue_risk"] == "low"

    def test_overheat_increases_fatigue(self, estimator):
        state = MockCollector(scenario="overheat_joint").collect()
        result = estimator.estimate(state)
        assert result["fatigue_score"] > 0.5
        assert result["fatigue_risk"] in ("medium", "high")

    def test_compute_overload_increases_fatigue(self, estimator):
        state = MockCollector(scenario="compute_overload").collect()
        result = estimator.estimate(state)
        assert result["fatigue_score"] > 0.0

    def test_ema_smoothing(self, estimator):
        hot = MockCollector(scenario="overheat_joint").collect()
        normal = MockCollector(scenario="normal").collect()
        # First hot sample pushes score up.
        r1 = estimator.estimate(hot, None, 1.0)
        # Following normal samples gradually bring it down, but not instantly.
        for _ in range(3):
            r2 = estimator.estimate(normal, hot, 1.0)
        assert r2["fatigue_score"] < r1["fatigue_score"]
        assert r2["fatigue_score"] > 0.0

    def test_empty_state_returns_low(self, estimator):
        state = BodyState(robot_id="empty", timestamp=0.0)
        result = estimator.estimate(state)
        assert result["fatigue_score"] == 0.0
        assert result["fatigue_risk"] == "low"

    def test_reset(self, estimator):
        hot = MockCollector(scenario="overheat_joint").collect()
        estimator.estimate(hot)
        estimator.reset()
        assert estimator._ema_score == 0.0

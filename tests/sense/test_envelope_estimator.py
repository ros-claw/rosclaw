"""Tests for rosclaw.sense.estimators.envelope."""

import pytest

from rosclaw.sense.estimators.envelope import OperationalEnvelopeEstimator
from rosclaw.sense.explain import SenseExplainer
from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.risk import RiskEstimator
from rosclaw.sense.estimators.readiness import ReadinessEvaluator


class TestOperationalEnvelopeEstimator:
    @pytest.fixture
    def estimator(self):
        return OperationalEnvelopeEstimator()

    def _make_sense(self, scenario):
        state = MockCollector(scenario=scenario).collect()
        risk_summary, _ = RiskEstimator().evaluate(state, fatigue_risk="low")
        readiness = ReadinessEvaluator({}, {}).evaluate_all(state, risk_summary, {})
        return SenseExplainer().summarize(state, risk_summary, readiness)

    def test_ready_full_scale(self, estimator):
        sense = self._make_sense("normal")
        envelope = estimator.estimate(sense, fatigue_risk="low")
        assert envelope["max_velocity_scale"] == 1.0
        assert envelope["sandbox_only"] is False

    def test_caution_reduces_velocity(self, estimator):
        sense = self._make_sense("normal")
        sense.overall_status = "caution"
        envelope = estimator.estimate(sense, fatigue_risk="low")
        assert envelope["max_velocity_scale"] == 0.5

    def test_not_ready_sandbox_only(self, estimator):
        sense = self._make_sense("kick_not_ready")
        envelope = estimator.estimate(sense, fatigue_risk="low")
        assert envelope["sandbox_only"] is True
        assert envelope["max_velocity_scale"] == 0.0

    def test_fatigue_high_further_derates(self, estimator):
        sense = self._make_sense("normal")
        envelope = estimator.estimate(sense, fatigue_risk="high")
        assert envelope["max_velocity_scale"] == 0.5

    def test_cooldown_required_for_hot_joint(self, estimator):
        sense = self._make_sense("hot_knee")
        envelope = estimator.estimate(sense)
        assert envelope["cooldown_required"] is True
        assert envelope["thermal_limited"] is True
        assert envelope["sandbox_only"] is True
        assert envelope["max_velocity_scale"] == 0.0

    def test_thermal_limited_derates_velocity(self, estimator):
        sense = self._make_sense("hot_knee")
        # Force status to caution so velocity scale starts at 0.5, then thermal halves it.
        sense.overall_status = "caution"
        envelope = estimator.estimate(sense, fatigue_risk="low")
        assert envelope["max_velocity_scale"] == 0.25

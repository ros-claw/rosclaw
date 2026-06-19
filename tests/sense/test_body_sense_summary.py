"""Tests for rosclaw.sense.explain summarizer."""

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.readiness import ReadinessEvaluator
from rosclaw.sense.estimators.risk import RiskEstimator
from rosclaw.sense.explain import SenseExplainer


class TestSenseExplainer:
    def _summarize(self, scenario):
        state = MockCollector(scenario=scenario).collect()
        risk = RiskEstimator().evaluate(state)[0]
        readiness = ReadinessEvaluator().evaluate_all(state, risk)
        return SenseExplainer().summarize(state, risk, readiness)

    def test_normal_summary_ready(self):
        sense = self._summarize("normal")
        assert sense.overall_status == "ready"
        assert not sense.blocked_capabilities

    def test_hot_knee_blocks_kick(self):
        sense = self._summarize("hot_knee")
        assert "kick_ball" in sense.blocked_capabilities
        assert any("right_knee" in reason for reason in sense.main_reasons)
        assert "cooldown" in sense.recommended_actions

    def test_numeric_evidence_present(self):
        sense = self._summarize("hot_knee")
        assert sense.evidence.get("max_joint_temperature_c") == 78.2

    def test_explain_block(self):
        state = MockCollector(scenario="hot_knee").collect()
        risk = RiskEstimator().evaluate(state)[0]
        readiness = ReadinessEvaluator().evaluate(state, risk, task="kick_ball")
        text = SenseExplainer().explain_block("kick_ball", readiness)
        assert "right_knee" in text
        assert "78.2" in text

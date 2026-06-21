"""G1 mock sense demo — safe, synthetic body-sense reasoning.

This script exercises rosclaw.sense with mock data and prints BodySense
summaries plus operational envelopes for a few G1 scenarios. It never talks
to real hardware.
"""

from __future__ import annotations

from rosclaw.sense.collectors.mock_collector import MockCollector
from rosclaw.sense.estimators.envelope import OperationalEnvelopeEstimator
from rosclaw.sense.estimators.fatigue import FatigueEstimator
from rosclaw.sense.estimators.readiness import ReadinessEvaluator
from rosclaw.sense.estimators.risk import RiskEstimator
from rosclaw.sense.explain import SenseExplainer
from rosclaw.sense.thresholds import DEFAULT_SENSE_THRESHOLDS


def main() -> None:
    scenarios = ["normal", "hot_knee", "kick_not_ready", "camera_degraded"]

    risk_estimator = RiskEstimator(DEFAULT_SENSE_THRESHOLDS)
    readiness_evaluator = ReadinessEvaluator(DEFAULT_SENSE_THRESHOLDS)
    fatigue_estimator = FatigueEstimator(DEFAULT_SENSE_THRESHOLDS)
    envelope_estimator = OperationalEnvelopeEstimator()
    explainer = SenseExplainer()

    print("=== G1 Mock Sense Demo ===")
    for scenario in scenarios:
        collector = MockCollector(robot_id="g1_lab_01", scenario=scenario)
        state = collector.collect()

        fatigue = fatigue_estimator.estimate(state, prev_state=None, dt=1.0)
        risk_summary, _ = risk_estimator.evaluate(
            state, fatigue_risk=fatigue["fatigue_risk"]
        )
        readiness = readiness_evaluator.evaluate_all(state, risk_summary)
        sense = explainer.summarize(state, risk_summary, readiness)
        envelope = envelope_estimator.estimate(
            sense, fatigue_risk=fatigue["fatigue_risk"]
        )

        print(f"\nScenario: {scenario}")
        print(f"  overall_status: {sense.overall_status}")
        print(f"  blocked: {sense.blocked_capabilities}")
        print(f"  degraded: {sense.degraded_capabilities}")
        print(f"  fatigue: {fatigue['fatigue_risk']} ({fatigue['fatigue_score']})")
        print(f"  summary: {sense.natural_language_summary}")
        print(f"  envelope: {envelope}")


if __name__ == "__main__":
    main()

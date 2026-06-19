"""Tests for rosclaw.sense.collectors.mock_collector."""

import pytest

from rosclaw.sense.collectors.mock_collector import SCENARIOS, MockCollector


class TestMockCollector:
    def test_all_scenarios_exist(self):
        assert "normal" in SCENARIOS
        assert "hot_knee" in SCENARIOS
        assert "kick_not_ready" in SCENARIOS

    @pytest.mark.parametrize("scenario", list(SCENARIOS))
    def test_collect_returns_state(self, scenario):
        collector = MockCollector(robot_id="g1", scenario=scenario)
        state = collector.collect()
        assert state.robot_id == "g1"
        assert state.timestamp > 0

    def test_hot_knee_temperature(self):
        collector = MockCollector(scenario="hot_knee")
        state = collector.collect()
        assert state.joints["right_knee"].temperature_c == 78.2

    def test_kick_not_ready_combined(self):
        collector = MockCollector(scenario="kick_not_ready")
        state = collector.collect()
        assert state.joints["right_knee"].temperature_c == 78.2
        assert state.balance.support_margin == 0.09
        assert state.perception.target_detector_confidence == 0.71
        assert state.energy.battery_percent == 33.0

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError):
            MockCollector(scenario="nonexistent")

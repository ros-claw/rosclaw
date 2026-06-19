"""Tests for rosclaw.sense.thresholds."""

from rosclaw.sense.thresholds import (
    DEFAULT_SENSE_THRESHOLDS,
    get_capability_requirements,
    load_thresholds,
)


class TestLoadThresholds:
    def test_defaults_returned(self):
        thresholds = load_thresholds()
        assert thresholds["battery"]["low"] == DEFAULT_SENSE_THRESHOLDS["battery"]["low"]

    def test_robot_specific_override(self):
        thresholds = load_thresholds(robot_family="unitree_g1")
        assert thresholds["battery"]["low"] == 30.0
        assert thresholds["joint_temperature_c"]["hot"] == 75.0

    def test_capability_requirements_loaded(self):
        caps = get_capability_requirements(robot_family="unitree_g1")
        assert "kick_ball" in caps
        assert caps["kick_ball"]["max_leg_joint_temp_c"] == 72.0

"""Missing measurements cannot certify a robot's diagnostic health."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model

NOW = datetime(2026, 10, 3, tzinfo=UTC)


@pytest.fixture
def observed():
    root = Path(__file__).resolve().parents[2] / "fixtures/ros_expert/nav2"

    def load(name):
        return json.loads((root / f"{name}.json").read_text())

    return build_system_model(
        RosGraphSnapshot.from_dict(load("graph")),
        robot_id="fixture_base",
        body=load("body"),
        native=load("native"),
    )


@pytest.mark.parametrize("profile", ["all", "navigation", "sensors"])
def test_incomplete_graph_cannot_certify_health(observed, profile):
    observed.completeness["graph"] = False
    report = diagnose(observed, profile=profile, now=NOW)
    assert report["status"] == "UNKNOWN"
    assert "graph" in report["unknown_checks"]


@pytest.mark.parametrize("profile", ["all", "navigation", "sensors"])
def test_missing_declared_topic_type_is_unknown(observed, profile):
    observed.body["required_topic_types"] = {"/scan": "sensor_msgs/msg/LaserScan"}
    next(t for t in observed.graph["topics"] if t["name"] == "/scan").pop("msg_type")
    report = diagnose(observed, profile=profile, now=NOW)
    assert report["status"] == "UNKNOWN"
    assert "topic_type:/scan" in report["unknown_checks"]
    assert "ROS_TOPIC_005" not in {i["issue_code"] for i in report["issues"]}


@pytest.mark.parametrize("missing_signal", [False, True])
def test_missing_declared_rate_measurement_is_unknown(observed, missing_signal):
    if missing_signal:
        observed.signals = [s for s in observed.signals if s.topic != "/scan"]
    else:
        next(s for s in observed.signals if s.topic == "/scan").rate_hz = None
    report = diagnose(observed, profile="sensors", now=NOW)
    assert report["status"] == "UNKNOWN"
    assert "signal_rate:/scan" in report["unknown_checks"]


@pytest.mark.parametrize("profile", ["all", "navigation"])
def test_unknown_lifecycle_state_is_not_active(observed, profile):
    observed.lifecycle[0].state = "UNKNOWN"
    report = diagnose(observed, profile=profile, now=NOW)
    assert report["status"] == "UNKNOWN"
    assert f"lifecycle:{observed.lifecycle[0].name}" in report["unknown_checks"]


def test_unrelated_profile_does_not_require_sensor_or_lifecycle_evidence(observed):
    observed.completeness["graph"] = False
    observed.signals = []
    observed.lifecycle[0].state = "UNKNOWN"
    assert diagnose(observed, profile="tf", now=NOW)["status"] == "HEALTHY"


def test_missing_and_known_bad_evidence_remain_distinct(observed):
    observed.completeness["graph"] = False
    observed.navigation["localization_ready"] = False
    report = diagnose(observed, now=NOW)
    assert report["status"] == "BLOCKED"
    assert "graph" in report["unknown_checks"]
    assert "NAV2_LOCALIZATION_001" in {i["issue_code"] for i in report["issues"]}


def test_actual_observations_restore_health_without_mutation(observed):
    observed.body["required_topic_types"] = {"/scan": "sensor_msgs/msg/LaserScan"}
    before = observed.to_dict()
    report = diagnose(observed, now=NOW)
    assert report["status"] == "HEALTHY"
    assert report["unknown_checks"] == []
    assert observed.to_dict() == before


@pytest.mark.parametrize("topic", ["/range_front", "/fleet/cart/lidar"])
def test_sensor_remapping_preserves_diagnose_and_recheck(observed, topic):
    observed.body["required_topics"] = [
        topic if name == "/scan" else name for name in observed.body["required_topics"]
    ]
    observed.body["minimum_rates"] = {topic: 5}
    observed.body["required_topic_types"] = {topic: "sensor_msgs/msg/LaserScan"}
    for entry in observed.graph["topics"]:
        if entry["name"] == "/scan":
            entry["name"] = topic
    signal = next(s for s in observed.signals if s.topic == "/scan")
    signal.topic = topic
    assert diagnose(observed, profile="sensors", now=NOW)["status"] == "HEALTHY"
    original_rate = signal.rate_hz
    signal.rate_hz = None
    report = diagnose(observed, profile="sensors", now=NOW)
    assert report["status"] == "UNKNOWN"
    assert report["unknown_checks"] == [f"signal_rate:{topic}"]
    signal.rate_hz = original_rate
    assert diagnose(observed, profile="sensors", now=NOW)["status"] == "HEALTHY"

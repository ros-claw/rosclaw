"""A terminal navigator is not proof of an idle controller or physical stop."""

from datetime import UTC, datetime, timedelta

import pytest

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence import RosSystemModel

NOW = datetime(2026, 10, 9, tzinfo=UTC)


def model(nav=6, control=2, age=0, namespace="/robot", current_nav=None):
    captured = (NOW - timedelta(seconds=age)).isoformat()
    goals = [{"goal_uuid": [1] * 16, "status": nav}]
    if current_nav is not None:
        goals.append({"goal_uuid": [3] * 16, "status": current_nav})
    return RosSystemModel(
        robot_id="sim",
        snapshot_id="fixture",
        captured_at=NOW,
        observations={
            "action_statuses": {
                namespace + "/navigate_to_pose": {"captured_at": captured, "goals": goals},
                namespace + "/follow_path": {
                    "captured_at": captured,
                    "goals": [{"goal_uuid": [2] * 16, "status": control}],
                },
            }
        },
    )


def findings(m, profile="all"):
    return [
        i
        for i in diagnose(m, now=NOW, profile=profile)["issues"]
        if i["issue_code"] == "NAV2_ACTION_001"
    ]


@pytest.mark.parametrize("nav,control", [(6, 1), (6, 2), (6, 3), (5, 2)])
def test_failed_navigator_with_active_controller_warns_without_ownership_or_stop_claim(
    nav, control
):
    item = findings(model(nav, control))[0]
    assert item["severity"] == "warning"
    assert item["ownership_verified"] is False
    assert item["physical_stop_verified"] is False
    assert item["runtime_mutation_required"] is False
    assert item["official_sources"]


@pytest.mark.parametrize("kwargs", [{"age": 6}, {"nav": 4}, {"control": 5}, {"current_nav": 2}])
def test_stale_normal_completed_or_new_navigation_does_not_warn(kwargs):
    assert not findings(model(**kwargs))


def test_namespaces_cannot_correlate_different_robots():
    m = model()
    state = m.observations["action_statuses"]
    state["/other/follow_path"] = state.pop("/robot/follow_path")
    assert not findings(m)


def test_missing_or_naive_times_and_missing_evidence_remain_uninferred():
    m = model()
    m.observations["action_statuses"]["/robot/follow_path"]["captured_at"] = "2026-10-09T00:00:00"
    assert not findings(m)
    m.observations = {}
    assert not findings(m)
    assert not findings(model(), profile="tf")

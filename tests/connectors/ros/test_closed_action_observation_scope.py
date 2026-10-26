"""Closed compiled action dependencies do not give unknown actions a waiver."""

import pytest

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence.system_model import Lifecycle, Signal
from rosclaw.connectors.ros.resolver.semantics import issue_applies_to
from tests.connectors.ros.test_action_observation_contracts import bind_motion_dependencies
from tests.connectors.ros.test_expert_harness import NOW, nav
from tests.connectors.ros.test_expert_harness import healthy as _healthy


@pytest.fixture
def model():
    value = _healthy.__wrapped__()
    bind_motion_dependencies(value)
    value.lifecycle.append(
        Lifecycle(
            source="fixture", captured_at=NOW, name="/robot/controller_server", state="ACTIVE"
        )
    )
    for contract in value.body["action_observation_requirements"]["capabilities"].values():
        contract["complete_for_action"] = True
    value.graph["topics"].append(
        {"name": "/extra/pulse", "msg_type": "std_msgs/msg/Bool", "publishers": ["/extra"]}
    )
    value.signals.append(
        Signal(
            source="native:fixture",
            captured_at=NOW,
            topic="/extra/pulse",
            last_message_age_ms=8000,
            publisher_count=1,
        )
    )
    return value


def stale_issue(model):
    return next(
        r
        for r in diagnose(model, now=NOW)["issues"]
        if r["issue_code"] == "ROS_TOPIC_002"
        and r["evidence"][0]["observation"].get("topic") == "/extra/pulse"
    )


def test_unbound_silent_debug_retains_diagnostic_without_vetoing_closed_navigation(model):
    assert nav(model)["status"] == "AVAILABLE"
    issue = stale_issue(model)
    assert issue["severity"] == "blocking"
    assert issue["evidence"][0]["observation"]["last_message_age_ms"] == 8000
    assert issue_applies_to(issue, "navigation.navigate_to_pose") is False
    assert issue_applies_to(issue, "unconfigured.new_action") is True


@pytest.mark.parametrize("closed", [False, None, "true"])
def test_open_missing_or_invalid_completeness_never_exempts_unbound_topics(model, closed):
    for contract in model.body["action_observation_requirements"]["capabilities"].values():
        if closed is None:
            contract.pop("complete_for_action")
        else:
            contract["complete_for_action"] = closed
    assert nav(model)["status"] == "BLOCKED"


def test_an_open_action_keeps_the_unbound_fault_global(model):
    model.body["action_observation_requirements"]["capabilities"]["motion.twist"][
        "complete_for_action"
    ] = False
    issue = stale_issue(model)
    assert issue_applies_to(issue, "navigation.navigate_to_pose") is False
    assert issue_applies_to(issue, "motion.twist") is True


def test_declared_critical_input_still_blocks_closed_navigation(model):
    scan = next(r for r in model.signals if r.topic == "/robot/scan")
    scan.last_message_age_ms = 8000
    assert nav(model)["status"] == "BLOCKED"
    issue = next(
        r
        for r in diagnose(model, now=NOW)["issues"]
        if r["issue_code"] == "ROS_TOPIC_002"
        and r["evidence"][0]["observation"].get("topic") == "/robot/scan"
    )
    assert issue_applies_to(issue, "navigation.navigate_to_pose") is True
    assert issue_applies_to(issue, "unconfigured.new_action") is True


def test_globally_required_but_omitted_signal_is_a_contract_conflict_not_a_waiver(model):
    model.body.setdefault("required_topics", []).append("/extra/pulse")
    assert nav(model)["status"] == "BLOCKED"


def test_global_time_and_controller_hazards_remain_blocking(model):
    assert issue_applies_to(
        {"issue_code": "ROS_TIME_001", "severity": "blocking"}, "navigation.navigate_to_pose"
    )
    model.navigation["controller_progress"] = False
    assert nav(model)["status"] == "BLOCKED"


@pytest.mark.parametrize("hazard", ["time", "collision"])
def test_actual_global_hazard_is_not_narrowed_by_closed_dependencies(model, hazard):
    if hazard == "time":
        model.observations["node_use_sim_time"] = {"/controller": True, "/other": False}
        code = "ROS_TIME_001"
    else:
        model.navigation["collision_monitor_ready"] = False
        code = "NAV2_COLLISION_001"
    assert nav(model)["status"] == "BLOCKED"
    issue = next(r for r in diagnose(model, now=NOW)["issues"] if r["issue_code"] == code)
    assert "unaffected_capabilities" not in issue
    assert issue_applies_to(issue, "navigation.navigate_to_pose") is True


def test_a_closed_assertion_without_compiled_body_identity_is_not_trusted(model):
    model.body.pop("effective_body_hash")
    assert nav(model)["status"] == "BLOCKED"


def test_missing_required_lifecycle_keeps_unknown_after_debug_scope_is_fixed(model):
    model.lifecycle = [r for r in model.lifecycle if r.name != "/robot/controller_server"]
    assert nav(model)["status"] == "UNKNOWN"


def test_native_projection_retains_closed_scope_for_hash_and_model(monkeypatch, model):
    from types import SimpleNamespace

    from rosclaw.connectors.ros.context import native

    monkeypatch.setattr(native, "inspect_system", lambda **_kwargs: model)
    monkeypatch.setattr(native, "datetime", SimpleNamespace(now=lambda *_args: NOW))
    service = SimpleNamespace(
        _config=SimpleNamespace(raw={"agent": {"ros_expert": {"enabled": True}}})
    )
    mission = SimpleNamespace(body_binding=SimpleNamespace(body_id="fixture"))
    observed = native.native_ros_observations(service, mission, None)
    facts = observed["admission_facts"]
    assert facts["capability_facts"]["navigation.navigate_to_pose"]["readiness"] == "AVAILABLE"
    pulse = next(h for h in facts["global_hazards"] if h["identity"].get("topic") == "/extra/pulse")
    assert "navigation.navigate_to_pose" in pulse["unaffected_capabilities"]
    assert "unaffected_capabilities=" in observed["summary"]
    assert "complete_for_action=true" in observed["summary"]


def test_stale_clock_source_remains_global_even_if_omitted_from_closed_topics(model):
    next(t for t in model.graph["topics"] if t["name"] == "/extra/pulse")["msg_type"] = (
        "rosgraph_msgs/msg/Clock"
    )
    assert nav(model)["status"] == "BLOCKED"
    assert issue_applies_to(stale_issue(model), "navigation.navigate_to_pose") is True

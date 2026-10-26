"""Normal/negative paired regressions for scoped, Body-owned observation facts."""

import pytest

from rosclaw.connectors.ros.diagnosis import diagnose
from rosclaw.connectors.ros.intelligence.system_model import Lifecycle, Signal
from rosclaw.connectors.ros.resolver.semantics import observation_contract
from tests.connectors.ros.test_expert_harness import NOW, nav
from tests.connectors.ros.test_expert_harness import healthy as _healthy_fixture


@pytest.fixture
def healthy():
    return _healthy_fixture.__wrapped__()


def add_contract(model, namespace="/test", publication="conditional", required=None):
    topic, publisher = namespace + "/telemetry", namespace + "/controller"
    model.body["effective_body_hash"] = "compiled-fixture-body"
    model.body.setdefault("observation_contracts", {})[topic] = {
        "schema_version": "rosclaw.observation_contract.v1",
        "semantic_role": "controller.execution_telemetry",
        "interface_type": "std_msgs/msg/Bool",
        "publication_model": publication,
        "required_for": required or [],
        "expected_when": {
            "action_server": namespace + "/follow_path",
            "action_type": "nav2_msgs/action/FollowPath",
            "phases": ["EXECUTING", "CANCELLING"],
        },
        "producer_binding": {
            "publisher": publisher,
            "package": "test_controller",
            "version": "1.3.13",
            "plugin_parameter": "FollowPath.plugin",
            "plugin": "test::Controller",
        },
    }
    model.observations.setdefault("package_versions", {})["test_controller"] = "1.3.13"
    model.observations.setdefault("node_parameters", {})[publisher] = {
        "FollowPath.plugin": "test::Controller"
    }
    model.graph["topics"].append(
        {"name": topic, "msg_type": "std_msgs/msg/Bool", "publishers": [publisher]}
    )
    model.signals.append(
        Signal(
            source="native:fixture",
            captured_at=NOW,
            topic=topic,
            last_message_age_ms=20000,
            publisher_count=1,
        )
    )
    return topic


@pytest.mark.parametrize(
    "phase", ["IDLE", "STARTING", "EXECUTING", "CANCELLING", "FAILED", "STOPPED"]
)
def test_auxiliary_telemetry_does_not_replace_independent_controller_health(healthy, phase):
    topic = add_contract(healthy)
    status = {
        "IDLE": None,
        "STARTING": 1,
        "EXECUTING": 2,
        "CANCELLING": 3,
        "FAILED": 6,
        "STOPPED": 4,
    }[phase]
    healthy.graph.setdefault("actions", []).append(
        {"name": "/test/follow_path", "action_type": "nav2_msgs/action/FollowPath"}
    )
    healthy.observations["action_statuses"] = {
        "/test/follow_path": {
            "captured_at": NOW.isoformat(),
            "goals": [] if status is None else [{"status": status}],
        }
    }
    healthy.lifecycle.append(
        Lifecycle(source="fixture", captured_at=NOW, name="/test/controller", state="ACTIVE")
    )
    assert nav(healthy)["status"] == "AVAILABLE"
    result = diagnose(healthy, now=NOW)
    issue = next(i for i in result["issues"] if i["issue_code"] == "ROS_TOPIC_002")
    assert issue["severity"] == "warning"
    assert issue["observation_contract"]["state"] == "UNKNOWN"
    assert issue["observation_contract"]["phase"] == phase
    assert issue["observation_contract"]["publication_expected"] is (
        phase in ["EXECUTING", "CANCELLING"]
    )
    assert topic in issue["evidence"][0]["observation"]["topic"]
    # A real independently observed controller failure still vetoes navigation.
    healthy.navigation["controller_progress"] = False
    assert nav(healthy)["status"] == "BLOCKED"
    assert any(
        i["issue_code"] == "NAV2_CONTROLLER_002" and i["severity"] == "blocking"
        for i in diagnose(healthy, now=NOW)["issues"]
    )


@pytest.mark.parametrize("namespace", ["/robot", "/renamed/robot"])
def test_optional_debug_invariance_and_namespace_binding(healthy, namespace):
    before = nav(healthy)["status"]
    add_contract(healthy, namespace=namespace, publication="optional")
    assert nav(healthy)["status"] == before == "AVAILABLE"


@pytest.mark.parametrize(
    "mismatch",
    ["body", "type", "publisher", "version", "plugin", "namespace", "rate", "required_topic"],
)
def test_names_and_boolean_types_alone_never_grant_exemption(healthy, mismatch):
    topic = add_contract(healthy)
    if mismatch == "body":
        healthy.body.pop("effective_body_hash")
    elif mismatch == "type":
        healthy.graph["topics"][-1]["msg_type"] = "std_msgs/msg/String"
    elif mismatch == "publisher":
        healthy.graph["topics"][-1]["publishers"] = ["/imposter"]
    elif mismatch == "version":
        healthy.observations["package_versions"]["test_controller"] = "unqualified"
    elif mismatch == "plugin":
        healthy.observations["node_parameters"]["/test/controller"]["FollowPath.plugin"] = "Other"
    elif mismatch == "namespace":
        healthy.graph["topics"][-1]["name"] = "/unbound/telemetry"
        healthy.signals[-1].topic = "/unbound/telemetry"
    elif mismatch == "rate":
        healthy.body.setdefault("minimum_rates", {})[topic] = 20
    elif mismatch == "required_topic":
        healthy.body.setdefault("required_topics", []).append(topic)
    assert observation_contract(healthy, topic) is None
    assert nav(healthy)["status"] == "BLOCKED"


@pytest.mark.parametrize("kind", ["scan", "depth", "odom", "tf", "controller"])
def test_required_inputs_still_fail_closed_with_optional_debug_present(healthy, kind):
    add_contract(healthy, publication="optional")
    if kind in ["scan", "odom"]:
        ros_type = "sensor_msgs/msg/LaserScan" if kind == "scan" else "nav_msgs/msg/Odometry"
        topic = next(t["name"] for t in healthy.graph["topics"] if t["msg_type"] == ros_type)
        next(s for s in healthy.signals if s.topic == topic).last_message_age_ms = 10000
    elif kind == "depth":
        topic = add_contract(
            healthy,
            namespace="/depth",
            publication="periodic",
            required=["navigation.navigate_to_pose"],
        )
        healthy.graph["topics"][-1]["msg_type"] = "sensor_msgs/msg/PointCloud2"
        healthy.body["observation_contracts"][topic]["interface_type"] = (
            "sensor_msgs/msg/PointCloud2"
        )
    elif kind == "tf":
        next(t for t in healthy.transforms if not t.static).age_ms = 10000
    else:
        healthy.navigation["controller_progress"] = False
    assert nav(healthy)["status"] == "BLOCKED"


def test_missing_required_observation_is_unknown_not_available(healthy):
    topic = add_contract(healthy, publication="periodic", required=["navigation.navigate_to_pose"])
    healthy.signals = [s for s in healthy.signals if s.topic != topic]
    assert nav(healthy)["status"] == "UNKNOWN"
    healthy.graph["topics"] = [t for t in healthy.graph["topics"] if t["name"] != topic]
    assert nav(healthy)["status"] == "BLOCKED"  # Missing typed binding fails closed.


def test_scoped_fault_does_not_veto_unrelated_capability(healthy):
    topic = add_contract(healthy, publication="periodic", required=["coverage.execute"])
    assert observation_contract(healthy, topic)
    assert diagnose(healthy, now=NOW)["status"] == "BLOCKED"  # Full diagnostic retained.
    assert nav(healthy)["status"] == "AVAILABLE"
    # A scope containing the current action is a genuine veto.
    healthy.body["observation_contracts"][topic]["required_for"].append(
        "navigation.navigate_to_pose"
    )
    assert nav(healthy)["status"] == "BLOCKED"


def test_unbound_new_topic_remains_conservative(healthy):
    topic = add_contract(healthy)
    healthy.body["observation_contracts"] = {}
    assert observation_contract(healthy, topic) is None
    assert nav(healthy)["status"] == "BLOCKED"

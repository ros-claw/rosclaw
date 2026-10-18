"""Synthetic anonymous interfaces and manifests; never driver/physical evidence."""

import copy
import hashlib
from datetime import timedelta

import pytest

from rosclaw.connectors.ros.resolver.component_compatibility import review_component_compatibility
from rosclaw.connectors.ros.resolver.task_capabilities import resolve_task_capability_graph
from tests.connectors.ros.test_intervention_plan import NOW, inputs


def model_and_control():
    _, model, _ = inputs()
    model.graph.update(
        topics=[{"name": "/arbitrary/base/input", "msg_type": "geometry_msgs/msg/TwistStamped"}]
    )
    model.qos.update(
        endpoints=[
            {
                "topic": "/arbitrary/base/input",
                "node": "/arbitrary/base/consumer",
                "kind": "subscriber",
            }
        ]
    )
    model.environment.update(ros_generation="ros2", distro="jazzy")
    model.observations.update(package_inventory=["anonymous_driver", "nav2_bringup"])
    model.completeness.update(qos=True)
    control = {
        "command_topic": "/arbitrary/base/input",
        "consumer_node": "/arbitrary/base/consumer",
        "message_type": "geometry_msgs/msg/TwistStamped",
        "driver_package": "anonymous_driver",
    }
    return model.seal(), control


def manifests(version="1.2.3"):
    xml = f'<package format="3"><name>nav2_bringup</name><version>{version}</version></package>'
    return {"nav2_bringup": {"xml": xml, "sha256": hashlib.sha256(xml.encode()).hexdigest()}}


def policies(version="1.2.3"):
    return [
        {
            "package": "nav2_bringup",
            "supported_ros": ["ros2"],
            "supported_distros": ["jazzy"],
            "required_version": version,
            "policy_source_sha256": "a" * 64,
        }
    ]


def test_existing_explicit_endpoint_presence_is_no_driver_physics_or_authority():
    model, control = model_and_control()
    before = copy.deepcopy(model.to_dict())
    result = resolve_task_capability_graph(model, "navigate_to_pose", control, now=NOW)
    assert result["control_prerequisite"]["status"] == "OBSERVED_ROS2_CONTROL_INTERFACE"
    assert result["execution_authority"] == "NONE" and result["physical_acceptance"] == "NOT_RUN"
    assert (
        result["control_prerequisite"]["driver_physics"] == "NOT_ESTABLISHED_BY_ENDPOINT_PRESENCE"
    )
    assert model.to_dict() == before
    assert not any(n["semantic_id"].startswith(("coverage.", "cleaning.")) for n in result["nodes"])


@pytest.mark.parametrize("missing", ["declaration", "topic", "endpoint", "package"])
def test_missing_control_prerequisite_requests_driver_adaptation_without_conventional_topic(
    missing,
):
    model, control = model_and_control()
    if missing == "declaration":
        control = None
    elif missing == "topic":
        model.graph["topics"] = []
    elif missing == "endpoint":
        model.qos["endpoints"] = []
    else:
        model.observations["package_inventory"].remove("anonymous_driver")
    result = resolve_task_capability_graph(model.seal(), "navigate_to_pose", control, now=NOW)
    assert result["control_prerequisite"]["status"] == "NEEDS_DRIVER"
    assert result["execution_authority"] == "NONE"


@pytest.mark.parametrize(
    "condition",
    [
        "stale",
        "future",
        "incomplete_graph",
        "incomplete_qos",
        "missing_inventory",
        "ambiguous_endpoint",
    ],
)
def test_incomplete_or_ambiguous_originals_remain_unknown(condition):
    model, control = model_and_control()
    now = NOW
    if condition == "stale":
        now += timedelta(milliseconds=501)
    elif condition == "future":
        now -= timedelta(milliseconds=1)
    elif condition == "incomplete_graph":
        model.completeness["graph"] = False
    elif condition == "incomplete_qos":
        model.completeness["qos"] = False
    elif condition == "missing_inventory":
        model.observations.pop("package_inventory")
    else:
        model.qos["endpoints"] *= 2
    result = resolve_task_capability_graph(model.seal(), "navigate_to_pose", control, now=now)
    assert result["control_prerequisite"]["status"] == "UNKNOWN"


def test_wrong_observed_command_type_is_incompatible():
    model, control = model_and_control()
    model.graph["topics"][0]["msg_type"] = "std_msgs/msg/String"
    assert (
        resolve_task_capability_graph(model.seal(), "navigate_to_pose", control, now=NOW)[
            "control_prerequisite"
        ]["status"]
        == "INCOMPATIBLE"
    )


def test_cleaning_is_a_separate_explicit_dependency_graph():
    model, control = model_and_control()
    result = resolve_task_capability_graph(model, "complete_area_cleaning", control, now=NOW)
    assert {"coverage.verify", "cleaning.enable", "cleaning.disable"} <= {
        n["semantic_id"] for n in result["nodes"]
    }
    assert {"consumer": "coverage.execute", "requires": "cleaning.disable"} in result[
        "dependencies"
    ]


def test_unknown_task_and_tampered_snapshot_are_rejected():
    model, control = model_and_control()
    with pytest.raises(ValueError):
        resolve_task_capability_graph(model, "invented", control, now=NOW)
    model.observations["package_inventory"] = []
    with pytest.raises(ValueError, match="sealed"):
        resolve_task_capability_graph(model, "navigate_to_pose", control, now=NOW)


def test_exact_version_distro_match_is_no_runtime_acceptance():
    model, _ = model_and_control()
    original = manifests()
    before = copy.deepcopy(original)
    result = review_component_compatibility(model, policies(), original)
    assert result["all_declared_requirements_match"] is True
    assert (
        result["runtime_compatibility"] == "NOT_VERIFIED"
        and result["execution_authority"] == "NONE"
    )
    assert original == before


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("inventory", "UNKNOWN"),
        ("package", "MISSING_DEPENDENCY"),
        ("distro", "UNSUPPORTED_DECLARED_ENVIRONMENT"),
        ("manifest", "UNKNOWN"),
        ("policy_version", "UNKNOWN"),
        ("wrong_version", "VERSION_MISMATCH"),
    ],
)
def test_missing_or_conflicting_compatibility_is_explicit(condition, expected):
    model, _ = model_and_control()
    original = manifests()
    policy = policies()
    if condition == "inventory":
        model.observations.pop("package_inventory")
    elif condition == "package":
        model.observations["package_inventory"].remove("nav2_bringup")
    elif condition == "distro":
        model.environment["distro"] = "unreviewed"
    elif condition == "manifest":
        original = {}
    elif condition == "policy_version":
        policy = policies(None)
    else:
        original = manifests("9.0.0")
    result = review_component_compatibility(model.seal(), policy, original)
    assert (
        result["components"][0]["status"] == expected
        and result["all_declared_requirements_match"] is False
    )


@pytest.mark.parametrize(
    "condition",
    [
        "sha",
        "identity",
        "DTD",
        "duplicate_version",
        "invalid_version",
        "duplicate_policy",
        "bad_policy",
    ],
)
def test_manifest_or_policy_tampering_is_rejected(condition):
    model, _ = model_and_control()
    original = manifests()
    policy = policies()
    if condition == "sha":
        original["nav2_bringup"]["sha256"] = "0" * 64
    elif condition == "duplicate_policy":
        policy *= 2
    elif condition == "bad_policy":
        policy = [None]
    else:
        xml = original["nav2_bringup"]["xml"]
        if condition == "identity":
            xml = xml.replace("<name>nav2_bringup</name>", "<name>other_package</name>")
        elif condition == "DTD":
            xml = "<!DOCTYPE package>" + xml
        elif condition == "duplicate_version":
            xml = xml.replace("</package>", "<version>9.0.0</version></package>")
        else:
            xml = xml.replace("1.2.3", "$(exec)")
        original["nav2_bringup"] = {"xml": xml, "sha256": hashlib.sha256(xml.encode()).hexdigest()}
    with pytest.raises(ValueError):
        review_component_compatibility(model, policy, original)

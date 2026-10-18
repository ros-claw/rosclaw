"""Anonymous synthetic application reuse; no World or new Body acceptance."""

import copy
import hashlib
import json
from datetime import timedelta

import pytest
import yaml

from rosclaw.connectors.ros.engineering.application_plan import compile_application_plan
from rosclaw.connectors.ros.engineering.application_staging import compile_application_staging
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.connectors.ros.resolver.component_compatibility import COMPONENTS
from rosclaw.contracts.common import content_hash
from tests.connectors.ros.test_navigation_body_workspace import NOW, source


def inputs():
    model, urdf, requirements, _ = source(
        namespace="/unbranded/platform", base="moving_frame", sensor="range_frame"
    )
    packages = sorted(set(sum(COMPONENTS.values(), ()) + ("anonymous_driver",)))
    model.observations["package_inventory"] = packages
    model.qos["endpoints"] = [
        {
            "topic": requirements["velocity_output_topic"],
            "node": requirements["velocity_consumer_node"],
            "kind": "subscriber",
        }
    ]
    model.environment["distro"] = "jazzy"
    model.seal()
    graph = compile_operational_graph(
        model, urdf, requirements, source_ref="fixture://synthetic-original", now=NOW
    )
    control = {
        "command_topic": requirements["velocity_output_topic"],
        "consumer_node": requirements["velocity_consumer_node"],
        "message_type": "geometry_msgs/msg/Twist",
        "driver_package": "anonymous_driver",
    }
    policies = [
        {
            "package": p,
            "supported_ros": ["ros2"],
            "supported_distros": ["jazzy"],
            "required_version": "1.2.3",
            "policy_source_sha256": "a" * 64,
        }
        for p in packages
    ]
    manifests = {}
    for p in packages:
        xml = f"<package><name>{p}</name><version>1.2.3</version></package>"
        manifests[p] = {"xml": xml, "sha256": hashlib.sha256(xml.encode()).hexdigest()}
    request = {
        "task_class": "navigate_to_pose",
        "control": control,
        "source_revision": "b" * 40,
        "component_policies": policies,
    }
    return graph, model, urdf, requirements, manifests, request


def artifacts(plan):
    rows = []
    parameters = {}
    # Synthetic concrete selected node selectors; no historical model values.
    for role in ("localization", "planner", "controller", "collision_monitor", "navigator"):
        parameters["/unbranded/platform/" + role] = {"ros__parameters": {"use_sim_time": True}}

    files = [
        ("NAV2_PARAMETERS", "nav2_bringup", "params/navigation.yaml", yaml.safe_dump(parameters)),
        (
            "BRINGUP_DESCRIPTOR",
            "nav2_bringup",
            "config/bringup.json",
            json.dumps({"namespace": "/unbranded/platform"}),
        ),
        (
            "BEHAVIOR_TREE",
            "nav2_bt_navigator",
            "behavior_trees/selected.xml",
            '<root><BehaviorTree ID="Selected"/></root>',
        ),
        (
            "LAUNCH_REFERENCE",
            "nav2_bringup",
            "references/launch.json",
            json.dumps(
                {
                    "package": "nav2_bringup",
                    "launch_file": "launch/bringup_launch.py",
                    "arguments": {},
                }
            ),
        ),
        (
            "EXISTING_DRIVER_REFERENCE",
            "anonymous_driver",
            "references/control.json",
            json.dumps(plan.request.control.model_dump(mode="json")),
        ),
    ]
    for role, package, path, raw in files:
        rows.append(
            {
                "role": role,
                "package": package,
                "package_version": "1.2.3",
                "original_package_path": path,
                "content": raw,
                "sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "target_snapshot_hash": plan.snapshot_hash,
                "source_urdf_sha256": plan.source_urdf_sha256,
            }
        )
    return rows


def test_source_bound_reuse_plan_and_bundle_are_unexecuted_and_inputs_unchanged():
    args = inputs()
    before = copy.deepcopy(args[1].to_dict())
    plan = compile_application_plan(*args)
    assert plan.status == "CONDITIONAL_REUSE_STAGING"
    result = compile_application_staging(plan, *args[:-1], artifacts(plan))
    assert result["driver_generated"] is False and result["execution_authority"] == "NONE"
    assert result["physical_acceptance"] == "NOT_RUN" and result["current_l1"] == "NOT_EVALUATED"
    assert len(result["files"]) == 5 and args[1].to_dict() == before


@pytest.mark.parametrize(
    "condition,expected",
    [
        ("driver", "NEEDS_DRIVER"),
        ("missing_package", "NEEDS_DEPENDENCIES"),
        ("version_mismatch", "INCOMPATIBLE"),
        ("missing_manifest", "NEEDS_READ_ONLY_PROBE"),
        ("missing_policy", "NEEDS_READ_ONLY_PROBE"),
        ("missing_stack", "NEEDS_APPLICATION_BRINGUP"),
        ("coverage", "NEEDS_SEPARATE_COVERAGE_BINDING"),
    ],
)
def test_missing_prerequisites_never_emit_partial_staging(condition, expected):
    graph, model, raw, requirements, manifests, request = inputs()
    if condition == "driver":
        request["control"] = None
    elif condition == "missing_package":
        model.observations["package_inventory"].remove("nav2_amcl")
    elif condition == "version_mismatch":
        request["component_policies"][0]["required_version"] = "2.0.0"
    elif condition == "missing_manifest":
        manifests.pop("nav2_amcl")
    elif condition == "missing_policy":
        request["component_policies"] = [
            p for p in request["component_policies"] if p["package"] != "nav2_amcl"
        ]
    elif condition == "missing_stack":
        model.lifecycle = []
    else:
        request["task_class"] = "complete_area_cleaning"
    model.seal()
    graph = compile_operational_graph(
        model, raw, requirements, source_ref="fixture://synthetic-original", now=NOW
    )
    plan = compile_application_plan(graph, model, raw, requirements, manifests, request)
    assert plan.status == expected
    with pytest.raises(ValueError, match="prerequisite"):
        compile_application_staging(
            plan,
            graph,
            model,
            raw,
            requirements,
            manifests,
            artifacts(plan) if request["control"] else [],
        )


@pytest.mark.parametrize("condition", ["stale", "model_tamper", "manifest_tamper", "graph_tamper"])
def test_original_bindings_cannot_be_renewed_or_tampered(condition):
    graph, model, raw, requirements, manifests, request = inputs()
    if condition == "stale":
        graph = compile_operational_graph(
            model,
            raw,
            requirements,
            source_ref="fixture://synthetic-original",
            now=NOW + timedelta(seconds=20),
        )
        assert (
            compile_application_plan(graph, model, raw, requirements, manifests, request).status
            == "ABSTAIN"
        )
        return
    if condition == "model_tamper":
        model.environment["distro"] = "altered"
    elif condition == "manifest_tamper":
        manifests["nav2_amcl"]["xml"] += " "
    else:
        graph.__dict__["source_ref"] = "changed"
    with pytest.raises(ValueError):
        compile_application_plan(graph, model, raw, requirements, manifests, request)


@pytest.mark.parametrize(
    "condition",
    [
        "missing_role",
        "duplicate_role",
        "artifact_sha",
        "version",
        "snapshot",
        "urdf",
        "path",
        "wrong_driver",
        "launch_path",
        "BT_DTD",
        "rehashed_plan",
    ],
)
def test_artifact_and_plan_attacks_are_rejected(condition):
    args = inputs()
    plan = compile_application_plan(*args)
    rows = artifacts(plan)
    if condition == "missing_role":
        rows.pop()
    elif condition == "duplicate_role":
        rows[1]["role"] = rows[0]["role"]
    elif condition == "artifact_sha":
        rows[0]["sha256"] = "0" * 64
    elif condition == "version":
        rows[0]["package_version"] = "2.0.0"
    elif condition == "snapshot":
        rows[0]["target_snapshot_hash"] = "other"
    elif condition == "urdf":
        rows[0]["source_urdf_sha256"] = "0" * 64
    elif condition == "path":
        rows[0]["original_package_path"] = "../outside"
    elif condition == "wrong_driver":
        rows[-1]["content"] = "{}"
        rows[-1]["sha256"] = hashlib.sha256(b"{}").hexdigest()
    elif condition == "launch_path":
        rows[-2]["content"] = json.dumps(
            {"package": "nav2_bringup", "launch_file": "../evil.py", "arguments": {}}
        )
        rows[-2]["sha256"] = hashlib.sha256(rows[-2]["content"].encode()).hexdigest()
    elif condition == "BT_DTD":
        rows[2]["content"] = "<!DOCTYPE root><root><BehaviorTree/></root>"
        rows[2]["sha256"] = hashlib.sha256(rows[2]["content"].encode()).hexdigest()
    else:
        payload = plan.model_dump(mode="json")
        payload.pop("plan_hash")
        payload["source_ref"] = "altered"
        payload["plan_hash"] = content_hash("rosapplication", payload)
        plan = type(plan).from_dict(payload)
    with pytest.raises(ValueError):
        compile_application_staging(plan, *args[:-1], rows)


def test_official_style_behavior_tree_comments_are_preserved_as_original_bytes():
    args = inputs()
    plan = compile_application_plan(*args)
    rows = artifacts(plan)
    raw = '<!-- ordinary installed Nav2 documentation -->\n<root BTCPP_format="4"><BehaviorTree ID="Selected"/></root>'
    rows[2]["content"] = raw
    rows[2]["sha256"] = hashlib.sha256(raw.encode()).hexdigest()
    result = compile_application_staging(plan, *args[:-1], rows)
    assert result["files"]["behavior-tree.xml"] == raw
    assert result["file_sha256"]["behavior-tree.xml"] == hashlib.sha256(raw.encode()).hexdigest()

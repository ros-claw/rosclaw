"""Synthetic source/capability contracts; no SDK, model or ROS process starts."""

import copy
import hashlib
import json

import pytest
import yaml

from benchmarks.ros_expert_ablation.rsk_application_aci import ApplicationPhysicalACI
from benchmarks.ros_expert_ablation.rsk_sensor_composition import (
    INPUTS_V2,
    PACKAGE_BYTES,
    PACKAGE_SHA256,
    SDK_IMAGE,
    compose_source_sensor_component,
)
from rosclaw.connectors.ros.intelligence.operational_graph import compile_operational_graph
from rosclaw.contracts.common import content_hash
from tests.benchmarks import test_rsk_application_aci as applications
from tests.benchmarks.test_rsk_baselines import generic, request
from tests.connectors.ros.test_application_bootstrap import inputs as original_inputs
from tests.connectors.ros.test_navigation_body_workspace import NOW


def expanded_inputs(namespace="/anonymous/base"):
    graph, model, raw, bringup, template, manifests, args = original_inputs(namespace)
    sensor = (
        b'<gazebo reference="range_frame"><sensor name="original" type="gpu_lidar">'
        b"<gz_frame_id>range_frame</gz_frame_id></sensor></gazebo>"
    )
    raw = raw.replace(b"</robot>", sensor + b"</robot>")
    for value in model.observations["urdf_descriptions"].values():
        value.update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    model.seal()
    graph = compile_operational_graph(model, source_ref="fixture:declared-source", now=NOW)
    args["source_urdf_sha256"] = hashlib.sha256(raw).hexdigest()
    return graph, model, raw, bringup, template, manifests, args


def component(data):
    bridge = yaml.safe_dump(
        [
            {
                "ros_topic_name": data["driver_bringup"]["scan_ros_topic"],
                "gz_topic_name": "/original/source_scan",
                "direction": "GZ_TO_ROS",
                "gz_type_name": "gz.msgs.LaserScan",
                "ros_type_name": "sensor_msgs/msg/LaserScan",
            }
        ]
    )
    return {
        "schema_version": "rosclaw.source_envelope_sensor_component.v1",
        "scope": "EXPLICIT_OPERATOR_OWNED_DISPOSABLE_SIM_ONLY",
        "component": "laser_filters/StaticLaserScanPolygonFilter",
        "availability": "PINNED_ARTIFACT_NOT_CURRENTLY_INSTALLED_OR_ADMITTED",
        "sdk_image": SDK_IMAGE,
        "artifact": {"path": "fixture:fixed-operator-artifact", "sha256": PACKAGE_SHA256},
        "artifact_bytes": PACKAGE_BYTES,
        "source_urdf_sha256": data["request"]["source_urdf_sha256"],
        "scan_bridge_yaml": bridge,
        "scan_bridge_sha256": hashlib.sha256(bridge.encode()).hexdigest(),
    }


def fixture(tmp_path, monkeypatch, namespace="/anonymous/base", fault=None):
    monkeypatch.setattr(applications, "inputs", expanded_inputs)
    old, core, data = applications.fixture(tmp_path, namespace=namespace, fault=fault)
    versioned = {**data, "schema_version": INPUTS_V2, "sensor_processing": component(data)}
    return ApplicationPhysicalACI(old.base, application_inputs=versioned, arm="B"), core, versioned


@pytest.mark.parametrize("namespace", ["/anonymous/base", "/renamed/independent"])
def test_opt_in_composition_derives_frames_preserves_all_other_core_files(
    tmp_path, monkeypatch, namespace
):
    aci, core, data = fixture(tmp_path, monkeypatch, namespace)
    saved = copy.deepcopy(data)
    plan = aci.compile_plan()
    assert data == saved and aci.base.validation_count == 0
    assert plan["core_compiler_plan_hash"] == core["plan_hash"]
    for name in ("nav2.yaml", "requirements.json", "navigation-policy.json"):
        assert plan["files"][name] == core["files"][name]
    before, after = [json.loads(p["files"]["bringup.json"]) for p in [core, plan]]
    assert {k: v for k, v in after.items() if k != "source_body_scan_filter"} == before
    assert after["source_body_scan_filter"]["base_frame"] == "moving_frame"
    assert after["source_body_scan_filter"]["scan_frame"] == "range_frame"
    assert plan["sensor_component"]["artifact_load_verified"] is False
    assert plan["sensor_component"]["current_authority"] == "NONE"
    assert (
        plan["sensor_component"]["component_rollback"]["restored_file_sha256"]
        == core["file_sha256"]["bringup.json"]
    )
    assert plan["plan_hash"] == content_hash(
        "rskcomposeddriverbootstrap", {k: v for k, v in plan.items() if k != "plan_hash"}
    )
    proposed = applications.call(aci)
    assert proposed["proposal"]["files"] == plan["files"]
    assert proposed["proposal"]["admission"] == "NOT_ADMITTED"


def test_legacy_input_schema_and_no_component_output_remain_exact(tmp_path):
    old, core, _ = applications.fixture(tmp_path)
    assert old.compile_plan() == core
    assert "optional-sensor-component.json" not in old.original_input_files()
    schema = old.request_schema()
    assert (
        schema["discriminator"]["mapping"]["ros_engineer.read_application_input"]
        == "#/$defs/ApplicationInputCall"
    )


def test_generic_and_typed_receive_identical_component_original_and_output_contract(
    tmp_path, monkeypatch
):
    expert, _, data = fixture(tmp_path, monkeypatch)
    baseline = generic(expert.base, application_inputs=data)
    raw = expert.original_input_files()["optional-sensor-component.json"]
    assert baseline._original_files()["optional-sensor-component.json"] == raw
    read = expert.call(
        request(
            expert,
            "ros_engineer.read_application_input",
            file="optional-sensor-component.json",
            offset=0,
            limit=16000,
        )
    )
    assert read["text"] == raw and read["source_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    files = expert.compile_plan()["files"]
    proposed = baseline.call(
        request(
            baseline,
            "staging.propose_application",
            nav2_yaml=files["nav2.yaml"],
            bringup_json=files["bringup.json"],
            requirements_json=files["requirements.json"],
            navigation_policy_json=files["navigation-policy.json"],
        )
    )
    assert proposed["generic_caller_supplied_files"] and proposed["proposal"]["files"] == files
    assert not baseline.base.results and baseline.base.validation_count == 0


@pytest.mark.parametrize(
    "change",
    [
        {"scope": "REAL"},
        {"availability": "CURRENTLY_ADMITTED"},
        {"polygon_padding": 1.0},
        {"artifact_bytes": True},
        {"sdk_image": "sha256:" + "f" * 64},
    ],
)
def test_closed_opt_in_catalog_cannot_grant_permission_or_change_geometry(
    tmp_path, monkeypatch, change
):
    aci, _, data = fixture(tmp_path, monkeypatch)
    data = copy.deepcopy(data)
    data["sensor_processing"].update(change)
    with pytest.raises(ValueError):
        ApplicationPhysicalACI(aci.base, application_inputs=data, arm="B")
    assert not aci.base.proposals and not aci.base.results


@pytest.mark.parametrize(
    "fault",
    ["wrong_urdf", "wrong_package", "wrong_bridge_bytes", "wrong_bridge_type", "ambiguous_bridge"],
)
def test_mismatched_source_dependency_or_producer_is_rejected_before_staging(
    tmp_path, monkeypatch, fault
):
    aci, _, data = fixture(tmp_path, monkeypatch)
    altered = copy.deepcopy(data)
    catalog = altered["sensor_processing"]
    if fault == "wrong_urdf":
        catalog["source_urdf_sha256"] = "0" * 64
    elif fault == "wrong_package":
        catalog["artifact"]["sha256"] = "0" * 64
    elif fault == "wrong_bridge_bytes":
        catalog["scan_bridge_yaml"] += "\n"
    else:
        rows = yaml.safe_load(catalog["scan_bridge_yaml"])
        if fault == "wrong_bridge_type":
            rows[0]["ros_type_name"] = "sensor_msgs/msg/PointCloud2"
        else:
            rows += copy.deepcopy(rows)
        catalog["scan_bridge_yaml"] = yaml.safe_dump(rows)
        catalog["scan_bridge_sha256"] = hashlib.sha256(
            catalog["scan_bridge_yaml"].encode()
        ).hexdigest()
    with pytest.raises(ValueError):
        candidate = ApplicationPhysicalACI(aci.base, application_inputs=altered, arm="B")
        candidate.compile_plan()
    assert not aci.base.proposals and aci.base.validation_count == 0


def test_core_driver_failure_never_becomes_composed_plan(tmp_path, monkeypatch):
    aci, _, _ = fixture(tmp_path, monkeypatch, fault="missing_driver")
    plan = aci.compile_plan()
    assert plan["status"] == "NEEDS_DRIVER" and plan["files"] is None
    assert "sensor_component" not in plan and not aci.base.proposals


def test_tampered_core_hash_is_not_accepted_as_source_plan(tmp_path, monkeypatch):
    aci, core, data = fixture(tmp_path, monkeypatch)
    core["files"]["bringup.json"] = "{}"
    with pytest.raises(ValueError, match="Core application plan changed"):
        compose_source_sensor_component(
            core,
            aci.base.model,
            aci.base.tools.files["robot.urdf"].encode(),
            data["sensor_processing"],
        )

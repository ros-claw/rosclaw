"""Generic Nav2 parameter contracts using actual installed template bytes.

Robot XML is explicitly synthetic; no asset, Node, World or action is used.
"""

import hashlib
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from rosclaw.connectors.ros.context.sim_controller_source import prepare_sim_controller_source
from rosclaw.connectors.ros.context.sim_navigation_source import (
    NODE_ROLES,
    prepare_sim_navigation_source,
)
from tests.connectors.ros import test_sim_controller_source as controlled

ROOT = Path(__file__).resolve().parents[2] / "fixtures/ros/navigation_source"
ATTACHMENT = {
    "schema_version": "rosclaw.sim_cleaning_attachment.v1",
    "evidence_domain": "SIMULATION",
    "kind": "SIMULATED_CLEANING",
    "declaration_source": "explicit_synthetic_source_contract",
    "cleaning_polygon": [[-0.16, -0.16], [0.16, -0.16], [0.16, 0.16], [-0.16, 0.16]],
}
POLICY = {
    "source": "simulator_operator_fixture_policy",
    "approved": True,
    "evidence_domain": "SIMULATION",
    "frames": {
        "base": "platform",
        "odom": "local_motion",
        "map": "declared_map",
        "lidar": "range_frame",
    },
    "nodes": {role: "/explicit_nav/" + role for role in NODE_ROLES},
    "topics": {
        role: "/source_topics/" + role
        for role in ["lidar", "odom", "map", "cmd_vel_raw", "cmd_vel_smoothed", "cmd_vel_safe"]
    },
    "endpoints": {
        role: "/explicit_actions/" + role
        for role in [
            "navigate_to_pose",
            "navigate_through_poses",
            "navigate_complete_coverage",
            "follow_path",
            "compute_path_to_pose",
            "compute_coverage_path",
            "set_initial_pose",
        ]
    },
    "spawn_xyyaw": [0.1, -0.2, 0.3],
    "operation_width_m": 0.30,
    "map_resolution": 0.05,
    "map_yaml_path": "/evidence/map.yaml",
    "coverage_bt_xml_path": "/installed/source/coverage.xml",
}


@pytest.fixture
def source():
    urdf = ET.fromstring(controlled.URDF)
    sdf = ET.fromstring(controlled.SDF)
    collision = ET.SubElement(urdf.find("link"), "collision", name="body_envelope")
    ET.SubElement(ET.SubElement(collision, "geometry"), "box", size=".4 .3 .2")
    ET.SubElement(urdf, "link", name="range_frame")
    joint = ET.SubElement(urdf, "joint", name="range_mount", type="fixed")
    ET.SubElement(joint, "parent", link="platform")
    ET.SubElement(joint, "child", link="range_frame")
    model = sdf.find("model")
    collision = ET.SubElement(model.find("link"), "collision", name="body_envelope")
    ET.SubElement(
        ET.SubElement(ET.SubElement(collision, "geometry"), "box"), "size"
    ).text = ".4 .3 .2"
    ET.SubElement(model, "link", name="range_frame")
    joint = ET.SubElement(model, "joint", name="range_mount", type="fixed")
    ET.SubElement(joint, "parent").text = "platform"
    ET.SubElement(joint, "child").text = "range_frame"
    result = prepare_sim_controller_source(ET.tostring(urdf), ET.tostring(sdf), controlled.POLICY)
    return result["source_files"]["robot.urdf"], result["report"]


def generate(source, declaration=None, attachment=None):
    return prepare_sim_navigation_source(
        source[0],
        (ROOT / "nav2_params.sdk.yaml").read_bytes(),
        (ROOT / "coverage_params.sdk.yaml").read_bytes(),
        attachment=attachment or ATTACHMENT,
        declaration=declaration or POLICY,
        controller_report=source[1],
    )


def test_real_installed_templates_use_source_body_frames_limits_and_exact_namespace(source):
    result = generate(source)
    radius = result["report"]["geometry"]["physical_radius_m"]
    assert radius == 0.25
    params = result["parameters"]
    for name in [
        "/explicit_nav/global_costmap/global_costmap",
        "/explicit_nav/local_costmap/local_costmap",
    ]:
        row = params[name]["ros__parameters"]
        assert row["robot_radius"] == radius and row["robot_base_frame"] == "platform"
        assert row["footprint"] == ""
        for layer in row["plugins"]:
            if row[layer].get("plugin") in {
                "nav2_costmap_2d::VoxelLayer",
                "nav2_costmap_2d::ObstacleLayer",
            }:
                assert row[layer]["scan"]["topic"] == "/source_topics/lidar"
    follow = params[POLICY["nodes"]["controller"]]["ros__parameters"]["FollowPath"]
    assert follow["desired_linear_vel"] == 0.12 and follow["rotate_to_heading_angular_vel"] == 0.8
    localization = params[POLICY["nodes"]["localization"]]["ros__parameters"]
    assert localization["global_frame_id"] == "declared_map"
    assert localization["set_initial_pose"] is False and "initial_pose" not in localization
    coverage = params[POLICY["nodes"]["coverage"]]["ros__parameters"]
    assert coverage["robot_width"] == 0.5 and coverage["operation_width"] == 0.3
    assert result["report"]["requires_actual_Graph_TF_Body_source_admission"] is True
    assert result["report"]["authorization"] is False
    assert result["report"]["physical_acceptance"] == "NOT_RUN"
    assert {node["namespace"] for node in result["launch_nodes"]} == {"/explicit_nav"}


@pytest.mark.parametrize(
    "fault",
    [
        "missing_geometry",
        "unsupported_mesh",
        "wrong_urdf",
        "wrong_controller_frame",
        "missing_frame",
        "wrong_lidar",
        "wrong_namespace",
        "topic_alias",
        "missing_map",
        "wide_cleaner",
        "implicit_spawn",
        "unbounded_resolution",
        "real",
        "path_escape",
    ],
)
def test_incomplete_or_mismatched_source_refuses_without_known_profile_fallback(source, fault):
    policy = deepcopy(POLICY)
    urdf, report = source[0], deepcopy(source[1])
    if fault in {"missing_geometry", "unsupported_mesh"}:
        robot = ET.fromstring(urdf)
        if fault == "missing_geometry":
            robot.find("link").remove(robot.find("link/collision"))
        else:
            geom = robot.find("link/collision/geometry")
            geom.remove(geom[0])
            ET.SubElement(geom, "mesh", filename="package://unresolved/body.stl")
        urdf = ET.tostring(robot)
        report["output_hashes"]["robot.urdf"] = hashlib.sha256(urdf).hexdigest()
    elif fault == "wrong_urdf":
        report["output_hashes"]["robot.urdf"] = "0" * 64
    elif fault == "wrong_controller_frame":
        report["frames"]["odom"] = "foreign_frame"
    elif fault == "missing_frame":
        policy["frames"].pop("odom")
    elif fault == "wrong_lidar":
        policy["frames"]["lidar"] = "not_in_actual_urdf"
    elif fault == "wrong_namespace":
        policy["nodes"]["planner"] = "/foreign/planner"
    elif fault == "topic_alias":
        policy["topics"]["lidar"] = policy["topics"]["odom"]
    elif fault == "missing_map":
        policy["topics"].pop("map")
    elif fault == "wide_cleaner":
        policy["operation_width_m"] = 1.0
    elif fault == "implicit_spawn":
        policy.pop("spawn_xyyaw")
    elif fault == "unbounded_resolution":
        policy["map_resolution"] = float("nan")
    elif fault == "real":
        policy["evidence_domain"] = "REAL"
    else:
        policy["map_yaml_path"] = "/evidence/../outside.yaml"
    with pytest.raises(ValueError):
        generate((urdf, report), policy)


@pytest.mark.parametrize(
    "raw",
    [
        b"amcl: {}\namcl: {}",
        b"amcl: &a {}\nother: *a",
        b"[broken",
        b"a: " + b"[" * 33 + b"0" + b"]" * 33,
    ],
)
def test_ambiguous_or_unbounded_original_template_is_refused(source, raw):
    with pytest.raises(ValueError):
        prepare_sim_navigation_source(
            source[0],
            raw,
            (ROOT / "coverage_params.sdk.yaml").read_bytes(),
            attachment=ATTACHMENT,
            declaration=POLICY,
            controller_report=source[1],
        )


@pytest.mark.parametrize("fault", ["missing", "incomplete", "nonfinite", "zero_excluded"])
def test_incomplete_controller_bounds_cannot_generate_navigation(source, fault):
    report = deepcopy(source[1])
    if fault == "missing":
        report.pop("resolved_motion_limits")
    elif fault == "incomplete":
        report["resolved_motion_limits"].pop("linear_acceleration")
    elif fault == "nonfinite":
        report["resolved_motion_limits"]["linear_velocity"]["min"] = float("nan")
    else:
        report["resolved_motion_limits"]["linear_velocity"]["min"] = 0.1
    with pytest.raises(ValueError):
        generate((source[0], report))

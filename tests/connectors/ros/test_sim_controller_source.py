"""Generic controller source derivation; synthetic XML, no asset or transport."""

from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest

from rosclaw.connectors.ros.context.sim_controller_source import prepare_sim_controller_source

URDF = b"""<robot name="synthetic_not_a_heldout_asset"><link name="platform"/><link name="left_wheel"/><link name="right_wheel"/><joint name="left_axis" type="continuous"><parent link="platform"/><child link="left_wheel"/><axis xyz="0 1 0"/></joint><joint name="right_axis" type="continuous"><parent link="platform"/><child link="right_wheel"/><axis xyz="0 1 0"/></joint></robot>"""
SDF = b"""<sdf version="1.9"><model name="synthetic_source_model"><link name="platform"/><link name="left_wheel"/><link name="right_wheel"/><joint name="left_axis" type="revolute"><parent>platform</parent><child>left_wheel</child><axis><xyz>0 1 0</xyz></axis></joint><joint name="right_axis" type="revolute"><parent>platform</parent><child>right_wheel</child><axis><xyz>0 1 0</xyz></axis></joint><plugin filename="gz-sim-diff-drive-system" name="gz::sim::systems::DiffDrive"><left_joint>left_axis</left_joint><right_joint>right_axis</right_joint><wheel_separation>.37</wheel_separation><wheel_radius>.063</wheel_radius><max_linear_velocity>.12</max_linear_velocity><max_angular_velocity>.8</max_angular_velocity></plugin></model></sdf>"""
POLICY = {
    "source": "simulator_operator_fixture_policy",
    "approved": True,
    "evidence_domain": "SIMULATION",
    "body_model_name": "synthetic_source_model",
    "frames": {"base": "platform", "odom": "local_motion"},
    "limits": {
        "linear_velocity": 0.2,
        "angular_velocity": 1.0,
        "linear_acceleration": 0.4,
        "angular_acceleration": 0.8,
    },
    "controller_manager": "/declared_scope/custom_manager",
    "drive_controller": "/declared_scope/custom_drive",
    "joint_state_broadcaster": "/declared_scope/custom_joint_states",
    "odom_topic": "/explicit_motion",
    "drive_velocity_topic": "/explicit_drive_stamped",
    "robot_description_topic": "/actual_source_description",
    "controller_parameters_path": "/evidence/source-controller.yaml",
}


def test_actual_source_wheels_dimensions_frames_and_limits_are_preserved_without_profile():
    result = prepare_sim_controller_source(URDF, SDF, POLICY)
    report = result["report"]
    assert report["source_wheel_names"] == {"left": ["left_axis"], "right": ["right_axis"]}
    assert report["source_dimensions"] == {"wheel_separation": 0.37, "wheel_radius": 0.063}
    params = result["controller_parameters"][POLICY["drive_controller"]]["ros__parameters"]
    assert params["base_frame_id"] == "platform" and params["odom_frame_id"] == "local_motion"
    assert params["tf_frame_prefix_enable"] is False and params["cmd_vel_timeout"] == 0.2
    assert params["linear.x.max_velocity"] == 0.12 and params["angular.z.max_velocity"] == 0.8
    assert params["linear.x.min_velocity"] == -0.2
    assert report["source_noncontroller_structure_preserved"] is True
    assert report["live_controller_admitted"] is False and report["authorization"] is False
    plugin = ET.fromstring(result["source_files"]["robot.sdf"]).find("model/plugin")
    assert plugin.findtext("controller_manager_name") == "custom_manager"
    assert plugin.findtext("ros/namespace") == "/declared_scope"
    assert "robot_description:=/actual_source_description" in [
        e.text for e in plugin.findall("ros/remapping")
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_dimension",
        "nonfinite_dimension",
        "zero_dimension",
        "missing_wheel",
        "wheel_alias",
        "unknown_joint",
        "wrong_joint_kind",
        "already_controlled",
        "foreign_plugin",
        "nested_model",
        "lost_limit",
        "positive_min_velocity",
    ],
)
def test_unresolved_or_unsupported_original_source_is_refused(fault):
    sdf = ET.fromstring(SDF)
    drive = sdf.find("model/plugin")
    if fault == "missing_dimension":
        drive.remove(drive.find("wheel_radius"))
    elif fault == "nonfinite_dimension":
        drive.find("wheel_radius").text = "nan"
    elif fault == "zero_dimension":
        drive.find("wheel_radius").text = "0"
    elif fault == "missing_wheel":
        drive.remove(drive.find("left_joint"))
    elif fault == "wheel_alias":
        drive.find("right_joint").text = "left_axis"
    elif fault == "unknown_joint":
        drive.find("left_joint").text = "invented"
    elif fault == "wrong_joint_kind":
        sdf.find("model/joint").set("type", "fixed")
    elif fault == "already_controlled":
        ET.SubElement(sdf.find("model"), "ros2_control")
    elif fault == "foreign_plugin":
        drive.set("filename", "unverified.so")
    elif fault == "nested_model":
        ET.SubElement(sdf.find("model"), "model", name="nested")
    elif fault == "lost_limit":
        ET.SubElement(drive, "max_jerk").text = ".1"
    else:
        ET.SubElement(drive, "min_linear_velocity").text = ".01"
    with pytest.raises(ValueError):
        prepare_sim_controller_source(URDF, ET.tostring(sdf), POLICY)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_base",
        "relative_node",
        "wrong_namespace",
        "topic_alias",
        "unbounded_limit",
        "missing_limit",
        "path_escape",
        "real",
        "implicit_namespace",
    ],
)
def test_incomplete_or_unsafe_operator_declaration_is_refused(fault):
    policy = deepcopy(POLICY)
    if fault == "unknown_base":
        policy["frames"]["base"] = "not_in_urdf"
    elif fault == "relative_node":
        policy["controller_manager"] = "manager"
    elif fault == "wrong_namespace":
        policy["drive_controller"] = "/another/drive"
    elif fault == "topic_alias":
        policy["odom_topic"] = policy["drive_velocity_topic"]
    elif fault == "unbounded_limit":
        policy["limits"]["linear_velocity"] = float("inf")
    elif fault == "missing_limit":
        policy["limits"].pop("angular_acceleration")
    elif fault == "path_escape":
        policy["controller_parameters_path"] = "/evidence/../outside.yaml"
    elif fault == "real":
        policy["evidence_domain"] = "REAL"
    else:
        policy.pop("robot_description_topic")
    with pytest.raises(ValueError):
        prepare_sim_controller_source(URDF, SDF, policy)


def test_four_actual_source_wheels_need_no_two_joint_model_template():
    urdf, sdf = ET.fromstring(URDF), ET.fromstring(SDF)
    for source, joint_name, link_name, side in [
        (urdf, "left_rear_axis", "left_rear", "left"),
        (urdf, "right_rear_axis", "right_rear", "right"),
    ]:
        ET.SubElement(source, "link", name=link_name)
        joint = ET.SubElement(source, "joint", name=joint_name, type="continuous")
        ET.SubElement(joint, "parent", link="platform")
        ET.SubElement(joint, "child", link=link_name)
        model = sdf.find("model")
        ET.SubElement(model, "link", name=link_name)
        joint = ET.SubElement(model, "joint", name=joint_name, type="revolute")
        ET.SubElement(joint, "parent").text = "platform"
        ET.SubElement(joint, "child").text = link_name
        ET.SubElement(model.find("plugin"), side + "_joint").text = joint_name
    result = prepare_sim_controller_source(ET.tostring(urdf), ET.tostring(sdf), POLICY)
    assert result["report"]["source_wheel_names"] == {
        "left": ["left_axis", "left_rear_axis"],
        "right": ["right_axis", "right_rear_axis"],
    }
    assert (
        len(ET.fromstring(result["source_files"]["robot.urdf"]).findall("ros2_control/joint")) == 4
    )

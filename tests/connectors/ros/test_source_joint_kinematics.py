"""Actual observed link DOFs replace zero-joint assumptions without tolerance changes."""

import math
from copy import deepcopy

import pytest

from rosclaw.connectors.ros.context.source_kinematics import (
    compile_source_kinematics,
    verify_source_kinematics,
)


def fixture(kind="revolute", q=0.2):
    sdf = f'''<sdf version="1.9"><model name="anonymous"><link name="base"/><link name="wheel"><pose>1 0 0 0 0 0</pose><collision name="offset"><pose>0.2 0 0 0 0 0</pose><geometry><box><size>0.1 0.2 0.3</size></box></geometry></collision></link><joint name="hinge" type="{kind}"><parent>base</parent><child>wheel</child><pose relative_to="wheel">0 0 0 0 0 0</pose><axis><xyz>0 1 0</xyz><limit><lower>-1</lower><upper>1</upper></limit></axis></joint></model></sdf>'''.encode()
    ancestry = [
        {"entity": 2, "name": "anonymous", "model": True},
        {"entity": 1, "name": "world", "world": True},
    ]
    wheel = (
        [1, 0, 0, 0, q, 0]
        if kind == "revolute"
        else [1, q, 0, 0, 0, 0]
        if kind == "prismatic"
        else [1, 0, 0, 0, 0, 0]
    )
    sample = {
        "sample": 3,
        "models": [{"entity": 2, "name": "anonymous", "world_pose": [0] * 6}],
        "joint_kinematics": {
            "schema_version": "rosclaw.ecm_joint_kinematics.v1",
            "complete": True,
            "links": [
                {"entity": 3, "name": "base", "ancestry": ancestry, "world_pose": [0] * 6},
                {"entity": 4, "name": "wheel", "ancestry": ancestry, "world_pose": wheel},
            ],
            "joints": [
                {
                    "entity": 5,
                    "name": "hinge",
                    "ancestry": ancestry,
                    "kind": kind,
                    "parent": "base",
                    "child": "wheel",
                    "axis": [0, 1, 0],
                    "axis_frame": "",
                    "bounds": [-1, 1],
                    "position": [q],
                    "pose": [0] * 6,
                }
            ],
        },
    }
    return compile_source_kinematics(sdf, robot_entity="anonymous"), sample


def review(plan, sample):
    return verify_source_kinematics(
        plan, sample, 2, position_tolerance=0.00001, rotation_tolerance=0.001
    )


def test_rotated_box_member_and_sensor_use_original_joint_dof_not_shape_symmetry():
    plan, sample = fixture()
    members, result = review(plan, sample)
    p = members[("wheel", "offset")][1]
    assert p == pytest.approx((1 + 0.2 * math.cos(0.2), 0, -0.2 * math.sin(0.2)))
    assert (
        result["execution_authority"] == "NONE"
        and result["status"] == "MATCHED_DECLARED_SOURCE_JOINT_DOF"
    )


@pytest.mark.parametrize("kind", ["prismatic", "fixed"])
def test_declared_other_supported_dofs_are_checked(kind):
    plan, sample = fixture(kind)
    members, result = review(plan, sample)
    assert members[("wheel", "offset")][1] == pytest.approx(
        (1.2, 0.2 if kind == "prismatic" else 0, 0)
    )
    assert result["status"] == "MATCHED_DECLARED_SOURCE_JOINT_DOF"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "incomplete",
        "missing_link",
        "missing_joint",
        "wrong_axis",
        "wrong_parent",
        "wrong_kind",
        "rotary_translation",
        "rotation_wrong_axis",
        "position_missing",
        "unwrapped_outside_bounds",
        "position_pose_mismatch",
        "duplicate_link",
        "moved_root",
    ],
)
def test_wrong_kinematics_do_not_relax_source_precision(fault):
    plan, sample = fixture()
    k = sample["joint_kinematics"]
    j = k["joints"][0]
    if fault == "missing":
        sample.pop("joint_kinematics")
    elif fault == "incomplete":
        k["complete"] = False
    elif fault == "missing_link":
        k["links"].pop()
    elif fault == "missing_joint":
        k["joints"] = []
    elif fault == "wrong_axis":
        j["axis"] = [1, 0, 0]
    elif fault == "wrong_parent":
        j["parent"] = "other"
    elif fault == "wrong_kind":
        j["kind"] = "fixed"
    elif fault == "rotary_translation":
        k["links"][1]["world_pose"][2] = 0.001
    elif fault == "rotation_wrong_axis":
        k["links"][1]["world_pose"][3:] = [0.2, 0, 0]
    elif fault == "position_missing":
        j["position"] = None
    elif fault == "unwrapped_outside_bounds":
        j["position"] = [2 * math.pi + 0.2]
    elif fault == "position_pose_mismatch":
        j["position"] = [0.3]
    elif fault == "duplicate_link":
        k["links"].append(deepcopy(k["links"][0]))
    else:
        k["links"][0]["world_pose"][0] = 0.001
    with pytest.raises(ValueError):
        review(plan, sample)


def test_joint_anchor_is_not_an_unchecked_degree_of_freedom():
    plan, sample = fixture()
    sample["joint_kinematics"]["joints"][0]["pose"][0] = 0.00002
    with pytest.raises(ValueError, match="child anchor"):
        review(plan, sample)


def test_optional_source_witness_capture_is_read_only_and_explicit():
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[3]
        / "integrations/ros_probe/acceptance/gazebo_source_witness/SourceWitness.cc"
    ).read_text()
    assert (
        "observeJointKinematics = false" in text
        and 'sdf->HasElement("observe_joint_kinematics")' in text
    )
    assert "components::JointAxis" in text and "components::Link" in text
    for forbidden in (
        "CreateComponent(",
        "SetComponentData(",
        "SetData(",
        "Advertise(",
        ".Publish(",
        ".Request(",
    ):
        assert forbidden not in text

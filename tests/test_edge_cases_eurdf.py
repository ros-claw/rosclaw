"""Edge case tests for e-URDF modules — Sprint 2 release guard."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from rosclaw.e_urdf.parser import EURDFParser, JointSpec, LinkSpec, RobotModel
from rosclaw.runtime.eurdf_loader import EURDFLoader, RobotRegistry

# ── parser edge cases ──

def test_joint_spec_defaults():
    """JointSpec with minimal args defaults correctly."""
    j = JointSpec(name="j")
    assert j.joint_type == "fixed"
    assert j.parent == ""
    assert j.child == ""
    assert np.array_equal(j.axis, np.array([0, 0, 1]))
    assert np.array_equal(j.origin, np.eye(4))


def test_joint_spec_limits_partial():
    """Only lower or only limit should not crash."""
    j = JointSpec(name="j", lower_limit=-1.0)
    assert j.limits["lower"] == -1.0
    assert "upper" not in j.limits


def test_link_spec_defaults():
    """LinkSpec with minimal args defaults correctly."""
    link = LinkSpec(name="l")
    assert link.mass == 0.0
    assert np.array_equal(link.inertia, np.eye(3))
    assert link.semantic_tags == []


def test_robot_model_empty_joints_end_effector():
    """Empty robot should return None for end effector."""
    model = RobotModel(name="empty")
    assert model.get_end_effector_link() is None


def test_robot_model_joint_limits_incomplete():
    """Joints with only lower or only upper limit are skipped silently."""
    model = RobotModel(name="m")
    model.joints["j1"] = JointSpec(name="j1", joint_type="revolute", limits={"lower": -1.0})
    model.joints["j2"] = JointSpec(name="j2", joint_type="revolute", limits={"upper": 1.0})
    model.joints["j3"] = JointSpec(name="j3", joint_type="revolute", limits={"lower": -1.0, "upper": 1.0})
    limits = model.get_joint_limits()
    assert "j1" not in limits
    assert "j2" not in limits
    assert "j3" in limits


def test_eurdf_parser_non_robot_root():
    """Parser should reject non-robot root element."""
    bad = Path("/tmp/bad_robot.urdf")
    bad.write_text("<bad_root name='test'></bad_root>")
    with pytest.raises(ValueError, match="Expected root element 'robot'"):
        EURDFParser(str(bad))


def test_eurdf_parser_empty_file():
    """Parser should reject empty file."""
    empty = Path("/tmp/empty.urdf")
    empty.write_text("")
    with pytest.raises(ET.ParseError):  # noqa: B017
        EURDFParser(str(empty))


# ── loader edge cases ──

def test_loader_empty_robot_id():
    """Empty robot_id should be handled gracefully."""
    loader = EURDFLoader(zoo_path="/tmp/nonexistent")
    with pytest.raises(FileNotFoundError):
        loader.load("")


def test_registry_empty_robot_id():
    """Registry get with empty string should return None."""
    reg = RobotRegistry()
    assert reg.get("") is None


def test_registry_list_available_empty_zoo():
    """Registry with empty zoo still exposes builtin Python profiles."""
    reg = RobotRegistry(loader=EURDFLoader(zoo_path="/tmp/nonexistent_zoo"))
    available = reg.list_available()
    assert "franka_panda" in available
    assert "realsense-d405" in available
    assert "realsense-d435i" in available
    assert "realsense-dual" in available


def test_loader_validate_empty_robot_id(tmp_path):
    """Validate with empty robot_id should fail gracefully."""
    loader = EURDFLoader(zoo_path=tmp_path)
    result = loader.validate("")
    assert result["valid"] is False


def test_loader_load_yaml_missing():
    """_load_yaml with missing file returns None."""
    assert EURDFLoader._load_yaml(Path("/tmp/nonexistent.yaml")) is None


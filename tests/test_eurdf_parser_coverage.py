"""Comprehensive coverage tests for rosclaw.e_urdf.parser.

Targets: boost coverage from ~71% to 85%+ by exercising edge cases
in JointSpec, RobotModel, and EURDFParser.
"""


import numpy as np
import pytest

from rosclaw.e_urdf.parser import EURDFParser, JointSpec, LinkSpec, RobotModel, SensorSpec


class TestJointSpec:
    """Coverage for JointSpec.__init__ parameter combinations."""

    def test_defaults_all_none(self):
        j = JointSpec(name="j1")
        assert j.joint_type == "fixed"
        assert j.parent == ""
        assert j.child == ""
        assert np.array_equal(j.axis, np.array([0, 0, 1]))
        assert np.array_equal(j.origin, np.eye(4))
        assert j.limits == {}
        assert j.dynamics == {}
        assert j.safety == {}
        assert j.control == {}

    def test_type_alias_overrides_joint_type(self):
        """When only 'type' is provided it fills joint_type."""
        j = JointSpec(name="j2", type="revolute")
        assert j.joint_type == "revolute"

    def test_joint_type_takes_precedence_over_type(self):
        """joint_type wins when both are provided."""
        j = JointSpec(name="j3", joint_type="prismatic", type="revolute")
        assert j.joint_type == "prismatic"

    def test_lower_limit_injected_into_limits(self):
        j = JointSpec(name="j4", lower_limit=-1.5)
        assert j.limits == {"lower": -1.5}

    def test_upper_limit_injected_into_limits(self):
        j = JointSpec(name="j5", upper_limit=2.0)
        assert j.limits == {"upper": 2.0}

    def test_both_limits_injected(self):
        j = JointSpec(name="j6", lower_limit=-0.5, upper_limit=0.5)
        assert j.limits == {"lower": -0.5, "upper": 0.5}

    def test_limits_dict_and_lower_upper_together(self):
        """lower_limit / upper_limit should merge into existing limits dict."""
        j = JointSpec(name="j7", limits={"velocity": 1.0}, lower_limit=-0.1, upper_limit=0.1)
        assert j.limits["velocity"] == 1.0
        assert j.limits["lower"] == -0.1
        assert j.limits["upper"] == 0.1

    def test_explicit_arrays(self):
        axis = np.array([1, 0, 0])
        origin = np.eye(4) * 2
        j = JointSpec(name="j8", axis=axis, origin=origin)
        assert np.array_equal(j.axis, axis)
        assert np.array_equal(j.origin, origin)

    def test_explicit_dicts(self):
        j = JointSpec(
            name="j9",
            dynamics={"damping": 0.1},
            safety={"k_position": 10.0},
            control={"p": 50.0},
        )
        assert j.dynamics == {"damping": 0.1}
        assert j.safety == {"k_position": 10.0}
        assert j.control == {"p": 50.0}


class TestRobotModelMethods:
    """Coverage for RobotModel helpers."""

    def test_get_joint_names_empty(self):
        model = RobotModel(name="empty")
        assert model.get_joint_names() == []

    def test_get_joint_names_ordered(self):
        model = RobotModel(name="bot")
        model.joints["a"] = JointSpec(name="a")
        model.joints["b"] = JointSpec(name="b")
        assert model.get_joint_names() == ["a", "b"]

    def test_get_actuated_joints_filters_fixed(self):
        model = RobotModel(name="bot")
        model.joints["j1"] = JointSpec(name="j1", joint_type="revolute")
        model.joints["j2"] = JointSpec(name="j2", joint_type="fixed")
        model.joints["j3"] = JointSpec(name="j3", joint_type="prismatic")
        assert model.get_actuated_joints() == ["j1", "j3"]

    def test_get_joint_limits_only_full_pairs(self):
        model = RobotModel(name="bot")
        model.joints["a"] = JointSpec(name="a", limits={"lower": -1, "upper": 1})
        model.joints["b"] = JointSpec(name="b", limits={"lower": -2})
        model.joints["c"] = JointSpec(name="c", limits={"upper": 2})
        assert model.get_joint_limits() == {"a": (-1.0, 1.0)}

    def test_get_end_effector_link_single_chain(self):
        """get_end_effector_link returns parent_links - child_links.

        In a chain base->link1->link2, parent_links={base,link1},
        child_links={link1,link2}, so the result is {base}.
        """
        model = RobotModel(name="arm")
        model.joints["j1"] = JointSpec(name="j1", parent="base", child="link1")
        model.joints["j2"] = JointSpec(name="j2", parent="link1", child="link2")
        assert model.get_end_effector_link() == "base"

    def test_get_end_effector_link_empty(self):
        model = RobotModel(name="none")
        assert model.get_end_effector_link() is None

    def test_to_llm_context_no_sensors_no_semantic_tags(self):
        model = RobotModel(name="simple")
        model.joints["j1"] = JointSpec(
            name="j1", joint_type="revolute", parent="base", child="l1",
            limits={"lower": -1.0, "upper": 1.0},
        )
        model.links["l1"] = LinkSpec(name="l1", mass=0.5)
        ctx = model.to_llm_context()
        assert "Robot: simple" in ctx
        assert "Degrees of Freedom: 1" in ctx
        assert "Joints: j1" in ctx
        assert "Joint Limits:" in ctx
        assert "j1: [-1.000, 1.000] rad" in ctx
        assert "Links:" in ctx
        assert "l1: mass=0.500kg" in ctx
        assert "Sensors:" not in ctx

    def test_to_llm_context_with_sensors_and_semantic_tags(self):
        model = RobotModel(name="rich")
        model.joints["j1"] = JointSpec(
            name="j1", joint_type="revolute", parent="base", child="l1",
            limits={"lower": -1.57, "upper": 1.57},
        )
        link = LinkSpec(name="l1", mass=1.2)
        link.semantic_tags = ["gripper", "tool"]
        model.links["l1"] = link
        model.sensors["cam1"] = SensorSpec(
            name="cam1", sensor_type="camera", parent_link="l1"
        )
        ctx = model.to_llm_context()
        assert "rich" in ctx
        assert "l1: mass=1.200kg (gripper, tool)" in ctx
        assert "Sensors:" in ctx
        assert "cam1: camera on l1" in ctx

    def test_to_llm_context_no_actuated_joints(self):
        model = RobotModel(name="static")
        model.links["base"] = LinkSpec(name="base", mass=5.0)
        ctx = model.to_llm_context()
        assert "Degrees of Freedom: 0" in ctx
        assert "Joints:" in ctx
        assert "Joint Limits:" in ctx


class TestEURDFParserErrors:
    """Coverage for EURDFParser error paths."""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            EURDFParser("/this/does/not/exist.urdf")

    def test_wrong_root_tag(self, tmp_path):
        bad = tmp_path / "bad.urdf"
        bad.write_text("<world name='earth'></world>")
        with pytest.raises(ValueError, match="Expected root element 'robot'"):
            EURDFParser(str(bad))

    def test_get_model_runtime_error(self, tmp_path):
        """Simulate a scenario where model is None.

        We force it by creating a valid parser then clobbering model.
        """
        valid = tmp_path / "valid.urdf"
        valid.write_text('<robot name="r"></robot>')
        parser = EURDFParser(str(valid))
        parser.model = None
        with pytest.raises(RuntimeError, match="Model not parsed"):
            parser.get_model()


class TestEURDFParserParseLink:
    """Coverage for EURDFParser._parse_link."""

    def test_visual_and_material(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="base">
                <visual>
                    <mesh filename="base.dae"/>
                    <material name="red"/>
                </visual>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["base"]
        assert link.visual_mesh == "base.dae"
        assert link.material == "red"

    def test_collision_mesh(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="link1">
                <collision>
                    <mesh filename="link1_col.stl"/>
                </collision>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["link1"]
        assert link.collision_mesh == "link1_col.stl"

    def test_inertial_mass_and_inertia(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="arm">
                <inertial>
                    <mass value="2.5"/>
                    <inertia ixx="0.1" ixy="0.01" ixz="0.02"
                             iyy="0.2" iyz="0.03" izz="0.3"/>
                </inertial>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["arm"]
        assert link.mass == 2.5
        expected = np.array([
            [0.1, 0.01, 0.02],
            [0.01, 0.2, 0.03],
            [0.02, 0.03, 0.3],
        ])
        assert np.allclose(link.inertia, expected)

    def test_inertial_defaults(self, tmp_path):
        """Inertial element present but missing mass/inertia children."""
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="lite">
                <inertial></inertial>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["lite"]
        assert link.mass == 0.0
        assert np.array_equal(link.inertia, np.eye(3))

    def test_semantic_tags(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="gripper">
                <semantic>
                    <tag name="end_effector"/>
                    <tag name="tool"/>
                </semantic>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["gripper"]
        assert link.semantic_tags == ["end_effector", "tool"]

    def test_grasp_points(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="gripper">
                <semantic>
                    <grasp_point name="top" x="0" y="0" z="0.1" approach="top"/>
                    <grasp_point name="side" x="0.05" y="0" z="0" approach="side"/>
                </semantic>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        link = parser.get_model().links["gripper"]
        assert len(link.grasp_points) == 2
        assert link.grasp_points[0]["name"] == "top"
        assert link.grasp_points[0]["position"] == [0.0, 0.0, 0.1]
        assert link.grasp_points[0]["approach"] == "top"
        assert link.grasp_points[1]["approach"] == "side"

    def test_grasp_point_defaults(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test">
            <link name="gripper">
                <semantic>
                    <grasp_point name="default"/>
                </semantic>
            </link>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        gp = parser.get_model().links["gripper"].grasp_points[0]
        assert gp["position"] == [0.0, 0.0, 0.0]
        assert gp["approach"] == "top"


class TestEURDFParserParseJoint:
    """Coverage for EURDFParser._parse_joint."""

    def test_origin_axis_limit_dynamics_safety_control(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0 0 1" rpy="0 0 0"/>
                <axis xyz="0 1 0"/>
                <limit lower="-1.57" upper="1.57" velocity="2.0" effort="10.0"/>
                <dynamics damping="0.1" friction="0.01"/>
                <safety_controller soft_lower_limit="-1.5"
                                   soft_upper_limit="1.5"
                                   k_position="100"
                                   k_velocity="10"/>
                <control p="50" i="5" d="5" imax="2"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        j = parser.get_model().joints["j1"]
        assert j.joint_type == "revolute"
        assert j.parent == "base"
        assert j.child == "link1"
        assert np.allclose(j.origin[:3, 3], [0, 0, 1])
        assert np.array_equal(j.axis, np.array([0, 1, 0]))
        assert j.limits == {"lower": -1.57, "upper": 1.57, "velocity": 2.0, "effort": 10.0}
        assert j.dynamics == {"damping": 0.1, "friction": 0.01}
        assert j.safety == {"lower": -1.5, "upper": 1.5, "k_position": 100.0, "k_velocity": 10.0}
        assert j.control == {"p": 50.0, "i": 5.0, "d": 5.0, "imax": 2.0}

    def test_safety_defaults_to_joint_limits(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <limit lower="-2.0" upper="2.0" velocity="1.0" effort="5.0"/>
                <safety_controller k_position="10" k_velocity="1"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        j = parser.get_model().joints["j1"]
        assert j.safety["lower"] == -2.0
        assert j.safety["upper"] == 2.0
        assert j.safety["k_position"] == 10.0
        assert j.safety["k_velocity"] == 1.0

    def test_control_defaults(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="arm">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <control p="200"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        j = parser.get_model().joints["j1"]
        assert j.control == {"p": 200.0, "i": 10.0, "d": 10.0, "imax": 1.0}

    def test_no_parent_child(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="arm">
            <joint name="j1" type="fixed"></joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        j = parser.get_model().joints["j1"]
        assert j.parent == ""
        assert j.child == ""


class TestEURDFParserParseSensor:
    """Coverage for EURDFParser._parse_sensor."""

    def test_sensor_with_origin_and_parameters(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <sensor name="cam1" type="camera" parent_link="head">
                <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
                <parameter name="width" value="640"/>
                <parameter name="height" value="480"/>
            </sensor>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        s = parser.get_model().sensors["cam1"]
        assert s.sensor_type == "camera"
        assert s.parent_link == "head"
        assert np.allclose(s.origin[:3, 3], [0.1, 0.2, 0.3])
        assert s.parameters == {"width": "640", "height": "480"}

    def test_sensor_defaults(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <sensor name="imu1" type="imu"/>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        s = parser.get_model().sensors["imu1"]
        assert s.parent_link == ""
        assert np.array_equal(s.origin, np.eye(4))
        assert s.parameters == {}


class TestEURDFParserParseMetadata:
    """Coverage for EURDFParser._parse_metadata."""

    def test_metadata_parsing(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <metadata>
                <author>Test Author</author>
                <version>1.0.0</version>
                <description>Test robot</description>
                <empty_tag></empty_tag>
            </metadata>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        meta = parser.get_model().metadata
        assert meta["author"] == "Test Author"
        assert meta["version"] == "1.0.0"
        assert meta["description"] == "Test robot"
        assert meta["empty_tag"] == ""

    def test_no_metadata(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text('<robot name="bot"></robot>')
        parser = EURDFParser(str(urdf))
        assert parser.get_model().metadata == {}


class TestEURDFParserParseOrigin:
    """Coverage for EURDFParser._parse_origin rotation matrix construction."""

    def test_origin_zero(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="fixed">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0 0 0" rpy="0 0 0"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        T = parser.get_model().joints["j1"].origin
        assert np.allclose(T, np.eye(4))

    def test_origin_translation_only(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="fixed">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="1 2 3" rpy="0 0 0"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        T = parser.get_model().joints["j1"].origin
        assert np.allclose(T[:3, 3], [1, 2, 3])
        assert np.allclose(T[:3, :3], np.eye(3))

    def test_origin_rotation_only(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="fixed">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0 0 0" rpy="1.5708 0 0"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        T = parser.get_model().joints["j1"].origin
        assert np.allclose(T[0, 0], 1.0, atol=1e-3)
        assert np.allclose(T[1, 1], 0.0, atol=1e-3)
        assert np.allclose(T[2, 2], 0.0, atol=1e-3)

    def test_origin_full_transform(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="bot">
            <link name="base"/>
            <link name="link1"/>
            <joint name="j1" type="fixed">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0.5 -0.2 1.0" rpy="0.1 0.2 0.3"/>
            </joint>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        T = parser.get_model().joints["j1"].origin
        assert np.allclose(T[:3, 3], [0.5, -0.2, 1.0])
        # Rotation matrix should be orthonormal
        R = T[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.allclose(np.linalg.det(R), 1.0, atol=1e-5)


class TestEURDFParserFullRobot:
    """End-to-end parser tests with complete e-URDF documents."""

    def test_full_eurdf_document(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text("""
        <robot name="test_arm">
            <link name="base">
                <visual><mesh filename="base.dae"/></visual>
                <collision><mesh filename="base_col.stl"/></collision>
                <inertial>
                    <mass value="5.0"/>
                    <inertia ixx="0.5" ixy="0" ixz="0" iyy="0.5" iyz="0" izz="0.5"/>
                </inertial>
            </link>
            <link name="link1">
                <semantic>
                    <tag name="arm_segment"/>
                </semantic>
            </link>
            <link name="gripper">
                <semantic>
                    <grasp_point name="default" x="0" y="0" z="0.05"/>
                </semantic>
            </link>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="link1"/>
                <origin xyz="0 0 0.5" rpy="0 0 0"/>
                <axis xyz="0 0 1"/>
                <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
            </joint>
            <joint name="j2" type="prismatic">
                <parent link="link1"/>
                <child link="gripper"/>
                <origin xyz="0 0 0.2" rpy="0 0 0"/>
                <axis xyz="1 0 0"/>
                <limit lower="0" upper="0.1" velocity="0.5" effort="5.0"/>
                <control p="100" i="10" d="10"/>
            </joint>
            <sensor name="wrist_cam" type="camera" parent_link="link1">
                <origin xyz="0.05 0 0" rpy="0 0 0"/>
                <parameter name="fov" value="60"/>
            </sensor>
            <metadata>
                <author>Test</author>
            </metadata>
        </robot>
        """)
        parser = EURDFParser(str(urdf))
        model = parser.get_model()
        assert model.name == "test_arm"
        assert len(model.links) == 3
        assert len(model.joints) == 2
        assert len(model.sensors) == 1
        assert model.metadata["author"] == "Test"
        # get_end_effector_link returns parent_links - child_links = {base}
        assert model.get_end_effector_link() == "base"
        assert model.get_actuated_joints() == ["j1", "j2"]

    def test_get_model_success(self, tmp_path):
        urdf = tmp_path / "robot.urdf"
        urdf.write_text('<robot name="r"></robot>')
        parser = EURDFParser(str(urdf))
        model = parser.get_model()
        assert isinstance(model, RobotModel)
        assert model.name == "r"

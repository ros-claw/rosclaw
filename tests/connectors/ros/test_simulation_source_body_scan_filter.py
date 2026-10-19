"""Source geometry and closed sensor routing; no ROS initialization or World."""

import copy
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance")
    )
    return importlib.import_module("simulation_source_body_scan_filter")


def source(size=".4 .2 .1", joint_type="fixed", sensor_frame="laser"):
    return f"""<robot name="independent_body">
      <link name="base_link"><collision><geometry><box size="{size}"/>
      </geometry></collision></link><link name="laser"/>
      <joint name="mount" type="{joint_type}"><parent link="base_link"/>
      <child link="laser"/><origin xyz=".03 0 .2"/></joint>
      <gazebo reference="laser"><sensor name="original_scan" type="gpu_lidar">
      <gz_frame_id>{sensor_frame}</gz_frame_id></sensor></gazebo></robot>""".encode()


def descriptor(urdf):
    return {
        "namespace": "/independent/robot",
        "scan_ros_topic": "/independent/robot/original_scan",
        "tf_topics_relative": True,
        "source_body_scan_filter": {
            "schema_version": "rosclaw.sdk_source_body_scan_filter.v1",
            "source_urdf_sha256": hashlib.sha256(urdf).hexdigest(),
            "base_frame": "base_link",
            "scan_frame": "laser",
        },
    }


def test_two_distinct_current_geometries_recompute_mask_without_padded_footprint(module):
    radii = []
    for size in (".4 .2 .1", ".8 .6 .1"):
        urdf = source(size)
        params, facts = module.source_body_filter_parameters(descriptor(urdf), urdf)
        radius = facts["geometry"]["physical_radius_m"]
        polygon = json.loads(params["filter1"]["params"]["polygon"])
        assert len(polygon) == 32
        assert all(math.isclose(math.hypot(*p), radius) for p in polygon)
        assert params["filter1"]["params"]["polygon_padding"] == 0
        assert params["filter1"]["params"]["invert"] is False
        assert facts["execution_authority"] == "NONE"
        radii.append(radius)
    assert radii[1] > 2 * radii[0]


def test_only_collision_monitor_sensor_input_is_remapped_without_changing_protection(module):
    d = descriptor(source())
    before = copy.deepcopy(d)
    assert module.source_body_scan_remappings(d, "collision_monitor") == [
        ("/independent/robot/original_scan", "/independent/robot/original_scan_body_filtered")
    ]
    for name in ("amcl", "controller_server", "velocity_smoother", "robot_state_publisher"):
        assert module.source_body_scan_remappings(d, name) == []
    assert d == before
    assert module.source_body_scan_remappings(None, "collision_monitor") == []
    assert module.source_body_scan_remappings({}, "collision_monitor") == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("polygon", "[[100,100]]"),
        ("radius", 100),
        ("padding", 1),
        ("source_urdf_sha256", "0" * 64),
        ("base_frame", "../base_link"),
        ("scan_frame", "/laser"),
        ("scan_frame", "base_link"),
        ("schema_version", "unrestricted"),
    ],
)
def test_unbound_source_user_geometry_and_unclosed_fields_rejected(module, key, value):
    urdf = source()
    d = descriptor(urdf)
    d["source_body_scan_filter"][key] = value
    with pytest.raises(ValueError):
        module.source_body_filter_parameters(d, urdf)


@pytest.mark.parametrize(
    "key,value",
    [
        ("namespace", "/other/robot"),
        ("namespace", "/independent/../robot"),
        ("scan_ros_topic", "/control/cmd_vel"),
        ("scan_ros_topic", "/independent/robot/scan/../../cmd_vel"),
        ("start_navigation_stack", False),
    ],
)
def test_wrong_scope_and_driver_only_mode_rejected(module, key, value):
    d = descriptor(source())
    d[key] = value
    with pytest.raises(ValueError):
        module.source_body_scan_remappings(d, "collision_monitor")


@pytest.mark.parametrize(
    "urdf",
    [source("0 .2 .1"), source(sensor_frame="unbound"), source(joint_type="continuous")],
)
def test_incomplete_geometry_or_missing_fixed_sensor_binding_abstains(module, urdf):
    with pytest.raises(ValueError):
        module.source_body_filter_parameters(descriptor(urdf), urdf)


def test_missing_original_mesh_cannot_use_default_or_declared_radius(module):
    urdf = source().replace(
        b'<box size=".4 .2 .1"/>', b'<mesh filename="package://unknown/a.stl"/>'
    )
    with pytest.raises(ValueError, match="collision geometry"):
        module.source_body_filter_parameters(descriptor(urdf), urdf)


def test_no_declaration_preserves_existing_stack_and_creates_no_extraction(module, tmp_path):
    assert (
        module.prepare_source_body_scan_filter({}, {}, tmp_path, tmp_path, tmp_path / "new") is None
    )
    assert list(tmp_path.iterdir()) == []


def test_default_nav2_launch_dependency_loads_without_a_core_install(module):
    code = """import importlib.abc, runpy, sys
class NoCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rosclaw' or fullname.startswith('rosclaw.'):
            raise ImportError('non-prepared Nav2 launch must not require Core')
sys.meta_path.insert(0, NoCore())
loaded = runpy.run_path(sys.argv[1])
assert loaded['source_body_scan_remappings'](None, 'collision_monitor') == []
assert not any(n == 'rosclaw' or n.startswith('rosclaw.') for n in sys.modules)
"""
    subprocess.run([sys.executable, "-I", "-c", code, module.__file__], check=True, timeout=10)


def test_unauthorized_environment_rejected_before_package_extraction(module, tmp_path, monkeypatch):
    monkeypatch.delenv("ROS_LOCALHOST_ONLY", raising=False)
    monkeypatch.setattr(
        module.subprocess, "run", lambda *_a, **_kw: pytest.fail("must not extract")
    )
    with pytest.raises(ValueError, match="isolated"):
        module.prepare_source_body_scan_filter(
            descriptor(source()), {}, tmp_path, tmp_path, tmp_path / "new"
        )
    assert list(tmp_path.iterdir()) == []


def test_wrong_package_size_rejected_before_hash_read_or_extraction(module, tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda p: True if str(p) == "/.dockerenv" else original_is_file(p)
    )
    package = tmp_path / "source-body-laser-filters.deb"
    package.write_bytes(b"wrong size")
    (tmp_path / "run_id.txt").write_text("owned-run")
    original_read = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda p: (
            pytest.fail("wrong-size package must not be read") if p == package else original_read(p)
        ),
    )
    monkeypatch.setattr(
        module.subprocess, "run", lambda *_a, **_kw: pytest.fail("must not extract")
    )
    with pytest.raises(ValueError, match="exact externally signature-verified"):
        module.prepare_source_body_scan_filter(
            descriptor(source()),
            {"generic_source_observer": True, "governed_sink": {}, "run_id": "owned-run"},
            tmp_path,
            tmp_path,
            tmp_path / "new",
        )


@pytest.mark.parametrize(
    "alteration", ["wrong_type", "wrong_direction", "duplicate", "output_taken"]
)
def test_control_or_ambiguous_bridge_cannot_become_filter_source(module, alteration):
    d = descriptor(source())
    bridge = [
        {
            "direction": "GZ_TO_ROS",
            "ros_topic_name": d["scan_ros_topic"],
            "gz_type_name": "gz.msgs.LaserScan",
            "ros_type_name": "sensor_msgs/msg/LaserScan",
        }
    ]
    module.check_original_scan_bridge(d, bridge)
    if alteration == "wrong_type":
        bridge[0]["ros_type_name"] = "geometry_msgs/msg/Twist"
    elif alteration == "wrong_direction":
        bridge[0]["direction"] = "BIDIRECTIONAL"
    elif alteration == "duplicate":
        bridge.append(copy.deepcopy(bridge[0]))
    else:
        bridge.append({"ros_topic_name": d["scan_ros_topic"] + "_body_filtered"})
    with pytest.raises(ValueError, match="unique original typed scan bridge"):
        module.check_original_scan_bridge(d, bridge)

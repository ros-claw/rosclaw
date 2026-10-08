"""Source instrumentation contracts; no third asset, transport or physics run."""

import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3] / "integrations/ros_probe/acceptance"
URDF = b'<robot name="anonymous"><link name="ground_base"/><link name="chassis"><collision name="body_shape"><geometry><box size=".4 .3 .2"/></geometry></collision><collision name="bumper"><geometry><sphere radius=".1"/></geometry></collision></link><joint name="fixed_base" type="fixed"><parent link="ground_base"/><child link="chassis"/></joint></robot>'
SDF = b'<sdf version="1.9"><model name="source_model"><link name="ground_base"/><link name="chassis"><collision name="bumper"><geometry><sphere><radius>.1</radius></sphere></geometry></collision><collision name="body_shape"><geometry><box><size>.4 .3 .2</size></box></geometry></collision><sensor name="front_range" type="gpu_lidar"><topic>/unseen/scan</topic></sensor></link><joint name="fixed_base" type="fixed"><parent>ground_base</parent><child>chassis</child></joint></model></sdf>'
BRIDGE = b"- ros_topic_name: /unseen/scan\n  gz_topic_name: /unseen/scan\n  ros_type_name: sensor_msgs/msg/LaserScan\n  gz_type_name: gz.msgs.LaserScan\n  direction: GZ_TO_ROS\n"
POLICY = {
    "source": "simulator_operator_fixture_policy",
    "approved": True,
    "evidence_domain": "SIMULATION",
    "world_name": "generic_world",
    "body_model_name": "anonymous_cleaner",
    "contact_prefix": "/unseen/contacts",
    "independent_pose_topic": "/unseen/actual_pose",
}


@pytest.fixture
def renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location(
        "generic_contact_test", ROOT / "generic_contact_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_named_collisions_get_one_sensor_each_without_geometry_or_urdf_changes(
    renderer, tmp_path
):
    out = tmp_path / "fixture"
    result = renderer.prepare_contact_fixture(
        out, urdf_bytes=URDF, sdf_bytes=SDF, bridge_bytes=BRIDGE, declaration=POLICY
    )
    assert result["physical_acceptance"] == "NOT_RUN"
    assert result["status"] == "PREPARED_NOT_SOURCE_ADMITTED"
    assert result["support_contact_topics"].startswith("UNKNOWN")
    assert (out / "robot.urdf").read_bytes() == URDF
    original = ET.fromstring(SDF).find("model/link[@name='chassis']")
    loaded = ET.parse(out / "robot.sdf").find("model/link[@name='chassis']")
    assert [ET.tostring(c) for c in original.findall("collision")] == [
        ET.tostring(c) for c in loaded.findall("collision")
    ]
    assert loaded.find("sensor[@name='front_range']").findtext("topic") == "/unseen/scan"
    streams = result["collision_streams"]
    assert [r["collision_name"] for r in streams] == ["body_shape", "bumper"]
    assert [r["collision_index"] for r in streams] == [0, 1]
    bridge = yaml.safe_load((out / "bridge.yaml").read_bytes())
    for row in streams:
        sensor = loaded.find("sensor[@name='" + row["sensor_name"] + "']")
        assert len(sensor.findall("contact/collision")) == 1
        assert sensor.findtext("contact/collision") == row["collision_name"]
        assert sensor.findtext("topic") == row["gz_topic"]
        assert any(
            b.get("ros_topic_name") == row["topic"] and b.get("gz_topic_name") == row["gz_topic"]
            for b in bridge
        )
    for name, record in result["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == record["sha256"]
    assert json.loads((out / "contact-fixture.json").read_bytes()) == result
    with pytest.raises(FileExistsError):
        renderer.prepare_contact_fixture(
            out, urdf_bytes=URDF, sdf_bytes=SDF, bridge_bytes=BRIDGE, declaration=POLICY
        )


@pytest.mark.parametrize(
    "fault",
    [
        "source_name",
        "dimensions",
        "offset",
        "rotation",
        "relative_frame",
        "pose_nan",
        "pose_duplicate",
        "pose_degrees",
        "mesh",
        "missing_link",
        "extra_link",
        "duplicate_link",
        "duplicate_collision",
        "unnamed_collision",
        "missing_collision",
        "existing_sensor",
        "existing_contact",
        "existing_pose",
        "topic_alias",
        "truth_alias",
        "bridge_type",
        "entity",
        "utf16",
        "nan",
        "zero",
        "model_nested",
        "unresolved",
        "unapproved",
        "real",
        "prefix",
        "policy_extra",
    ],
)
def test_unknown_or_conflicting_sources_refuse_before_creating_fixture(renderer, tmp_path, fault):
    urdf, sdf, bridge, policy = URDF, SDF, BRIDGE, POLICY.copy()
    if fault == "source_name":
        sdf = sdf.replace(b"body_shape", b"changed_shape")
    elif fault == "dimensions":
        sdf = sdf.replace(b".4 .3 .2", b".5 .3 .2")
    elif fault in {
        "offset",
        "rotation",
        "relative_frame",
        "pose_nan",
        "pose_duplicate",
        "pose_degrees",
    }:
        pose = {
            "offset": b"<pose>.1 0 0 0 0 0</pose>",
            "rotation": b"<pose>0 0 0 0 0 1</pose>",
            "relative_frame": b'<pose relative_to="unobserved">0 0 0 0 0 0</pose>',
            "pose_nan": b"<pose>nan 0 0 0 0 0</pose>",
            "pose_duplicate": b"<pose>0 0 0 0 0 0</pose><pose>0 0 0 0 0 0</pose>",
            "pose_degrees": b'<pose degrees="true">0 0 0 0 0 0</pose>',
        }[fault]
        sdf = sdf.replace(b'<collision name="body_shape">', b'<collision name="body_shape">' + pose)
    elif fault == "mesh":
        urdf = urdf.replace(b'<box size=".4 .3 .2"/>', b'<mesh filename="unknown.stl"/>')
    elif fault == "missing_link":
        sdf = sdf.replace(b'<link name="ground_base"/>', b"")
    elif fault == "extra_link":
        sdf = sdf.replace(b"</model>", b'<link name="extra"/></model>')
    elif fault == "duplicate_link":
        sdf = sdf.replace(b"</model>", b'<link name="ground_base"/></model>')
    elif fault == "duplicate_collision":
        urdf = urdf.replace(b'name="bumper"', b'name="body_shape"')
    elif fault == "unnamed_collision":
        urdf = urdf.replace(b' name="bumper"', b"")
    elif fault == "missing_collision":
        sdf = sdf.replace(
            b'<collision name="bumper"><geometry><sphere><radius>.1</radius></sphere></geometry></collision>',
            b"",
        )
    elif fault == "existing_sensor":
        sdf = sdf.replace(b"front_range", b"rosclaw_contact_0")
    elif fault == "existing_contact":
        sdf = sdf.replace(b'type="gpu_lidar"', b'type="contact"')
    elif fault == "existing_pose":
        sdf = sdf.replace(
            b"</model>", b'<plugin name="gz::sim::systems::PosePublisher" filename="pose"/></model>'
        )
    elif fault == "topic_alias":
        bridge = bridge.replace(b"/unseen/scan", b"/unseen/contacts/chassis/collision_0")
    elif fault == "truth_alias":
        bridge = bridge.replace(b"/unseen/scan", b"/unseen/actual_pose")
    elif fault == "bridge_type":
        bridge = b"{}"
    elif fault == "entity":
        urdf = b'<!DOCTYPE robot [<!ENTITY secret SYSTEM "file:///private">]>' + urdf
    elif fault == "utf16":
        urdf = urdf.decode().encode("utf-16")
    elif fault == "nan":
        urdf = urdf.replace(b".4 .3 .2", b"nan .3 .2")
    elif fault == "zero":
        urdf = urdf.replace(b".4 .3 .2", b"0 .3 .2")
    elif fault == "model_nested":
        sdf = sdf.replace(b"</model>", b'<model name="nested"/></model>')
    elif fault == "unresolved":
        sdf = sdf.replace(b"</model>", b"<include><uri>unresolved</uri></include></model>")
    elif fault == "unapproved":
        policy["approved"] = False
    elif fault == "real":
        policy["evidence_domain"] = "REAL"
    elif fault == "prefix":
        policy["contact_prefix"] = "relative"
    else:
        policy["extra"] = "unknown"
    out = tmp_path / "fixture"
    with pytest.raises((ValueError, ET.ParseError)):
        renderer.prepare_contact_fixture(
            out, urdf_bytes=urdf, sdf_bytes=sdf, bridge_bytes=bridge, declaration=policy
        )
    assert not out.exists()


def test_same_nonzero_collision_origin_and_cylinder_are_preserved(renderer, tmp_path):
    urdf = URDF.replace(
        b'<collision name="body_shape">',
        b'<collision name="body_shape"><origin xyz=".1 -.1 .2" rpy="0 0 1"/>',
    ).replace(b'<box size=".4 .3 .2"/>', b'<cylinder radius=".2" length=".3"/>')
    sdf = SDF.replace(
        b'<collision name="body_shape">',
        b'<collision name="body_shape"><pose relative_to="chassis">.1 -.1 .2 0 0 1</pose>',
    ).replace(
        b"<box><size>.4 .3 .2</size></box>",
        b"<cylinder><radius>.2</radius><length>.3</length></cylinder>",
    )
    result = renderer.prepare_contact_fixture(
        tmp_path / "fixture",
        urdf_bytes=urdf,
        sdf_bytes=sdf,
        bridge_bytes=BRIDGE,
        declaration=POLICY,
    )
    assert result["physical_acceptance"] == "NOT_RUN"
    loaded = ET.parse(tmp_path / "fixture/robot.sdf").find(
        "model/link[@name='chassis']/collision[@name='body_shape']"
    )
    assert loaded.findtext("pose") == ".1 -.1 .2 0 0 1"
    assert loaded.findtext("geometry/cylinder/radius") == ".2"

"""Source joining on synthetic primitives and actual installed templates only."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import yaml

from tests.connectors.ros import test_sim_controller_source as controller
from tests.connectors.ros import test_sim_navigation_source as navigation

ROOT = Path(__file__).resolve().parents[3]


def synthetic_stack_inputs():
    urdf, sdf = ET.fromstring(controller.URDF), ET.fromstring(controller.SDF)
    for model, is_urdf in ((urdf, True), (sdf.find("model"), False)):
        collision = ET.SubElement(model.find("link"), "collision", name="source_body")
        geom = ET.SubElement(collision, "geometry")
        if is_urdf:
            ET.SubElement(geom, "box", size=".4 .3 .2")
        else:
            ET.SubElement(ET.SubElement(geom, "box"), "size").text = ".4 .3 .2"
        link = ET.SubElement(model, "link", name="range_frame")
        joint = ET.SubElement(model, "joint", name="range_mount", type="fixed")
        if is_urdf:
            ET.SubElement(joint, "parent", link="platform")
            ET.SubElement(joint, "child", link="range_frame")
        else:
            ET.SubElement(joint, "parent").text = "platform"
            ET.SubElement(joint, "child").text = "range_frame"
            sensor = ET.SubElement(link, "sensor", name="actual_source_lidar", type="gpu_lidar")
            ET.SubElement(sensor, "topic").text = "/declared_gz_lidar"
            lidar = ET.SubElement(sensor, "lidar")
            horizontal = ET.SubElement(ET.SubElement(lidar, "scan"), "horizontal")
            for key, value in {
                "samples": "360",
                "resolution": "1",
                "min_angle": "-3.14",
                "max_angle": "3.14",
            }.items():
                ET.SubElement(horizontal, key).text = value
            range_ = ET.SubElement(lidar, "range")
            for key, value in {"min": ".1", "max": "10", "resolution": ".01"}.items():
                ET.SubElement(range_, key).text = value
    ctrl, nav = deepcopy(controller.POLICY), deepcopy(navigation.POLICY)
    ctrl["controller_parameters_path"] = "/evidence/controller_params.yaml"
    nav["topics"]["odom"] = ctrl["odom_topic"]
    contact = {
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "world_name": "explicit_world",
        "body_model_name": ctrl["body_model_name"],
        "contact_prefix": "/explicit_contacts",
        "independent_pose_topic": "/explicit_truth",
    }
    map_source = {
        "image": "source_map.pgm",
        "resolution": 0.05,
        "origin": [-2.0, -1.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.25,
    }
    return {
        "urdf_bytes": ET.tostring(urdf),
        "sdf_bytes": ET.tostring(sdf),
        "world_bytes": b'<sdf version="1.9"><world name="explicit_world"><model name="source_ground"><static>true</static><link name="ground"><collision name="surface"><geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry></collision></link></model></world></sdf>',
        "bridge_bytes": yaml.safe_dump(
            [
                {
                    "ros_topic_name": nav["topics"]["lidar"],
                    "gz_topic_name": "/declared_gz_lidar",
                    "ros_type_name": "sensor_msgs/msg/LaserScan",
                    "gz_type_name": "gz.msgs.LaserScan",
                    "direction": "GZ_TO_ROS",
                }
            ]
        ).encode(),
        "map_yaml_bytes": yaml.safe_dump(map_source).encode(),
        "map_image_bytes": b"P5\n2 2\n255\n" + bytes([254, 254, 254, 254]),
        "nav2_bytes": (navigation.ROOT / "nav2_params.sdk.yaml").read_bytes(),
        "coverage_bytes": (navigation.ROOT / "coverage_params.sdk.yaml").read_bytes(),
        "attachment": deepcopy(navigation.ATTACHMENT),
        "controller_declaration": ctrl,
        "navigation_declaration": nav,
        "contact_declaration": contact,
    }


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "integrations/ros_probe/acceptance"))
    from generic_stack_source import prepare_generic_stack_source

    return prepare_generic_stack_source


def test_join_preserves_actual_world_map_and_original_assets_and_seals_generated_files(
    generator, tmp_path
):
    data = synthetic_stack_inputs()
    original = deepcopy(data)
    out = tmp_path / "stack"
    manifest = generator(out, **data)
    assert data == original
    assert (out / "world.sdf").read_bytes() == data["world_bytes"]
    assert (out / "map.yaml").read_bytes() == data["map_yaml_bytes"]
    assert (out / "source_map.pgm").read_bytes() == data["map_image_bytes"]
    params = yaml.safe_load((out / "controller_params.yaml").read_bytes())
    assert (
        params[data["controller_declaration"]["drive_controller"]]["ros__parameters"][
            "cmd_vel_timeout"
        ]
        == 0.2
    )
    for name, sha in manifest["output_hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    assert manifest["physical_acceptance"] == "NOT_RUN" and manifest["authorization"] is False
    assert manifest["heldout_asset"] == "NOT_SELECTED"
    assert manifest["map_pixels_and_dimensions"] == "REQUIRES_ACTUAL_SDK_DECODER_AND_LIVE_MAP"
    robot = ET.fromstring((out / "robot.sdf").read_bytes())
    assert robot.find("model/link/collision/geometry/box/size").text == ".4 .3 .2"
    assert len(robot.findall("model/link/sensor[@type='contact']")) == 1
    assert (
        json.loads((out / "navigation-source-report.json").read_bytes())[
            "requires_actual_Graph_TF_Body_source_admission"
        ]
        is True
    )


def test_prepared_handoff_captures_verified_bytes_without_live_admission(generator, tmp_path):
    from generic_stack_source import read_prepared_generic_stack

    out = tmp_path / "stack"
    manifest = generator(out, **synthetic_stack_inputs())
    verified = read_prepared_generic_stack(out)
    assert verified["manifest"] == manifest
    assert verified["captured_files"]["world.sdf"] == (out / "world.sdf").read_bytes()
    assert verified["live_admission"] is False and verified["authorization"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "modified",
        "missing",
        "escape",
        "parent_link",
        "file_link",
        "authority",
        "omitted_original",
        "omitted_map_image",
    ],
)
def test_prepared_handoff_refuses_unreviewed_or_escaped_sources(generator, tmp_path, fault):
    from generic_stack_source import read_prepared_generic_stack

    from rosclaw.connectors.ros.diagnosis.coverage_audit import digest

    out = tmp_path / "stack"
    manifest = generator(out, **synthetic_stack_inputs())
    if fault == "modified":
        (out / "world.sdf").write_text("unreviewed replacement")
    elif fault == "missing":
        (out / "world.sdf").unlink()
    elif fault == "escape":
        manifest["output_hashes"]["../outside"] = "0" * 64
    elif fault == "parent_link":
        (out / "original-sources").rename(tmp_path / "external-sources")
        (out / "original-sources").symlink_to(tmp_path / "external-sources")
    elif fault == "file_link":
        (out / "world.sdf").rename(tmp_path / "external-world.sdf")
        (out / "world.sdf").symlink_to(tmp_path / "external-world.sdf")
    elif fault == "authority":
        manifest["authorization"] = True
    elif fault == "omitted_original":
        del manifest["output_hashes"]["original-sources/robot.original.urdf"]
    else:
        del manifest["output_hashes"]["source_map.pgm"]
    manifest["artifact_hash"] = digest({k: v for k, v in manifest.items() if k != "artifact_hash"})
    (out / "generic-stack-source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, OSError)):
        read_prepared_generic_stack(out)


@pytest.mark.parametrize(
    "fault",
    [
        "world_identity",
        "world_include",
        "body_alias",
        "odom_alias",
        "lidar_bridge",
        "lidar_sensor",
        "map_escape",
        "map_rotated",
        "map_resolution",
        "map_negate",
        "map_thresholds",
        "wrong_path",
    ],
)
def test_unknown_or_conflicting_sources_refuse_before_writing_stack(generator, tmp_path, fault):
    data = synthetic_stack_inputs()
    if fault == "world_identity":
        data["contact_declaration"]["world_name"] = "foreign_world"
    elif fault == "world_include":
        data["world_bytes"] = data["world_bytes"].replace(
            b"</world>", b"<include><uri>model://external</uri></include></world>"
        )
    elif fault == "body_alias":
        data["contact_declaration"]["body_model_name"] = "source_ground"
    elif fault == "odom_alias":
        data["navigation_declaration"]["topics"]["odom"] = "/other_odom"
    elif fault == "lidar_bridge":
        data["bridge_bytes"] = b"[]"
    elif fault == "lidar_sensor":
        data["sdf_bytes"] = data["sdf_bytes"].replace(b"/declared_gz_lidar", b"/foreign")
    elif fault == "wrong_path":
        data["controller_declaration"]["controller_parameters_path"] = "/evidence/other.yaml"
    else:
        map_ = yaml.safe_load(data["map_yaml_bytes"])
        if fault == "map_escape":
            map_["image"] = "../outside.pgm"
        elif fault == "map_rotated":
            map_["origin"][2] = 0.2
        elif fault == "map_resolution":
            map_["resolution"] = 0.1
        elif fault == "map_negate":
            map_["negate"] = True
        else:
            map_["occupied_thresh"] = map_["free_thresh"]
        data["map_yaml_bytes"] = yaml.safe_dump(map_).encode()
    out = tmp_path / "refused"
    with pytest.raises(ValueError):
        generator(out, **data)
    assert not out.exists()


def test_existing_workspace_is_preserved(generator, tmp_path):
    out = tmp_path / "protected"
    out.mkdir()
    (out / "keep").write_bytes(b"original")
    with pytest.raises(FileExistsError):
        generator(out, **synthetic_stack_inputs())
    assert (out / "keep").read_bytes() == b"original"


def test_joined_source_retains_explicit_operator_initialization_prior(generator, tmp_path):
    data = synthetic_stack_inputs()
    data["localization_initialization_declaration"] = {
        "schema_version": "rosclaw.sim_localization_initial_prior.v1",
        "source": "simulator_operator_fixture_policy",
        "approved": True,
        "evidence_domain": "SIMULATION",
        "source_pose_kind": "OPERATOR_FROZEN_SPAWN_PRIOR",
        "world_name": data["contact_declaration"]["world_name"],
        "map_frame": data["navigation_declaration"]["frames"]["map"],
        "world_to_map_xyyaw": [0, 0, 0],
    }
    out = tmp_path / "explicit-prior"
    manifest = generator(out, **data)
    config = yaml.safe_load((out / "nav2.yaml").read_bytes())
    node = data["navigation_declaration"]["nodes"]["localization"]
    assert config[node]["ros__parameters"]["set_initial_pose"] is True
    assert "localization-initialization-declaration.json" in manifest["output_hashes"]
    assert (
        json.loads((out / "localization-initialization-declaration.json").read_bytes())
        == data["localization_initialization_declaration"]
    )
    assert manifest["physical_acceptance"] == "NOT_RUN"
    assert manifest["authorization"] is False

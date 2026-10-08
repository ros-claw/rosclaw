"""Source assembly with explicit synthetic robot/ELF; no World or transport."""

import hashlib
import importlib
import json
from xml.etree import ElementTree as ET

import pytest
import yaml

from tests.connectors.ros import test_backend_probe_world as world_contract
from tests.connectors.ros.test_backend_probe_fixture import declaration

world_fixture = world_contract.fixture


@pytest.fixture
def bundle(world_fixture):
    _, scene, _, _, library, out = world_fixture
    (scene / "robot.sdf").write_text("""<sdf version="1.9"><model name="active_robot">
    <link name="base"><collision name="body"><geometry><box><size>.2 .2 .1</size></box></geometry></collision>
    <sensor name="touch" type="contact"><topic>/robot/contact</topic><contact><collision>body</collision><topic>/qualified/robot/contact</topic></contact></sensor></link>
    <plugin name="gz::sim::systems::PosePublisher"><topic>/robot/pose</topic><publish_model_pose>true</publish_model_pose><use_pose_vector_msg>true</use_pose_vector_msg></plugin>
    </model></sdf>""")
    (scene / "brush_binding.json").write_text(
        json.dumps(
            {
                "run_id": "synthetic_run",
                "body_snapshot_hash": "synthetic_robot_hash",
                "attachment_hash": "synthetic_brush_hash",
                "producer_id": "synthetic_actor_not_live",
            }
        )
    )
    bridge = [
        {
            "ros_topic_name": "/robot/pose",
            "gz_topic_name": "/robot/pose",
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "gz_type_name": "gz.msgs.Pose_V",
            "direction": "GZ_TO_ROS",
        },
        {
            "ros_topic_name": "/robot/contact",
            "gz_topic_name": "/qualified/robot/contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        },
    ]
    (scene / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    module = importlib.import_module("backend_world_bundle")
    options = {
        "scene_directory": scene,
        "declaration": declaration(),
        "contact_library": library,
        "robot_support_topics": ["/robot/contact"],
        "robot_ground_collisions": ["ground::link::collision"],
        "robot_pose_topic": "/robot/pose",
        "robot_pose_frame": "synthetic_world",
        "probe_pose_frame": "synthetic_world",
    }
    return module, options, out


def test_assembles_disjoint_all_step_robot_and_sampled_probe_without_runtime_admission(bundle):
    module, options, output = bundle
    source = options["scene_directory"]
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    result = module.prepare_backend_world(output, **options)
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    world = ET.parse(output / "world-source/world.sdf").find("world")
    contacts = [p for p in world.findall("plugin") if p.get("name") == "rosclaw::PassiveContacts"]
    assert {(p.findtext("body_model_name"), p.findtext("topic")) for p in contacts} == {
        ("active_robot", "/rosclaw_sim/contact_components"),
        ("isolated_instrument", "/rosclaw_sim/backend_probe_components"),
    }
    assert len(contacts) == 2
    policy = json.loads((output / "robot-source/native-policy.json").read_text())
    assert policy["sampling_semantics"] == "ALL_POSTUPDATE_PHYSICS_STEPS"
    config = json.loads((output / "observer-source/backend_actor_constraint.json").read_text())
    assert config["constraint_policy_hash"] == result["constraint_policy_hash"]
    for path, sha in result["output_hashes"].items():
        assert hashlib.sha256((output / path).read_bytes()).hexdigest() == sha
    assert not result["actual_world_source_ownership_admitted"]
    assert not result["backend_health_admitted"] and not result["authorization"]
    assert result["qualified_Native_launcher"] == "NOT_JOINED"
    with pytest.raises(FileExistsError):
        module.prepare_backend_world(output, **options)


@pytest.mark.parametrize(
    "fault",
    [
        "run",
        "body",
        "brush",
        "region",
        "producer_alias",
        "library",
        "bridge_alias",
        "support",
        "frame",
    ],
)
def test_wrong_source_or_role_is_refused_and_originals_remain_reviewable(bundle, fault):
    module, options, output = bundle
    scene = options["scene_directory"]
    if fault in {"run", "body", "region"}:
        key = {"run": "run_id", "body": "robot_model_name", "region": "cleaning_polygon"}[fault]
        options["declaration"][key] = (
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]
            if fault == "region"
            else "other_source"
        )
    elif fault == "brush":
        p = scene / "brush_binding.json"
        data = json.loads(p.read_text())
        data["attachment_hash"] = "other_brush"
        p.write_text(json.dumps(data))
    elif fault == "producer_alias":
        p = scene / "world.sdf"
        tree = ET.parse(p)
        ET.SubElement(
            tree.find("world"), "plugin", name="rosclaw::PassiveContacts", filename="not_loaded"
        )
        tree.write(p)
    elif fault == "library":
        options["contact_library"].write_bytes(b"not an ELF")
    elif fault == "bridge_alias":
        p = scene / "bridge.yaml"
        bridge = yaml.safe_load(p.read_text())
        bridge.append(
            {"ros_topic_name": "/rosclaw_sim/contact_components", "gz_topic_name": "/wrong"}
        )
        p.write_text(yaml.safe_dump(bridge))
    elif fault == "support":
        options["robot_support_topics"] = ["/unobserved_support"]
    else:
        options["robot_pose_frame"] = ""
    before = {p.name: p.read_bytes() for p in scene.iterdir()}
    with pytest.raises(ValueError):
        module.prepare_backend_world(output, **options)
    assert before == {p.name: p.read_bytes() for p in scene.iterdir()}
    assert not (output / "backend-world-bundle.json").exists()


def test_separate_pose_bridge_and_shorthand_observations_preserve_declared_roles(bundle):
    module, options, output = bundle
    scene = options["scene_directory"]
    bridge = yaml.safe_load((scene / "bridge.yaml").read_text())
    (scene / "truth_bridge.yaml").write_text(yaml.safe_dump([bridge.pop(0)]))
    bridge.append(
        {
            "topic_name": "renamed_sensor",
            "ros_type_name": "sensor_msgs/msg/Imu",
            "gz_type_name": "gz.msgs.IMU",
            "direction": "GZ_TO_ROS",
        }
    )
    (scene / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    result = module.prepare_backend_world(output, **options)
    rows = yaml.safe_load((output / "robot-source/bridge.yaml").read_text())
    row = next(r for r in rows if r["ros_topic_name"] == "renamed_sensor")
    assert row["gz_topic_name"] == "renamed_sensor" and "topic_name" not in row
    assert "truth_bridge.yaml" in result["source_hashes"]
    assert sum(r["ros_topic_name"] == "/robot/pose" for r in rows) == 1


@pytest.mark.parametrize(
    "fault", ["control_direction", "service", "mixed_shorthand", "missing_type", "relative_alias"]
)
def test_ambiguous_or_control_bridge_never_becomes_observation_bundle(bundle, fault):
    module, options, output = bundle
    path = options["scene_directory"] / "bridge.yaml"
    bridge = yaml.safe_load(path.read_text())
    if fault == "control_direction":
        bridge[0]["direction"] = "BIDIRECTIONAL"
    elif fault == "service":
        bridge[0]["service_name"] = "/world/set_pose"
    elif fault == "mixed_shorthand":
        bridge[0]["topic_name"] = "/robot/pose"
    elif fault == "missing_type":
        bridge[0].pop("ros_type_name")
    else:
        copy = bridge[0].copy()
        copy["ros_topic_name"] = "robot/pose"
        copy["gz_topic_name"] = "robot/pose"
        bridge.append(copy)
    path.write_text(yaml.safe_dump(bridge))
    with pytest.raises(ValueError):
        module.prepare_backend_world(output, **options)
    assert not output.exists()


def test_duplicate_visual_names_repaired_only_in_candidate_nonvisual_bytes_identical(bundle):
    module, options, output = bundle
    path = options["scene_directory"] / "world.sdf"
    tree = ET.parse(path)
    link = tree.find("world/model/link")
    ET.SubElement(link, "visual", name="collision")
    tree.write(path)
    raw = path.read_bytes()
    result = module.prepare_backend_world(output, **options)
    assert result["candidate_visual_name_repairs"] == [
        {
            "link": "link",
            "original": "collision",
            "candidate": "rosclaw_observer_visual_0_collision",
        }
    ]
    assert len(result["nonvisual_world_source_sha256_before_and_after_visual_repair"]) == 64
    assert path.read_bytes() == raw
    assert (
        ET.parse(output / "world-source/world.sdf").find("world/model/link/collision").get("name")
        == "collision"
    )


def test_ambiguous_visual_frame_references_remain_unsupported(bundle):
    module, options, output = bundle
    path = options["scene_directory"] / "world.sdf"
    tree = ET.parse(path)
    link = tree.find("world/model/link")
    ET.SubElement(link, "visual", name="collision")
    ET.SubElement(
        tree.find("world"),
        "frame",
        name="ambiguous_visual_reference",
        attached_to="ground::link::collision",
    )
    tree.write(path)
    with pytest.raises(ValueError, match="ambiguous visual frame reference"):
        module.prepare_backend_world(output, **options)
    assert not (output / "backend-world-bundle.json").exists()

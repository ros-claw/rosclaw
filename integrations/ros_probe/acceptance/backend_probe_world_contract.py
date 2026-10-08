"""Owned source-world rendering and installed SDF parser contract, not physics."""

import argparse
import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from backend_probe_cdr_contract import fixture
from backend_probe_world import prepare_probe_world


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--plugin", required=True, type=Path)
    args = parser.parse_args()
    args.directory.mkdir()
    instrument = args.directory / "instrument"
    policy, _ = fixture(instrument, args.plugin)
    scene = args.directory / "scene"
    scene.mkdir()
    sdf = ET.Element("sdf", version="1.9")
    world = ET.SubElement(sdf, "world", name="fixture_world")
    for name, filename in (
        ("gz::sim::systems::Physics", "gz-sim-physics-system"),
        ("gz::sim::systems::Contact", "gz-sim-contact-system"),
        ("gz::sim::systems::UserCommands", "gz-sim-user-commands-system"),
    ):
        ET.SubElement(world, "plugin", name=name, filename=filename)
    body = "separate_synthetic_robot"
    binding = {
        "run_id": "synthetic_probe_cdr_run",
        "world_name": "fixture_world",
        "body_model_name": body,
        "body_snapshot_hash": "synthetic_robot_body_not_admitted",
        "world_to_map_xyyaw": [0, 0, 0],
        "frame_transform_source": "simulator_operator_fixture_policy",
        "map_world_identity_approved": True,
        "attachment_hash": "synthetic_robot_attachment_not_admitted",
        "grid": {"cleaning_polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]]},
        "scene_model_names": ["support_plane", "parked_box", body],
        "obstacle_names": ["parked_box"],
    }
    passive = ET.SubElement(
        world, "plugin", name="rosclaw::PassivePhysics", filename="librosclaw_passive_physics.so"
    )
    for key in ("run_id", "body_snapshot_hash", "attachment_hash", "body_model_name"):
        ET.SubElement(passive, key).text = binding[key]
    ET.SubElement(passive, "static_model").text = "support_plane"
    ET.SubElement(passive, "obstacle_model").text = "parked_box"
    ground = ET.SubElement(world, "model", name="support_plane")
    ET.SubElement(ground, "static").text = "true"
    link = ET.SubElement(ground, "link", name="floor_link")
    collision = ET.SubElement(link, "collision", name="floor_collision")
    geometry = ET.SubElement(collision, "geometry")
    plane = ET.SubElement(geometry, "plane")
    ET.SubElement(plane, "normal").text = "0 0 1"
    ET.SubElement(plane, "size").text = "20 20"
    box = ET.SubElement(world, "model", name="parked_box")
    ET.SubElement(box, "static").text = "true"
    ET.SubElement(box, "pose").text = "4 0 .1 0 0 0"
    link = ET.SubElement(box, "link", name="box_link")
    geometry = ET.SubElement(ET.SubElement(link, "collision", name="box_collision"), "geometry")
    ET.SubElement(ET.SubElement(geometry, "box"), "size").text = ".2 .2 .2"
    ET.ElementTree(sdf).write(scene / "world.sdf")
    (scene / "physics_binding.json").write_text(json.dumps(binding))
    (scene / "bridge.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "ros_topic_name": "/robot/physics_snapshot",
                    "gz_topic_name": "/rosclaw_sim/physics_snapshot",
                    "ros_type_name": "std_msgs/msg/String",
                    "gz_type_name": "gz.msgs.StringMsg",
                    "direction": "GZ_TO_ROS",
                }
            ]
        )
    )
    before = {p.name: p.read_bytes() for p in scene.iterdir()}
    result = prepare_probe_world(
        args.directory / "candidate",
        scene_directory=scene,
        instrument_directory=instrument,
        policy=policy,
        plugin_path=instrument / "libcontacts.so",
    )
    assert {p.name: p.read_bytes() for p in scene.iterdir()} == before
    checked = subprocess.run(
        ["gz", "sdf", "-k", str(args.directory / "candidate/world.sdf")],
        capture_output=True,
        timeout=15,
    )
    (args.directory / "sdf-parser-stdout.txt").write_bytes(checked.stdout)
    (args.directory / "sdf-parser-stderr.txt").write_bytes(checked.stderr)
    if checked.returncode != 0 or b"Valid." not in checked.stdout:
        raise ValueError("owned source-world candidate rejected by installed SDF parser")
    summary = {
        "status": "PASS_OFFLINE_SOURCE_WORLD_AND_SDF_PARSER",
        "result": result,
        "source_scene_unchanged": True,
        "sdf_parser_returncode": checked.returncode,
        "fixture_models": "SYNTHETIC_INSTRUMENT_AND_SCENE_NOT_THIRD_ROBOT_ASSET",
        "gazebo_world_started": False,
        "node_or_dds_started": False,
        "scene_service_executed": False,
        "backend_health_admitted": False,
        "physical_acceptance": "NOT_RUN",
    }
    (args.directory / "contract-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

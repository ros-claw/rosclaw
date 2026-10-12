"""Operator-owned known-fixture staging; no goals, velocities or live edits.

All repairs are confined to a fresh staging tree. Collision dimensions are
never copied from a vendor SDF: physics starts from exact URDF conversion.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import yaml

from rosclaw.connectors.ros.context.geometry import IDENTITY, ZERO, _compose, _origin
from rosclaw.connectors.ros.context.navigation_workspace_review import _json
from rosclaw.connectors.ros.context.sim_source_binding import sdf_inventory


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_prepared_assets(root, expected_hash, output):
    """Require an external inventory pin and an actual read-only asset mount."""
    root, output = Path(root), Path(output)
    if root.is_symlink() or not os.statvfs(root).f_flag & os.ST_RDONLY:
        raise ValueError("prepared assets must be independently mounted read-only")
    manifest = root / "prepared-assets.json"
    if manifest.is_symlink() or digest(manifest) != expected_hash:
        raise ValueError("operator-pinned prepared asset inventory required")
    inventory = _json(manifest.read_bytes())
    actual = {
        str(p.relative_to(root)): digest(p)
        for p in root.rglob("*")
        if p.is_file() and p != manifest
    }
    if any(p.is_symlink() for p in root.rglob("*")) or actual != inventory["file_sha256"]:
        raise ValueError("prepared assets changed")
    output.mkdir(exist_ok=True)
    for p in root.iterdir():
        if p.is_file() and p != manifest:
            target = output / p.name
            if target.exists() or target.is_symlink():
                raise ValueError("fresh fixture output required")
            target.symlink_to(p)


def frame_pose(urdf, frame, root):
    joints = {j.find("child").get("link"): j for j in urdf.findall("joint")}
    chain, seen = [], set()
    while frame != root:
        if frame in seen or frame not in joints:
            raise ValueError("unique fixed sensor mount path required")
        seen.add(frame)
        joint = joints[frame]
        if joint.get("type") != "fixed":
            raise ValueError("moving sensor mount needs explicit supported kinematics")
        chain.append(_origin(joint))
        frame = joint.find("parent").get("link")
    transform = (IDENTITY, ZERO)
    for item in reversed(chain):
        transform = _compose(transform, item)
    r, p = transform
    angles = (
        math.atan2(r[2][1], r[2][2]),
        math.asin(max(-1.0, min(1.0, -r[2][0]))),
        math.atan2(r[1][0], r[0][0]),
    )
    return " ".join(format(x, ".17g") for x in (*p, *angles))


def stage_visual_resources(urdf, output, *, repair_missing):
    from ament_index_python.packages import get_package_share_directory

    repairs, copied = [], set()

    def copy_resource(path, package, share):
        path = path.resolve(strict=True)
        relative = path.relative_to(share)
        dest = output / "resources" / package / relative
        if path not in copied:
            copied.add(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            if path.suffix.lower() == ".dae":
                for node in ET.parse(path).getroot().iter():
                    if (
                        node.tag.rsplit("}", 1)[-1] == "init_from"
                        and node.text
                        and not node.text.startswith("#")
                    ):
                        reference = node.text.strip()
                        if "://" in reference or Path(reference).is_absolute():
                            raise ValueError("unresolved external visual resource")
                        copy_resource(path.parent / reference, package, share)
        return "/assets/" + str(dest.relative_to(output))

    for mesh in urdf.findall("./link/visual/geometry/mesh"):
        original = mesh.get("filename", "")
        parsed = urlparse(original)
        if parsed.scheme != "package" or not parsed.netloc:
            raise ValueError("explicit ROS-package visual resources required")
        share = Path(get_package_share_directory(parsed.netloc)).resolve(strict=True)
        path = share / parsed.path.lstrip("/")
        if not path.is_file():
            candidates = list(share.rglob(Path(parsed.path).name))
            if not repair_missing or len(candidates) != 1 or not candidates[0].is_file():
                raise ValueError("unresolved or ambiguous visual URI: " + original)
            path = candidates[0]
        mesh.set("filename", copy_resource(path, parsed.netloc, share))
        repairs.append(
            {
                "original_uri": original,
                "staged_uri": mesh.get("filename"),
                "image_source_path": str(path),
                "sha256": digest(path),
                "collision_geometry_changed": False,
            }
        )
    return repairs


def prepare(run_id, *, repair_visual_uris=False):
    import stack

    output = stack.OUTPUT
    if output.exists() and any(output.iterdir()):
        raise ValueError("fresh staging output required")
    stack.prepare(controller_watchdog=True)
    urdf = ET.parse(output / "robot.urdf").getroot()
    repairs = stage_visual_resources(urdf, output, repair_missing=repair_visual_uris)
    ET.ElementTree(urdf).write(output / "robot.urdf")
    # A CLI YAML plain scalar folds XML whitespace. A typed YAML string keeps
    # the live parameter byte-identical to the independently pinned asset.
    (output / "robot-state-params.yaml").write_text(
        yaml.safe_dump(
            {
                "robot_state_publisher": {
                    "ros__parameters": {
                        "use_sim_time": True,
                        "robot_description": (output / "robot.urdf").read_text(),
                    }
                }
            }
        )
    )
    # The actual conversion command is recorded and its result is preserved.
    argv = ["gz", "sdf", "-p", "/assets/robot.urdf"]
    # /assets is a read-only bind of this fresh staging tree during prep;
    # preparation writes through /evidence. Runtime has no writable asset alias.
    with (
        (output / "converted.sdf").open("xb") as out,
        (output / "conversion-stderr.log").open("xb") as err,
    ):
        result = subprocess.run(argv, stdout=out, stderr=err, timeout=30)
    if result.returncode:
        raise ValueError("URDF conversion failed; original output retained")
    converted = ET.parse(output / "converted.sdf").getroot().find("model")
    original = ET.parse(output / "robot.sdf").getroot().find("model")
    model_name = converted.get("name")
    links = {link.get("name"): link for link in converted.findall("link")}
    parents = {j.find("child").get("link") for j in urdf.findall("joint")}
    roots = {link.get("name") for link in urdf.findall("link")} - parents
    if len(roots) != 1 or next(iter(roots)) not in links:
        raise ValueError("explicit unique converted root required")
    base = next(iter(roots))
    for sensor in original.findall(".//sensor"):
        if sensor.get("type") not in {"gpu_lidar", "imu"}:
            continue
        frame = sensor.findtext("gz_frame_id")
        if not frame:
            raise ValueError("explicit source sensor frame required")
        pose = sensor.find("pose")
        if pose is None:
            pose = ET.SubElement(sensor, "pose")
        pose.attrib.clear()
        pose.text = frame_pose(urdf, frame, base)
        links[base].append(sensor)
    topics = []
    for name, link in links.items():
        collisions = link.findall("collision")
        if not collisions:
            continue
        topic = "/rosclaw_sim/contacts/" + name
        topics.append(topic)
        sensor = ET.SubElement(link, "sensor", name="expert_contacts", type="contact")
        ET.SubElement(sensor, "always_on").text = "true"
        ET.SubElement(sensor, "update_rate").text = "20"
        ET.SubElement(sensor, "topic").text = topic
        contact = ET.SubElement(sensor, "contact")
        for collision in collisions:
            ET.SubElement(contact, "collision").text = collision.get("name")
    control = urdf.find("ros2_control")
    if control is None:
        raise ValueError("existing fixture control contract required")
    converted.append(ET.fromstring(ET.tostring(control)))
    for plugin in original.findall("plugin"):
        converted.append(plugin)
    world = ET.parse(output / "world.sdf").getroot()
    pose = ET.SubElement(converted, "pose")
    pose.text = "0 0 0.05 0 0 0"
    world.find("world").append(converted)
    plugin = ET.SubElement(
        world.find("world"),
        "plugin",
        filename="/observer/librosclaw_source_witness.so",
        name="rosclaw::SourceWitness",
    )
    ET.SubElement(plugin, "output").text = "/evidence/ecm-original.jsonl"
    ET.SubElement(plugin, "run_id").text = run_id
    ET.ElementTree(world).write(output / "world.sdf")
    # Existing bridge inputs are reused; only the actual converted contact links differ.
    bridge = yaml.safe_load((output / "bridge.yaml").read_text())
    bridge = [
        x for x in bridge if not x.get("ros_topic_name", "").startswith("/rosclaw_sim/contacts/")
    ]
    bridge += [
        {
            "ros_topic_name": topic,
            "gz_topic_name": f"/world/ros_expert/model/{model_name}/link/{topic.rsplit('/', 1)[-1]}/sensor/expert_contacts/contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        }
        for topic in topics
    ]
    (output / "bridge.yaml").write_text(yaml.safe_dump(bridge))
    (output / "contact_topics.json").write_text(json.dumps(topics))
    params = yaml.safe_load((output / "nav2.yaml").read_text())
    # This zero-motion source fixture needs genuine scan-driven AMCL updates.
    # Zero odometry thresholds process measured settling/jitter without goals,
    # initialpose publication, covariance widening, or timestamp fabrication.
    params["amcl"]["ros__parameters"]["update_min_d"] = 0.0
    params["amcl"]["ros__parameters"]["update_min_a"] = 0.0
    # Explicit operator initialization, separate from passive source observers.
    # The 5 mm / 0.01 rad uncertainty describes this known disposable fixture;
    # it does not change any review threshold or confer action authority.
    (output / "source-initialization.json").write_text(
        json.dumps(
            {
                "schema_version": "rosclaw.owned_sim_source_initialization.v1",
                "run_id": run_id,
                "service": "/set_initial_pose",
                "frame_id": "map",
                "pose": params["amcl"]["ros__parameters"]["initial_pose"],
                "covariance_diagonal": [0.000025, 0.000025, 0.0001],
                "operator_owned_disposable_sim_only": True,
            },
            indent=2,
        )
        + "\n"
    )
    navigator = params["bt_navigator"]["ros__parameters"]
    navigator["navigators"] = ["navigate_to_pose"]
    navigator.pop("navigate_complete_coverage", None)
    navigator.pop("default_coverage_bt_xml", None)
    (output / "nav2.yaml").write_text(yaml.safe_dump(params))
    expected = sdf_inventory(
        (output / "world.sdf").read_bytes(), robot_entity=model_name, world_name="ros_expert"
    )
    converted_inventory = sdf_inventory(
        (output / "converted.sdf").read_bytes(), robot_entity=model_name
    )
    if (
        expected["collisions"] != converted_inventory["collisions"]
        or expected["structure"] != converted_inventory["structure"]
    ):
        raise ValueError(
            "staging must preserve every converted collision and kinematic/inertial field"
        )
    (output / "source-preparation.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "conversion": {
                    "argv": argv,
                    "exit_code": result.returncode,
                    "input_sha256": digest(output / "robot.urdf"),
                    "output_sha256": digest(output / "converted.sdf"),
                },
                "visual_resource_repairs": repairs,
                "collision_inventory_count": len(expected["collisions"]),
                "source_sensor_mounts_from_URDF": True,
                "goals_sent": 0,
                "velocity_commands_sent": 0,
            },
            indent=2,
        )
        + "\n"
    )
    inventory = {
        "schema_version": "rosclaw.prepared_sim_assets.v1",
        "file_sha256": {
            str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()
        },
    }
    path = output / "prepared-assets.json"
    path.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"inventory_sha256": digest(path), "run_id": run_id}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repair-visual-uris", action="store_true")
    args = parser.parse_args()
    prepare(args.run_id, repair_visual_uris=args.repair_visual_uris)

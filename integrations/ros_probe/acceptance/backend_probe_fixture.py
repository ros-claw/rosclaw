"""Prepare one declared SIM instrument; no third robot, runtime or movement.

The probe's source hash binds an instrument artifact, not the active robot Body.
Its placement must remain outside the frozen cleaning region and body clearance.
World integration and actual component admission must be performed separately.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def validate_probe_declaration(declaration):
    keys = {
        "schema_version",
        "source",
        "approved",
        "evidence_domain",
        "run_id",
        "world_name",
        "robot_model_name",
        "probe_model_name",
        "probe_xy",
        "sphere_radius_m",
        "ground_z_m",
        "lift_z_m",
        "ground_collision_name",
        "cleaning_polygon",
        "maximum_robot_radius_m",
        "pose_topic",
        "component_topic",
        "contact_topic",
    }
    if (
        type(declaration) is not dict
        or set(declaration) != keys
        or declaration["schema_version"] != "rosclaw.backend_probe_fixture_declaration.v1"
    ):
        raise ValueError("closed explicit probe fixture declaration required")
    if (
        declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
    ):
        raise ValueError("SIM instrument fixture only; no REAL authority")
    for key in ("world_name", "robot_model_name", "probe_model_name"):
        if type(declaration[key]) is not str or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]{0,63}", declaration[key]
        ):
            raise ValueError("explicit bounded source model names required")
    if declaration["probe_model_name"] == declaration["robot_model_name"]:
        raise ValueError("instrument must be distinct from the active robot")
    for key in ("run_id", "ground_collision_name"):
        if type(declaration[key]) is not str or not 0 < len(declaration[key]) <= 256:
            raise ValueError("bounded explicit source binding required")
    topics = [declaration[k] for k in ("pose_topic", "component_topic", "contact_topic")]
    if len(set(topics)) != 3 or any(
        type(v) is not str or not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", v) for v in topics
    ):
        raise ValueError("three distinct explicit observer topic roles required")
    xy = declaration["probe_xy"]
    polygon = declaration["cleaning_polygon"]
    if (
        type(xy) is not list
        or len(xy) != 2
        or type(polygon) is not list
        or not 3 <= len(polygon) <= 256
    ):
        raise ValueError("explicit bounded region and probe placement required")
    for point in [xy, *polygon]:
        if (
            type(point) is not list
            or len(point) != 2
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 20 for v in point
            )
        ):
            raise ValueError("finite bounded actual fixture coordinates required")
    for key, lo, hi in (
        ("sphere_radius_m", 0.02, 0.1),
        ("ground_z_m", -1, 1),
        ("lift_z_m", 5, 12),
        ("maximum_robot_radius_m", 0.01, 10),
    ):
        value = declaration[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("bounded explicit fixture geometry required")
    # Stay outside an expanded bounding box of the entire declared polygon.
    # This is conservative for arbitrary polygons and does not infer robot size.
    clearance = declaration["maximum_robot_radius_m"] + declaration["sphere_radius_m"] + 0.5
    if (
        min(p[0] for p in polygon) - clearance <= xy[0] <= max(p[0] for p in polygon) + clearance
        and min(p[1] for p in polygon) - clearance
        <= xy[1]
        <= max(p[1] for p in polygon) + clearance
    ):
        raise ValueError("probe must be outside frozen work region plus full body clearance")
    return declaration


def prepare_probe_fixture(output, declaration):
    validate_probe_declaration(declaration)
    xy = declaration["probe_xy"]
    name = declaration["probe_model_name"]
    radius = declaration["sphere_radius_m"]
    sdf = ET.Element("sdf", version="1.9")
    model = ET.SubElement(sdf, "model", name=name)
    ET.SubElement(model, "pose").text = " ".join(
        str(v) for v in [*xy, declaration["ground_z_m"] + radius, 0, 0, 0]
    )
    ET.SubElement(model, "static").text = "false"
    link = ET.SubElement(model, "link", name="probe_link")
    ET.SubElement(link, "gravity").text = "true"
    inertia = ET.SubElement(link, "inertial")
    ET.SubElement(inertia, "mass").text = "1"
    tensor = ET.SubElement(inertia, "inertia")
    for key in ("ixx", "iyy", "izz"):
        ET.SubElement(tensor, key).text = str(0.4 * radius * radius)
    for key in ("ixy", "ixz", "iyz"):
        ET.SubElement(tensor, key).text = "0"
    collision = ET.SubElement(link, "collision", name="probe_collision")
    geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(ET.SubElement(geometry, "sphere"), "radius").text = str(radius)
    sensor = ET.SubElement(link, "sensor", name="probe_contact", type="contact")
    ET.SubElement(sensor, "always_on").text = "true"
    ET.SubElement(sensor, "update_rate").text = "20"
    ET.SubElement(sensor, "topic").text = declaration["contact_topic"]
    contact = ET.SubElement(sensor, "contact")
    ET.SubElement(contact, "collision").text = "probe_collision"
    ET.SubElement(contact, "topic").text = "/rosclaw_sim/backend_probe_contact"
    pose = ET.SubElement(
        model,
        "plugin",
        filename="gz-sim-pose-publisher-system",
        name="gz::sim::systems::PosePublisher",
    )
    for key, value in {
        "publish_model_pose": "true",
        "publish_link_pose": "false",
        "use_pose_vector_msg": "true",
        "update_frequency": "-1",
        "topic": declaration["pose_topic"],
    }.items():
        ET.SubElement(pose, key).text = value
    raw = ET.tostring(sdf, encoding="utf-8", xml_declaration=True)
    binding = {
        "run_id": declaration["run_id"],
        "world_name": declaration["world_name"],
        "body_model_name": name,
        "body_snapshot_hash": digest(
            {
                "role": "SIM_INSTRUMENT_ARTIFACT_BINDING_NOT_ROBOT_BODY",
                "source_sdf_sha256": hashlib.sha256(raw).hexdigest(),
            }
        ),
        "attachment_hash": digest(
            {"role": "SIM_INSTRUMENT_SENSOR_BINDING", "sensor": ET.tostring(sensor).decode()}
        ),
        "producer_id": "backend_probe_" + digest(declaration)[:24],
    }
    bridge = [
        {
            "ros_topic_name": declaration["contact_topic"],
            "gz_topic_name": "/rosclaw_sim/backend_probe_contact",
            "ros_type_name": "ros_gz_interfaces/msg/Contacts",
            "gz_type_name": "gz.msgs.Contacts",
            "direction": "GZ_TO_ROS",
        },
        {
            "ros_topic_name": declaration["pose_topic"],
            "gz_topic_name": declaration["pose_topic"],
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "gz_type_name": "gz.msgs.Pose_V",
            "direction": "GZ_TO_ROS",
        },
        {
            "ros_topic_name": declaration["component_topic"],
            "gz_topic_name": "/rosclaw_sim/backend_probe_components",
            "ros_type_name": "std_msgs/msg/String",
            "gz_type_name": "gz.msgs.StringMsg",
            "direction": "GZ_TO_ROS",
        },
    ]
    import yaml

    files = {
        "robot.sdf": raw,
        "physics_binding.json": json.dumps(binding, indent=2).encode(),
        "brush_binding.json": json.dumps(
            {
                k: binding[k]
                for k in ("run_id", "body_snapshot_hash", "attachment_hash", "producer_id")
            },
            indent=2,
        ).encode(),
        "bridge.yaml": yaml.safe_dump(bridge).encode(),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for filename, data in files.items():
        (output / filename).write_bytes(data)
    result = {
        "schema_version": "rosclaw.backend_probe_fixture.v1",
        "status": "PREPARED_NOT_SOURCE_ADMITTED",
        "evidence_role": "SIM_INSTRUMENT_NOT_ROBOT_BODY_OR_AUTHORITY",
        "declaration": declaration,
        "binding": binding,
        "physical_acceptance": "NOT_RUN",
        "world_integration": "NOT_IMPLEMENTED",
        "gravity_and_contact_behavior": "REQUIRES_ACTUAL_PHYSICS",
        "files": {
            filename: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for filename, data in files.items()
        },
    }
    (output / "probe-fixture.json").write_text(json.dumps(result, indent=2) + "\n")
    return result

"""Materialize explicit contact instrumentation from captured URDF/SDF bytes.

No profiles, ROS imports, actions or physical acceptance. Geometry, link and
collision names come from both supplied sources. Mismatches are refused rather
than normalized into a fabricated Body. PostUpdate admission remains required.
"""

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint


def _xml(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 2_000_000 or b"\x00" in raw:
        raise ValueError("bounded materialized UTF-8 XML source required")
    text = raw.decode("utf-8")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("external entities are not source geometry")
    return ET.fromstring(text)


def _name(value):
    if type(value) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", value):
        raise ValueError("explicit bounded source name required")
    return value


def _numbers(text, count):
    if type(text) is not str or len(text) > 256:
        raise ValueError("bounded primitive dimensions required")
    values = tuple(float(v) for v in text.split())
    if len(values) != count or any(not math.isfinite(v) or not 0 < v <= 100 for v in values):
        raise ValueError("finite positive primitive dimensions required")
    return values


def _primitive(collision, *, urdf):
    geometries = collision.findall("geometry")
    if len(geometries) != 1 or len(geometries[0]) != 1:
        raise ValueError("one explicit primitive per source collision required")
    shape = geometries[0][0]
    if shape.tag == "box":
        return ("box", _numbers(shape.get("size") if urdf else shape.findtext("size"), 3))
    if shape.tag == "sphere":
        return ("sphere", _numbers(shape.get("radius") if urdf else shape.findtext("radius"), 1))
    if shape.tag == "cylinder":
        return (
            "cylinder",
            _numbers(shape.get("radius") if urdf else shape.findtext("radius"), 1),
            _numbers(shape.get("length") if urdf else shape.findtext("length"), 1),
        )
    raise ValueError("unsupported collision geometry remains UNKNOWN")


def _local_pose(collision, *, urdf, link_name):
    nodes = collision.findall("origin" if urdf else "pose")
    if len(nodes) > 1:
        raise ValueError("unique source collision-local pose required")
    if not nodes:
        return (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    node = nodes[0]
    if urdf:
        values = (node.get("xyz", "0 0 0") + " " + node.get("rpy", "0 0 0")).split()
    else:
        if (
            node.get("relative_to", link_name) not in ("", link_name)
            or node.get("rotation_format", "euler_rpy") != "euler_rpy"
            or node.get("degrees", "false") != "false"
        ):
            raise ValueError("unsupported collision-local frame is UNKNOWN")
        values = (node.text or "0 0 0 0 0 0").split()
    if len(values) != 6 or any(len(v) > 64 for v in values):
        raise ValueError("bounded explicit collision-local transform required")
    xyzrpy = tuple(float(v) for v in values)
    if any(not math.isfinite(v) or abs(v) > 100 for v in xyzrpy):
        raise ValueError("finite bounded collision-local transform required")
    roll, pitch, yaw = xyzrpy[3:]
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return xyzrpy[:3], (
        cy * cp,
        cy * sp * sr - sy * cr,
        cy * sp * cr + sy * sr,
        sy * cp,
        sy * sp * sr + cy * cr,
        sy * sp * cr - cy * sr,
        -sp,
        cp * sr,
        cp * cr,
    )


def prepare_contact_fixture(output, *, urdf_bytes, sdf_bytes, bridge_bytes, declaration):
    """Write a new instrumented source proposal; never overwrite original assets."""
    if type(declaration) is not dict or set(declaration) != {
        "source",
        "approved",
        "evidence_domain",
        "world_name",
        "body_model_name",
        "contact_prefix",
        "independent_pose_topic",
    }:
        raise ValueError("closed explicit contact instrumentation declaration required")
    if (
        declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
    ):
        raise ValueError("explicit simulator-owned instrumentation policy required")
    world_name, model_name = (_name(declaration[k]) for k in ("world_name", "body_model_name"))
    prefix = absolute_endpoint(declaration["contact_prefix"])
    truth = absolute_endpoint(declaration["independent_pose_topic"])
    if prefix == truth:
        raise ValueError("distinct source topic roles required")
    robot, sdf = _xml(urdf_bytes), _xml(sdf_bytes)
    if robot.tag != "robot" or sdf.tag != "sdf" or len(sdf.findall("model")) != 1 or len(sdf) != 1:
        raise ValueError("materialized URDF and single explicit SDF model required")
    model = sdf.find("model")
    if model.findall("model") or model.findall("include"):
        raise ValueError("nested or unresolved simulator models are UNKNOWN")
    original_model_name = _name(model.get("name"))
    urdf_links, sdf_links = {}, {}
    for source, destination in ((robot, urdf_links), (model, sdf_links)):
        for link in source.findall("link"):
            name = _name(link.get("name"))
            if name in destination:
                raise ValueError("duplicate source link names are ambiguous")
            destination[name] = link
    if not 1 <= len(urdf_links) <= 512 or set(urdf_links) != set(sdf_links):
        raise ValueError("exact unfused URDF/SDF link correspondence required")
    if type(bridge_bytes) is not bytes or not 0 < len(bridge_bytes) <= 2_000_000:
        raise ValueError("bounded original bridge bytes required")
    bridge = yaml.safe_load(bridge_bytes)
    if (
        type(bridge) is not list
        or len(bridge) > 1024
        or any(type(row) is not dict for row in bridge)
    ):
        raise ValueError("typed original bridge list required")
    used_topics = {row.get("ros_topic_name", row.get("topic_name")) for row in bridge}
    if truth in used_topics:
        raise ValueError("independent pose topic already has an unreviewed bridge")
    streams = []
    model.set("name", model_name)
    for link_name, urdf_link in urdf_links.items():
        sdf_link = sdf_links[link_name]
        if any(sensor.get("type") == "contact" for sensor in sdf_link.findall("sensor")):
            raise ValueError("existing contact sources require explicit reconciliation")
        source_collisions, loaded_collisions = (
            urdf_link.findall("collision"),
            sdf_link.findall("collision"),
        )
        source_names = [_name(c.get("name")) for c in source_collisions]
        loaded_names = [_name(c.get("name")) for c in loaded_collisions]
        if (
            len(source_names) != len(set(source_names))
            or len(loaded_names) != len(set(loaded_names))
            or set(source_names) != set(loaded_names)
        ):
            raise ValueError("exact named URDF/SDF collision correspondence required")
        by_name = {c.get("name"): c for c in loaded_collisions}
        sensor_names = {_name(s.get("name")) for s in sdf_link.findall("sensor")}
        for index, collision in enumerate(source_collisions):
            name = collision.get("name")
            if _primitive(collision, urdf=True) != _primitive(by_name[name], urdf=False):
                raise ValueError("source primitive dimensions differ from simulator geometry")
            source_pose = _local_pose(collision, urdf=True, link_name=link_name)
            loaded_pose = _local_pose(by_name[name], urdf=False, link_name=link_name)
            if any(
                abs(a - b) > 1e-9
                for source_values, loaded_values in zip(source_pose, loaded_pose, strict=True)
                for a, b in zip(source_values, loaded_values, strict=True)
            ):
                raise ValueError("source collision-local transform differs from simulator geometry")
            sensor_name = "rosclaw_contact_" + str(index)
            if sensor_name in sensor_names:
                raise ValueError("instrumentation would overwrite an existing source sensor")
            topic = absolute_endpoint(prefix + "/" + link_name + "/collision_" + str(index))
            if topic in used_topics or topic == truth:
                raise ValueError("source contact topic would alias an existing bridge")
            used_topics.add(topic)
            sensor = ET.SubElement(sdf_link, "sensor", name=sensor_name, type="contact")
            ET.SubElement(sensor, "always_on").text = "true"
            ET.SubElement(sensor, "update_rate").text = "20"
            ET.SubElement(sensor, "topic").text = topic
            ET.SubElement(ET.SubElement(sensor, "contact"), "collision").text = name
            streams.append(
                {
                    "link": link_name,
                    "collision_index": index,
                    "collision_name": name,
                    "sensor_name": sensor_name,
                    "topic": topic,
                    "gz_topic": topic,
                }
            )
            bridge.append(
                {
                    "ros_topic_name": topic,
                    "gz_topic_name": topic,
                    "ros_type_name": "ros_gz_interfaces/msg/Contacts",
                    "gz_type_name": "gz.msgs.Contacts",
                    "direction": "GZ_TO_ROS",
                }
            )
    if not 1 <= len(streams) <= 256:
        raise ValueError("one to 256 source collisions required by component observer bound")
    if any(
        plugin.get("name") == "gz::sim::systems::PosePublisher"
        for plugin in model.findall("plugin")
    ):
        raise ValueError("existing independent pose producer requires explicit reconciliation")
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
        "update_frequency": "30",
        "topic": truth,
    }.items():
        ET.SubElement(pose, key).text = value
    bridge.append(
        {
            "ros_topic_name": truth,
            "gz_topic_name": truth,
            "ros_type_name": "tf2_msgs/msg/TFMessage",
            "gz_type_name": "gz.msgs.Pose_V",
            "direction": "GZ_TO_ROS",
        }
    )
    output = Path(output)
    manifest = {
        "schema_version": "rosclaw.generic_contact_fixture.v1",
        "status": "PREPARED_NOT_SOURCE_ADMITTED",
        "physical_acceptance": "NOT_RUN",
        "world_name": world_name,
        "body_model_name": model_name,
        "original_model_name": original_model_name,
        "original_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "original_sdf_sha256": hashlib.sha256(sdf_bytes).hexdigest(),
        "original_bridge_sha256": hashlib.sha256(bridge_bytes).hexdigest(),
        "collision_streams": streams,
        "support_contact_topics": "UNKNOWN_REQUIRES_ACTUAL_CONTINUOUS_CONTACT_OBSERVATION",
        "controller_watchdog": "NOT_VERIFIED_REQUIRES_SEPARATE_SOURCE_ADMISSION",
        "files": {},
    }
    files = {
        "robot.urdf": urdf_bytes,
        "robot.sdf": ET.tostring(sdf, encoding="utf-8", xml_declaration=True),
        "bridge.yaml": yaml.safe_dump(bridge).encode(),
        "instrumentation-policy.json": (
            json.dumps(declaration, sort_keys=True, indent=2) + "\n"
        ).encode(),
    }
    output.mkdir(exist_ok=False)
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
        manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    with (output / "contact-fixture.json").open("x") as stream:
        stream.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return manifest

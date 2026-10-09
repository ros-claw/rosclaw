"""Explicit SIM-only World system proposal; no launch or sensor evidence.

Gazebo's core server defaults do not establish a rendered lidar/contact stream.
An operator may request these standard systems while retaining every original
World/model source. A plugin declaration never substitutes for actual messages.
"""

import hashlib
from xml.etree import ElementTree as ET

SYSTEMS = {
    "physics": ("gz-sim-physics-system", "gz::sim::systems::Physics"),
    "commands": ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
    "scene": ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
    "sensors": ("gz-sim-sensors-system", "gz::sim::systems::Sensors"),
    "contact": ("gz-sim-contact-system", "gz::sim::systems::Contact"),
}


def _xml(raw):
    if (
        type(raw) is not bytes
        or not 0 < len(raw) <= 16_000_000
        or b"\x00" in raw
        or b"<!DOCTYPE" in raw.upper()
        or b"<!ENTITY" in raw.upper()
    ):
        raise ValueError("bounded materialized XML source required")
    return ET.fromstring(raw)


def prepare_world_system_source(world_bytes, robot_sdf_bytes, declaration):
    if type(declaration) is not dict or set(declaration) != {
        "source",
        "approved",
        "evidence_domain",
        "world_name",
        "render_engine",
    }:
        raise ValueError("closed explicit World system declaration required")
    if (
        declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
        or declaration["render_engine"] != "ogre2"
    ):
        raise ValueError("explicit SIM-only standard Ogre2 system policy required")
    root, robot = _xml(world_bytes), _xml(robot_sdf_bytes)
    if root.tag != "sdf" or len(root) != 1 or len(root.findall("world")) != 1:
        raise ValueError("one materialized World required")
    world = root.find("world")
    if world.get("name") != declaration["world_name"] or root.findall(".//include"):
        raise ValueError("exact resolved declared World identity required")
    if robot.tag != "sdf" or len(robot) != 1 or len(robot.findall("model")) != 1:
        raise ValueError("one materialized robot source model required")
    sensor_types = sorted({s.get("type", "UNKNOWN") for s in robot.findall(".//sensor")})
    required = ["physics", "commands", "scene"]
    if set(sensor_types) & {"gpu_lidar", "lidar"}:
        required.append("sensors")
    if "contact" in sensor_types:
        required.append("contact")
    added, configured = [], []
    for role in required:
        filename, name = SYSTEMS[role]
        aliases = {filename, "lib" + filename + ".so"}
        matches = [
            p
            for p in world.findall("plugin")
            if p.get("name") == name or p.get("filename") in aliases
        ]
        if len(matches) > 1 or any(
            p.get("name") != name or p.get("filename") not in aliases for p in matches
        ):
            raise ValueError("duplicate or rebound standard World system: " + role)
        if matches:
            plugin = matches[0]
        else:
            plugin = ET.SubElement(world, "plugin", filename=filename, name=name)
            added.append(role)
        if role == "sensors":
            engines = plugin.findall("render_engine")
            if len(engines) > 1 or (engines and engines[0].text != "ogre2"):
                raise ValueError(
                    "existing sensor render engine differs; explicit reconciliation required"
                )
            if not engines:
                ET.SubElement(plugin, "render_engine").text = "ogre2"
                configured.append(role + ".render_engine")
    result = (
        ET.tostring(root, encoding="utf-8", xml_declaration=True)
        if added or configured
        else world_bytes
    )
    report = {
        "schema_version": "rosclaw.generic_world_system_source.v1",
        "source_world_sha256": hashlib.sha256(world_bytes).hexdigest(),
        "source_robot_sdf_sha256": hashlib.sha256(robot_sdf_bytes).hexdigest(),
        "prepared_world_sha256": hashlib.sha256(result).hexdigest(),
        "world_name": world.get("name"),
        "materialized_robot_sensor_types": sensor_types,
        "required_standard_roles": required,
        "added_standard_roles": added,
        "configured_standard_parameters": configured,
        "other_sensor_types_require_separate_runtime_evidence": sorted(
            set(sensor_types) - {"gpu_lidar", "lidar", "contact"}
        ),
        "actual_system_loader_validation": "REQUIRED_NOT_RUN",
        "actual_sensor_messages": "REQUIRED_NOT_OBSERVED",
        "World_started": False,
        "authorization": False,
        "physical_acceptance": "NOT_RUN",
    }
    return {"world_bytes": result, "report": report}

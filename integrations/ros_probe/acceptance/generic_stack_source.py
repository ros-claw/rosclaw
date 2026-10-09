"""Join actual generic robot/world/map sources into an unlaunched SIM workspace.

No Body identity, live Graph, map denominator or source admission is invented.
The operator supplies complete expanded sources and all interface declarations.
"""

import hashlib
import json
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml
from backend_world_bundle import observation_bridge_rows
from generic_contact_fixture import prepare_contact_fixture

from rosclaw.connectors.ros.context.sim_controller_source import prepare_sim_controller_source
from rosclaw.connectors.ros.context.sim_localization_source import apply_frozen_localization_prior
from rosclaw.connectors.ros.context.sim_navigation_source import (
    _yaml,
    prepare_sim_navigation_source,
)
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def prepare_generic_stack_source(
    output,
    *,
    urdf_bytes,
    sdf_bytes,
    world_bytes,
    bridge_bytes,
    map_yaml_bytes,
    map_image_bytes,
    nav2_bytes,
    coverage_bytes,
    attachment,
    controller_declaration,
    navigation_declaration,
    contact_declaration,
    localization_initialization_declaration=None,
):
    source = {
        "robot.original.urdf": urdf_bytes,
        "robot.original.sdf": sdf_bytes,
        "world.original.sdf": world_bytes,
        "bridge.original.yaml": bridge_bytes,
        "map.original.yaml": map_yaml_bytes,
        "map.original.image": map_image_bytes,
        "nav2.original.yaml": nav2_bytes,
        "coverage.original.yaml": coverage_bytes,
    }
    if any(type(raw) is not bytes or not 0 < len(raw) <= 16_000_000 for raw in source.values()):
        raise ValueError("bounded complete expanded source bytes required")
    if b"<!DOCTYPE" in world_bytes.upper() or b"<!ENTITY" in world_bytes.upper():
        raise ValueError("external world resources remain unresolved")
    world = ET.fromstring(world_bytes)
    if world.tag != "sdf" or len(world) != 1 or len(world.findall("world")) != 1:
        raise ValueError("one materialized complete source World required")
    actual = world.find("world")
    if actual.get("name") != contact_declaration["world_name"] or world.findall(".//include"):
        raise ValueError("exact explicit World identity without unresolved includes required")
    body_name = controller_declaration["body_model_name"]
    if body_name != contact_declaration["body_model_name"] or any(
        m.get("name") == body_name for m in actual.findall(".//model")
    ):
        raise ValueError("distinct World and separately materialized robot identity required")
    controlled = prepare_sim_controller_source(urdf_bytes, sdf_bytes, controller_declaration)
    navigation = prepare_sim_navigation_source(
        controlled["source_files"]["robot.urdf"],
        nav2_bytes,
        coverage_bytes,
        attachment=attachment,
        declaration=navigation_declaration,
        controller_report=controlled["report"],
    )
    if localization_initialization_declaration is not None:
        navigation = apply_frozen_localization_prior(
            navigation,
            world_name=actual.get("name"),
            map_frame=navigation_declaration["frames"]["map"],
            declaration=localization_initialization_declaration,
        )
    if navigation_declaration["topics"]["odom"] != controller_declaration["odom_topic"]:
        raise ValueError("actual navigation/controller odometry source must be identical")
    if (
        navigation_declaration["map_yaml_path"] != "/evidence/map.yaml"
        or controller_declaration["controller_parameters_path"]
        != "/evidence/controller_params.yaml"
    ):
        raise ValueError("source resource paths must match generated owned workspace paths")
    bridge = observation_bridge_rows(bridge_bytes)
    sensor_rows = [
        row for row in bridge if row["ros_topic_name"] == navigation_declaration["topics"]["lidar"]
    ]
    if (
        len(sensor_rows) != 1
        or sensor_rows[0]["ros_type_name"] != "sensor_msgs/msg/LaserScan"
        or sensor_rows[0]["gz_type_name"] != "gz.msgs.LaserScan"
    ):
        raise ValueError("one exact actual declared typed lidar bridge source required")
    robot_sdf = ET.fromstring(sdf_bytes)
    lidar_links = [
        link
        for link in robot_sdf.findall("model/link")
        if link.get("name") == navigation_declaration["frames"]["lidar"]
    ]
    sensors = [
        sensor
        for link in lidar_links
        for sensor in link.findall("sensor")
        if sensor.get("type") in {"lidar", "gpu_lidar"}
    ]
    if len(sensors) != 1 or sensors[0].findtext("topic") != sensor_rows[0]["gz_topic_name"]:
        raise ValueError("actual materialized sensor link/topic must match explicit bridge source")
    map_source = _yaml(map_yaml_bytes)
    if type(map_source) is not dict or set(map_source) - {
        "image",
        "resolution",
        "origin",
        "negate",
        "occupied_thresh",
        "free_thresh",
        "mode",
    }:
        raise ValueError("closed source map-server metadata required")
    required = {"image", "resolution", "origin", "negate", "occupied_thresh", "free_thresh"}
    if not required <= set(map_source):
        raise ValueError("complete original map metadata required")
    image = map_source["image"]
    if type(image) is not str or not re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.-]{0,127}\.(pgm|png)", image
    ):
        raise ValueError("one explicit local source map image required")
    origin = map_source["origin"]
    if (
        type(origin) is not list
        or len(origin) != 3
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 10000 for v in origin
        )
        or origin[2] != 0
    ):
        raise ValueError(
            "explicit finite unrotated map origin required; rotated map remains UNKNOWN"
        )
    resolution = map_source["resolution"]
    if (
        type(resolution) not in (int, float)
        or not math.isfinite(resolution)
        or resolution != navigation_declaration["map_resolution"]
    ):
        raise ValueError("map/navigation source resolution mismatch")
    free, occupied = map_source["free_thresh"], map_source["occupied_thresh"]
    if (
        type(free) not in (int, float)
        or type(occupied) not in (int, float)
        or not 0 <= free < occupied <= 1
        or type(map_source["negate"]) is not int
        or map_source["negate"] not in (0, 1)
        or type(map_source.get("mode", "trinary")) is not str
        or map_source.get("mode", "trinary") not in {"trinary", "scale", "raw"}
    ):
        raise ValueError("explicit bounded map classification metadata required")
    # This preserves original image bytes. Only the installed map decoder and
    # live map observation may establish dimensions, content and denominator.
    output = Path(output)
    output.mkdir(exist_ok=False, mode=0o700)
    original = output / "original-sources"
    original.mkdir()
    for name, raw in source.items():
        (original / name).write_bytes(raw)
    contact = prepare_contact_fixture(
        output / "contact-source",
        urdf_bytes=controlled["source_files"]["robot.urdf"],
        sdf_bytes=controlled["source_files"]["robot.sdf"],
        bridge_bytes=yaml.safe_dump(bridge).encode(),
        declaration=contact_declaration,
    )
    for name, raw in controlled["source_files"].items():
        (output / name).write_bytes(raw)
    for name in ("robot.sdf", "bridge.yaml"):
        (output / name).write_bytes((output / "contact-source" / name).read_bytes())
    (output / "world.sdf").write_bytes(world_bytes)
    (output / "map.yaml").write_bytes(map_yaml_bytes)
    (output / image).write_bytes(map_image_bytes)
    (output / "nav2.yaml").write_text(yaml.safe_dump(navigation["parameters"]))
    (output / "controller_params.yaml").write_text(
        yaml.safe_dump(controlled["controller_parameters"])
    )
    for name, data in [
        ("controller-source-report.json", controlled["report"]),
        ("navigation-source-report.json", navigation["report"]),
        ("navigation-launch-source.json", navigation["launch_nodes"]),
    ]:
        (output / name).write_text(json.dumps(data, indent=2) + "\n")
    declarations = {
        "controller": controller_declaration,
        "navigation": navigation_declaration,
        "contact": contact_declaration,
        "attachment": attachment,
    }
    if localization_initialization_declaration is not None:
        declarations["localization_initialization"] = localization_initialization_declaration
        (output / "localization-initialization-declaration.json").write_text(
            json.dumps(localization_initialization_declaration, indent=2) + "\n"
        )
    manifest = {
        "schema_version": "rosclaw.generic_stack_source.v1",
        "status": "PREPARED_NOT_LAUNCHED_OR_ADMITTED",
        "world_name": contact_declaration["world_name"],
        "body_model_name": body_name,
        "source_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in source.items()},
        "declaration_hash": digest(declarations),
        "output_hashes": {
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "contact_source_report": contact,
        "map_pixels_and_dimensions": "REQUIRES_ACTUAL_SDK_DECODER_AND_LIVE_MAP",
        "requires_actual_Graph_TF_Body_and_loaded_source_admission": True,
        "physical_acceptance": "NOT_RUN",
        "heldout_asset": "NOT_SELECTED",
        "authorization": False,
    }
    manifest["artifact_hash"] = digest(manifest)
    (output / "generic-stack-source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest

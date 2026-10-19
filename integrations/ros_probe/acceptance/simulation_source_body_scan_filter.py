"""Optional fixed-package sensor processing in owned disposable SDK fixtures.

This constructs no driver, Runtime, permission, goal or velocity publisher.
The raw scan and all collision-monitor protection parameters remain intact.
"""

import hashlib
import json
import math
import os
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PACKAGE_SHA256 = "49756059a9f5bde5416c678260feb4513dc825751ed4f4ff44c40612a52634d9"
PACKAGE_BYTES = 574144
SCHEMA = "rosclaw.sdk_source_body_scan_filter.v1"
OPTION = "source_body_scan_filter"


def filter_declaration(descriptor):
    if descriptor is None or OPTION not in descriptor:
        return None
    value = descriptor[OPTION]
    if (
        type(value) is not dict
        or set(value) != {"schema_version", "source_urdf_sha256", "base_frame", "scan_frame"}
        or value["schema_version"] != SCHEMA
        or type(value["source_urdf_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", value["source_urdf_sha256"]) is None
        or any(
            type(value[key]) is not str
            or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value[key]) is None
            for key in ("base_frame", "scan_frame")
        )
        or value["base_frame"] == value["scan_frame"]
        or descriptor.get("start_navigation_stack", True) is not True
        or type(descriptor.get("namespace")) is not str
        or re.fullmatch(r"(/[A-Za-z][A-Za-z0-9_]*)+", descriptor["namespace"]) is None
        or type(descriptor.get("scan_ros_topic")) is not str
        or not descriptor["scan_ros_topic"].startswith(descriptor["namespace"] + "/")
        or re.fullmatch(r"(/[A-Za-z][A-Za-z0-9_]*)+", descriptor["scan_ros_topic"]) is None
    ):
        raise ValueError("closed current-source SDK scan filter declaration required")
    return value


def source_body_scan_remappings(descriptor, node_name):
    """Route only CollisionMonitor's sensor input; never remap control or TF."""
    if filter_declaration(descriptor) is None or node_name != "collision_monitor":
        return []
    raw = descriptor["scan_ros_topic"]
    return [(raw, raw + "_body_filtered")]


def source_body_filter_parameters(descriptor, urdf_bytes, mesh_originals=None):
    from rosclaw.connectors.ros.context.geometry import derive_collision_envelope

    value = filter_declaration(descriptor)
    if value is None:
        raise ValueError("explicit optional filter declaration required")
    if (
        type(urdf_bytes) is not bytes
        or hashlib.sha256(urdf_bytes).hexdigest() != value["source_urdf_sha256"]
    ):
        raise ValueError("filter source URDF changed")
    geometry = derive_collision_envelope(
        urdf_bytes, base_frame=value["base_frame"], mesh_originals=mesh_originals
    )
    radius = geometry["physical_radius_m"]
    if not geometry["complete"] or type(radius) not in (int, float) or not 0 < radius <= 10:
        raise ValueError("complete current original collision geometry required")
    # Require the fixed scan transform and its actual source sensor frame.
    from gazebo_source_witness.prepare import frame_pose

    urdf = ET.fromstring(urdf_bytes)
    pose = frame_pose(urdf, value["scan_frame"], value["base_frame"])
    sensors = [
        sensor
        for sensor in urdf.findall("./gazebo/sensor")
        if sensor.get("type") in {"gpu_lidar", "lidar"}
        and sensor.findtext("gz_frame_id") == value["scan_frame"]
    ]
    if len(sensors) != 1:
        raise ValueError("unique original fixed scan sensor frame required")
    polygon = [
        [radius * math.cos(i * 2 * math.pi / 32), radius * math.sin(i * 2 * math.pi / 32)]
        for i in range(32)
    ]
    params = {
        "use_sim_time": True,
        "filter1": {
            "name": "current_source_body_only",
            "type": "laser_filters/StaticLaserScanPolygonFilter",
            "params": {
                "polygon_frame": value["base_frame"],
                "polygon": json.dumps(polygon),
                "polygon_padding": 0.0,
                "invert": False,
                "transform_timeout": 0.2,
                "footprint_topic": descriptor["namespace"] + "/source_body_filter_unused",
            },
        },
    }
    return params, {
        "source_urdf_sha256": value["source_urdf_sha256"],
        "geometry": geometry,
        "source_fixed_scan_pose": pose,
        "mask_scope": "INSCRIBED_POLYGON_OF_CURRENT_COLLISION_ENVELOPE_NO_PADDING",
        "physical_body_occupancy_of_each_masked_point": "NOT_ESTABLISHED_BY_ENVELOPE",
        "execution_authority": "NONE",
    }


def check_original_scan_bridge(descriptor, bridge):
    """Only an existing one-way native LaserScan source can feed this filter."""
    if filter_declaration(descriptor) is None:
        return
    raw = descriptor["scan_ros_topic"]
    if type(bridge) is not list:
        raise ValueError("unique original typed scan bridge required")
    selected = [row for row in bridge if type(row) is dict and row.get("ros_topic_name") == raw]
    if (
        len(selected) != 1
        or selected[0].get("direction") != "GZ_TO_ROS"
        or selected[0].get("gz_type_name") != "gz.msgs.LaserScan"
        or selected[0].get("ros_type_name") != "sensor_msgs/msg/LaserScan"
        or any(
            type(row) is dict and row.get("ros_topic_name") == raw + "_body_filtered"
            for row in bridge
        )
    ):
        raise ValueError("unique original typed scan bridge and unused filter output required")


def prepare_source_body_scan_filter(descriptor, settings, assets, output, extraction):
    """Verify before launching anything; extract one exact signed package locally."""
    value = filter_declaration(descriptor)
    if value is None:
        return None
    if (
        not Path("/.dockerenv").is_file()
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
        or settings.get("generic_source_observer") is not True
        or "governed_sink" not in settings
        or not settings.get("run_id")
        or (assets / "run_id.txt").read_text().strip() != settings["run_id"]
        or extraction.exists()
    ):
        raise ValueError("fresh isolated source-owned SDK fixture required")
    package = assets / "source-body-laser-filters.deb"
    if (
        package.is_symlink()
        or package.stat().st_size != PACKAGE_BYTES
        or hashlib.sha256(package.read_bytes()).hexdigest() != PACKAGE_SHA256
    ):
        raise ValueError("exact externally signature-verified SDK scan package required")
    check_original_scan_bridge(descriptor, yaml.safe_load((assets / "bridge.yaml").read_text()))
    mesh = assets / "collision-asset-originals.json"
    params, facts = source_body_filter_parameters(
        descriptor,
        (assets / "robot.urdf").read_bytes(),
        json.loads(mesh.read_text()) if mesh.is_file() else None,
    )
    subprocess.run(["dpkg-deb", "--extract", str(package), str(extraction)], check=True, timeout=10)
    prefix = extraction / "opt/ros/jazzy"
    binary = prefix / "lib/laser_filters/scan_to_scan_filter_chain"
    if not binary.is_file() or binary.is_symlink():
        raise ValueError("fixed scan filter executable absent")
    # Every process, including the plugin's internal TF listener, uses SIM time.
    configuration = {
        "/**": {"ros__parameters": {"use_sim_time": True}},
        descriptor["namespace"] + "/scan_to_scan_filter_chain": {"ros__parameters": params},
    }
    config = output / "source-body-filter-parameters.json"
    with config.open("x") as stream:
        json.dump(configuration, stream, indent=2)
    facts.update(package_sha256=PACKAGE_SHA256, host_or_underlay_install=False)
    with (output / "source-body-filter-source-facts.json").open("x") as stream:
        json.dump(facts, stream, indent=2)
    raw = descriptor["scan_ros_topic"]
    argv = [
        str(binary),
        "--ros-args",
        "--params-file",
        str(config),
        "-r",
        "__ns:=" + descriptor["namespace"],
        "-r",
        "scan:=" + raw,
        "-r",
        "scan_filtered:=" + raw + "_body_filtered",
    ]
    if descriptor["tf_topics_relative"]:
        argv += ["-r", "/tf:=tf", "-r", "/tf_static:=tf_static"]
    env = dict(os.environ)
    env["AMENT_PREFIX_PATH"] = str(prefix) + ":" + env.get("AMENT_PREFIX_PATH", "")
    env["LD_LIBRARY_PATH"] = str(prefix / "lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    return argv, env

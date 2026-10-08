"""Source-derived SIM differential-drive watchdog candidate, no transport.

No robot template, wheel dimensions, frame prefix or controller namespace is
inferred. Original geometry is preserved and all changes remain unexecuted
operator-fixture proposals, never hardware/provider or rosclawd permission.
"""

import hashlib
import math
from copy import deepcopy
from xml.etree import ElementTree as ET

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint


def _xml(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 5_000_000:
        raise ValueError("bounded original expanded XML source required")
    raw.decode("utf-8", errors="strict")
    if b"\x00" in raw or b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("external XML entities are not source geometry")
    return ET.fromstring(raw)


def _number(element, key):
    values = element.findall(key)
    if len(values) != 1:
        raise ValueError("one explicit source drive dimension required: " + key)
    try:
        value = float(values[0].text)
    except (ValueError, TypeError) as error:
        raise ValueError("finite source drive dimension required") from error
    if not math.isfinite(value) or not 0.001 <= value <= 10:
        raise ValueError("bounded positive source drive dimension required")
    return value


def _structure(element):
    return (
        element.tag,
        dict(element.attrib),
        (element.text or "").strip(),
        [_structure(child) for child in element],
    )


def prepare_sim_controller_source(urdf_bytes, sdf_bytes, declaration):
    """Convert the materialized SDK DiffDrive source to a bounded SIM deadman.

    Already controlled, nested, merged or unsupported source remains UNKNOWN
    by refusal; this function never fabricates absent wheel/joint information.
    """
    keys = {
        "source",
        "approved",
        "evidence_domain",
        "body_model_name",
        "frames",
        "limits",
        "controller_manager",
        "drive_controller",
        "joint_state_broadcaster",
        "odom_topic",
        "drive_velocity_topic",
        "robot_description_topic",
        "controller_parameters_path",
    }
    if type(declaration) is not dict or set(declaration) != keys:
        raise ValueError("closed explicit SIM controller source declaration required")
    if (
        declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
    ):
        raise ValueError("explicit approved simulator-only controller candidate required")
    frames = declaration["frames"]
    if (
        type(frames) is not dict
        or set(frames) != {"base", "odom"}
        or any(
            type(v) is not str or not v or len(v) > 256 or v.startswith("/")
            for v in frames.values()
        )
        or frames["base"] == frames["odom"]
    ):
        raise ValueError("distinct exact source frame names required")
    nodes = {
        key: absolute_endpoint(declaration[key])
        for key in ("controller_manager", "drive_controller", "joint_state_broadcaster")
    }
    topics = {
        key: absolute_endpoint(declaration[key])
        for key in ("odom_topic", "drive_velocity_topic", "robot_description_topic")
    }
    if len(set(nodes.values()) | set(topics.values())) != 6:
        raise ValueError("controller node and topic roles must not alias")
    path = declaration["controller_parameters_path"]
    if (
        type(path) is not str
        or not path.startswith("/evidence/")
        or not path.endswith(".yaml")
        or ".." in path.split("/")
        or len(path) > 256
    ):
        raise ValueError("explicit bounded owned controller parameter source path required")
    urdf, sdf = _xml(urdf_bytes), _xml(sdf_bytes)
    if urdf.tag != "robot" or sdf.tag != "sdf" or len(sdf) != 1 or len(sdf.findall("model")) != 1:
        raise ValueError("expanded actual URDF and one materialized SDF model required")
    model = sdf.find("model")
    if (
        model.get("name") != declaration["body_model_name"]
        or model.findall("model")
        or model.findall("include")
        or model.findall("ros2_control")
        or urdf.findall("ros2_control")
        or any("ros2control" in p.get("name", "").lower() for p in model.findall("plugin"))
    ):
        raise ValueError("exclusive uncontrolled source model required")
    links = [link.get("name") for link in urdf.findall("link")]
    if frames["base"] not in links or len(links) != len(set(links)):
        raise ValueError("exact unambiguous source Body base frame required")
    drive = [p for p in model.findall("plugin") if p.get("name") == "gz::sim::systems::DiffDrive"]
    if len(drive) != 1:
        raise ValueError("one actual SDK differential-drive source plugin required")
    drive = drive[0]
    if drive.get("filename") not in {"gz-sim-diff-drive-system", "libgz-sim8-diff-drive-system.so"}:
        raise ValueError("actual pinned Gazebo8 SDK DiffDrive source library name required")
    wheels = {side: [e.text for e in drive.findall(side + "_joint")] for side in ("left", "right")}
    wheel_names = wheels["left"] + wheels["right"]
    if (
        not all(1 <= len(v) <= 8 for v in wheels.values())
        or any(type(v) is not str or not v for v in wheel_names)
        or len(wheel_names) != len(set(wheel_names))
    ):
        raise ValueError("explicit disjoint bounded left/right source wheel joints required")
    for source in (urdf, model):
        joints = source.findall("joint")
        names = [joint.get("name") for joint in joints]
        if len(names) != len(set(names)):
            raise ValueError("ambiguous source joint identity")
        for name in wheel_names:
            matches = [j for j in joints if j.get("name") == name]
            if len(matches) != 1 or matches[0].get("type") not in {"continuous", "revolute"}:
                raise ValueError("declared wheel must be one actual rotational URDF/SDF joint")
    dimensions = {key: _number(drive, key) for key in ("wheel_separation", "wheel_radius")}
    limits = declaration["limits"]
    limit_keys = {
        "linear_velocity",
        "angular_velocity",
        "linear_acceleration",
        "angular_acceleration",
    }
    if (
        type(limits) is not dict
        or set(limits) != limit_keys
        or any(
            type(v) not in (int, float) or not math.isfinite(v) or not 0.001 <= v <= 10
            for v in limits.values()
        )
    ):
        raise ValueError("complete bounded explicit SIM motion limits required")
    resolved_limits = {}
    for name, cap in limits.items():
        bounds = {"max": float(cap), "min": -float(cap)}
        for side in ("max", "min"):
            rows = drive.findall(side + "_" + name)
            if len(rows) > 1:
                raise ValueError("ambiguous original SDK drive motion limit")
            if rows:
                try:
                    original_limit = float(rows[0].text)
                except (ValueError, TypeError) as error:
                    raise ValueError("finite original source drive limit required") from error
                if not math.isfinite(original_limit):
                    raise ValueError("finite original source drive limit required")
                bounds[side] = (min if side == "max" else max)(bounds[side], original_limit)
        if not bounds["min"] <= 0 < bounds["max"]:
            raise ValueError("source motion limits must permit a real zero command")
        resolved_limits[name] = bounds
    source_fields = {
        "left_joint",
        "right_joint",
        "wheel_separation",
        "wheel_radius",
        "topic",
        "odom_topic",
        "tf_topic",
        "frame_id",
        "child_frame_id",
        "odom_publish_frequency",
    } | {side + "_" + name for side in ("max", "min") for name in limit_keys}
    if any(child.tag not in source_fields for child in drive):
        raise ValueError("unsupported original SDK drive option cannot be silently discarded")
    control = ET.Element("ros2_control", name="ObservedSIMSourceSystem", type="system")
    ET.SubElement(
        ET.SubElement(control, "hardware"), "plugin"
    ).text = "gz_ros2_control/GazeboSimSystem"
    for name in wheel_names:
        joint = ET.SubElement(control, "joint", name=name)
        ET.SubElement(joint, "command_interface", name="velocity")
        ET.SubElement(joint, "state_interface", name="position")
        ET.SubElement(joint, "state_interface", name="velocity")
    before_urdf = _structure(urdf)
    model.remove(drive)
    before_model = _structure(model)
    urdf.append(deepcopy(control))
    model.append(deepcopy(control))
    plugin = ET.SubElement(
        model,
        "plugin",
        filename="libgz_ros2_control-system.so",
        name="gz_ros2_control::GazeboSimROS2ControlPlugin",
    )
    ET.SubElement(plugin, "parameters").text = path
    manager_ns, manager_name = nodes["controller_manager"].rsplit("/", 1)
    # Decompose the exact declared absolute name; never invent a namespace.
    ET.SubElement(plugin, "controller_manager_name").text = manager_name
    ros = ET.SubElement(plugin, "ros")
    ET.SubElement(ros, "namespace").text = manager_ns or "/"
    for key in ("drive_controller", "joint_state_broadcaster"):
        if nodes[key].rsplit("/", 1)[0] != manager_ns:
            raise ValueError(
                "SDK controller manager requires exactly the declared shared namespace"
            )
    ET.SubElement(ros, "remapping").text = "robot_description:=" + topics["robot_description_topic"]
    drive_name = nodes["drive_controller"].rsplit("/", 1)[1]
    broadcaster_name = nodes["joint_state_broadcaster"].rsplit("/", 1)[1]
    ET.SubElement(ros, "remapping").text = (
        nodes["drive_controller"] + "/odom:=" + topics["odom_topic"]
    )
    ET.SubElement(ros, "remapping").text = (
        nodes["drive_controller"] + "/cmd_vel:=" + topics["drive_velocity_topic"]
    )
    parameters = {
        nodes["controller_manager"]: {
            "ros__parameters": {
                "update_rate": 100,
                "use_sim_time": True,
                broadcaster_name: {"type": "joint_state_broadcaster/JointStateBroadcaster"},
                drive_name: {"type": "diff_drive_controller/DiffDriveController"},
            }
        },
        nodes["drive_controller"]: {
            "ros__parameters": {
                "left_wheel_names": wheels["left"],
                "right_wheel_names": wheels["right"],
                **dimensions,
                "base_frame_id": frames["base"],
                "odom_frame_id": frames["odom"],
                "tf_frame_prefix_enable": False,
                "enable_odom_tf": True,
                "publish_rate": 50.0,
                "cmd_vel_timeout": 0.2,
                "use_sim_time": True,
            }
        },
        nodes["joint_state_broadcaster"]: {"ros__parameters": {"use_sim_time": True}},
    }
    drive_params = parameters[nodes["drive_controller"]]["ros__parameters"]
    for group, prefix in (("linear", "linear.x"), ("angular", "angular.z")):
        for kind in ("velocity", "acceleration"):
            bounds = resolved_limits[group + "_" + kind]
            drive_params[prefix + ".has_" + kind + "_limits"] = True
            drive_params[prefix + ".max_" + kind] = bounds["max"]
            drive_params[prefix + ".min_" + kind] = bounds["min"]
    check_urdf, check_model = deepcopy(urdf), deepcopy(model)
    check_urdf.remove(check_urdf.find("ros2_control"))
    check_model.remove(check_model.find("ros2_control"))
    check_model.remove(
        next(p for p in check_model.findall("plugin") if p.get("name") == plugin.get("name"))
    )
    if _structure(check_urdf) != before_urdf or _structure(check_model) != before_model:
        raise ValueError(
            "source collision/joint/dynamic/sensor geometry changed during controller preparation"
        )
    outputs = {
        "robot.urdf": ET.tostring(urdf, encoding="utf-8", xml_declaration=True),
        "robot.sdf": ET.tostring(sdf, encoding="utf-8", xml_declaration=True),
    }
    report = {
        "schema_version": "rosclaw.sim_controller_source_candidate.v1",
        "role": "SOURCE_CANDIDATE_NOT_DRIVER_ADMISSION",
        "body_model_name": model.get("name"),
        "original_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "original_sdf_sha256": hashlib.sha256(sdf_bytes).hexdigest(),
        "declaration_sha256": digest(declaration),
        "source_wheel_names": wheels,
        "source_dimensions": dimensions,
        "resolved_motion_limits": resolved_limits,
        "source_noncontroller_structure_preserved": True,
        "output_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in outputs.items()},
        "controller_parameters_sha256": digest(parameters),
        "controller_watchdog_sec": 0.2,
        "physical_acceptance": "NOT_RUN",
        "live_controller_admitted": False,
        "usable_for_real_execution": False,
        "authorization": False,
    }
    return {"report": report, "source_files": outputs, "controller_parameters": parameters}

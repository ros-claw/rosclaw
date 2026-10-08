"""Generic Nav2/F2C source parameters from actual URDF and explicit SIM roles.

Only source proposals are produced. No known-body profile, ROS Node, transport,
world, compiler admission, task execution or hardware permission is created.
"""

import hashlib
import math
from copy import deepcopy
from xml.etree import ElementTree as ET

import yaml

from rosclaw.connectors.ros.context.geometry import derive_collision_envelope
from rosclaw.connectors.ros.context.sim_attachment import validate_sim_attachment
from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.sim_endpoints import absolute_endpoint, freeze_sim_spawn

NODE_ROLES = {
    "map_server": ("nav2_map_server", "map_server", "map_server"),
    "localization": ("nav2_amcl", "amcl", "amcl"),
    "controller": ("nav2_controller", "controller_server", "controller_server"),
    "planner": ("nav2_planner", "planner_server", "planner_server"),
    "smoother": ("nav2_smoother", "smoother_server", "smoother_server"),
    "behavior": ("nav2_behaviors", "behavior_server", "behavior_server"),
    "navigator": ("nav2_bt_navigator", "bt_navigator", "bt_navigator"),
    "velocity_smoother": ("nav2_velocity_smoother", "velocity_smoother", "velocity_smoother"),
    "collision_monitor": ("nav2_collision_monitor", "collision_monitor", "collision_monitor"),
    "coverage": ("opennav_coverage", "opennav_coverage", "coverage_server"),
}
TOPIC_ROLES = {"lidar", "odom", "map", "cmd_vel_raw", "cmd_vel_smoothed", "cmd_vel_safe"}
ACTION_ROLES = {
    "navigate_to_pose",
    "navigate_through_poses",
    "navigate_complete_coverage",
    "follow_path",
    "compute_path_to_pose",
    "compute_coverage_path",
    "set_initial_pose",
}


def _yaml(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= 2_000_000:
        raise ValueError("bounded original installed navigation template bytes required")

    # Source templates are immutable plain mappings. Refuse aliases and
    # duplicate keys rather than silently changing a source parameter.
    class SourceLoader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = self.construct_object(key_node, deep=deep)
                if type(key) is not str or key in mapping:
                    raise ValueError("unique string navigation parameter keys required")
                mapping[key] = self.construct_object(value_node, deep=deep)
            return mapping

    try:
        depth = 0
        for count, event in enumerate(yaml.parse(raw), start=1):
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
            if depth > 32:
                raise ValueError("bounded navigation template nesting required")
            if count > 20000 or isinstance(event, yaml.AliasEvent):
                raise ValueError("bounded navigation template without YAML aliases required")
        value = yaml.load(raw, Loader=SourceLoader)
    except yaml.YAMLError as error:
        raise ValueError("valid original navigation YAML required") from error
    if type(value) is not dict:
        raise ValueError("actual installed navigation parameter object required")
    return value


def prepare_sim_navigation_source(
    urdf_bytes, nav2_bytes, coverage_bytes, *, attachment, declaration, controller_report
):
    keys = {
        "source",
        "approved",
        "evidence_domain",
        "frames",
        "nodes",
        "topics",
        "endpoints",
        "spawn_xyyaw",
        "operation_width_m",
        "map_resolution",
        "map_yaml_path",
        "coverage_bt_xml_path",
    }
    if (
        type(declaration) is not dict
        or set(declaration) != keys
        or declaration["source"] != "simulator_operator_fixture_policy"
        or declaration["approved"] is not True
        or declaration["evidence_domain"] != "SIMULATION"
    ):
        raise ValueError("closed explicit SIM navigation source declaration required")
    frames = declaration["frames"]
    if (
        type(frames) is not dict
        or set(frames) != {"base", "odom", "map", "lidar"}
        or any(
            type(frame) is not str or not frame or len(frame) > 256 or frame.startswith("/")
            for frame in frames.values()
        )
        or len({frames[k] for k in ("base", "odom", "map")}) != 3
    ):
        raise ValueError("complete distinct exact frame roles required")
    geometry = derive_collision_envelope(urdf_bytes, base_frame=frames["base"])
    if geometry["complete"] is not True:
        raise ValueError(
            "unresolved actual collision geometry remains UNKNOWN: "
            + str(geometry["unsupported_reasons"])
        )
    robot = ET.fromstring(urdf_bytes)
    if frames["lidar"] not in {link.get("name") for link in robot.findall("link")}:
        raise ValueError("declared lidar frame must be an actual source URDF link")
    brush = validate_sim_attachment(attachment)
    width, resolution = declaration["operation_width_m"], declaration["map_resolution"]
    if (
        type(width) not in (int, float)
        or not math.isfinite(width)
        or not 0 < width <= 2 * brush["inscribed_radius_m"]
        or type(resolution) not in (int, float)
        or not math.isfinite(resolution)
        or not 0.001 <= resolution <= 1
    ):
        raise ValueError(
            "operation width must fit the explicit cleaner and map resolution must be bounded"
        )
    if (
        type(controller_report) is not dict
        or controller_report.get("schema_version") != "rosclaw.sim_controller_source_candidate.v1"
        or controller_report.get("source_noncontroller_structure_preserved") is not True
        or controller_report.get("physical_acceptance") != "NOT_RUN"
        or controller_report.get("authorization") is not False
        or controller_report.get("output_hashes", {}).get("robot.urdf")
        != hashlib.sha256(urdf_bytes).hexdigest()
    ):
        raise ValueError("navigation must use the exact candidate controlled URDF source")
    if controller_report.get("frames") != {key: frames[key] for key in ("base", "odom")}:
        raise ValueError("navigation frames must equal the exact controller source frames")
    limits = controller_report.get("resolved_motion_limits")
    if (
        type(limits) is not dict
        or set(limits)
        != {"linear_velocity", "angular_velocity", "linear_acceleration", "angular_acceleration"}
        or any(
            type(bounds) is not dict
            or set(bounds) != {"min", "max"}
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not -10 <= value <= 10
                for value in bounds.values()
            )
            or not bounds["min"] <= 0 < bounds["max"]
            for bounds in limits.values()
        )
    ):
        raise ValueError(
            "complete finite bounded controller motion limits with feasible zero required"
        )
    linear, angular = limits["linear_velocity"]["max"], limits["angular_velocity"]["max"]
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 10
        for value in (linear, angular)
    ):
        raise ValueError("exact bounded controller source velocity limits required")
    if type(declaration["nodes"]) is not dict or set(declaration["nodes"]) != set(NODE_ROLES):
        raise ValueError("all explicit navigation node roles required")
    nodes = {key: absolute_endpoint(name) for key, name in declaration["nodes"].items()}
    if len(set(nodes.values())) != len(nodes):
        raise ValueError("navigation node roles cannot alias")
    namespaces = {name.rsplit("/", 1)[0] for name in nodes.values()}
    if len(namespaces) != 1:
        raise ValueError(
            "explicit shared navigation namespace required for SDK BT/costmap internals"
        )
    if type(declaration["topics"]) is not dict or set(declaration["topics"]) != TOPIC_ROLES:
        raise ValueError("all explicit actual topic roles required")
    topics = {key: absolute_endpoint(name) for key, name in declaration["topics"].items()}
    if type(declaration["endpoints"]) is not dict or set(declaration["endpoints"]) != ACTION_ROLES:
        raise ValueError("all explicit navigation action/service roles required")
    endpoints = {key: absolute_endpoint(name) for key, name in declaration["endpoints"].items()}
    names = list(topics.values()) + list(endpoints.values())
    if len(set(names)) != len(names):
        raise ValueError("explicit navigation topic/action/service roles cannot alias")
    spawn = freeze_sim_spawn(declaration["spawn_xyyaw"])
    for key, root, suffix in (
        ("map_yaml_path", "/evidence/", ".yaml"),
        ("coverage_bt_xml_path", "/", ".xml"),
    ):
        path = declaration[key]
        if (
            type(path) is not str
            or not path.startswith(root)
            or not path.endswith(suffix)
            or ".." in path.split("/")
            or len(path) > 1024
        ):
            raise ValueError("explicit bounded source resource path required")
    nav, coverage = _yaml(nav2_bytes), _yaml(coverage_bytes)
    params = deepcopy(nav)
    # Current installed bringup sets map-server YAML in launch rather than in
    # nav2_params.yaml. Its complete source parameters are explicit here.
    params["map_server"] = {
        "ros__parameters": {
            "yaml_filename": declaration["map_yaml_path"],
            "frame_id": frames["map"],
            "use_sim_time": True,
        }
    }
    params["coverage_server"] = deepcopy(coverage["coverage_server"])
    params["controller_server"]["ros__parameters"].update(
        deepcopy(coverage["controller_server"]["ros__parameters"])
    )
    radius = geometry["physical_radius_m"]
    config = {
        role: params[source_key]["ros__parameters"]
        for role, (_, _, source_key) in NODE_ROLES.items()
    }
    for row in config.values():
        row["use_sim_time"] = True
    config["map_server"]["yaml_filename"] = declaration["map_yaml_path"]
    config["localization"].update(
        base_frame_id=frames["base"],
        odom_frame_id=frames["odom"],
        global_frame_id=frames["map"],
        scan_topic=topics["lidar"],
        set_initial_pose=False,
    )
    config["localization"].pop("initial_pose", None)
    config["navigator"].update(
        global_frame=frames["map"],
        robot_base_frame=frames["base"],
        odom_topic=topics["odom"],
        navigators=["navigate_to_pose", "navigate_through_poses", "navigate_complete_coverage"],
        navigate_complete_coverage={"plugin": "opennav_coverage_navigator/CoverageNavigator"},
        plugin_lib_names=deepcopy(coverage["bt_navigator"]["ros__parameters"]["plugin_lib_names"]),
        default_coverage_bt_xml=declaration["coverage_bt_xml_path"],
    )
    config["behavior"].update(
        local_frame=frames["odom"], global_frame=frames["map"], robot_base_frame=frames["base"]
    )
    config["collision_monitor"].update(
        base_frame_id=frames["base"],
        odom_frame_id=frames["odom"],
        cmd_vel_in_topic=topics["cmd_vel_smoothed"],
        cmd_vel_out_topic=topics["cmd_vel_safe"],
        observation_sources=["scan"],
    )
    config["collision_monitor"]["scan"].update(topic=topics["lidar"])
    # The software stop polygon contains the entire conservative Body envelope.
    config["collision_monitor"]["polygons"] = ["source_body_stop"]
    config["collision_monitor"]["source_body_stop"] = {
        "type": "polygon",
        "action_type": "stop",
        "points": str(
            [
                [radius + resolution, radius + resolution],
                [radius + resolution, -radius - resolution],
                [-radius - resolution, -radius - resolution],
                [-radius - resolution, radius + resolution],
            ]
        ),
        "min_points": 4,
        "visualize": False,
        "enabled": True,
    }
    config["velocity_smoother"].update(
        odom_topic=topics["odom"],
        max_velocity=[linear, 0.0, angular],
        min_velocity=[limits["linear_velocity"]["min"], 0.0, limits["angular_velocity"]["min"]],
        max_accel=[
            limits["linear_acceleration"]["max"],
            0.0,
            limits["angular_acceleration"]["max"],
        ],
        max_decel=[
            limits["linear_acceleration"]["min"],
            0.0,
            limits["angular_acceleration"]["min"],
        ],
    )
    config["controller"]["FollowPath"].update(
        desired_linear_vel=linear, rotate_to_heading_angular_vel=angular
    )
    config["coverage"].update(
        robot_width=2 * radius, operation_width=float(width), default_headland_width=2 * radius
    )
    output = {nodes[role]: {"ros__parameters": row} for role, row in config.items()}
    namespace = next(iter(namespaces))
    for costmap, global_frame in (
        ("global_costmap", frames["map"]),
        ("local_costmap", frames["odom"]),
    ):
        row = deepcopy(params[costmap][costmap]["ros__parameters"])
        row.update(
            use_sim_time=True,
            global_frame=global_frame,
            robot_base_frame=frames["base"],
            robot_radius=radius,
            resolution=float(resolution),
            footprint="",
        )
        row["inflation_layer"].update(inflation_radius=radius + resolution)
        observed_layers = 0
        for layer_name in row["plugins"]:
            layer = row[layer_name]
            if layer.get("plugin") in {
                "nav2_costmap_2d::ObstacleLayer",
                "nav2_costmap_2d::VoxelLayer",
            }:
                layer["observation_sources"] = "scan"
                layer["scan"].update(topic=topics["lidar"])
                observed_layers += 1
            elif layer.get("plugin") == "nav2_costmap_2d::StaticLayer":
                layer["map_topic"] = topics["map"]
            elif layer.get("plugin") != "nav2_costmap_2d::InflationLayer":
                raise ValueError("unsupported installed source costmap plugin remains UNKNOWN")
        if observed_layers != 1:
            raise ValueError("one explicit supported actual laser costmap source required")
        # SDK costmap node namespaces follow their owning server's namespace,
        # while the names are the official costmap names, not a Body prefix.
        output[namespace + "/" + costmap + "/" + costmap] = {"ros__parameters": row}
    launch = []
    action_remaps = list(endpoints.items()) + [("map", topics["map"])]
    for role, (package, executable, _) in NODE_ROLES.items():
        node_ns, name = nodes[role].rsplit("/", 1)
        remaps = list(action_remaps)
        if role in {"controller", "behavior"}:
            remaps.append(("cmd_vel", topics["cmd_vel_raw"]))
        if role == "velocity_smoother":
            remaps.extend(
                [
                    ("cmd_vel", topics["cmd_vel_raw"]),
                    ("cmd_vel_smoothed", topics["cmd_vel_smoothed"]),
                ]
            )
        launch.append(
            {
                "role": role,
                "package": package,
                "executable": executable,
                "name": name,
                "namespace": node_ns or "/",
                "remappings": remaps,
            }
        )
    report = {
        "schema_version": "rosclaw.sim_navigation_source_candidate.v1",
        "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "source_nav2_template_sha256": hashlib.sha256(nav2_bytes).hexdigest(),
        "source_coverage_template_sha256": hashlib.sha256(coverage_bytes).hexdigest(),
        "declaration_hash": digest(declaration),
        "controller_source_report_hash": digest(controller_report),
        "geometry": geometry,
        "attachment": brush,
        "nodes": nodes,
        "topics": topics,
        "spawn_xyyaw": list(spawn),
        "parameters_hash": digest(output),
        "launch_spec_hash": digest(launch),
        "requires_actual_Graph_TF_Body_source_admission": True,
        "physical_acceptance": "NOT_RUN",
        "usable_for_real_execution": False,
        "authorization": False,
    }
    return {"report": report, "parameters": output, "launch_nodes": launch}

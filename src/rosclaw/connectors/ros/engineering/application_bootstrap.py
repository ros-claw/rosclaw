"""Build a proposed Nav2 SIM stack from observed base interfaces, never execute."""

import copy
import hashlib
import json
import math
import re
from datetime import datetime

import yaml
from pydantic import Field

from rosclaw.connectors.ros.context.discovery import _fresh_frame_path, discover_body_candidate
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash

from ..intelligence.graph_contracts import GraphRecord, RosOperationalGraphV1
from ..resolver.component_compatibility import review_component_compatibility
from ..resolver.task_capabilities import ExistingRos2ControlV1, resolve_task_capability_graph

REQUIRED_COMPONENTS = (
    "nav2_bringup",
    "nav2_map_server",
    "nav2_amcl",
    "nav2_controller",
    "nav2_planner",
    "nav2_smoother",
    "nav2_behaviors",
    "nav2_bt_navigator",
    "nav2_velocity_smoother",
    "nav2_collision_monitor",
    "nav2_lifecycle_manager",
    "nav2_regulated_pure_pursuit_controller",
    "nav2_costmap_2d",
    "nav2_navfn_planner",
    "diff_drive_controller",
    "robot_state_publisher",
)
SERVERS = (
    "map_server",
    "amcl",
    "controller_server",
    "smoother_server",
    "planner_server",
    "behavior_server",
    "bt_navigator",
    "velocity_smoother",
    "collision_monitor",
)


class DriverBootstrapRequestV1(GraphRecord):
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_urdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    control: ExistingRos2ControlV1
    controller_manager: str = Field(min_length=2, max_length=512)
    namespace: str = Field(min_length=2, max_length=512)
    map_frame: str = Field(min_length=1, max_length=128)
    body_id: str = Field(min_length=1, max_length=128)
    navigation_limits: dict
    component_policies: list[dict] = Field(min_length=1, max_length=32)


def _bounded_json(value, limit=5_000_000):
    encoded = json.dumps(value, allow_nan=False)
    if len(encoded.encode()) > limit:
        raise ValueError("bounded finite original JSON required")
    return copy.deepcopy(value)


def _readback(model, node, now, max_age):
    stamp = model.observations.get("parameter_captured_at", {}).get(node)
    try:
        at = datetime.fromisoformat(stamp)
        if at.tzinfo is None or not 0 <= (now - at).total_seconds() * 1000 <= max_age:
            return None
    except (TypeError, ValueError):
        return None
    value = model.observations.get("node_parameters", {}).get(node)
    return value if type(value) is dict else None


def _parameters(nodes, name):
    row = nodes.get(name)
    if (
        type(row) is not dict
        or set(row) != {"ros__parameters"}
        or type(row["ros__parameters"]) is not dict
    ):
        raise ValueError("concrete original template node parameters required: " + name)
    return row["ros__parameters"]


def _set_nested(mapping, keys, value):
    for key in keys[:-1]:
        if key not in mapping:
            mapping[key] = {}
        if type(mapping[key]) is not dict:
            raise ValueError("unambiguous nested software template required")
        mapping = mapping[key]
    mapping[keys[-1]] = value


def compile_driver_bootstrap(
    graph, model, urdf_bytes, driver_bringup, software_template, original_manifests, request
):
    """Propose a bounded differential-drive Nav2 RPP application without Nav2.

    Only actual fresh odometry/laser/TF/description/control observations are
    prerequisites. The map frame is an explicit target supplied by the operator,
    not a claim that a map, localization or navigation action exists already.
    New World source/Body/full L1 and canonical physics remain mandatory.
    """
    graph = RosOperationalGraphV1.from_dict(graph.model_dump(mode="json"))
    request = DriverBootstrapRequestV1.model_validate(request)
    for name in (request.namespace, request.controller_manager):
        if not re.fullmatch(r"/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*", name):
            raise ValueError("explicit concrete ROS namespace and controller manager required")
    if any(ord(c) < 33 for c in request.map_frame) or request.map_frame.startswith("/"):
        raise ValueError("explicit target map frame without implicit slash normalization required")
    if type(urdf_bytes) is not bytes or not 0 < len(urdf_bytes) <= 5_000_000:
        raise ValueError("bounded original expanded URDF required")
    if (
        model.snapshot_hash != graph.snapshot_hash
        or model.compute_snapshot_hash() != graph.snapshot_hash
    ):
        raise ValueError("unchanged original model and graph required")
    digest = hashlib.sha256(urdf_bytes).hexdigest()
    if digest != request.source_urdf_sha256:
        raise ValueError("request differs from original URDF bytes")
    original_manifests = _bounded_json(original_manifests, 3_000_000)
    software_template = _bounded_json(software_template)
    driver_bringup = _bounded_json(driver_bringup)
    limits = _bounded_json(request.navigation_limits, 8192)
    control = resolve_task_capability_graph(
        model, "navigate_to_pose", request.control.model_dump(mode="json"), now=graph.reviewed_at
    )
    components = review_component_compatibility(
        model, request.component_policies, original_manifests
    )
    result = {
        "schema_version": "rosclaw.driver_navigation_bootstrap.v1",
        "snapshot_hash": model.snapshot_hash,
        "graph_hash": graph.graph_hash,
        "source_ref": graph.source_ref,
        "source_revision": request.source_revision,
        "source_urdf_sha256": digest,
        "reviewed_at": graph.reviewed_at.isoformat(),
        "evidence_domain": graph.evidence_domain,
        "request": request.model_dump(mode="json"),
        "original_inputs_hash": content_hash(
            "bootstrapinputs",
            {
                "driver_bringup": driver_bringup,
                "software_template": software_template,
                "installed_manifests": original_manifests,
            },
        ),
        "current_observations": control,
        "component_compatibility": components,
        "files": None,
        "file_sha256": None,
        "requirements": None,
        "source_policy_material": None,
        "target_capabilities": [
            "mapping.occupancy_map",
            "localization.pose_estimate",
            "navigation.navigate_to_pose",
            "safety.collision_monitor",
        ],
        "target_status": "PLANNED_NOT_OBSERVED_OR_ADMITTED",
        "execution_authority": "NONE",
        "current_l1": "NOT_EVALUATED",
        "physical_acceptance": "NOT_RUN",
        "driver_generated": False,
        "coverage_cleaning": "SEPARATE_UNSUPPORTED_BY_THIS_NAVIGATION_COMPILER",
        "validation_plan": [
            "review source-bound generated parameters BT map and existing driver adapter",
            "constructor-only type checks are not lifecycle or physics acceptance",
            "new independent owned SIM World and actual map/world/source binding",
            "fresh full L1 and new immutable Body before motion",
            "two canonical MCP rosclawd navigation receipts",
            "independent stop zero contacts source continuity unchanged500ms source250ms proof",
            "record all operator initialization and source/driver prerequisites",
        ],
    }

    def finish(status, reason):
        data = {**result, "status": status, "reasons": [reason]}
        return copy.deepcopy({"plan_hash": content_hash("driverbootstrap", data), **data})

    if (
        graph.source_status == "MISMATCH"
        or graph.snapshot_freshness != "FRESH"
        or not 0 <= model.age_ms(graph.reviewed_at) <= 500
    ):
        return finish("ABSTAIN", "original source mismatch or stale/future capture; no files")
    if (
        model.environment.get("ros_generation") != "ros2"
        or model.environment.get("use_sim_time") is not True
    ):
        return finish("NEEDS_ADAPTER", "first bootstrap profile is explicitly ROS2 simulation only")
    state = control["control_prerequisite"]["status"]
    if state == "NEEDS_DRIVER":
        return finish("NEEDS_DRIVER", control["control_prerequisite"]["reason"])
    if state != "OBSERVED_ROS2_CONTROL_INTERFACE":
        return finish("NEEDS_READ_ONLY_PROBE", control["control_prerequisite"]["reason"])
    if request.control.driver_package != "diff_drive_controller":
        return finish(
            "NEEDS_ADAPTER",
            "first bootstrap profile requires an observed existing differential ROS2 driver; no other kinematics guessed",
        )
    if type(driver_bringup) is not dict or driver_bringup.get("namespace") != request.namespace:
        return finish(
            "NEEDS_ADAPTER",
            "owned source driver namespace must explicitly match target SDK namespace",
        )
    controllers = driver_bringup.get("controllers")
    if (
        driver_bringup.get("schema_version") != "rosclaw.prepared_source_bringup.v1"
        or type(controllers) is not dict
        or set(controllers) != {"manager", "names"}
        or controllers["manager"] != request.controller_manager
        or type(controllers["names"]) is not list
        or request.control.consumer_node.rsplit("/", 1)[-1] not in controllers["names"]
        or type(driver_bringup.get("tf_topics_relative")) is not bool
        or type(driver_bringup.get("autostart")) is not bool
    ):
        return finish(
            "NEEDS_ADAPTER",
            "original closed driver bringup must name the selected observed manager/controller",
        )
    if request.control.consumer_node.rsplit("/", 1)[0] != request.namespace:
        return finish(
            "NEEDS_ADAPTER", "separate driver namespace requires an explicit supported adapter"
        )
    statuses = {row["status"] for row in components["components"]}
    if "MISSING_DEPENDENCY" in statuses:
        return finish(
            "NEEDS_DEPENDENCIES", "required installed component missing; no installation authority"
        )
    if statuses & {"VERSION_MISMATCH", "UNSUPPORTED_DECLARED_ENVIRONMENT"}:
        return finish(
            "INCOMPATIBLE_COMPONENTS", "original declared version/distro comparison failed"
        )
    missing = set(REQUIRED_COMPONENTS) - {p["package"] for p in request.component_policies}
    if missing or not components["all_declared_requirements_match"]:
        return finish(
            "NEEDS_READ_ONLY_PROBE",
            "all required exact-version policies and original installed manifests required",
        )
    body = discover_body_candidate(model, urdf_bytes, now=graph.reviewed_at)
    frames = body["frames"]
    geometry = body.get("geometry") or {}
    result["current_body_candidate"] = body
    if (
        geometry.get("complete") is not True
        or len(body.get("urdf_provenance", [])) != 1
        or not {"base", "odom", "lidar"} <= set(frames)
    ):
        return finish(
            "NEEDS_READ_ONLY_PROBE",
            "current original odometry laser frames matching description and complete collision geometry required; missing map/action is a future dependency",
        )
    stamp = body["urdf_provenance"][0].get("captured_at")
    try:
        age = (graph.reviewed_at - datetime.fromisoformat(stamp)).total_seconds() * 1000
        if not 0 <= age <= 500:
            return finish("NEEDS_READ_ONLY_PROBE", "original description readback exceeds 500ms")
    except (TypeError, ValueError):
        return finish("NEEDS_READ_ONLY_PROBE", "original description receive clock required")
    for role in ["state.odometry", "sensing.lidar"]:
        rows = body["interfaces"].get(role, [])
        if len(rows) != 1:
            return finish(
                "NEEDS_READ_ONLY_PROBE", "one current typed base stream required: " + role
            )
        topic = rows[0]["name"]
        signals = [s for s in model.signals if s.topic == topic]
        if (
            len(signals) != 1
            or signals[0].last_message_age_ms is None
            or not 0 <= signals[0].last_message_age_ms <= 500
            or not 0 <= (graph.reviewed_at - signals[0].captured_at).total_seconds() * 1000 <= 500
        ):
            return finish(
                "NEEDS_READ_ONLY_PROBE",
                "original base stream receive clocks must remain fresh; no renewed envelope",
            )
    if not _fresh_frame_path(
        model, frames["odom"], frames["base"], graph.reviewed_at
    ) or not _fresh_frame_path(model, frames["base"], frames["lidar"], graph.reviewed_at):
        return finish(
            "NEEDS_READ_ONLY_PROBE", "current unique directed odom/base/laser TF chains required"
        )
    # Retain original receive clocks for each required dynamic TF edge.
    for parent, child in ((frames["odom"], frames["base"]), (frames["base"], frames["lidar"])):
        current = child
        while current != parent:
            edge = next(e for e in model.transforms if e.child == current)
            if not edge.static and (
                edge.age_ms is None
                or not 0 <= edge.age_ms <= 500
                or not 0 <= (graph.reviewed_at - edge.captured_at).total_seconds() * 1000 <= 500
            ):
                return finish(
                    "NEEDS_READ_ONLY_PROBE", "required dynamic TF exceeds original 500ms budget"
                )
            current = edge.parent
    manager = _readback(model, request.controller_manager, graph.reviewed_at, 500)
    driver = _readback(model, request.control.consumer_node, graph.reviewed_at, 500)
    name = request.control.consumer_node.rsplit("/", 1)[-1]
    if manager is None or driver is None:
        return finish(
            "NEEDS_READ_ONLY_PROBE",
            "fresh original manager and driver parameter readbacks required",
        )
    if manager.get(name + ".type") != "diff_drive_controller/DiffDriveController":
        return finish(
            "NEEDS_ADAPTER",
            "actual selected controller class is missing or differs; no kinematics inferred from topic type",
        )
    rate = manager.get("update_rate")
    linear = driver.get("linear.x.max_velocity")
    angular = driver.get("angular.z.max_velocity")
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0
        for v in (rate, linear, angular)
    ):
        return finish(
            "NEEDS_READ_ONLY_PROBE",
            "actual finite positive controller rate and driver speed bounds required",
        )
    # Same pure limit validation as existing Body compilation; no Body is made.
    ns = request.namespace
    action = ns + "/navigate_to_pose"
    SimulationNavigationBinding(
        request.body_id,
        "0" * 64,
        action,
        ns + "/lease",
        ns + "/observation",
        request.map_frame,
        "proposed-not-a-live-run",
        **limits,
    )
    if limits["max_speed_mps"] > linear or limits["max_yaw_rate_rps"] > angular:
        return finish(
            "UNSUPPORTED_POLICY", "declared new-Body limits exceed measured existing driver limits"
        )
    params = copy.deepcopy(software_template)
    if (
        len(params) == 1
        and type(next(iter(params.values()))) is dict
        and "amcl" in next(iter(params.values()))
    ):
        params = next(iter(params.values()))
    # Nav2's installed default template omits map_server because its launch
    # supplies yaml_filename. This profile explicitly generates that node.
    params.setdefault("map_server", {"ros__parameters": {}})
    if not all(server in params for server in SERVERS):
        raise ValueError(
            "explicit reusable Nav2 software template required for every target server"
        )
    # Only this profile's selected servers are launched; do not carry unused
    # docking/route/coverage nodes or their predecessor frame references.
    selected = set(SERVERS) | {"global_costmap", "local_costmap"}
    params = {name: row for name, row in params.items() if name in selected}
    base = frames["base"]
    odom = frames["odom"]
    map_frame = request.map_frame
    scan = body["interfaces"]["sensing.lidar"][0]["name"]
    odometry = body["interfaces"]["state.odometry"][0]["name"]
    radius = geometry["physical_radius_m"]
    nav_radius = radius + max(0.01, radius * 0.02)
    amcl = _parameters(params, "amcl")
    amcl.update(
        use_sim_time=True,
        base_frame_id=base,
        odom_frame_id=odom,
        global_frame_id=map_frame,
        scan_topic=scan,
        tf_broadcast=True,
        set_initial_pose=False,
        update_min_d=0.0,
        update_min_a=0.0,
    )
    amcl.pop("initial_pose", None)
    _parameters(params, "map_server").update(
        use_sim_time=True, yaml_filename="/assets/map.yaml", frame_id=map_frame
    )
    controller = _parameters(params, "controller_server")
    controller.update(
        use_sim_time=True, controller_frequency=float(rate), controller_plugins=["FollowPath"]
    )
    controller["FollowPath"] = {
        "plugin": "nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController",
        "desired_linear_vel": float(limits["max_speed_mps"] * 0.8),
        "rotate_to_heading_angular_vel": float(limits["max_yaw_rate_rps"] * 0.8),
        "lookahead_dist": float(radius * 0.5),
        "min_lookahead_dist": float(radius * 0.25),
        "max_lookahead_dist": float(radius * 0.75),
        "curvature_lookahead_dist": float(radius * 0.5),
        "approach_velocity_scaling_dist": float(radius * 0.4),
        "min_approach_linear_velocity": float(limits["max_speed_mps"] * 0.1),
        "regulated_linear_scaling_min_radius": float(radius),
        "regulated_linear_scaling_min_speed": float(limits["max_speed_mps"] * 0.2),
        "use_collision_detection": True,
        "use_regulated_linear_velocity_scaling": True,
        "use_rotate_to_heading": True,
        "allow_reversing": False,
    }
    for value in controller.values():
        if type(value) is dict and "xy_goal_tolerance" in value:
            # A reusable software profile may intentionally reserve more of
            # the physical acceptance budget for localization error. Never
            # loosen its explicit tighter goal checker when binding a Body.
            for key, limit in (
                ("xy_goal_tolerance", limits["goal_position_tolerance_m"]),
                ("yaw_goal_tolerance", limits["goal_yaw_tolerance_rad"]),
            ):
                original = value.get(key)
                if (
                    type(original) not in (int, float)
                    or not math.isfinite(original)
                    or original <= 0
                ):
                    raise ValueError("finite positive original goal checker tolerances required")
                value[key] = float(min(original, limit * 0.5))
        if type(value) is dict and "required_movement_radius" in value:
            value.update(
                required_movement_radius=float(limits["goal_position_tolerance_m"] * 2),
                movement_time_allowance=float(min(5.0, limits["max_duration_sec"])),
            )
    for role in ["global_costmap", "local_costmap"]:
        if type(params.get(role)) is not dict or type(params[role].get(role)) is not dict:
            raise ValueError("explicit nested Nav2 costmap template required")
        cm = params[role][role]["ros__parameters"]
        requested_publish = cm.get("publish_frequency", rate / 2)
        if (
            type(requested_publish) not in (int, float)
            or not math.isfinite(requested_publish)
            or not 0 < requested_publish <= rate
        ):
            raise ValueError(
                "software costmap publication must be positive and within the observed driver-rate profile ceiling"
            )
        cm.update(
            use_sim_time=True,
            global_frame=map_frame if role == "global_costmap" else odom,
            robot_base_frame=base,
            robot_radius=float(nav_radius),
            footprint="[]",
            update_frequency=float(rate / 2),
            publish_frequency=float(max(rate / 2, requested_publish)),
        )
        if role == "local_costmap":
            cm.update(width=max(1, math.ceil(radius * 4)), height=max(1, math.ceil(radius * 4)))
        for layer in ["obstacle_layer", "voxel_layer"]:
            if layer in cm:
                _set_nested(cm, [layer, "scan", "topic"], scan)
                _set_nested(cm, [layer, "scan", "sensor_frame"], "")
        if "inflation_layer" in cm:
            _set_nested(cm, ["inflation_layer", "inflation_radius"], float(nav_radius * 1.25))
    navigator = _parameters(params, "bt_navigator")
    navigator.update(
        use_sim_time=True,
        global_frame=map_frame,
        robot_base_frame=base,
        odom_topic=odometry,
        navigators=["navigate_to_pose"],
        default_nav_to_pose_bt_xml="/assets/behavior-tree.xml",
    )
    # An empty YAML array has no ROS parameter type; leave built-in BT plugins
    # at the installed navigator default instead of emitting an untyped value.
    navigator.pop("plugin_lib_names", None)
    navigator.pop("navigate_through_poses", None)
    navigator.pop("navigate_complete_coverage", None)
    navigator.pop("default_coverage_bt_xml", None)
    _parameters(params, "planner_server").update(
        planner_plugins=["GridBased"],
        GridBased={
            "plugin": "nav2_navfn_planner::NavfnPlanner",
            "tolerance": float(limits["goal_position_tolerance_m"]),
            "use_astar": False,
            "allow_unknown": True,
        },
    )
    smoother = _parameters(params, "velocity_smoother")
    smoother.update(
        use_sim_time=True,
        odom_topic=odometry,
        max_velocity=[
            float(limits["max_speed_mps"] * 0.8),
            0.0,
            float(limits["max_yaw_rate_rps"] * 0.8),
        ],
        min_velocity=[
            -float(limits["max_speed_mps"] * 0.8),
            0.0,
            -float(limits["max_yaw_rate_rps"] * 0.8),
        ],
    )
    monitor = _parameters(params, "collision_monitor")
    monitor.update(
        use_sim_time=True,
        base_frame_id=base,
        odom_frame_id=odom,
        cmd_vel_in_topic=ns + "/cmd_vel_smoothed",
        cmd_vel_out_topic=ns + "/nav_cmd_vel",
    )
    monitor["polygons"] = ["FootprintApproach"]
    monitor["FootprintApproach"] = {
        "type": "polygon",
        "action_type": "approach",
        "footprint_topic": ns + "/local_costmap/published_footprint",
        "time_before_collision": 1.2,
        "simulation_time_step": 0.1,
        "min_points": 6,
        "visualize": False,
        "enabled": True,
    }
    monitor["observation_sources"] = ["scan"]
    _set_nested(monitor, ["scan", "topic"], scan)
    for value in monitor.values():
        if type(value) is dict and "footprint_topic" in value:
            value["footprint_topic"] = ns + "/local_costmap/published_footprint"
    behavior = _parameters(params, "behavior_server")
    behavior.update(
        use_sim_time=True,
        global_frame=map_frame,
        local_frame=odom,
        max_rotational_vel=float(limits["max_yaw_rate_rps"] * 0.8),
        min_rotational_vel=float(limits["max_yaw_rate_rps"] * 0.1),
        robot_base_frame=base,
        local_costmap_topic=ns + "/local_costmap/costmap_raw",
        global_costmap_topic=ns + "/global_costmap/costmap_raw",
        local_footprint_topic=ns + "/local_costmap/published_footprint",
        global_footprint_topic=ns + "/global_costmap/published_footprint",
    )
    bringup = copy.deepcopy(driver_bringup)
    bringup.update(
        namespace=ns,
        autostart=True,
        start_navigation_stack=True,
        scan_ros_topic=scan,
        initialization_boundary="LOCALIZATION_ACTIVE_BEFORE_NAVIGATION",
    )
    requirements = {
        "interface_names": {
            "mapping.occupancy_map": ns + "/map",
            "state.odometry": odometry,
            "sensing.lidar": scan,
            "navigation.navigate_to_pose": action,
        },
        "lifecycle_nodes": {
            "localization": ns + "/amcl",
            "planner": ns + "/planner_server",
            "controller": ns + "/velocity_smoother",
            "collision_monitor": ns + "/collision_monitor",
            "navigator": ns + "/bt_navigator",
        },
        "costmap_topics": [ns + "/local_costmap/costmap", ns + "/global_costmap/costmap"],
        "localization_topic": ns + "/amcl_pose",
        "velocity_input_topic": ns + "/cmd_vel_smoothed",
        "velocity_output_topic": ns + "/nav_cmd_vel",
        "velocity_consumer_node": ns + "/rosclaw_governed_sink",
    }
    result.update(
        requirements=requirements,
        source_policy_material={
            "schema_version": "rosclaw.sdk_navigation_policy_input.v1",
            "scope": "OPERATOR_OWNED_DISPOSABLE_SIM_NEW_BODY",
            "body_id": request.body_id,
            "policy_id": "bootstrap-source-" + digest[:16],
            "source_urdf_sha256": digest,
            "navigation_requirements_hash": content_hash("navrequirements", requirements),
            "navigation_limits": limits,
        },
        derived_frames={
            "base": base,
            "odom": odom,
            "lidar": frames["lidar"],
            "target_map": map_frame,
        },
        physical_radius_m=radius,
        nav2_radius_m=nav_radius,
    )
    allowed = {scan, odometry, "/assets/map.yaml", "/assets/behavior-tree.xml"}

    def check_references(value):
        if (
            type(value) is str
            and value.startswith("/")
            and value not in allowed
            and not value.startswith(ns + "/")
        ):
            raise ValueError(
                "unbound predecessor topic/frame/path retained in software template: " + value
            )
        if type(value) is dict:
            for v in value.values():
                check_references(v)
        elif type(value) is list:
            for v in value:
                check_references(v)

    # Fill all selected software odometry hooks before rejecting old prefixes.
    for row in params.values():
        if type(row) is dict and type(row.get("ros__parameters")) is dict:
            values = row["ros__parameters"]
            values["use_sim_time"] = True
            if "odom_topic" in values:
                values["odom_topic"] = odometry
    check_references(params)
    files = {
        "nav2.yaml": yaml.safe_dump({ns: params}, sort_keys=False),
        "bringup.json": json.dumps(bringup, indent=2) + "\n",
        "requirements.json": json.dumps(requirements, indent=2) + "\n",
        "navigation-policy.json": json.dumps(result["source_policy_material"], indent=2) + "\n",
    }
    result.update(
        files=files,
        file_sha256={name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in files.items()},
    )
    return finish(
        "CONDITIONAL_DRIVER_BOOTSTRAP_STAGING",
        "new Nav2 target generated from current base observations; map/action/lifecycle absence was not treated as a robot fault; independent map/BT/source/Body/L1 and guarded physics remain required",
    )

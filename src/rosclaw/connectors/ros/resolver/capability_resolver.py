"""Interface presence is separate from readiness, and never authorization."""

from __future__ import annotations

from datetime import UTC, datetime

from rosclaw.connectors.ros.diagnosis.engine import frame_connected
from rosclaw.connectors.ros.intelligence import RosSystemModel

from .semantics import SEMANTIC_TYPES, is_initial_pose_command


def resolve_capabilities(model: RosSystemModel, *, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(UTC)
    interfaces: dict[str, list[dict]] = {}
    for kind, type_key in [("topics", "msg_type"), ("actions", "action_type")]:
        for interface in model.graph.get(kind, []):
            if kind == "topics" and is_initial_pose_command(interface):
                continue
            semantic = SEMANTIC_TYPES.get(interface.get(type_key, ""))
            if semantic:
                interfaces.setdefault(semantic, []).append({**interface, "kind": kind})
    frames = model.body.get("frames", {})
    states = {x.name: x for x in model.lifecycle}
    signals = {x.topic: x for x in model.signals}
    bindings = model.body.get("ros_capability_bindings", {})
    if not isinstance(bindings, dict):
        bindings = {}

    def health(interface):
        signal = signals.get(interface["name"])
        if signal is None or signal.last_message_age_ms is None:
            return None
        if not -100 <= (now - signal.captured_at).total_seconds() * 1000 <= 5000:
            return False
        minimum_rate = model.body.get("minimum_rates", {}).get(signal.topic)
        if minimum_rate is not None:
            if signal.rate_hz is None:
                return None
            if signal.rate_hz < minimum_rate:
                return False
        return signal.publisher_count != 0 and (
            signal.freshness_policy == "latched" or signal.last_message_age_ms <= signal.max_age_ms
        )

    def nav_requirements(interface):
        from rosclaw.connectors.ros.diagnosis import diagnose

        namespace = interface["name"].rsplit("/", 1)[0]
        requirements = {
            "tf.fresh": None
            if any(not edge.static and edge.age_ms is None for edge in model.transforms)
            else all(
                edge.static or -edge.future_tolerance_ms <= edge.age_ms <= 1000
                for edge in model.transforms
            ),
            "runtime_diagnostics": not any(
                issue["severity"] == "blocking"
                and issue["issue_code"].startswith(("ROS_", "NAV2_"))
                for issue in diagnose(model, now=now)["issues"]
            ),
            "tf.map_to_odom": frame_connected(
                model, frames.get("map", "map"), frames.get("odom", "odom")
            ),
            "tf.odom_to_base": frame_connected(
                model, frames.get("odom", "odom"), frames.get("base", "base_link")
            ),
        }
        for role in [
            "mapping.occupancy_map",
            "localization.pose_estimate",
            "state.odometry",
            "sensing.lidar",
        ]:
            candidates = interfaces.get(role, [])
            checks = [health(i) for i in candidates]
            requirements[role] = True if True in checks else (None if None in checks else False)
        if interface.get("action_type", "").startswith("nav2_msgs/") or interface.get(
            "action_type", ""
        ).startswith("opennav_"):
            for node in ["planner_server", "controller_server", "bt_navigator"]:
                observed = states.get(namespace + "/" + node)
                requirements[node + ".active"] = (
                    None
                    if observed is None or observed.state == "UNKNOWN"
                    else observed.state == "ACTIVE"
                    and (now - observed.captured_at).total_seconds() <= 5
                )
            for key in ["localization_ready", "costmaps_fresh", "obstacle_source_configured"]:
                requirements[key] = model.navigation.get(key)
        return requirements

    results = []
    for semantic in sorted(
        set(SEMANTIC_TYPES.values())
        | {"coverage.verify", "safety.collision_monitor", "cleaning.enable", "cleaning.disable"}
    ):
        candidates: list[dict] = []
        binding = bindings.get(semantic)
        if isinstance(binding, dict) and semantic in {
            "coverage.verify",
            "safety.collision_monitor",
        }:
            requirements = {
                "body.bound": bool(model.body.get("effective_body_hash")),
                "snapshot.fresh": -100 <= model.age_ms(now) <= 5000,
            }
            if semantic == "coverage.verify":
                from rosclaw.connectors.ros.verification.coverage import CoverageVerifier

                requirements["verifier.known"] = (
                    binding.get("name") == "rosclaw.coverage_verifier.v1"
                )
                try:
                    CoverageVerifier(
                        width=1,
                        height=1,
                        resolution=1,
                        accessible_cells=[0],
                        cleaning_polygon=binding.get("cleaning_polygon", []),
                    )
                    requirements["cleaning_geometry.valid"] = True
                except (ValueError, TypeError, KeyError):
                    requirements["cleaning_geometry.valid"] = False
            else:
                observed = states.get(binding.get("name"))
                requirements["monitor.active"] = (
                    None
                    if observed is None
                    else (
                        observed.state == "ACTIVE"
                        and -0.1 <= (now - observed.captured_at).total_seconds() <= 5
                    )
                )
                parameters = model.observations.get("node_parameters", {}).get(binding.get("name"))
                if not isinstance(parameters, dict):
                    parameters = None
                requirements["monitor.configured"] = (
                    None
                    if parameters is None
                    else (
                        parameters.get("cmd_vel_in_topic") == binding.get("input_topic")
                        and parameters.get("cmd_vel_out_topic") == binding.get("output_topic")
                        and bool(parameters.get("polygons"))
                        and "scan" in parameters.get("observation_sources", [])
                        and parameters.get("scan.topic") == binding.get("sensor_topic")
                    )
                )
                sensor = next(
                    (
                        t
                        for t in interfaces.get("sensing.lidar", [])
                        if t["name"] == binding.get("sensor_topic")
                    ),
                    None,
                )
                requirements["sensor.fresh"] = None if sensor is None else health(sensor)
                for key, direction in [
                    ("input_topic", "subscribers"),
                    ("output_topic", "publishers"),
                ]:
                    topic = next(
                        (
                            t
                            for t in model.graph.get("topics", [])
                            if t.get("name") == binding.get(key)
                        ),
                        None,
                    )
                    requirements[key + ".connected"] = (
                        None if topic is None else (binding.get("name") in topic.get(direction, []))
                    )
            status = (
                "BLOCKED"
                if False in requirements.values()
                else ("UNKNOWN" if None in requirements.values() else "AVAILABLE")
            )
            candidates.append(
                {
                    "status": status,
                    "interface": {**binding, "kind": "bound"},
                    "requirements": requirements,
                }
            )
        if semantic in {"cleaning.enable", "cleaning.disable"} and isinstance(binding, dict):
            # A generic SetBool service does not identify a cleaner. Only the
            # matching compiled Body can declare this fixed interface and its
            # independent state observation. Availability grants no authority.
            service = next(
                (
                    s
                    for s in model.graph.get("services", [])
                    if s.get("name") == binding.get("name")
                ),
                None,
            )
            topic = next(
                (
                    t
                    for t in model.graph.get("topics", [])
                    if t.get("name") == binding.get("state_topic")
                ),
                None,
            )
            requirements = {
                "body.bound": bool(model.body.get("effective_body_hash")),
                "service.typed": service is not None
                and service.get("srv_type") == binding.get("srv_type") == "std_srvs/srv/SetBool",
                "command.typed": binding.get("data") is (semantic == "cleaning.enable"),
                "state.typed": None
                if topic is None
                else topic.get("msg_type") == binding.get("state_type") == "std_msgs/msg/Bool",
                "state.fresh": None if topic is None else health(topic),
                "snapshot.fresh": -100 <= model.age_ms(now) <= 5000,
            }
            status = (
                "BLOCKED"
                if False in requirements.values()
                else ("UNKNOWN" if None in requirements.values() else "AVAILABLE")
            )
            candidates.append(
                {
                    "status": status,
                    "interface": {**binding, "kind": "services"},
                    "requirements": requirements,
                }
            )
        for interface in interfaces.get(semantic, []):
            requirements = (
                nav_requirements(interface)
                if semantic.startswith(("navigation.", "coverage."))
                else {"signal.fresh": health(interface)}
            )
            requirements["snapshot.fresh"] = -100 <= model.age_ms(now) <= 5000
            status = (
                "BLOCKED"
                if False in requirements.values()
                else ("UNKNOWN" if None in requirements.values() else "AVAILABLE")
            )
            candidates.append(
                {"status": status, "interface": interface, "requirements": requirements}
            )
        rank = {"AVAILABLE": 0, "UNKNOWN": 1, "BLOCKED": 2}
        candidates.sort(key=lambda c: (rank[c["status"]], c["interface"]["name"]))
        best = (
            candidates[0]
            if candidates
            else {"status": "MISSING", "interface": None, "requirements": {}}
        )
        results.append(
            {
                "semantic_id": semantic,
                **best,
                "alternatives": candidates[1:],
                "usable_for_real_execution": False,
                "execution_entry": "request_action",
            }
        )
    return results

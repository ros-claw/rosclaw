"""Deterministic failure taxonomy. Unobserved checks remain UNKNOWN."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from rosclaw.connectors.ros.intelligence import RosSystemModel

PROFILES = {"all", "navigation", "tf", "qos", "sensors", "time"}


def frame_connected(model: RosSystemModel, parent: str, child: str) -> bool | None:
    if not model.completeness.get("tf", False):
        return None
    adjacency: dict[str, set[str]] = {}
    for edge in model.transforms:
        adjacency.setdefault(edge.parent, set()).add(edge.child)
    pending, visited = [parent], set()
    while pending:
        frame = pending.pop()
        if frame == child:
            return True
        if frame in visited:
            continue
        visited.add(frame)
        pending.extend(adjacency.get(frame, ()))
    return False


def diagnose(
    model: RosSystemModel,
    *,
    profile: str = "all",
    now: datetime | None = None,
    max_age_ms: float = 5000,
    event_bus=None,
) -> dict[str, Any]:
    if profile not in PROFILES:
        raise ValueError(f"unknown diagnostic profile: {profile}")
    now = now or datetime.now(UTC)
    issues: list[dict[str, Any]] = []

    def add(code: str, group: str, observation: Any, repair: str, severity: str = "blocking"):
        if profile not in {"all", "navigation", group}:
            return
        issues.append(
            {
                "issue_code": code,
                "severity": severity,
                "evidence": [
                    {
                        "source": model.snapshot_id,
                        "observation": observation,
                        "timestamp": model.captured_at.isoformat(),
                    }
                ],
                "confidence": 1.0,
                "hypotheses": [],
                "next_checks": [repair],
                "recommended_repairs": [repair],
                "runtime_mutation_required": False,
            }
        )

    age = model.age_ms(now)
    if age > max_age_ms or age < -100:
        add("ROS_GRAPH_001", "time", {"snapshot_age_ms": age}, "Refresh the read-only snapshot.")
    for error in dict.fromkeys(model.errors):
        add("ROS_ENV_001", "sensors", error, "Inspect the failed probe read.", "warning")

    frames = model.body.get("frames", {})
    map_frame, odom_frame, base_frame = (
        frames.get("map", "map"),
        frames.get("odom", "odom"),
        frames.get("base", "base_link"),
    )
    for code, parent, child in [
        ("ROS_TF_001", map_frame, odom_frame),
        ("ROS_TF_002", odom_frame, base_frame),
    ]:
        if frame_connected(model, parent, child) is False:
            add(code, "tf", {"parent": parent, "child": child}, "Inspect transform publishers.")
    for sensor in model.body.get("sensors", []):
        frame = sensor.get("frame")
        if frame and frame_connected(model, base_frame, frame) is False:
            add("ROS_TF_003", "tf", {"sensor_frame": frame}, "Inspect sensor frame binding.")
    parents: dict[str, set[str]] = {}
    for edge in model.transforms:
        parents.setdefault(edge.child, set()).add(edge.parent)
        if not edge.static and edge.age_ms is not None:
            if edge.age_ms > 1000:
                add(
                    "ROS_TF_004",
                    "tf",
                    edge.model_dump(mode="json"),
                    "Inspect TF timestamp and rate.",
                )
            if edge.age_ms < -edge.future_tolerance_ms:
                add(
                    "ROS_TIME_002",
                    "time",
                    edge.model_dump(mode="json"),
                    "Inspect ROS clock domains.",
                )
    for child, names in sorted(parents.items()):
        if len(names) > 1:
            add(
                "ROS_TF_005",
                "tf",
                {"child": child, "parents": sorted(names)},
                "Remove conflicting TF publishers through reviewed configuration.",
            )
    # Walk parent links, detecting cycles even when transforms are stale.
    for start in sorted(parents):
        pending: list[tuple[str, set[str]]] = [(start, set())]
        while pending:
            node, path = pending.pop()
            if node in path:
                add("ROS_TF_006", "tf", {"frame": start}, "Inspect cyclic frame configuration.")
                break
            pending.extend((p, path | {node}) for p in parents.get(node, ()))
    roots = sorted({e.parent for e in model.transforms} - set(parents))
    if len(roots) > 1:
        add("ROS_TF_007", "tf", {"roots": roots}, "Inspect disconnected frame trees.", "warning")

    topics = {t["name"] for t in model.graph.get("topics", [])}
    for expected in model.body.get("required_topics", []):
        if model.completeness.get("graph") and expected not in topics:
            add(
                "ROS_TOPIC_001", "sensors", {"topic": expected}, "Inspect required topic publisher."
            )
    from rosclaw.connectors.ros.resolver.semantics import is_initial_pose_command

    command_topics = {
        topic["name"] for topic in model.graph.get("topics", []) if is_initial_pose_command(topic)
    }
    for signal in model.signals:
        if signal.topic in command_topics:
            continue
        data = signal.model_dump(mode="json")
        if (
            signal.freshness_policy != "latched"
            and signal.last_message_age_ms is not None
            and signal.last_message_age_ms > signal.max_age_ms
        ):
            add("ROS_TOPIC_002", "sensors", data, "Inspect sensor freshness and upstream health.")
        min_rate = model.body.get("minimum_rates", {}).get(signal.topic)
        if min_rate is not None and signal.rate_hz is not None and signal.rate_hz < min_rate:
            add("ROS_TOPIC_003", "sensors", data, "Inspect sensor rate and middleware losses.")
        if signal.publisher_count == 0:
            add("ROS_TOPIC_004", "sensors", data, "Inspect missing publisher.")

    endpoints = model.qos.get("endpoints", [])
    for pub in endpoints:
        if pub.get("kind") != "publisher":
            continue
        for sub in endpoints:
            if sub.get("kind") != "subscriber" or sub.get("topic") != pub.get("topic"):
                continue
            incompatible = (
                pub.get("reliability") == "BEST_EFFORT" and sub.get("reliability") == "RELIABLE"
            ) or (
                pub.get("durability") == "VOLATILE" and sub.get("durability") == "TRANSIENT_LOCAL"
            )
            if incompatible:
                add(
                    "ROS_QOS_001",
                    "qos",
                    {"publisher": pub, "subscriber": sub},
                    "Inspect requested/offered QoS compatibility.",
                )
    for pair in model.qos.get("incompatible_pairs", []):
        add("ROS_QOS_001", "qos", pair, "Inspect native QoS compatibility evidence.")
    sim_times = model.observations.get("node_use_sim_time", {})
    if len(set(sim_times.values())) > 1:
        add(
            "ROS_TIME_001",
            "time",
            sim_times,
            "Align use_sim_time in reviewed launch configuration.",
        )
    if any(sim_times.values()) and model.observations.get("clock_advancing") is False:
        add(
            "ROS_TIME_003",
            "time",
            {"clock_advancing": False},
            "Inspect simulation clock publisher.",
        )
    if profile in {"all", "navigation"}:
        from rosclaw.connectors.ros.diagnosis.action_lifecycle import action_lifecycle_issues

        issues.extend(action_lifecycle_issues(model, now=now))
    for node in model.lifecycle:
        if node.state not in {"ACTIVE", "UNKNOWN"}:
            add(
                "ROS_LIFECYCLE_001",
                "navigation",
                node.model_dump(mode="json"),
                "Inspect lifecycle manager configuration; activation requires guarded execution.",
            )
    for key, code, repair in [
        (
            "localization_ready",
            "NAV2_LOCALIZATION_001",
            "Inspect localization, pose and TF evidence.",
        ),
        (
            "obstacle_source_configured",
            "NAV2_COSTMAP_001",
            "Inspect costmap observation_sources configuration.",
        ),
        ("costmaps_fresh", "NAV2_COSTMAP_002", "Inspect costmap updates and sensor consumers."),
        (
            "controller_progress",
            "NAV2_CONTROLLER_002",
            "Inspect controller progress and blocked routes.",
        ),
        ("controller_stable", "NAV2_CONTROLLER_001", "Inspect controller oscillation evidence."),
        (
            "collision_monitor_ready",
            "NAV2_COLLISION_001",
            "Inspect collision monitor state and sensor input.",
        ),
        (
            "coverage_plan_valid",
            "COVERAGE_PLAN_001",
            "Inspect coverage polygon, exclusions and footprint.",
        ),
        (
            "coverage_complete",
            "COVERAGE_VERIFY_001",
            "Revisit missed reachable regions and verify again.",
        ),
    ]:
        if model.navigation.get(key) is False:
            add(code, "navigation", {key: False}, repair)
    known_groups = {
        "tf": "tf",
        "qos": "qos",
        "sensors": "signals",
        "navigation": "lifecycle",
        "time": "time",
    }
    selected = (
        list(known_groups.values()) if profile in {"all", "navigation"} else [known_groups[profile]]
    )
    unknown = sorted(k for k in selected if not model.completeness.get(k, False))
    if profile in {"all", "navigation", "tf"} and any(
        not edge.static and edge.age_ms is None for edge in model.transforms
    ):
        unknown.append("tf_timing")
    status = (
        "BLOCKED"
        if any(i["severity"] == "blocking" for i in issues)
        else ("UNKNOWN" if unknown else "DEGRADED" if issues else "HEALTHY")
    )
    result = {
        "schema_version": "rosclaw.ros_diagnosis.v1",
        "snapshot_id": model.snapshot_id,
        "snapshot_hash": model.snapshot_hash,
        "profile": profile,
        "status": status,
        "unknown_checks": unknown,
        "issues": issues,
    }
    from rosclaw.connectors.ros.intelligence.evidence import emit_expert_evidence

    emit_expert_evidence(
        event_bus,
        "rosclaw.ros.diagnosis.created",
        {"robot_id": model.robot_id, "snapshot_id": model.snapshot_id, "diagnosis": result},
    )
    return result

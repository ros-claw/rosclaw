"""Provenance-bound read-only Body candidates; never installs or authorizes."""

import hashlib
from datetime import UTC, datetime

from rosclaw.connectors.ros.context.geometry import derive_collision_envelope
from rosclaw.connectors.ros.intelligence import RosSystemModel
from rosclaw.connectors.ros.resolver.semantics import SEMANTIC_TYPES


def _fresh_frame_path(model, parent, child, now):
    """Require one fresh directed ancestor chain for the requested frame pair.

    Unrelated World/model or camera transforms do not establish or invalidate
    this chain. Missing, repeated or conflicting required edges remain unknown.
    This checks observations only and grants no Body binding or action authority.
    """
    if not model.completeness.get("tf") or parent == child:
        return False
    by_child = {}
    for edge in model.transforms:
        by_child.setdefault(edge.child, []).append(edge)
    visited = set()
    frame = child
    while frame != parent:
        edges = by_child.get(frame, ())
        if frame in visited or len(edges) != 1:
            return False
        visited.add(frame)
        edge = edges[0]
        if not edge.static and not (
            edge.age_ms is not None
            and -edge.future_tolerance_ms <= edge.age_ms <= 1000
            and -100 <= (now - edge.captured_at).total_seconds() * 1000 <= 5000
        ):
            return False
        frame = edge.parent
    return True


def discover_body_candidate(model: RosSystemModel, urdf_bytes: bytes, *, now=None):
    """Select only unique fresh typed streams and matching captured URDF bytes.

    No default frame, topic, model profile, cleaner or command interface exists.
    Ambiguity and missing provenance remain explicit UNKNOWN fields. Readiness
    and immutable verified binding require subsequent existing runtime checks.
    """
    now = now or datetime.now(UTC)
    if model.snapshot_hash != model.compute_snapshot_hash():
        raise ValueError("ROS snapshot hash mismatch")
    if type(urdf_bytes) is not bytes:
        raise TypeError("expanded URDF bytes required")
    result = {
        "schema_version": "rosclaw.body_discovery_candidate.v1",
        "status": "UNKNOWN",
        "source_snapshot_id": model.snapshot_id,
        "source_snapshot_hash": model.snapshot_hash,
        "source_urdf_sha256": hashlib.sha256(urdf_bytes).hexdigest(),
        "frames": {},
        "interfaces": {},
        "geometry": None,
        "unknown_fields": [],
        "authorization": False,
        "capabilities_granted": [],
        "binding_verified": False,
        "cleaning_attachment": None,
        "physical_acceptance_level": "NOT_RUN",
    }
    if not -100 <= model.age_ms(now) <= 5000 or not model.completeness.get("graph"):
        result["unknown_fields"].append("fresh_complete_graph")
        return result
    signals = {s.topic: s for s in model.signals}
    metadata = model.observations.get("message_frames", {})

    def current(timestamp, max_age=5000):
        try:
            stamp = datetime.fromisoformat(timestamp)
            return -100 <= (now - stamp).total_seconds() * 1000 <= max_age
        except (ValueError, TypeError):
            return False

    def candidates(semantic):
        rows = []
        for topic in model.graph.get("topics", []):
            name = topic.get("name")
            if SEMANTIC_TYPES.get(topic.get("msg_type")) != semantic:
                continue
            signal = signals.get(name)
            frames = metadata.get(name, {})
            if (
                signal is not None
                and -100 <= (now - signal.captured_at).total_seconds() * 1000 <= 5000
                and signal.publisher_count is not None
                and signal.publisher_count > 0
                and signal.last_message_age_ms is not None
                and (
                    signal.freshness_policy == "latched"
                    or signal.last_message_age_ms <= signal.max_age_ms
                )
                and frames.get("source") == name
                and frames.get("frame_id")
                and (signal.freshness_policy == "latched" or current(frames.get("captured_at")))
            ):
                rows.append({"name": name, "ros_type": topic["msg_type"], "frames": dict(frames)})
        return rows

    selected = {}
    for role in ("state.odometry", "mapping.occupancy_map", "sensing.lidar"):
        rows = candidates(role)
        result["interfaces"][role] = rows
        if len(rows) == 1:
            selected[role] = rows[0]
        else:
            result["unknown_fields"].append(role + ".unique_fresh_typed_stream")
    odom = selected.get("state.odometry", {}).get("frames", {})
    mapping = selected.get("mapping.occupancy_map", {}).get("frames", {})
    lidar = selected.get("sensing.lidar", {}).get("frames", {})
    for key, value in {
        "base": odom.get("child_frame_id"),
        "odom": odom.get("frame_id"),
        "map": mapping.get("frame_id"),
        "lidar": lidar.get("frame_id"),
    }.items():
        if isinstance(value, str) and value:
            result["frames"][key] = value
        else:
            result["unknown_fields"].append("frames." + key)
    for a, b in (("map", "odom"), ("odom", "base"), ("base", "lidar")):
        if (
            a not in result["frames"]
            or b not in result["frames"]
            or not _fresh_frame_path(model, result["frames"][a], result["frames"][b], now)
        ):
            result["unknown_fields"].append(f"tf.{a}_to_{b}")
    descriptions = [
        {"node": node, **value}
        for node, value in model.observations.get("urdf_descriptions", {}).items()
        if value.get("sha256") == result["source_urdf_sha256"]
        and value.get("complete") is True
        and value.get("size_bytes") == len(urdf_bytes)
        and value.get("source") == node + "/get_parameters"
        and current(value.get("captured_at"))
    ]
    result["urdf_provenance"] = descriptions
    base = result["frames"].get("base")
    if len(descriptions) != 1:
        result["unknown_fields"].append("urdf.unique_matching_live_description")
    elif base:
        result["geometry"] = derive_collision_envelope(urdf_bytes, base_frame=base)
        if not result["geometry"].get("model_identity"):
            result["unknown_fields"].append("URDF.model_identity")
        if not result["geometry"]["complete"]:
            result["unknown_fields"].append("conservative_collision_geometry")
    actions = [
        dict(a)
        for a in model.graph.get("actions", [])
        if a.get("action_type") == "nav2_msgs/action/NavigateToPose"
    ]
    result["interfaces"]["navigation.navigate_to_pose"] = actions
    if len(actions) != 1:
        result["unknown_fields"].append("navigation.unique_typed_action")
    if not result["unknown_fields"]:
        result["status"] = "PROPOSED"
    return result

"""Fresh, pure SIM L1 review. No runtime, transport, admission or authority."""

import math
from datetime import UTC, datetime

from rosclaw.body.schema import EffectiveBody
from rosclaw.connectors.ros.context.geometry import IDENTITY, ZERO, _apply, _compose, _inverse
from rosclaw.connectors.ros.context.navigation_workspace_review import _inspect, _json, _Reader
from rosclaw.connectors.ros.context.sim_source_binding import review_sim_source_binding
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.contracts.common import content_hash


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _fresh(stamp, now, age):
    try:
        return 0 <= (now - datetime.fromisoformat(stamp)).total_seconds() * 1000 <= age
    except (ValueError, TypeError):
        return False


def _contains_circle(points, radius):
    """Require a strictly convex polygon whose every half-plane contains circle."""
    if type(points) is not list or not 3 <= len(points) <= 128:
        raise ValueError("bounded convex footprint polygon required")
    if any(type(p) is not list or len(p) != 2 or not all(_number(x) for x in p) for p in points):
        raise ValueError("finite 2D footprint points required")
    signs = []
    for i, p in enumerate(points):
        q = points[(i + 1) % len(points)]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = math.hypot(dx, dy)
        if length == 0 or not math.isfinite(length):
            raise ValueError("nondegenerate footprint edges required")
        cross = dx * -p[1] - dy * -p[0]
        signs.append(cross)
        # Also excludes self-intersections and concavity, including star polygons.
        other = [
            dx * (r[1] - p[1]) - dy * (r[0] - p[0])
            for j, r in enumerate(points)
            if j not in {i, (i + 1) % len(points)}
        ]
        if not (all(x > 0 for x in other) or all(x < 0 for x in other)):
            raise ValueError("strictly convex footprint required")
        if abs(cross) / length < radius:
            return False
    return all(x > 0 for x in signs) or all(x < 0 for x in signs)


def _frame_transform(model, frame_id, base, stamp_sec, now, age, *, static_only=False):
    """Resolve original numeric TF, never infer an odom/map identity transform."""
    rows = model.observations.get("transform_values", [])
    if type(rows) is not list or len(rows) > 4096:
        raise ValueError("bounded original TF values required")
    children = [r["child"] for r in rows]
    if len(set(children)) != len(children):
        raise ValueError("unique TF parents required")
    graph = {}
    for row in rows:
        if static_only and row["static"] is not True:
            continue
        observations = [
            r for r in model.transforms if r.parent == row["parent"] and r.child == row["child"]
        ]
        if (
            len(observations) != 1
            or observations[0].source != row["source"]
            or observations[0].captured_at.isoformat() != row["captured_at"]
        ):
            raise ValueError("original TF value/observation binding required")
        p, q = row["translation"], row["rotation_xyzw"]
        if len(p) != 3 or len(q) != 4 or not all(_number(v) for v in [*p, *q]):
            raise ValueError("finite original TF pose required")
        if abs(sum(v * v for v in q) - 1) > 1e-6:
            raise ValueError("unit TF quaternion required")
        if not row["static"] and (
            not _fresh(row["captured_at"], now, age)
            or not _number(row.get("stamp_sec"))
            or abs(row["stamp_sec"] - stamp_sec) * 1000 > 1000 + observations[0].future_tolerance_ms
        ):
            raise ValueError("time-corresponding live numeric TF required")
        x, y, z, w = q
        rotation = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        transform = (rotation, tuple(p))
        graph.setdefault(row["child"], []).append((row["parent"], transform))
        graph.setdefault(row["parent"], []).append((row["child"], _inverse(transform)))
    queue, visited, target = [(frame_id, (IDENTITY, ZERO))], set(), None
    while queue:
        frame, transform = queue.pop()
        if frame in visited:
            continue
        visited.add(frame)
        if frame == base:
            target = transform
            break
        queue.extend(
            (n, _compose(t, transform)) for n, t in graph.get(frame, []) if n not in visited
        )
    if target is None:
        raise ValueError("complete footprint-to-Body TF required")
    return target


def _footprint_in_base(model, footprint, base, now, age):
    target = _frame_transform(model, footprint["frame_id"], base, footprint["stamp_sec"], now, age)
    points = footprint["points"]
    if type(points) is not list or not 3 <= len(points) <= 128:
        raise ValueError("bounded effective footprint required")
    polygon = []
    for point in points:
        if type(point) is not list or len(point) != 3 or not all(_number(v) for v in point):
            raise ValueError("finite original footprint points required")
        rotated = _apply(target[0], point)
        transformed = [rotated[i] + target[1][i] for i in range(3)]
        if abs(transformed[2]) > 0.01:
            raise ValueError("planar base-frame footprint required")
        polygon.append(transformed[:2])
    return polygon


def review_sim_navigation_readiness(workspace, request, *, policy, policy_hash, **source_inputs):
    """Recompute G1 from raw pinned inputs, then remove all L1 deferrals.

    A READY observation expires with its original capture. It is not consumed
    by the action executor as a permit. Policies are operator inputs, not Agent
    grants. The caller must independently pin their canonical content hash.
    """
    now = source_inputs.get("now") or datetime.now(UTC)
    source_inputs = {**source_inputs, "now": now}
    keys = {
        "schema_version",
        "policy_id",
        "rationale",
        "max_age_ms",
        "variance_limits",
        "costmaps",
        "footprint_topics",
    }
    if (
        type(policy) is not dict
        or set(policy) != keys
        or policy["schema_version"] != "rosclaw.sim_navigation_readiness_policy.v1"
        or policy_hash != content_hash("navreadinesspolicy", policy)
        or any(
            type(policy[k]) is not str or not policy[k].strip() for k in ("policy_id", "rationale")
        )
        or not _number(policy["max_age_ms"])
        or not 0 < policy["max_age_ms"] <= min(request["max_age_ms"], 5000)
        or type(policy["variance_limits"]) is not dict
        or set(policy["variance_limits"]) != {"x", "y", "yaw"}
        or any(not _number(v) or not 0 < v <= 0.25 for v in policy["variance_limits"].values())
        or type(policy["footprint_topics"]) is not dict
        or set(policy["footprint_topics"]) != set(policy["costmaps"])
        or len(set(policy["footprint_topics"].values())) != 2
        or any(
            type(n) is not str or not n.startswith("/") for n in policy["footprint_topics"].values()
        )
        or type(policy["costmaps"]) is not dict
        or len(policy["costmaps"]) != 2
        or len(set(policy["costmaps"].values())) != 2
        or any(type(n) is not str or not n.startswith("/") for n in policy["costmaps"].values())
    ):
        raise ValueError("independently pinned, closed bounded operator readiness policy required")
    source = review_sim_source_binding(workspace, request, **source_inputs)
    result = {
        "schema_version": "rosclaw.sim_navigation_readiness.v1",
        "status": "UNKNOWN",
        "source_binding": source,
        "policy_hash": policy_hash,
        "source_snapshot_hash": source_inputs["model"].snapshot_hash,
        "manifest_hash": request["manifest_hash"],
        "body_snapshot_hash": request["body_snapshot_hash"],
        "sim_run_id": request["sim_run_id"],
        "reviewed_at": now.isoformat(),
        "blocked_reasons": [],
        "unknown_reasons": [],
        "execution_authority": "NONE",
        "real_authority": False,
        "workspace_modified": False,
        "L2": "NOT_RUN",
        "unverified_checks": [
            "obstacle_response",
            "speed_limit_enforcement",
            "daemon_admission",
            "physical_navigation",
        ],
    }
    unknown, blocked = result["unknown_reasons"], result["blocked_reasons"]
    unknown.extend("SOURCE:" + r for r in source["unknown_reasons"])
    blocked.extend("SOURCE:" + r for r in source["mismatch_reasons"])
    review = source.get("navigation_observation_review")
    if review is None:
        unknown.append("FRESH_NAVIGATION_REVIEW_UNAVAILABLE")
    else:
        # No localization or other navigation check is deferred in L1.
        unknown.extend("NAVIGATION:" + r for r in review["unknown_fields"])
    try:
        model = source_inputs["model"]
        age = policy["max_age_ms"]
        if not 0 <= model.age_ms(now) <= age or not _fresh(source["observed_at"], now, age):
            unknown.append("STALE_L1_CAPTURE")
        with _Reader(workspace) as reader:
            files = reader.read()
            _inspect(files, request["manifest_hash"], request["body_snapshot_hash"], now)
            body = EffectiveBody.from_dict(_json(files["body/refs/effective_body.json"]))
            binding = SimulationNavigationBinding.from_body(body)
            if (
                binding.body_snapshot_hash != request["body_snapshot_hash"]
                or binding.run_id != request["sim_run_id"]
            ):
                blocked.append("BODY_RUN_MISMATCH")
            requirements = _json(files["navigation-requirements.json"])
            if set(policy["costmaps"]) != set(requirements["costmap_topics"]):
                raise ValueError("explicit exact costmap topic associations required")
            limits = policy["variance_limits"]
            if (
                max(limits["x"], limits["y"]) > binding.goal_position_tolerance_m**2
                or limits["yaw"] > binding.goal_yaw_tolerance_rad**2
            ):
                blocked.append("LOCALIZATION_POLICY_EXCEEDS_BODY_GOAL_PRECISION")
            localization = requirements["localization_topic"]
            quality = [
                q
                for q in model.observations.get("localization_quality", [])
                if q.get("source") == localization
            ]
            if len(quality) != 1 or not _fresh(quality[0].get("captured_at"), now, age):
                unknown.append("FRESH_ORIGINAL_LOCALIZATION_COVARIANCE_REQUIRED")
            else:
                covariance = quality[0].get("covariance")
                if (
                    type(covariance) is not list
                    or len(covariance) != 36
                    or not all(_number(v) for v in covariance)
                ):
                    unknown.append("COMPLETE_FINITE_LOCALIZATION_COVARIANCE_REQUIRED")
                else:
                    axes = (0, 1, 5)
                    c = [[covariance[i * 6 + j] for j in axes] for i in axes]
                    diagonal = [c[i][i] for i in range(3)]
                    if any(
                        quality[0].get(k + "_variance") != diagonal[i]
                        for i, k in enumerate(("x", "y", "yaw"))
                    ):
                        blocked.append("LOCALIZATION_COVARIANCE_SUMMARY_MISMATCH")
                    determinant = (
                        c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
                        - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
                        + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
                    )
                    if (
                        any(
                            not math.isclose(c[i][j], c[j][i], rel_tol=1e-12, abs_tol=1e-15)
                            for i in range(3)
                            for j in range(3)
                        )
                        or any(
                            v < 0 or v > limits[k]
                            for v, k in zip(diagonal, ("x", "y", "yaw"), strict=True)
                        )
                        or any(
                            c[i][i] * c[j][j] - c[i][j] ** 2 < 0
                            for i in range(3)
                            for j in range(i + 1, 3)
                        )
                        or determinant < 0
                    ):
                        blocked.append("UNSAFE_PLANAR_LOCALIZATION_COVARIANCE")
            if review:
                radius = review["candidate"]["geometry"]["physical_radius_m"]
                frames = review["candidate"]["frames"]
                lidar = review["candidate"]["interfaces"]["sensing.lidar"][0]["name"]
                for topic, node in policy["costmaps"].items():
                    params = model.observations.get("node_parameters", {}).get(node, {})
                    stamp = model.observations.get("parameter_captured_at", {}).get(node)
                    rows = [t for t in model.graph.get("topics", []) if t.get("name") == topic]
                    if (
                        not _fresh(stamp, now, age)
                        or len(rows) != 1
                        or rows[0].get("publishers") != [node]
                        or params.get("use_sim_time") is not True
                        or params.get("global_frame") not in {frames["map"], frames["odom"]}
                    ):
                        unknown.append("FRESH_BOUND_COSTMAP_CONFIGURATION:" + topic)
                    plugins = params.get("plugins", [])
                    if type(plugins) is not list or not any(
                        params.get(p + ".plugin")
                        in {"nav2_costmap_2d::ObstacleLayer", "nav2_costmap_2d::VoxelLayer"}
                        and params.get(p + ".enabled") is True
                        and type(params.get(p + ".observation_sources")) is str
                        and any(
                            params.get(p + "." + s + ".topic") == lidar
                            for s in params[p + ".observation_sources"].split()
                        )
                        for p in plugins
                    ):
                        unknown.append("COSTMAP_LIDAR_SOURCE_UNVERIFIED:" + topic)
                    base_transform = _frame_transform(
                        model,
                        params.get("robot_base_frame"),
                        frames["base"],
                        0,
                        now,
                        age,
                        static_only=True,
                    )
                    if (
                        any(abs(base_transform[0][2][i]) > 1e-6 for i in (0, 1))
                        or abs(base_transform[0][2][2] - 1) > 1e-6
                    ):
                        raise ValueError("planar fixed costmap base transform required")
                    footprint = params.get("footprint")
                    if type(footprint) is not str:
                        unknown.append("EXPLICIT_COSTMAP_FOOTPRINT_REQUIRED:" + topic)
                    else:
                        polygon = _json(footprint) if footprint.strip() else []
                        if polygon:
                            transformed = []
                            for point in polygon:
                                rotated = _apply(base_transform[0], [*point, 0])
                                transformed.append(
                                    [rotated[i] + base_transform[1][i] for i in range(2)]
                                )
                            contains = _contains_circle(transformed, radius)
                        else:
                            nav_radius = params.get("robot_radius")
                            if not _number(nav_radius) or nav_radius <= 0:
                                raise ValueError("finite positive costmap circle required")
                            contains = nav_radius >= radius + math.hypot(*base_transform[1][:2])
                        if not contains:
                            blocked.append("COSTMAP_FOOTPRINT_UNDERSIZES_BODY:" + topic)
                    footprint_topic = policy["footprint_topics"][topic]
                    observed = model.observations.get("costmap_footprints", {}).get(footprint_topic)
                    pub = [
                        t for t in model.graph.get("topics", []) if t.get("name") == footprint_topic
                    ]
                    header = model.observations.get("message_frames", {}).get(footprint_topic, {})
                    if (
                        not observed
                        or observed.get("source") != footprint_topic
                        or not _fresh(observed.get("captured_at"), now, age)
                        or len(pub) != 1
                        or pub[0].get("publishers") != [node]
                        or pub[0].get("msg_type") != "geometry_msgs/msg/PolygonStamped"
                        or not _number(header.get("stamp_age_ms"))
                        or not 0 <= header["stamp_age_ms"] <= min(age, 1000)
                        or observed.get("frame_id") != params.get("global_frame")
                    ):
                        unknown.append("EFFECTIVE_PUBLISHED_FOOTPRINT_READBACK_REQUIRED:" + topic)
                    else:
                        polygon = _footprint_in_base(model, observed, frames["base"], now, age)
                        if not _contains_circle(polygon, radius):
                            blocked.append("EFFECTIVE_FOOTPRINT_UNDERSIZES_BODY:" + topic)
            capture = _json(source_inputs["capture_bytes"])
            sample = capture["physics_witness"]["sample"]
            if not binding.stopped(sample):
                blocked.append("BODY_NOT_STOPPED_WITH_LEASE_CLOSED")
            reader.unchanged()
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OSError,
        OverflowError,
        IndexError,
    ) as exc:
        unknown.append("UNVERIFIED_L1_FACT:" + str(exc)[:500])
    result["unknown_reasons"] = sorted(set(unknown))
    result["blocked_reasons"] = sorted(set(blocked))
    result["status"] = "BLOCKED" if blocked else "UNKNOWN" if unknown else "READY_FOR_SIM_TEST"
    return result

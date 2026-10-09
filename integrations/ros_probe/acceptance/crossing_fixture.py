"""Bounded owned SIM blocker traversal; no robot transport or acceptance authority."""

import math
from pathlib import Path

from audit_cursor import AuditCursor
from prepared_obstacle import PlacementClearanceUnavailableError, pose_request


def crossing_waypoints(start, policy):
    if type(policy) is not dict or set(policy) != {
        "crossing_end_xy",
        "step_m",
        "interval_sim_sec",
        "maximum_crossing_sim_sec",
    }:
        raise ValueError("closed explicit crossing geometry and time policy required")
    end = policy["crossing_end_xy"]
    if any(
        type(point) is not list
        or len(point) != 2
        or any(type(v) not in (int, float) or not -1.2 <= v <= 1.2 for v in point)
        for point in (start, end)
    ):
        raise ValueError("bounded actual scene crossing endpoints required")
    step, interval, maximum = (
        policy[k] for k in ("step_m", "interval_sim_sec", "maximum_crossing_sim_sec")
    )
    if (
        type(step) not in (int, float)
        or not 0.01 <= step <= 0.05
        or type(interval) not in (int, float)
        or not 0.25 <= interval <= 1
        or type(maximum) not in (int, float)
        or not 10 <= maximum <= 30
    ):
        raise ValueError("bounded conservative crossing step, interval and duration required")
    distance = math.dist(start, end)
    count = math.ceil(distance / step)
    if not 0.5 <= distance <= 2.4 or not 10 <= count * interval <= maximum or count > 240:
        raise ValueError("nontrivial bounded crossing lasting 10..30 nominal SIM seconds required")
    return [[start[j] + (end[j] - start[j]) * i / count for j in (0, 1)] for i in range(count + 1)]


def crossing_pose_request(fixture, body, sample, *, name, previous_xy, target_xy):
    """Retain original target guard and screen the entire short scene-only segment."""
    if (
        type(previous_xy) not in (list, tuple)
        or len(previous_xy) != 2
        or any(type(v) not in (int, float) or not -100 <= v <= 100 for v in previous_xy)
    ):
        raise ValueError("finite actual previous crossing pose required")
    request = pose_request(fixture, body, sample, name=name, x=target_xy[0], y=target_xy[1])
    if math.dist(previous_xy, target_xy) > 0.050000001:
        raise ValueError("owned crossing cannot jump over intermediate scene positions")
    obstacle = next(row for row in fixture["obstacles"] if row["name"] == name)
    delta = [target_xy[i] - previous_xy[i] for i in (0, 1)]
    length2 = sum(v * v for v in delta)
    fraction = (
        0
        if not length2
        else min(
            1,
            max(
                0,
                sum((sample[k] - previous_xy[i]) * delta[i] for i, k in enumerate(("x", "y")))
                / length2,
            ),
        )
    )
    nearest = [previous_xy[i] + fraction * delta[i] for i in (0, 1)]
    original_margin = body["physical_radius_m"] + math.hypot(*obstacle["box_size"]) / 2 + 0.1
    if math.dist(nearest, [sample["x"], sample["y"]]) <= original_margin:
        raise PlacementClearanceUnavailableError(
            "crossing segment violates original body clearance"
        )
    return request


def crossing_intersects_swaths(start, end, row, *, frame_id):
    """Require a transverse interior intersection with actual passive swath markers."""
    p = row.get("payload", {})
    if (
        row.get("kind") != "swaths"
        or p.get("topic") != "/coverage_server/swaths"
        or p.get("frame_id") != frame_id
        or p.get("marker_type") != 5
        or p.get("marker_action") != 0
    ):
        return False
    points = p.get("points")
    if type(points) is not list or len(points) > 20000 or len(points) % 2:
        raise ValueError("bounded original LINE_LIST swath marker required")
    if any(
        type(q) is not list
        or len(q) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in q)
        for q in points
    ):
        raise ValueError("finite original swath points required")
    a = [end[i] - start[i] for i in (0, 1)]
    for left, right in zip(points[::2], points[1::2], strict=True):
        b = [right[i] - left[i] for i in (0, 1)]
        det = a[0] * b[1] - a[1] * b[0]
        if abs(det) <= 0.1 * math.hypot(*a) * math.hypot(*b) or abs(det) < 1e-12:
            continue
        offset = [left[i] - start[i] for i in (0, 1)]
        t = (offset[0] * b[1] - offset[1] * b[0]) / det
        u = (offset[0] * a[1] - offset[1] * a[0]) / det
        if 0.001 < t < 0.999 and 0.001 < u < 0.999:
            return True
    return False


class MainCoverageWindow:
    """Follow existing daemon diagnostics; no action dispatch or completion claims."""

    def __init__(self, root, binding):
        self.root, self.binding = Path(root), binding
        self.cursors, self.active = {}, {}
        self.coverage_actions = set()
        self.ever_started = False
        self.finished = False

    def poll(self):
        paths = list((self.root / "actions").glob("coverage-audit-*.jsonl"))
        if len(paths) > 16 or not set(self.cursors) <= set(paths):
            raise ValueError("bounded unreplaced daemon action audit set required")
        for path in paths:
            cursor = self.cursors.setdefault(path, AuditCursor(path, run_id=self.binding["run_id"]))
            for row in cursor.poll():
                if row.get("body_snapshot_hash") != self.binding["body_snapshot_hash"]:
                    raise ValueError("main coverage audit belongs to another Body")
                action = row.get("action_id")
                if type(action) is not str or not action:
                    raise ValueError("actual daemon action audit identity required")
                payload = row["payload"]
                if (
                    row["kind"] == "action_admitted"
                    and payload.get("capability_id") == "coverage.execute"
                ):
                    self.coverage_actions.add(action)
                if row["kind"] == "goal_started" and payload.get("stage") == "MAIN_COVERAGE":
                    goal = payload.get("nav_goal_id")
                    if (
                        action not in self.coverage_actions
                        or type(goal) is not str
                        or not goal
                        or self.active
                        or self.ever_started
                    ):
                        raise ValueError("one source-admitted main coverage goal required")
                    self.active[goal] = {
                        "action_id": action,
                        "nav_goal_id": goal,
                        "audit_event_sha256": row["artifact_sha256"],
                    }
                    self.ever_started = True
                if (
                    row["kind"] == "goal_ended"
                    and self.active.pop(payload.get("nav_goal_id"), None) is not None
                ):
                    self.finished = True
        if len(self.active) > 1:
            raise ValueError("ambiguous active main coverage goal")
        return next(iter(self.active.values()), None)

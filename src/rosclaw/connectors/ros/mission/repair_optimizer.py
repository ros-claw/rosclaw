"""Bounded pose ranking. Static predictions neither dispatch nor grant credit."""

import heapq
import math
import time
from dataclasses import dataclass

from rosclaw.connectors.ros.verification.coverage import point_in_polygon


@dataclass(frozen=True)
class RepairPose:
    x: float
    y: float
    yaw: float
    predicted_new_cells: tuple[int, ...]
    predicted_access_distance_m: float
    predicted_heading_change_rad: float
    estimated_cost_sec: float
    utility: float
    center_cell: int


@dataclass(frozen=True)
class RepairSelection:
    status: str
    poses: tuple[RepairPose, ...] = ()
    elapsed_ms: float = 0.0
    evaluated_poses: int = 0
    cost_model: str = "STATIC_LEGAL_CENTER_GRID_PREDICTION_ONLY"
    reward_model: str = "NOMINAL_SAMPLED_FOOTPRINT_PREDICTION_ONLY"
    predicted_dispatch_cost_sec: float | None = None


class _BudgetExceededError(Exception):
    pass


def _wrapped(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def rank_repair_poses(
    grid,
    legal_centers,
    remaining_cells,
    current_pose,
    *,
    attempts=None,
    swath_yaw=0.0,
    budget_ms=500.0,
    drive_speed_mps=0.2,
    turn_speed_radps=1.8,
    goal_overhead_sec=2.896155560857831,
    beam_width=3,
    shortlist_size=24,
    robust_footprint=False,
    shared_sequence_overhead=False,
):
    """Return at most two predicted poses for an explicitly selected dispatch mode.

    Legal centers come from the existing Body/map preflight. Eight-neighbor
    costs prohibit corner cutting. This static estimate does not establish
    live Nav2 reachability or motion permission. Each call copies the actual
    missed mask; only independent measured poses may update coverage.
    """
    if type(robust_footprint) is not bool:
        raise ValueError("robust footprint mode must be boolean")
    if type(shared_sequence_overhead) is not bool:
        raise ValueError("shared sequence overhead mode must be boolean")
    start = time.monotonic()
    if (
        not all(
            math.isfinite(x) and x > 0
            for x in [budget_ms, drive_speed_mps, turn_speed_radps, goal_overhead_sec]
        )
        or not 1 <= beam_width <= shortlist_size <= 64
    ):
        raise ValueError("ranking cost/budget parameters must be finite, positive and bounded")
    if not all(math.isfinite(current_pose[k]) for k in ["x", "y", "yaw"]):
        raise ValueError("current pose must be finite")
    if not math.isfinite(swath_yaw):
        raise ValueError("swath yaw must be finite")
    evaluated = 0

    def check_budget():
        if (time.monotonic() - start) * 1000 > budget_ms:
            raise _BudgetExceededError

    def finish(status, poses=(), dispatch_cost=None):
        return RepairSelection(
            status,
            tuple(poses),
            (time.monotonic() - start) * 1000,
            evaluated,
            cost_model=(
                "STATIC_LEGAL_CENTER_GRID_SHARED_SEQUENCE_OVERHEAD_PREDICTION_ONLY"
                if shared_sequence_overhead
                else "STATIC_LEGAL_CENTER_GRID_PREDICTION_ONLY"
            ),
            reward_model=(
                "NINE_ONE_CELL_TRANSLATIONS_NOT_CALIBRATED_PROBABILITY"
                if robust_footprint
                else "NOMINAL_SAMPLED_FOOTPRINT_PREDICTION_ONLY"
            ),
            predicted_dispatch_cost_sec=dispatch_cost if shared_sequence_overhead else None,
        )

    try:
        # Bound precomputation too, including unusually large externally supplied maps.
        if len(legal_centers) > 5000:
            return finish("BUDGET_EXCEEDED")
        width, height, res = grid["width"], grid["height"], grid["resolution"]
        ox, oy = grid["origin"]
        polygon = tuple(tuple(p) for p in grid["cleaning_polygon"])
        if len(polygon) > 256:
            return finish("BUDGET_EXCEEDED")
        remaining = frozenset(remaining_cells)
        retries = dict(attempts or {})
        nodes = {}
        for x, y in legal_centers:
            check_budget()
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError("legal centers must be finite")
            col, row = round((x - ox) / res - 0.5), round((y - oy) / res - 0.5)
            if not 0 <= col < width or not 0 <= row < height:
                raise ValueError("legal center lies outside the supplied map")
            if not math.isclose(x, ox + (col + 0.5) * res, abs_tol=1e-6) or not math.isclose(
                y, oy + (row + 0.5) * res, abs_tol=1e-6
            ):
                raise ValueError("legal centers must align with the supplied map cell centers")
            nodes[row * width + col] = (x, y)
        if not nodes or not remaining:
            return finish("NO_CANDIDATE")
        if not remaining <= set(grid["accessible_cells"]):
            raise ValueError("missed mask differs from fixed accessible denominator")

        def distances(source):
            costs, heap = {source: 0.0}, [(0.0, source)]
            popped = 0
            while heap:
                distance, cell = heapq.heappop(heap)
                if distance != costs[cell]:
                    continue
                popped += 1
                if popped % 32 == 0:
                    check_budget()
                col, row = cell % width, cell // width
                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]:
                    if not 0 <= col + dx < width or not 0 <= row + dy < height:
                        continue
                    neighbor = cell + dy * width + dx
                    if neighbor not in nodes:
                        continue
                    if dx and dy and (cell + dx not in nodes or cell + dy * width not in nodes):
                        continue
                    cost = distance + res * math.hypot(dx, dy)
                    if cost < costs.get(neighbor, math.inf):
                        costs[neighbor] = cost
                        heapq.heappush(heap, (cost, neighbor))
            return costs

        source = min(
            nodes,
            key=lambda c: (
                (nodes[c][0] - current_pose["x"]) ** 2 + (nodes[c][1] - current_pose["y"]) ** 2
            ),
        )
        access = distances(source)
        entry = math.hypot(
            nodes[source][0] - current_pose["x"], nodes[source][1] - current_pose["y"]
        )
        # Both initial heading and swath-aligned/diagonal orientations are considered.
        headings = tuple(
            dict.fromkeys(
                round(_wrapped(a), 8)
                for a in [
                    current_pose["yaw"],
                    swath_yaw,
                    swath_yaw + math.pi,
                    swath_yaw + math.pi / 2,
                    swath_yaw - math.pi / 2,
                    swath_yaw + math.pi / 4,
                    swath_yaw - math.pi / 4,
                    swath_yaw + 3 * math.pi / 4,
                    swath_yaw - 3 * math.pi / 4,
                ]
            )
        )
        radius = math.ceil(max(math.hypot(x, y) for x, y in polygon) / res)
        offsets = {}
        for yaw in headings:
            check_budget()
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotated = [(x * cosine - y * sine, x * sine + y * cosine) for x, y in polygon]
            selected = []
            for dy in range(-min(radius, height), min(radius, height) + 1):
                check_budget()
                for dx in range(-min(radius, width), min(radius, width) + 1):
                    if dx % 32 == 0:
                        check_budget()
                    if point_in_polygon(dx * res, dy * res, rotated):
                        selected.append((dx, dy))
            offsets[yaw] = tuple(selected)

        def gain(cells):
            return sum(max(0, 3 - retries.get(c, 0)) for c in cells)

        # These are fixed prediction scenarios, not localization estimates,
        # collision envelopes, reachable centers or measured cleaning credit.
        shifts = tuple((dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
        kernels = {}
        if robust_footprint:
            for yaw, footprint in offsets.items():
                counts = {}
                for sx, sy in shifts:
                    check_budget()
                    for dx, dy in footprint:
                        key = (dx + sx, dy + sy)
                        counts[key] = counts.get(key, 0) + 1
                kernels[yaw] = tuple(counts.items())

        def robust_gain(col, row, yaw):
            return sum(
                count * max(0, 3 - retries.get((row + dy) * width + col + dx, 0))
                for (dx, dy), count in kernels[yaw]
                if 0 <= col + dx < width
                and 0 <= row + dy < height
                and (row + dy) * width + col + dx in remaining
            ) / len(shifts)

        def shifted_footprints(pose):
            col, row = pose.center_cell % width, pose.center_cell // width
            return tuple(
                frozenset(
                    (row + dy + sy) * width + col + dx + sx
                    for dx, dy in offsets[pose.yaw]
                    if 0 <= col + dx + sx < width
                    and 0 <= row + dy + sy < height
                    and (row + dy + sy) * width + col + dx + sx in remaining
                )
                for sx, sy in shifts
            )

        def heading_cost(x, y, yaw, origin):
            distance = math.hypot(x - origin["x"], y - origin["y"])
            if distance < res / 2:
                return abs(_wrapped(yaw - origin["yaw"]))
            bearing = math.atan2(y - origin["y"], x - origin["x"])
            return abs(_wrapped(bearing - origin["yaw"])) + abs(_wrapped(yaw - bearing))

        ranked = []
        for cell, (x, y) in nodes.items():
            if cell not in access:
                continue
            col, row = cell % width, cell // width
            distance = access[cell] + entry
            for yaw in headings:
                check_budget()
                cells = tuple(
                    sorted(
                        (row + dy) * width + col + dx
                        for dx, dy in offsets[yaw]
                        if 0 <= col + dx < width
                        and 0 <= row + dy < height
                        and (row + dy) * width + col + dx in remaining
                    )
                )
                evaluated += 1
                reward = gain(cells)
                if not reward:
                    continue
                if robust_footprint:
                    reward = robust_gain(col, row, yaw)
                turn = heading_cost(x, y, yaw, current_pose)
                cost = distance / drive_speed_mps + turn / turn_speed_radps + goal_overhead_sec
                ranked.append(
                    RepairPose(x, y, yaw, cells, distance, turn, cost, reward / cost, cell)
                )
        ranked.sort(key=lambda p: (-p.utility, p.estimated_cost_sec, p.center_cell, p.yaw))
        # Many yaw variants can cover exactly the same cells. Filling a small
        # beam with those variants hides other holes and defeats lookahead.
        # Represent separate measured-hole components before adding global
        # utility leaders; predictions still grant no measured cleaning credit.
        components, unassigned = {}, set(remaining)
        while unassigned:
            check_budget()
            root = min(unassigned)
            pending = [root]
            unassigned.remove(root)
            while pending:
                check_budget()
                cell = pending.pop()
                components[cell] = root
                col, row = cell % width, cell // width
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    neighbor = cell + dy * width + dx
                    if 0 <= col + dx < width and 0 <= row + dy < height and neighbor in unassigned:
                        unassigned.remove(neighbor)
                        pending.append(neighbor)
        leaders = {}
        for pose in ranked:
            check_budget()
            for component in {components[c] for c in pose.predicted_new_cells}:
                leaders.setdefault(component, pose)
        diverse = sorted(
            leaders.values(), key=lambda p: (-p.utility, p.estimated_cost_sec, p.center_cell, p.yaw)
        )
        shortlist, signatures = [], set()
        for pose in diverse + ranked:
            check_budget()
            if pose.predicted_new_cells in signatures:
                continue
            signatures.add(pose.predicted_new_cells)
            shortlist.append(pose)
            if len(shortlist) == shortlist_size:
                break
        shortlist.sort(key=lambda p: (-p.utility, p.estimated_cost_sec, p.center_cell, p.yaw))
        if not shortlist:
            return finish("NO_CANDIDATE")
        best, best_utility = (shortlist[0],), shortlist[0].utility
        best_cost = shortlist[0].estimated_cost_sec
        scenarios = {p: shifted_footprints(p) for p in shortlist} if robust_footprint else {}
        seen = set()
        for first in shortlist:
            if first.center_cell in seen:
                continue
            seen.add(first.center_cell)
            if len(seen) > beam_width:
                break
            access_next = distances(first.center_cell)
            for second in shortlist:
                check_budget()
                if second.center_cell == first.center_cell or second.center_cell not in access_next:
                    continue
                added = set(second.predicted_new_cells) - set(first.predicted_new_cells)
                if not gain(added):
                    continue
                turn = heading_cost(
                    second.x, second.y, second.yaw, {"x": first.x, "y": first.y, "yaw": first.yaw}
                )
                cost = first.estimated_cost_sec + access_next[second.center_cell] / drive_speed_mps
                # One continuous Nav2 action pays its dispatch overhead once.
                # The legacy mode retains its original per-target estimate.
                cost += turn / turn_speed_radps + (
                    0.0 if shared_sequence_overhead else goal_overhead_sec
                )
                utility = (gain(first.predicted_new_cells) + gain(added)) / cost
                if robust_footprint:
                    utility = (
                        sum(
                            gain(a | b)
                            for a, b in zip(scenarios[first], scenarios[second], strict=True)
                        )
                        / len(shifts)
                        / cost
                    )
                if utility > best_utility:
                    best, best_utility = (first, second), utility
                    best_cost = cost
        check_budget()
        return finish("READY", best, best_cost)
    except _BudgetExceededError:
        # Partial rankings do not masquerade as completed search.
        return finish("BUDGET_EXCEEDED")

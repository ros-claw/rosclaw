"""Read-only bounded proposals over successfully time-paired SIM observations.

Route reachability must be supplied from the configured legal-center graph at
this same snapshot, not guessed from the absence of occupied brush cells.
This module never dispatches, waits, acknowledges or creates execution receipts.
"""

import math
import time
from collections import deque

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest
from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery
from rosclaw.connectors.ros.verification.coverage import point_in_polygon
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting, coverage_grid_hash


def paired_route_candidates(
    accounting, legal_centers, current, *, physical_radius_m, budget_ms=500
):
    """Filter frozen Body/map centers using the current accounted occupancy.

    This computes a conservative geometric route component, not a Nav2 path or
    permission. Occupied cell squares are dilated by the physical Body disc and
    an extra full cell diagonal so adjacent center segments are also clear.
    No corner cutting or guessed long entry from the current pose is allowed.
    Returned brush cells are predictions only and never enter measured visits.
    """
    if (
        accounting.fault
        or current["time_sec"] != accounting.previous_time
        or type(physical_radius_m) not in (int, float)
        or not 0 < physical_radius_m <= 10
        or type(budget_ms) not in (int, float)
        or not 0 < budget_ms <= 1000
    ):
        raise ValueError("paired occupancy and bounded physical Body radius required")
    verifier = accounting.verifier
    if len(legal_centers) > 5000 or len(verifier.polygon) > 256:
        raise ValueError("temporal route input exceeds bound")
    begin = time.monotonic()

    def check_budget():
        if (time.monotonic() - begin) * 1000 > budget_ms:
            raise ValueError("temporal route budget exhausted; no partial candidates")

    width, height, res = verifier.width, verifier.height, verifier.resolution
    ox, oy = verifier.origin
    reach = physical_radius_m + math.sqrt(2) * res
    span = math.ceil(reach / res)
    if span > 128:
        raise ValueError("temporal route dilation exceeds bound")
    offsets = [
        (dx, dy)
        for dy in range(-span, span + 1)
        for dx in range(-span, span + 1)
        if math.hypot(dx, dy) * res <= reach
    ]
    blocked_centers = set()
    operations = 0
    for cell in verifier.temporary_blocked:
        check_budget()
        for dx, dy in offsets:
            operations += 1
            if operations > 1_000_000:
                raise ValueError("temporal route dilation exceeds computation bound")
            col, row = cell % width + dx, cell // width + dy
            if 0 <= col < width and 0 <= row < height:
                blocked_centers.add(row * width + col)
    nodes = {}
    for x, y in legal_centers:
        check_budget()
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (x, y)):
            raise ValueError("finite frozen legal centers required")
        col, row = round((x - ox) / res - 0.5), round((y - oy) / res - 0.5)
        if (
            not 0 <= col < width
            or not 0 <= row < height
            or not math.isclose(x, ox + (col + 0.5) * res, abs_tol=1e-6)
            or not math.isclose(y, oy + (row + 0.5) * res, abs_tol=1e-6)
        ):
            raise ValueError("temporal centers must align with frozen map")
        cell = row * width + col
        if cell not in blocked_centers:
            nodes[cell] = (x, y)
    col, row = math.floor((current["x"] - ox) / res), math.floor((current["y"] - oy) / res)
    source = row * width + col
    visited = {source} if 0 <= col < width and 0 <= row < height and source in nodes else set()
    queue = deque(visited)
    while queue:
        check_budget()
        cell = queue.popleft()
        col, row = cell % width, cell // width
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            if not 0 <= col + dx < width or not 0 <= row + dy < height:
                continue
            neighbor = cell + dy * width + dx
            if neighbor not in nodes or neighbor in visited:
                continue
            if dx and dy and (cell + dx not in nodes or cell + dy * width not in nodes):
                continue
            visited.add(neighbor)
            queue.append(neighbor)
    # A heading union may admit a potential repair; only a fresh observed
    # enabled footprint can later credit it. The executor still ranks a target.
    footprint = set()
    brush_span = math.ceil(verifier.radius / res)
    if brush_span > 128:
        raise ValueError("temporal brush projection exceeds bound")
    for yaw in (current["yaw"], 0, math.pi / 4, math.pi / 2, 3 * math.pi / 4):
        check_budget()
        co, si = math.cos(yaw), math.sin(yaw)
        polygon = [(x * co - y * si, x * si + y * co) for x, y in verifier.polygon]
        footprint.update(
            (dx, dy)
            for dy in range(-brush_span, brush_span + 1)
            for dx in range(-brush_span, brush_span + 1)
            if point_in_polygon(dx * res, dy * res, polygon)
        )
    reachable = set()
    for cell in visited:
        check_budget()
        for dx, dy in footprint:
            operations += 1
            if operations > 1_000_000:
                raise ValueError("temporal route projection exceeds computation bound")
            col, row = cell % width + dx, cell // width + dy
            if 0 <= col < width and 0 <= row < height:
                candidate = row * width + col
                if (
                    candidate in accounting.denominator
                    and candidate not in verifier.temporary_blocked
                ):
                    reachable.add(candidate)
    centers = tuple(nodes[cell] for cell in sorted(visited))
    proof = {
        "snapshot_sequence": accounting.previous_sequence,
        "sim_time_sec": accounting.previous_time,
        "grid_hash": accounting.grid_hash,
        "physical_radius_m": physical_radius_m,
        "reachable_center_cells": sorted(visited),
        "reachable_cells": sorted(reachable),
        "prediction_only": True,
    }
    return centers, frozenset(reachable), {**proof, "route_evidence_hash": digest(proof)}


class TimePairedRecovery(MissedRegionRecovery):
    def __init__(self, accounting, *, admission_sim_time, deadline_sim_time, max_attempts=3):
        if not isinstance(accounting, OccupancyAccounting):
            raise TypeError("time-paired occupancy accounting required")
        if any(
            type(v) not in (int, float) or not math.isfinite(v)
            for v in [admission_sim_time, deadline_sim_time]
        ):
            raise ValueError("finite SIM admission/deadline required")
        if not 0 <= admission_sim_time < deadline_sim_time <= admission_sim_time + 1800:
            raise ValueError("immutable bounded SIM deadline required")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("original bounded retry policy required")
        super().__init__(accounting.verifier, max_attempts=max_attempts)
        self.accounting = accounting
        self.admission_sim_time, self.deadline_sim_time = admission_sim_time, deadline_sim_time

    def propose(self):
        raise ValueError(
            "temporal recovery requires propose_at with paired occupancy and route evidence"
        )

    def propose_at(self, *, sim_time_sec, snapshot_sequence, reachable_cells):
        """Partition actual pending cells, not entire mixed blocked components."""
        base = {
            "dispatched": False,
            "ready": [],
            "deferred": [],
            "exhausted": [],
            "deadline_sim_time": self.deadline_sim_time,
            "fixed_denominator_cells": len(self.accounting.denominator),
        }
        if (
            self.accounting.fault
            or self.accounting.previous_sequence is None
            or type(snapshot_sequence) is not int
            or snapshot_sequence != self.accounting.previous_sequence
            or type(sim_time_sec) not in (int, float)
            or not math.isfinite(sim_time_sec)
            or sim_time_sec != self.accounting.previous_time
            or sim_time_sec < self.admission_sim_time
            or type(reachable_cells) is not frozenset
            or any(type(c) is not int for c in reachable_cells)
            or not reachable_cells <= self.accounting.denominator
            or coverage_grid_hash(self.verifier) != self.accounting.grid_hash
        ):
            return {
                **base,
                "status": "UNKNOWN",
                "reason": "missing, faulty or unpaired occupancy/route evidence",
            }
        pending = self.accounting.denominator - set(self.verifier.visits)
        blocked = pending & self.verifier.temporary_blocked
        exhausted = {c for c in pending if self.attempts.get(c, 0) >= self.max_attempts}
        ready = pending & reachable_cells - blocked - exhausted
        deferred = pending - ready - exhausted
        for key, cells, reason in [
            ("ready", ready, "MEASURED_UNCLEANED_AND_ROUTE_REACHABLE"),
            ("deferred", deferred, "TEMP_BLOCKED_OR_CURRENT_ROUTE_UNAVAILABLE"),
            ("exhausted", exhausted, "BOUNDED_ATTEMPTS_EXHAUSTED"),
        ]:
            if cells:
                base[key] = [
                    {"cells": sorted(cells), "reason": reason, "execution_entry": "request_action"}
                ]
        if not pending:
            status = "COVERAGE_COMPLETE"
        elif sim_time_sec >= self.deadline_sim_time:
            status = "BLOCKED" if deferred else "REQUIRES_OPERATOR"
            base["ready"] = []  # no dispatch proposal survives the original deadline
        elif ready:
            status = "REPAIR_PENDING"
        elif deferred:
            status = "WAITING_FOR_OBSTACLE"
        else:
            status = "REQUIRES_OPERATOR"
        return {
            **base,
            "status": status,
            "snapshot_sequence": snapshot_sequence,
            "sim_time_sec": sim_time_sec,
        }

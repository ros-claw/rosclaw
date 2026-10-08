"""Read-only bounded proposals over successfully time-paired SIM observations.

Route reachability must be supplied from the configured legal-center graph at
this same snapshot, not guessed from the absence of occupied brush cells.
This module never dispatches, waits, acknowledges or creates execution receipts.
"""

import math

from rosclaw.connectors.ros.mission.recovery import MissedRegionRecovery
from rosclaw.connectors.ros.verification.occupancy import OccupancyAccounting, coverage_grid_hash


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

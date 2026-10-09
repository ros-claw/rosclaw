"""Bounded missed-region proposals. Dispatch remains in the existing task runtime."""

from rosclaw.connectors.ros.verification.coverage import CoverageVerifier


class MissedRegionRecovery:
    def __init__(self, verifier: CoverageVerifier, *, max_attempts: int = 3):
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        self.verifier = verifier
        self.max_attempts = max_attempts
        self.attempts: dict[int, int] = {}
        self.seen_action_ids: set[str] = set()

    def propose(self) -> dict:
        ready, deferred, exhausted = [], [], []
        for region in self.verifier.missed_regions():
            eligible = [
                cell for cell in region["cells"] if self.attempts.get(cell, 0) < self.max_attempts
            ]
            retries = min((self.attempts.get(cell, 0) for cell in region["cells"]), default=0)
            item = {
                **region,
                "retry_count": retries,
                "execution_entry": "request_action",
                "requires": ["coverage.compute_path", "coverage.execute", "coverage.verify"],
            }
            if not eligible:
                exhausted.append(item)
            elif region["deferred"]:
                deferred.append(item)
            else:
                item["cells"] = eligible
                item["area_m2"] = len(eligible) * self.verifier.resolution**2
                item["centroid"] = [
                    self.verifier.origin[0]
                    + sum(i % self.verifier.width + 0.5 for i in eligible)
                    / len(eligible)
                    * self.verifier.resolution,
                    self.verifier.origin[1]
                    + sum(i // self.verifier.width + 0.5 for i in eligible)
                    / len(eligible)
                    * self.verifier.resolution,
                ]
                ready.append(item)
        return {
            "ready": ready,
            "deferred": deferred,
            "exhausted": exhausted,
            "status": "REQUIRES_OPERATOR"
            if exhausted
            else "REPAIR_PENDING"
            if ready
            else "WAITING_FOR_OBSTACLE"
            if deferred
            else "COVERAGE_COMPLETE",
            "dispatched": False,
        }

    def record_attempt(self, cells: list[int], *, action_id: str) -> None:
        """Runtime hook after a canonical action; repeated delivery is idempotent.

        This records retry bookkeeping, not success. Only new measured poses
        can reduce missed regions in the independent verifier.
        """
        self.record_attempt_sequence([cells], action_id=action_id)

    def record_attempt_sequence(self, groups, *, action_id: str) -> None:
        """Count each planned footprint under one real goal id, idempotently.

        An overlapping pair uses two attempts for its common cells. This
        bookkeeping neither asserts arrival nor changes measured coverage.
        """
        if type(groups) not in (list, tuple) or not 1 <= len(groups) <= 2:
            raise ValueError("one or two bounded planned footprints required")
        unique = [set(cells) for cells in groups]
        if not action_id or any(not cells <= self.verifier.accessible for cells in unique):
            raise ValueError("attempt requires action id and accessible region cells")
        if action_id in self.seen_action_ids:
            return
        if len(unique) == 2:
            increments = {}
            for cells in unique:
                for cell in cells:
                    increments[cell] = increments.get(cell, 0) + 1
            if any(
                self.attempts.get(c, 0) + count > self.max_attempts
                for c, count in increments.items()
            ):
                raise ValueError("planned pair exceeds existing per-cell attempt budget")
        self.seen_action_ids.add(action_id)
        for cells in unique:
            for cell in cells:
                self.attempts[cell] = self.attempts.get(cell, 0) + 1

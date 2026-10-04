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
            retries = max((self.attempts.get(cell, 0) for cell in region["cells"]), default=0)
            item = {
                **region,
                "retry_count": retries,
                "execution_entry": "request_action",
                "requires": ["coverage.compute_path", "coverage.execute", "coverage.verify"],
            }
            if retries >= self.max_attempts:
                exhausted.append(item)
            elif region["deferred"]:
                deferred.append(item)
            else:
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
        if not action_id or not set(cells) <= self.verifier.accessible:
            raise ValueError("attempt requires action id and accessible region cells")
        if action_id in self.seen_action_ids:
            return
        self.seen_action_ids.add(action_id)
        for cell in set(cells):
            self.attempts[cell] = self.attempts.get(cell, 0) + 1

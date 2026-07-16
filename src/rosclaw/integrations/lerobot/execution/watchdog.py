"""Execution watchdog: communication and liveness supervision.

Tracks transport health and worker liveness between steps.  Any of these
conditions raises :class:`WatchdogTrip` and must drive the executor into
``COMMUNICATION_LOST`` / ``FAULT`` with the permit revoked:

- serial read failure or disconnect
- observation staleness beyond ``max_observation_age_ms``
- policy worker death / restart (worker_generation change)
- operator abort request
"""

from __future__ import annotations

import time
from typing import Any

from rosclaw.body.rh56.transport import RH56Transport, TransportIOError


class WatchdogTrip(RuntimeError):
    """Watchdog escalation; message starts with a machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class ExecutionWatchdog:
    """Supervise transport + worker liveness for one execution session."""

    def __init__(
        self,
        *,
        max_observation_age_ms: float = 300.0,
    ):
        self.max_observation_age_ms = max_observation_age_ms
        self._worker_generation: str | None = None
        self._abort_requested = False
        self.trips: list[str] = []

    # ------------------------------------------------------------------

    def track_worker(self, worker_generation: str | None) -> None:
        """Record the current worker generation (call at session start)."""
        self._worker_generation = worker_generation

    def check_worker(self, worker_generation: str | None) -> None:
        """Raise when the policy worker restarted since tracking began."""
        if self._worker_generation is None:
            self._worker_generation = worker_generation
            return
        if worker_generation != self._worker_generation:
            self._trip("worker_restart", "policy worker restarted during execution")

    # ------------------------------------------------------------------

    def check_transport(self, transport: RH56Transport) -> None:
        if not transport.is_connected():
            self._trip("serial_disconnect", "transport reports disconnected")

    def read_feedback(self, transport: RH56Transport):
        """Read transport feedback, converting I/O errors into WatchdogTrip."""
        try:
            feedback = transport.read_state()
        except TransportIOError as exc:
            self._trip("serial_io_error", str(exc))
        age_ms = max(0, (time.monotonic_ns() - feedback.timestamp_monotonic_ns) / 1e6)
        if age_ms > self.max_observation_age_ms:
            self._trip(
                "stale_observation",
                f"feedback age {age_ms:.1f} ms > {self.max_observation_age_ms} ms",
            )
        return feedback

    # ------------------------------------------------------------------

    def request_abort(self) -> None:
        self._abort_requested = True

    def check_abort(self) -> None:
        if self._abort_requested:
            self._trip("operator_abort", "operator requested abort")

    # ------------------------------------------------------------------

    def _trip(self, code: str, message: str) -> None:
        self.trips.append(code)
        raise WatchdogTrip(code, message)

    def status(self) -> dict[str, Any]:
        return {
            "worker_generation": self._worker_generation,
            "abort_requested": self._abort_requested,
            "trips": list(self.trips),
        }

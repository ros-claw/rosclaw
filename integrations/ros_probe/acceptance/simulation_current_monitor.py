"""SDK independent bounded observation worker; no ROS callback or authority.

Only original facts validated with an explicit age margin may be shared. Every
consumer checks the actual monotonic deadline. Any observed failure latches;
there is no last-good fallback after loss, worker stall, or expired observation.
"""

import copy
import threading
import time

from simulation_current_facts import CurrentSimulationFacts


class BoundedCurrentFactsMonitor:
    def __init__(self, *, facts, daemon_observer):
        if not isinstance(facts, CurrentSimulationFacts) or not callable(daemon_observer):
            raise ValueError("independent original SDK facts and actual daemon observer required")
        self.facts, self._daemon_observer = facts, daemon_observer
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self.failed_reason = None
        self.last_collection_started_ns = self.last_collection_completed_ns = None

    def _collect(self):
        self.last_collection_started_ns = time.monotonic_ns()
        facts = self.facts.collect(freshness_horizon_ms=100)
        facts["daemon_status"] = self._daemon_observer()
        self.last_collection_completed_ns = time.monotonic_ns()
        if time.monotonic_ns() >= facts["valid_until_monotonic_ns"]:
            raise ValueError("fresh observation expired during actual daemon check")
        return facts

    def start(self):
        if self._thread is not None or self._stop.is_set():
            raise ValueError("SDK current monitor cannot be restarted")
        initial = self._collect()
        with self._lock:
            self._latest = initial
        self._thread = threading.Thread(target=self._run, name="sdk-current-facts", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.wait(0.01):
            try:
                latest = self._collect()
                with self._lock:
                    if self.failed_reason is not None:
                        return
                    self._latest = latest
            except Exception as exc:
                with self._lock:
                    self.failed_reason = str(exc)[:2500]
                    self._latest = None
                return

    def get(self, *, now_monotonic_ns=None):
        now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
        with self._lock:
            if self.failed_reason is not None:
                raise ValueError("SDK current monitor permanently latched: " + self.failed_reason)
            if (
                type(now) is not int
                or self._latest is None
                or not self._latest["health_review_ns"]
                <= now
                < self._latest["valid_until_monotonic_ns"]
            ):
                self.failed_reason = (
                    "CURRENT_OBSERVATION_WINDOW_EXPIRED_OR_CLOCK_INVALID"
                    f" now={now} started={self.last_collection_started_ns}"
                    f" completed={self.last_collection_completed_ns}"
                )
                self._latest = None
                raise ValueError("SDK current monitor permanently latched: " + self.failed_reason)
            return copy.deepcopy(self._latest)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        with self._lock:
            self.failed_reason = self.failed_reason or "SDK_CURRENT_MONITOR_CLOSED"
            self._latest = None

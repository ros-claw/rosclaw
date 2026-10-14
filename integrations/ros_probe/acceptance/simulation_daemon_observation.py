"""Independent SDK read-only daemon observations; no runtime or action authority."""

import copy
import os
import threading
import time

from simulation_navigation_supervision import review_active_sim_daemon


class BoundedDaemonObservation:
    def __init__(self, *, client, admission, body_id):
        self.client, self.admission, self.body_id = client, admission, body_id
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self.failed_reason = None

    def _observe(self):
        started = time.monotonic_ns()
        a = self.admission
        if started >= a["expires_ns"]:
            raise ValueError("explicit SDK admission expired")
        status = self.client.get_runtime_status()
        if (
            status["daemon_peer"]["pid"] != a["daemon_pid"]
            or status["daemon_peer"]["uid"] != a["daemon_uid"]
            or status["client_peer"]["pid"] != os.getpid()
            or status["process_separated"] is not True
        ):
            raise ValueError("actual independent daemon socket peer changed")
        review_active_sim_daemon(
            status,
            instance_id=a["daemon_instance_id"],
            pid=a["daemon_pid"],
            uid=a["daemon_uid"],
            body_id=self.body_id,
            expected_running=None,
        )
        ended = time.monotonic_ns()
        if not 0 <= ended - started < 250_000_000:
            raise ValueError("actual daemon observation expired during RPC")
        return {
            "status": status,
            "started_ns": started,
            "completed_ns": ended,
            "valid_until_ns": min(started + 250_000_000, a["expires_ns"]),
        }

    def start(self):
        if self._thread is not None or self._stop.is_set() or self.failed_reason:
            raise ValueError("SDK daemon observer cannot restart")
        self._latest = self._observe()
        self._thread = threading.Thread(target=self._run, daemon=True, name="sdk-daemon-observer")
        self._thread.start()

    def _run(self):
        while not self._stop.wait(0.05):
            try:
                latest = self._observe()
                with self._lock:
                    if self.failed_reason:
                        return
                    self._latest = latest
            except Exception as exc:
                with self._lock:
                    self.failed_reason = str(exc)[:2500]
                    self._latest = None
                return

    def snapshot(self):
        with self._lock:
            if self.failed_reason:
                raise ValueError(
                    "SDK daemon observation permanently latched: " + self.failed_reason
                )
            now = time.monotonic_ns()
            if (
                self._latest is None
                or not self._latest["started_ns"] <= now < self._latest["valid_until_ns"]
            ):
                self.failed_reason = "DAEMON_OBSERVATION_EXPIRED_OR_CLOCK_INVALID"
                self._latest = None
                raise ValueError(
                    "SDK daemon observation permanently latched: " + self.failed_reason
                )
            return copy.deepcopy(self._latest)

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        with self._lock:
            self.failed_reason = self.failed_reason or "SDK_DAEMON_OBSERVER_CLOSED"
            self._latest = None

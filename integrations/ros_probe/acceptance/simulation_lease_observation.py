"""Read-only control-state observation for an independent SIM physics witness.

This reader cannot create, renew or revoke a lease. Status is written by the
separate SDK sink in the operator-owned fixture directory. Unknown status stays
unknown: a missing status file must never be interpreted as a verified revoke.
"""

import json
import re
import stat
import time
from pathlib import Path

from simulation_atomic_read import read_bounded_regular


class ExternalSimulationLeaseObservation:
    def __init__(self, directory, *, filename, run_id, body_snapshot_hash=None):
        if (
            type(filename) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]+\.json", filename) is None
            or type(run_id) is not str
            or not run_id
            or len(run_id) > 160
            or (
                body_snapshot_hash is not None
                and (
                    type(body_snapshot_hash) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", body_snapshot_hash) is None
                )
            )
        ):
            raise ValueError("explicit bounded independent SIM lease observation pins required")
        self.directory = Path(directory)
        info = self.directory.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("original fixture directory required")
        self._identity = info.st_dev, info.st_ino
        self.filename, self.run_id, self.body_hash = filename, run_id, body_snapshot_hash
        self._last_clock = self._last_status_clock = 0
        self.failed_reason = None

    def read(self, *, now_monotonic_ns=None):
        if self.failed_reason is not None:
            return {"known": False, "lease_enabled": None, "reason": self.failed_reason}
        try:
            now = time.monotonic_ns() if now_monotonic_ns is None else now_monotonic_ns
            if type(now) is not int or now <= 0 or now < self._last_clock:
                raise ValueError("lease observation clock invalid or rolled back")
            info = self.directory.stat(follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != self._identity:
                raise ValueError("lease observation fixture directory replaced")
            raw = read_bounded_regular(
                self.directory / self.filename, 4096, allow_atomic_unlink=True
            )
            if now_monotonic_ns is None:
                now = time.monotonic_ns()
                if now < self._last_clock:
                    raise ValueError("lease observation clock invalid or rolled back")
            status = json.loads(raw)
            keys = {
                "schema_version",
                "run_id",
                "body_snapshot_hash",
                "monotonic_ns",
                "lease_expires_ns",
                "terminal",
                "lease_updates",
                "received_commands",
                "published_velocity_messages",
                "stop_zero_messages",
                "action_id",
                "daemon_instance_id",
            }
            if (
                type(status) is not dict
                or set(status) != keys
                or status["schema_version"] != "rosclaw.sim_sink_status.v1"
            ):
                raise ValueError("closed original SDK sink status required")
            if status["run_id"] != self.run_id or (
                self.body_hash is not None and status["body_snapshot_hash"] != self.body_hash
            ):
                raise ValueError("observed sink Body/run identity changed")
            if (
                type(status["terminal"]) is not bool
                or any(
                    type(status[k]) is not int or status[k] < 0
                    for k in (
                        "monotonic_ns",
                        "lease_expires_ns",
                        "lease_updates",
                        "received_commands",
                        "published_velocity_messages",
                        "stop_zero_messages",
                    )
                )
                or not 0 <= now - status["monotonic_ns"] <= 250_000_000
                or status["monotonic_ns"] < self._last_status_clock
                or status["lease_expires_ns"] > status["monotonic_ns"] + 250_000_000
                or (status["terminal"] and status["lease_expires_ns"] != 0)
            ):
                raise ValueError("sink status stale, malformed, excessive lease or rolled back")
            self._last_clock, self._last_status_clock = now, status["monotonic_ns"]
            return {
                "known": True,
                "lease_enabled": not status["terminal"] and now < status["lease_expires_ns"],
                "reason": None,
                "status": status,
            }
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self.failed_reason = str(exc)[:300]
            return {"known": False, "lease_enabled": None, "reason": self.failed_reason}

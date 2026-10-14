"""Existing rosclawd's SDK-owned current-facts/proof producer for SIM only.

This adapter opens no ROS transport and creates no Runtime, Session, Lease or
Permit. The existing daemon supplies its actual status callback. The SDK sink
must independently enroll the same operator context and recompute local facts.
"""

import hashlib
import json
import math
import os
import stat
import threading
import time
from contextlib import ExitStack, suppress
from pathlib import Path

from simulation_current_facts import CurrentSimulationFacts
from simulation_execution_proof import (
    MAX_PROOF_LIFETIME_NS,
    ExecutionProofPins,
    sign_execution_proof,
)
from simulation_lease_observation import ExternalSimulationLeaseObservation

from rosclaw.connectors.ros.mission.navigation_supervision import NavigationExecutionSupervision


def review_active_sim_daemon(status, *, instance_id, pid, uid, body_id, expected_running=1):
    """Actual daemon-internal status, distinct from passive socket observation.

    An internal call has process_separated=False by construction; independent
    operator enrollment must verify real socket peer credentials separately.
    Active SIM actions may have Sessions/Leases. They may never have REAL work.
    """
    if type(status) is not dict or any(
        type(status.get(k)) is not dict
        for k in (
            "watchdog",
            "ledger",
            "recovery",
            "queue",
        )
    ):
        raise ValueError("actual current daemon status required")
    if (
        type(status.get("running")) is not bool
        or type(status.get("emergency_stop_latched")) is not bool
        or any(
            type(status.get(k)) is not int
            for k in ("daemon_pid", "daemon_uid", "hardware_actions_executed")
        )
    ):
        raise ValueError("typed actual daemon identity and execution facts required")
    if any(
        status.get(k) != v
        for k, v in (
            ("daemon_instance_id", instance_id),
            ("daemon_pid", pid),
            ("daemon_uid", uid),
            ("robot_id", body_id),
            ("running", True),
            ("southbound_owner", "rosclawd"),
            ("runtime_state", "RUNNING"),
            ("supervision_state", "DISARMED"),
            ("emergency_stop_latched", False),
            ("hardware_actions_executed", 0),
        )
    ):
        raise ValueError("actual SIM daemon identity, supervision or runtime changed")
    if (
        status["watchdog"].get("running") is not True
        or status["watchdog"].get("healthy") is not True
        or status["ledger"].get("integrity_verified") is not True
        or status["ledger"].get("write_failed") is not False
        or status["recovery"].get("required") is not False
        or type(status["queue"].get("RUNNING")) is not int
        or status["queue"]["RUNNING"]
        not in ({0, 1} if expected_running is None else {expected_running})
    ):
        raise ValueError("actual SIM daemon watchdog, ledger or active execution unhealthy")
    expected = {
        "sandbox.reach:SIMULATION",
        "sandbox.reach:FIXTURE",
        "navigation.navigate_to_pose:SIMULATION",
    }
    executors = status.get("registered_executors")
    if (
        type(executors) is not list
        or any(type(x) is not str or x not in expected for x in executors)
        or "navigation.navigate_to_pose:SIMULATION" not in executors
        or status.get("drivers") != ["gazebo_ros_expert"]
    ):
        raise ValueError("unexpected daemon execution or driver path")


class SdkNavigationExecutionSupervision(NavigationExecutionSupervision):
    def __init__(
        self,
        *,
        binding,
        facts,
        daemon_instance_id,
        daemon_status,
        key,
        proof_file,
        audit_file,
        lease_observation,
        allowed_action_ids,
    ):
        super().__init__(binding)
        if (
            not isinstance(facts, CurrentSimulationFacts)
            or not isinstance(lease_observation, ExternalSimulationLeaseObservation)
            or not callable(daemon_status)
            or type(key) is not bytes
            or len(key) != 32
            or type(allowed_action_ids) is not tuple
            or not 1 <= len(allowed_action_ids) <= 2
            or len(set(allowed_action_ids)) != len(allowed_action_ids)
        ):
            raise ValueError(
                "independent SDK facts, private fixture key and bounded operator actions required"
            )
        request = facts.enrollment["request"]
        if (
            request["body_snapshot_hash"] != binding.body_snapshot_hash
            or request["sim_run_id"] != binding.run_id
        ):
            raise ValueError("SDK facts differ from compiled daemon Body/run")
        self._pins = {
            "body_id": binding.body_id,
            "body_snapshot_hash": binding.body_snapshot_hash,
            "manifest_hash": request["manifest_hash"],
            "run_id": binding.run_id,
            "world_sha256": facts.enrollment["world_sha256"],
            "observer_epoch": facts.enrollment["source_pins"]["observer_epoch"],
            "daemon_instance_id": daemon_instance_id,
        }
        for action_id in allowed_action_ids:
            ExecutionProofPins(**self._pins, action_id=action_id)
        self.facts, self._status, self._key = facts, daemon_status, key
        self.lease_observation = lease_observation
        self.proof_file = Path(proof_file)
        info = self.proof_file.parent.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("private operator-owned proof directory required")
        self._directory_identity = info.st_dev, info.st_ino
        self._directory_fd = os.open(
            self.proof_file.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        opened = os.fstat(self._directory_fd)
        if (opened.st_dev, opened.st_ino) != self._directory_identity:
            os.close(self._directory_fd)
            raise ValueError("private proof directory replaced during enrollment")
        self._audit_stack = ExitStack()
        try:
            self._trace = self._audit_stack.enter_context(Path(audit_file).open("x"))  # noqa: SIM115 - ExitStack owns the full SDK guard lifetime
        except BaseException:
            os.close(self._directory_fd)
            raise
        self._closed = False
        self._trace_rows = 0
        self._allowed = allowed_action_ids
        self._used = set()
        self._lock = threading.Lock()
        self._action_id = None
        self._deadline = 0
        self._sequence = 0
        self.failed_reason = None

    def begin(self, action_id, *, duration_sec):
        with self._lock:
            try:
                if self.failed_reason is not None:
                    raise ValueError(self.failed_reason)
                if (
                    action_id not in self._allowed
                    or action_id in self._used
                    or type(duration_sec) not in (int, float)
                    or not math.isfinite(duration_sec)
                    or not 0 < duration_sec <= self.binding.max_duration_sec
                ):
                    raise ValueError("operator action scope, reuse or duration rejected")
                self.facts.collect()
                if not self.facts.reader.capacity_covers(
                    duration_ms=math.ceil(duration_sec * 1000) + 1000, max_sample_bytes=500_000
                ):
                    raise ValueError("bounded source budget cannot cover action and stop reserve")
                self._action_id = action_id
                self._used.add(action_id)
                self._deadline = time.monotonic_ns() + math.ceil(duration_sec * 1e9)
                self._check_locked(action_id)
            except Exception as exc:
                self._fail(exc)

    def _fail(self, exc):
        self.failed_reason = str(exc)[:2500]
        self._deadline = 0
        # A previously written proof expires within 250ms even if unlink fails.
        if not self._closed:
            with suppress(OSError):
                os.unlink(self.proof_file.name, dir_fd=self._directory_fd)
        raise ValueError(
            "SDK navigation supervision permanently latched: " + self.failed_reason
        ) from exc

    def check(self, action_id):
        with self._lock:
            try:
                self._check_locked(action_id)
            except Exception as exc:
                self._fail(exc)

    def _check_locked(self, action_id):
        started = time.monotonic_ns()
        if (
            self._closed
            or self.failed_reason is not None
            or action_id != self._action_id
            or started >= self._deadline
        ):
            raise ValueError("SDK supervision lost, expired or action changed")
        info = self.proof_file.parent.stat(follow_symlinks=False)
        if (
            (info.st_dev, info.st_ino) != self._directory_identity
            or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise ValueError("private proof directory changed")
        facts = self.facts.collect()
        sink = self.lease_observation.read()
        if not sink["known"] or sink["status"]["terminal"]:
            raise ValueError("independent SDK sink state unavailable or terminal")
        status = self._status()
        review_active_sim_daemon(
            status,
            instance_id=self._pins["daemon_instance_id"],
            pid=os.getpid(),
            uid=os.geteuid(),
            body_id=self.binding.body_id,
        )
        now = time.monotonic_ns()
        if not 0 <= now - started <= 250_000_000 or now >= self._deadline:
            raise ValueError("SDK proof production exceeded current execution window")
        self._sequence += 1
        payload = {
            **self._pins,
            "action_id": action_id,
            "schema_version": "rosclaw.sim_execution_supervision.v1",
            "sequence": self._sequence,
            "source_sample": facts["source"]["last_sample"],
            "health_sha256": hashlib.sha256(
                json.dumps(
                    facts["health"], sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode()
            ).hexdigest(),
            "source_review_ns": facts["source"]["reviewed_at_monotonic_ns"],
            "health_review_ns": facts["health_review_ns"],
            "issued_ns": now,
            "expires_ns": min(
                now + MAX_PROOF_LIFETIME_NS, self._deadline, facts["health_review_ns"] + 500_000_000
            ),
        }
        raw = sign_execution_proof(
            payload,
            pins=ExecutionProofPins(**self._pins, action_id=action_id),
            key=self._key,
            now_ns=now,
        )
        self._record(
            {
                "event": "PROOF_PREPARED",
                "facts": facts,
                "sink_observation": sink,
                "daemon_status": status,
                "proof": json.loads(raw),
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        temporary = self.proof_file.name + ".tmp"
        fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=self._directory_fd,
        )
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary,
                self.proof_file.name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
            )
            os.fsync(self._directory_fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self._directory_fd)
        if time.monotonic_ns() >= payload["expires_ns"]:
            raise ValueError("supervision proof expired before durable publication")

        self._record(
            {
                "event": "PROOF_PUBLISHED",
                "sequence": self._sequence,
                "proof_sha256": hashlib.sha256(raw).hexdigest(),
                "monotonic_ns": time.monotonic_ns(),
            }
        )
        if time.monotonic_ns() >= payload["expires_ns"]:
            raise ValueError("supervision proof expired during original audit persistence")

    def _record(self, row):
        raw = json.dumps(row, allow_nan=False) + "\n"
        if self._trace_rows >= 2000 or len(raw.encode()) > 32_000:
            raise ValueError("bounded original supervision audit capacity exhausted")
        self._trace.write(raw)
        self._trace.flush()
        os.fsync(self._trace.fileno())
        self._trace_rows += 1

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.failed_reason = self.failed_reason or "SDK_SUPERVISION_CLOSED"
            with suppress(OSError):
                os.unlink(self.proof_file.name, dir_fd=self._directory_fd)
            try:
                self.facts.close()
            finally:
                self._audit_stack.close()
                with suppress(OSError):
                    os.close(self._directory_fd)

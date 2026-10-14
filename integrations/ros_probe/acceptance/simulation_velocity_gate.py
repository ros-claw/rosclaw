"""SDK final SIM velocity gate state machine; this module never opens ROS.

An isolated sink must call this gate before forwarding any command and on its
steady-clock watchdog. A daemon proof alone is insufficient: collect_local must
independently recompute current ROS health and poll the pinned ECM reader.
The canonical lease RPC is handled by that sink, not by an Agent or this module.
"""

import math
import time

from simulation_execution_proof import (
    MAX_OBSERVATION_AGE_NS,
    ExecutionProofPins,
    ExecutionProofVerifier,
)

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


class SimulationVelocityGate:
    def __init__(
        self,
        *,
        binding,
        pins,
        key,
        collect_local,
        clock=time.monotonic_ns,
        collect_at_least_source=None,
    ):
        if (
            not isinstance(binding, SimulationNavigationBinding)
            or not isinstance(pins, ExecutionProofPins)
            or binding.body_id != pins.body_id
            or binding.body_snapshot_hash != pins.body_snapshot_hash
            or binding.run_id != pins.run_id
            or not callable(collect_local)
            or not callable(clock)
            or (collect_at_least_source is not None and not callable(collect_at_least_source))
        ):
            raise ValueError(
                "matching immutable SIM Body/run and independent SDK collector required"
            )
        self.binding, self.pins = binding, pins
        self._verifier = ExecutionProofVerifier(pins=pins, key=key)
        self._collect_local, self._clock = collect_local, clock
        self._collect_at_least_source = collect_at_least_source
        self._deadline = 0
        self._started = False
        self.failed_reason = None

    def _latch(self, reason):
        self.failed_reason = str(reason)[:300]
        self._deadline = 0
        raise ValueError("SIM velocity gate permanently latched: " + self.failed_reason)

    def revoke(self):
        """Normal revocation ends this action's gate; it cannot be re-enabled."""
        self._deadline = 0
        self.failed_reason = self.failed_reason or "ACTION_LEASE_REVOKED"

    def _check(self, raw):
        if self.failed_reason is not None:
            self._latch(self.failed_reason)
        try:
            now = self._clock()
            if self._started and now >= self._deadline:
                raise ValueError("ACTION_LEASE_EXPIRED")
            # Record the start before actual IO/review; collection time counts
            # against freshness and may not extend the authenticated deadline.
            started = now
            if self._collect_at_least_source is None:
                local = self._collect_local()
            else:
                # Authenticate before asking the independent process to catch
                # up. The same proof and old lease are checked AGAIN after
                # waiting, so synchronization never extends either deadline.
                requested = self._verifier.verify(raw, now_ns=now)
                local = self._collect_at_least_source(requested["source_sample"])
            now = self._clock()
            if self._started and now >= self._deadline:
                raise ValueError("ACTION_LEASE_EXPIRED_DURING_COLLECTION")
            proof = self._verifier.verify(raw, now_ns=now)
            if type(now) is not int or not 0 <= now - started <= MAX_OBSERVATION_AGE_NS:
                raise ValueError("LOCAL_SUPERVISION_CLOCK_OR_COLLECTION_EXPIRED")
            health, source = local["health"], local["source"]
            if (
                health["status"] != "HEALTHY_SIM_OBSERVATION_ONLY"
                or any(
                    health[k] != v
                    for k, v in (
                        ("body_snapshot_hash", self.pins.body_snapshot_hash),
                        ("manifest_hash", self.pins.manifest_hash),
                        ("sim_run_id", self.pins.run_id),
                    )
                )
                or source["continuous"] is not True
                or source["terminal"] is not False
                or source["reason"] is not None
                or type(source["last_sample"]) is not int
                or source["last_sample"] < proof["source_sample"]
            ):
                raise ValueError("INDEPENDENT_CURRENT_ROS_SOURCE_OR_IDENTITY_REJECTED")
            for timestamp in (local["health_review_ns"], source["reviewed_at_monotonic_ns"]):
                if type(timestamp) is not int or not 0 <= now - timestamp <= MAX_OBSERVATION_AGE_NS:
                    raise ValueError("LOCAL_SUPERVISION_OBSERVATION_EXPIRED")
            return proof, now
        except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
            self._latch(exc)

    def refresh_lease(self, raw):
        """Only an actual daemon lease callback may use this supplementary check."""
        proof, now = self._check(raw)
        self._deadline = proof["expires_ns"]
        self._started = True
        return (self._deadline - now) / 1e9

    def watchdog(self, raw):
        """Returns False before first lease; any loss after enable is terminal."""
        if self.failed_reason is not None:
            self._latch(self.failed_reason)
        if not self._started:
            return False
        self._check(raw)
        return True

    def validate_command(self, command, *, raw):
        """Return a copied bounded planar Twist only under a current lease.

        Reject excessive values rather than silently changing the action. The
        SDK ROS sink must publish zero on any raised exception and on revoke.
        """
        if self.failed_reason is not None:
            self._latch(self.failed_reason)
        if not self._started:
            return None
        self._check(raw)
        try:
            if type(command) is not dict or set(command) != {"linear", "angular"}:
                raise ValueError("CLOSED_PLANAR_TWIST_REQUIRED")
            for group in ("linear", "angular"):
                if type(command[group]) is not dict or set(command[group]) != {"x", "y", "z"}:
                    raise ValueError("EXPLICIT_SIX_AXIS_TWIST_REQUIRED")
                if any(
                    type(v) not in (int, float) or not math.isfinite(v)
                    for v in command[group].values()
                ):
                    raise ValueError("FINITE_TWIST_REQUIRED")
            if (
                command["linear"]["y"] != 0
                or command["linear"]["z"] != 0
                or command["angular"]["x"] != 0
                or command["angular"]["y"] != 0
                or abs(command["linear"]["x"]) > self.binding.max_speed_mps
                or abs(command["angular"]["z"]) > self.binding.max_yaw_rate_rps
            ):
                raise ValueError("BODY_PLANAR_VELOCITY_LIMIT_EXCEEDED")
            return {k: dict(v) for k, v in command.items()}
        except (ValueError, TypeError, KeyError) as exc:
            self._latch(exc)

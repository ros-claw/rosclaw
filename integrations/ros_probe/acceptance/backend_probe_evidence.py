"""Read-only contact-cache intervention qualification for an owned SIM probe.

The probe is an instrument outside the robot's work region. This module never
moves the probe or the robot, and never grants task, Body or action authority.
Actual runtime qualification and closed source replay remain separate gates.
"""

import hashlib
import math
import re
from collections import OrderedDict

from native_contact_evidence import decode_native_packet, native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


def probe_policy(policy):
    keys = {
        "schema_version",
        "source",
        "approved",
        "evidence_domain",
        "native_policy",
        "probe_xy",
        "sphere_radius_m",
        "ground_z_m",
        "lift_z_m",
        "stable_samples",
        "stable_sim_span_sec",
        "refresh_sim_sec",
        "refresh_wall_sec",
    }
    if (
        type(policy) is not dict
        or set(policy) != keys
        or policy["schema_version"] != "rosclaw.backend_cache_probe_policy.v1"
    ):
        raise ValueError("closed owned SIM backend probe policy required")
    if (
        policy["source"] != "simulator_operator_fixture_policy"
        or policy["approved"] is not True
        or policy["evidence_domain"] != "SIMULATION"
    ):
        raise ValueError("explicit SIM instrument policy required; no REAL authority")
    source = native_policy(policy["native_policy"])
    if source.get("component_gz_topic") != "/rosclaw_sim/backend_probe_components":
        raise ValueError("probe must use a separate simulator-owned component producer")
    base = source["contact_policy"]
    if (
        len(base["contacts"]) != 1
        or len(next(iter(base["contacts"].values()))) != 1
        or len(base["ground_collisions"]) != 1
    ):
        raise ValueError(
            "one actual probe sphere collision and one declared support ground required"
        )
    xy = policy["probe_xy"]
    if (
        type(xy) is not list
        or len(xy) != 2
        or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 20 for v in xy)
    ):
        raise ValueError("finite explicit instrument placement required")
    for key, low, high in (
        ("sphere_radius_m", 0.02, 0.1),
        ("ground_z_m", -1, 1),
        ("lift_z_m", 5, 12),
        ("stable_sim_span_sec", 0.2, 1),
        ("refresh_sim_sec", 5, 15),
        ("refresh_wall_sec", 10, 30),
    ):
        value = policy[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("bounded preregistered probe geometry and deadlines required")
    if type(policy["stable_samples"]) is not int or not 5 <= policy["stable_samples"] <= 20:
        raise ValueError("bounded distinct source sample requirement")
    return policy


class BackendCacheProbe:
    """Require real contact → acknowledged lift → measured clear → contact.

    ACK only arms a transition. Actual original packets and independent pose
    observations establish the measured transition. No empty ROS message or
    silence is inferred. A missed refresh or any source fault latches UNKNOWN.
    """

    def __init__(self, policy):
        self.policy = probe_policy(policy)
        self.phase = "WAIT_GROUND"
        self.previous = self.inventory_hash = self.pose_source = None
        self.pose_history = OrderedDict()
        self.lift_pose_confirmed = False
        self.last_received = None
        self.stable = []
        self.lift_ack = None
        self.cycles = 0
        self.cycle_sim = self.cycle_wall = None
        self.fault = None
        self.qualified_once = False

    def pose(self, sim_time, wall_time, values):
        try:
            if (
                type(values) is not list
                or len(values) != 7
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or abs(sum(v * v for v in values[3:]) - 1) > 1e-6
            ):
                raise ValueError("finite normalized independent probe world pose required")
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in (sim_time, wall_time)
            ):
                raise ValueError("finite independent probe source clock required")
            if self.pose_source is not None and (
                sim_time <= self.pose_source[0] or wall_time <= self.pose_source[1]
            ):
                raise ValueError("independent probe pose clock repeated/regressed")
            self.pose_source = (sim_time, wall_time, list(values))
            self.pose_history[round(sim_time * 1e9)] = self.pose_source
            while len(self.pose_history) > 32:
                self.pose_history.popitem(last=False)
        except ValueError as exc:
            self.fault = self.fault or str(exc)
            raise

    def acknowledge_lift(self, ack):
        try:
            if self.fault is not None:
                raise ValueError("backend probe source rejection remains latched")
            keys = {
                "source",
                "run_id",
                "model_name",
                "target_xyz",
                "acknowledged_at_unix_ns",
                "request_sha256",
                "service_success",
            }
            base = self.policy["native_policy"]["contact_policy"]
            if type(ack) is not dict or set(ack) != keys or self.phase != "READY_FOR_LIFT":
                raise ValueError(
                    "one lift acknowledgement only after measured stable ground contact"
                )
            if (
                ack["source"] != "owned_gazebo_world_set_pose_ack_not_pose_proof"
                or ack["run_id"] != base["run_id"]
                or ack["model_name"] != base["model_name"]
                or ack["target_xyz"] != [*self.policy["probe_xy"], self.policy["lift_z_m"]]
                or ack["service_success"] is not True
                or type(ack["acknowledged_at_unix_ns"]) is not int
                or self.previous is None
                or not 0 <= ack["acknowledged_at_unix_ns"] - self.previous[3] < 300_000_000
                or type(ack["request_sha256"]) is not str
                or not re.fullmatch(r"[0-9a-f]{64}", ack["request_sha256"])
            ):
                raise ValueError("exact original probe service acknowledgement required")
            self.lift_ack = dict(ack)
            self.lift_pose_confirmed = False
            self.phase, self.stable = "WAIT_CLEAR_AFTER_LIFT", []
        except ValueError as exc:
            self.fault = self.fault or str(exc)
            raise

    def _stable(self, sim, wall, eligible):
        if not eligible:
            self.stable = []
            return False
        self.stable.append((sim, wall))
        if len(self.stable) > 100:
            self.stable.pop(0)
        return (
            len(self.stable) >= self.policy["stable_samples"]
            and sim - self.stable[0][0] >= self.policy["stable_sim_span_sec"]
            and wall - self.stable[0][1] >= self.policy["stable_sim_span_sec"]
        )

    def observe(self, raw, *, received_monotonic_sec, received_unix_ns):
        try:
            if self.fault is not None:
                raise ValueError("backend probe source rejection remains latched")
            packet, mapped = decode_native_packet(raw, self.policy["native_policy"])
            sim, wall = packet["sim_time_sec"], received_monotonic_sec
            if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
                raise ValueError("original probe wall receipt required")
            if (
                type(received_unix_ns) is not int
                or not 0 <= received_unix_ns - packet["captured_at_unix_ns"] < 300_000_000
            ):
                raise ValueError("probe original PostUpdate receipt stale or future")
            current = (packet["sequence"], packet["iterations"], sim, packet["captured_at_unix_ns"])
            if self.qualified_once and (
                sim - self.cycle_sim > self.policy["refresh_sim_sec"]
                or wall - self.cycle_wall > self.policy["refresh_wall_sec"]
            ):
                raise ValueError("backend cache intervention refresh deadline expired")
            if self.previous is not None and (
                any(a <= b for a, b in zip(current, self.previous, strict=True))
                or not 0 < wall - self.last_received <= 0.3
                or not 0 < sim - self.previous[2] <= 0.3
            ):
                raise ValueError("probe component source repeated, regressed or has gaps")
            paired = self.pose_history.get(round(sim * 1e9))
            if paired is None or not -0.1 <= wall - paired[1] < 0.3:
                raise ValueError(
                    "probe component lacks exact-time original independent pose; interpolation refused"
                )
            pose = packet["body_world_pose"]
            if math.dist(pose[:3], paired[2][:3]) > 0.05 or abs(
                sum(a * b for a, b in zip(pose[3:], paired[2][3:], strict=True))
            ) < math.cos(0.05):
                raise ValueError(
                    "actual probe component and exact-time independent world pose differ"
                )
            if math.dist(pose[:2], self.policy["probe_xy"]) > 0.1:
                raise ValueError("actual instrument left preregistered isolated placement")
            identity = digest(
                {
                    k: packet[k]
                    for k in ("world_entity_id", "body_model_entity_id", "contact_sources")
                }
            )
            if self.inventory_hash is not None and identity != self.inventory_hash:
                raise ValueError("actual probe inventory changed")
            ground = self.policy["native_policy"]["contact_policy"]["ground_collisions"][0]
            pairs = next(iter(mapped.values()))
            if any(ground not in pair for pair in pairs):
                raise ValueError("probe contacted a non-ground object")
            touching = bool(pairs)
            grounded = (
                abs(pose[2] - (self.policy["ground_z_m"] + self.policy["sphere_radius_m"])) <= 0.05
            )
            if touching and not grounded:
                raise ValueError(
                    "contact cache and actual probe height are physically inconsistent"
                )
            if self.phase in {"WAIT_GROUND", "READY_FOR_LIFT"}:
                if not (touching and grounded):
                    self.phase = "WAIT_GROUND"
                if self._stable(sim, wall, touching and grounded):
                    self.phase = "READY_FOR_LIFT"
            elif self.phase == "WAIT_CLEAR_AFTER_LIFT":
                fresh = packet["captured_at_unix_ns"] > self.lift_ack["acknowledged_at_unix_ns"]
                if fresh and pose[2] >= self.policy["lift_z_m"] - 0.25:
                    self.lift_pose_confirmed = True
                if self._stable(
                    sim,
                    wall,
                    fresh
                    and self.lift_pose_confirmed
                    and not touching
                    and pose[2] > self.policy["ground_z_m"] + 2,
                ):
                    self.phase, self.stable = "WAIT_RECONTACT", []
            elif self.phase == "WAIT_RECONTACT":
                if self._stable(sim, wall, touching and grounded):
                    self.cycles += 1
                    self.cycle_sim, self.cycle_wall = sim, wall
                    self.qualified_once = True
                    self.phase, self.stable = "READY_FOR_LIFT", []
            else:
                raise ValueError("unknown probe intervention state")
            self.previous, self.last_received, self.inventory_hash = current, wall, identity
            return {
                "original_source_sha256": hashlib.sha256(raw).hexdigest(),
                "original_size_bytes": len(raw),
                "probe_policy_hash": digest(self.policy),
                "sim_time_sec": sim,
                "probe_phase": self.phase,
                "measured_ground_contact": touching,
                "actual_probe_world_pose": pose,
                "completed_cache_cycles": self.cycles,
            }
        except ValueError as exc:
            self.fault = self.fault or str(exc)
            raise

    def snapshot(self, wall_time):
        if type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time < 0:
            raise ValueError("finite probe observer clock required")
        fresh = (
            self.previous is not None
            and 0 <= wall_time - self.last_received < 0.3
            and self.pose_source is not None
            and 0 <= wall_time - self.pose_source[1] < 0.3
        )
        refreshed = (
            self.cycles > 0
            and 0 <= self.previous[2] - self.cycle_sim <= self.policy["refresh_sim_sec"]
            and 0 <= wall_time - self.cycle_wall <= self.policy["refresh_wall_sec"]
        )
        if self.qualified_once and (not fresh or not refreshed):
            self.fault = (
                self.fault or "backend cache intervention source lost or refresh deadline expired"
            )
        return {
            "evidence_role": "probe_cache_intervention_corroboration_not_task_acceptance",
            "probe_policy_hash": digest(self.policy),
            "cache_update_pattern_observed": bool(fresh and refreshed and self.fault is None),
            "phase": self.phase,
            "completed_cache_cycles": self.cycles,
            "source_fault": self.fault,
            "runtime_backend_admission": "NOT_VERIFIED_REQUIRES_OWNED_RUNTIME_AND_CLOSED_SOURCE_REPLAY",
            "physical_acceptance": "NOT_VERIFIED",
            "authorization": False,
        }

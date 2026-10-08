"""Additional live observation constraint; never a robot permission or acceptance.

An owned SIM actor can use this constraint in addition to its daemon lease.
Source correspondence, actual world/Body admission and canonical closed replay
remain mandatory independent requirements.
"""

import math

from backend_probe_evidence import probe_policy
from closed_backend_probe import ProbeEventReplay
from native_contact_evidence import NativeContactEvidence, native_policy

from rosclaw.connectors.ros.diagnosis.coverage_audit import digest


class BackendObservationGate:
    def __init__(self, robot_policy, instrument_policy, pose_frame):
        robot = native_policy(robot_policy)
        if robot.get("sampling_semantics") != "ALL_POSTUPDATE_PHYSICS_STEPS":
            raise ValueError(
                "live robot constraint requires an explicit all-physics-step contact policy"
            )
        probe_policy(instrument_policy)
        probe = instrument_policy["native_policy"]
        first, second = robot["contact_policy"], probe["contact_policy"]
        robot_topics = {robot["component_topic"], first["pose_topic"], *first["contacts"]}
        probe_topics = {probe["component_topic"], second["pose_topic"], *second["contacts"]}
        if (
            first["run_id"] != second["run_id"]
            or robot["world_name"] != probe["world_name"]
            or first["model_name"] == second["model_name"]
            or robot_topics & probe_topics
        ):
            raise ValueError(
                "same owned run/world with disjoint robot and instrument observation roles required"
            )
        self.robot = NativeContactEvidence(robot)
        self.probe = ProbeEventReplay(instrument_policy, pose_frame)
        self.policy_hash = digest(
            {"robot": robot, "probe": instrument_policy, "pose_frame": pose_frame}
        )
        self.opened = False
        self.fault = None
        self.last_sample = None

    def snapshot(self, wall_time):
        if type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time < 0:
            self.fault = self.fault or "malformed live gate sample clock"
            raise ValueError("finite original live gate sample time required")
        if self.last_sample is not None and wall_time <= self.last_sample:
            self.fault = self.fault or "live gate sample clock repeated/regressed"
        self.last_sample = wall_time
        try:
            self.probe.transaction_fresh(wall_time)
        except ValueError as exc:
            self.fault = self.fault or str(exc)
        if self.probe.pending and wall_time - self.probe.pending[0][1] >= 0.3:
            self.probe.fault = (
                self.probe.fault or "pending original instrument component lacks timely exact pose"
            )
        robot = self.robot.snapshot(wall_time)
        probe = self.probe.tracker.snapshot(wall_time)
        ready = (
            robot["observation_complete"]
            and robot["collision_count"] == 0
            and not robot["active_contact_topics"]
            and probe["cache_update_pattern_observed"]
            and self.probe.fault is None
        )
        a, b = self.robot.actual_source_identity, self.probe.tracker.actual_source_identity
        if a is not None and b is not None:
            if a["world_entity_id"] != b["world_entity_id"] or a["entity_ids"] & b["entity_ids"]:
                self.fault = self.fault or "robot/instrument actual world or entity roles alias"
            if abs(self.robot.last[2] - self.probe.tracker.previous[2]) > 0.15:
                ready = False
        else:
            ready = False
        source_fault = robot["source_fault"] or probe["source_fault"] or self.probe.fault
        if source_fault:
            self.fault = self.fault or source_fault
        if self.opened and not ready:
            self.fault = self.fault or "live robot/instrument observation lost after gate opened"
        if ready and self.fault is None:
            self.opened = True
        return {
            "schema_version": "rosclaw.backend_observation_constraint.v1",
            "evidence_role": "additional_live_SIM_observation_constraint_not_permission_or_acceptance",
            "constraint_policy_hash": self.policy_hash,
            "sampled_monotonic_sec": wall_time,
            "live_source_constraint_satisfied": bool(ready and self.fault is None),
            "robot_sim_time_sec": robot["sim_time_sec"],
            "probe_sim_time_sec": self.probe.tracker.previous[2]
            if self.probe.tracker.previous
            else None,
            "robot_collision_count": robot["collision_count"],
            "probe_completed_cache_cycles": probe["completed_cache_cycles"],
            "probe_phase": probe["phase"],
            "probe_lift_transaction_pending": self.probe.lift_transaction is not None,
            "source_fault": self.fault,
            "runtime_actor_integration": "NOT_IMPLEMENTED",
            "actual_world_and_body_admission": "NOT_VERIFIED",
            "backend_health_admitted": False,
            "physical_acceptance": "NOT_VERIFIED",
            "authorization": False,
        }

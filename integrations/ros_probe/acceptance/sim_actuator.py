"""Simulator-owned lease/cleaner/drive bridge, separate from passive evidence.

Only the isolated SIM fixture launcher runs this process. Agents still use
rosclawd request_action. This bridge neither observes physics nor verifies a
mission. A prepared frozen brush_binding.json is required for split mode.
"""

import base64
import hashlib
import json
import math
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import rclpy
from actuator_observation_constraint import ActuatorObservationConstraint, decode
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from runtime_policy import load_frozen_sim_runtime_policy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool

from rosclaw.connectors.ros.diagnosis.coverage_audit import CoverageAuditLog
from rosclaw.connectors.ros.verification.brush_timeline import BrushStateEvent


def load_binding(path):
    binding = json.loads(Path(path).read_text())
    keys = {"run_id", "body_snapshot_hash", "attachment_hash", "producer_id"}
    if type(binding) is not dict or set(binding) != keys:
        raise ValueError("exact frozen SIM brush binding required")
    if any(type(v) is not str or not v for v in binding.values()):
        raise ValueError("nonempty frozen brush identities required")
    if binding["run_id"] != Path("/evidence/run_id.txt").read_text().strip():
        raise ValueError("brush producer run identity differs from scene")
    return binding


class SimulationActuator(Node):
    def __init__(self):
        super().__init__("rosclaw_sim_actuator")
        runtime_policy = load_frozen_sim_runtime_policy(Path("/evidence"))
        self.runtime_policy = runtime_policy
        self.binding = load_binding("/evidence/brush_binding.json")
        self.cleaning, self.lease, self.lease_updates, self.sequence = False, 0, 0, 0
        self.fault = None
        required = self.declare_parameter("require_backend_observation", False).value
        if type(required) is not bool:
            raise ValueError("explicit backend observation interlock boolean required")
        self.observation_constraint = None
        if required:
            config = decode(Path("/evidence/backend_actor_constraint.json").read_text())
            self.observation_constraint = ActuatorObservationConstraint(config, self.binding)
            self.create_subscription(
                String, "/rosclaw_sim/backend_observation_constraint", self.observation, 128
            )
        self.last_event_time = None
        self.controller_watchdog = self.declare_parameter("controller_watchdog", True).value
        if runtime_policy is not None and not self.controller_watchdog:
            raise ValueError("generic SIM drive requires the declared stamped controller watchdog")
        topics = (
            runtime_policy["policy"]["topics"]
            if runtime_policy is not None
            else {
                "drive_velocity": "/drive_controller/cmd_vel"
                if self.controller_watchdog
                else "/cmd_vel",
                "cleaning_state": "/rosclaw_sim/cleaning_state",
                "brush_events": "/rosclaw_sim/brush_events",
                "nav_velocity": "/nav_cmd_vel",
            }
        )
        endpoints = (
            runtime_policy["endpoints"]
            if runtime_policy is not None
            else {
                "cleaning": "/rosclaw_sim/cleaning",
                "hold": "/rosclaw_sim/hold",
                "lease": "/rosclaw_sim/lease",
            }
        )
        self.velocity = self.create_publisher(
            TwistStamped if self.controller_watchdog else Twist,
            topics["drive_velocity"],
            10,
        )
        self.cleaning_state = self.create_publisher(Bool, topics["cleaning_state"], 10)
        self.brush_events = self.create_publisher(String, topics["brush_events"], 2048)
        self.audit = CoverageAuditLog(
            Path("/evidence") / f"brush-events-{time.time_ns()}.jsonl",
            context={
                **self.binding,
                "source": "simulator_owned_actuator",
                "evidence_domain": "SIMULATION",
            },
        )
        self.observation_audit = (
            CoverageAuditLog(
                Path("/evidence") / f"actor-observation-events-{time.time_ns()}.jsonl",
                context={
                    **self.binding,
                    "source": "SIM_actor_observation_interlock",
                    "constraint_policy_hash": self.observation_constraint.config[
                        "constraint_policy_hash"
                    ],
                    "evidence_domain": "SIMULATION",
                },
            )
            if required
            else None
        )
        self.create_subscription(Twist, topics["nav_velocity"], self.command, 10)
        self.create_service(SetBool, endpoints["cleaning"], self.set_cleaning)
        self.holding = False
        self.create_service(SetBool, endpoints["hold"], self.set_hold)
        self.create_service(SetBool, endpoints["lease"], self.heartbeat)
        self.create_timer(0.05, self.tick, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def event(self, kind):
        stamp = self.get_clock().now().nanoseconds / 1e9
        if not math.isfinite(stamp) or stamp < 0:
            raise ValueError("finite SIM actuator clock required")
        if self.last_event_time is not None and stamp < self.last_event_time:
            raise ValueError("SIM actuator clock reversed")
        event = BrushStateEvent(
            **self.binding,
            sequence=self.sequence,
            sim_time_sec=stamp,
            kind=kind,
            enabled=self.cleaning,
            captured_at=datetime.now(UTC).isoformat(),
            complete=True,
        )
        self.sequence += 1
        self.last_event_time = stamp
        payload = {
            "event": asdict(event),
            "artifact_hash": event.artifact_hash(),
            "lease_remaining_sec": self.lease - time.monotonic(),
            "lease_updates": self.lease_updates,
        }
        self.audit.emit("brush_state_event", payload, sim_time=stamp)
        self.brush_events.publish(String(data=json.dumps(payload)))

    def publish_velocity(self, message):
        if self.controller_watchdog:
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.twist = message
            self.velocity.publish(stamped)
        else:
            self.velocity.publish(message)

    def observation(self, message):
        constraint = self.observation_constraint
        if constraint is None:
            raise RuntimeError("backend observation interlock not configured")
        wall = time.monotonic()
        constraint.receive(message.data, wall)
        raw = message.data.encode("utf-8", errors="surrogatepass")
        payload = {
            "received_monotonic_sec": wall,
            "original_source_sha256": hashlib.sha256(raw).hexdigest(),
            "original_size_bytes": len(raw),
            "source_bytes_complete": len(raw) <= 4096,
            "source_fault": constraint.fault,
        }
        if len(raw) <= 4096:
            payload["original_source_base64"] = base64.b64encode(raw).decode("ascii")
        self.observation_audit.emit("actor_observation_envelope_received", payload)
        self.source_ready()

    def source_ready(self):
        constraint = self.observation_constraint
        if constraint is None:
            return True
        sampled, sim = time.monotonic(), self.get_clock().now().nanoseconds / 1e9
        ready = constraint.ready(sampled, sim)
        lease_before = self.lease
        if self.observation_audit.dropped or self.observation_audit.error:
            constraint.fault = constraint.fault or "actor observation source audit incomplete"
            ready = False
        if not ready:
            self.lease = 0
            if constraint.fault:
                self.fault = self.fault or constraint.fault
            self.stop()
            self.cleaning_state.publish(Bool(data=False))
        self.observation_audit.emit(
            "actor_observation_interlock_sample",
            {
                "sampled_monotonic_sec": sampled,
                "actor_sim_time_sec": sim,
                "source_ready": ready,
                "source_sequence": constraint.last["sequence"] if constraint.last else None,
                "source_fault": constraint.fault,
                "lease_before": lease_before,
                "lease_after": self.lease,
                "cleaning_enabled": self.cleaning,
                "authorization": False,
            },
        )
        return ready

    def command(self, message):
        if (
            self.source_ready()
            and self.fault is None
            and not self.holding
            and time.monotonic() < self.lease
        ):
            self.publish_velocity(message)

    def set_hold(self, request, response):
        ready = self.source_ready()
        if self.fault or (not request.data and (not ready or time.monotonic() >= self.lease)):
            response.success = False
            response.message = "SIM hold release requires a live healthy daemon lease"
            return response
        self.holding = request.data
        if self.holding:
            try:
                self.stop()
            except (ValueError, RuntimeError) as exc:
                self.fault = str(exc)
                self.lease = 0
                self.cleaning = False
                self.publish_velocity(Twist())
                response.success = False
                response.message = "SIM hold source failed; drive stopped"
                return response
        response.success = True
        response.message = "SIM drive hold recorded"
        return response

    def set_cleaning(self, request, response):
        ready = self.source_ready()
        if self.fault or (
            request.data and (not ready or self.holding or time.monotonic() >= self.lease)
        ):
            response.success = False
            response.message = "SIM cleaner requires a live daemon lease and healthy event source"
            return response
        if self.cleaning != request.data:
            self.cleaning = request.data
            try:
                self.event("TRANSITION")
            except (ValueError, RuntimeError) as exc:
                self.fault = str(exc)
                self.lease = 0
                self.cleaning = False
                self.publish_velocity(Twist())
                response.success = False
                response.message = "SIM brush event failure; drive stopped"
                return response
        response.success = True
        response.message = "SIMULATED_CLEANING transition recorded; no physical sensor claim"
        return response

    def heartbeat(self, request, response):
        ready = self.source_ready()
        self.lease = time.monotonic() + 1.5 if request.data and ready and not self.fault else 0
        self.lease_updates += 1
        if not request.data:
            self.stop()
        response.success = self.fault is None and (not request.data or ready)
        return response

    def stop(self):
        self.publish_velocity(Twist())
        if self.cleaning:
            self.cleaning = False
            self.event("TRANSITION")

    def tick(self):
        try:
            if not self.source_ready() or self.fault or time.monotonic() >= self.lease:
                self.stop()
            self.event("WATERMARK")
            self.cleaning_state.publish(Bool(data=self.cleaning))
        except (ValueError, RuntimeError) as exc:
            self.fault = str(exc)
            self.lease = 0
            self.cleaning = False
            self.publish_velocity(Twist())


def main():
    rclpy.init()
    node = SimulationActuator()
    try:
        rclpy.spin(node)
    finally:
        try:
            if rclpy.ok():
                node.stop()
        finally:
            if node.observation_audit is not None:
                observation_summary = node.observation_audit.close()
                Path(observation_summary["path"] + ".summary.json").write_text(
                    json.dumps(observation_summary) + "\n"
                )
            summary = node.audit.close()
            Path(summary["path"] + ".summary.json").write_text(json.dumps(summary) + "\n")
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()

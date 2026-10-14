"""Separate SDK SIM sink: bootstrap refusal, explicit admission, leased commands.

Run only inside the disposable SDK fixture. Model tools never open this ROS
process or write its private operator context. All actions still originate at
canonical MCP request_action backed by the independent existing rosclawd.
"""

import argparse
import hashlib
import hmac
import json
import os
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from simulation_current_facts import CurrentSimulationFacts, _read
from simulation_current_process import SdkCurrentFactsProcess
from simulation_execution_proof import ExecutionProofPins, _canonical
from simulation_navigation_supervision import review_active_sim_daemon
from simulation_sink_admission import authenticate_sink_admission
from simulation_source_capture import BoundedSourceJsonlCapture
from simulation_velocity_gate import SimulationVelocityGate
from simulation_witness_sources import SimulationWitnessSources
from std_srvs.srv import SetBool

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.daemon.client import DaemonClient


class GovernedSimulationSink(Node):
    def __init__(self, *, source_path, source_sha256, p03_sha256):
        super().__init__("rosclaw_governed_sink")
        self.root = Path("/evidence")
        self.control = self.root / "execution-control"
        self.sources = SimulationWitnessSources.from_bytes(
            _read(source_path, 128_000),
            expected_sha256=source_sha256,
            urdf_bytes=_read(Path("/assets/robot.urdf"), 5_000_000),
            world_sdf_bytes=_read(Path("/assets/world.sdf"), 20_000_000),
        )
        if self.sources.drive_twist_stamped is not True:
            raise ValueError("generic leased sink requires the actual stamped controller watchdog")
        self.p03_sha256 = p03_sha256
        self.binding = self.admission = self.facts = self.client = self.gate = None
        self.monitor = None
        self.key = self.velocity = None
        self.failed_reason = None
        self.deadline_ns = self.lease_updates = self.received = self.published = self.zeroes = 0
        self.action_id = None
        self.used_actions = set()
        self.events = BoundedSourceJsonlCapture(
            self.root / "governed-sink-events-original.jsonl",
            max_segments=1,
            max_bytes=30_000_000,
            max_rows=5000,
            max_line_bytes=32_000,
        )
        self.create_subscription(Twist, self.sources.topic("velocity_input"), self.command, 1)
        self.create_service(SetBool, self.sources.topic("lease"), self.heartbeat)
        self.wall_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.timer = self.create_timer(0.05, self.tick, clock=self.wall_clock)
        self._write_status()

    def _bootstrap_binding(self):
        if self.admission is not None:
            return
        workspace = self.root / (
            "generic-compiled"
            if (self.root / "generic-compiled/execution_config.json").exists()
            else "home"
        )
        config_path = (
            (workspace / "execution_config.json")
            if workspace.name == "generic-compiled"
            else self.root / "execution_config.json"
        )
        try:
            config = json.loads(_read(config_path, 16_000))
            body = BodyResolver(workspace=workspace, body_id=config["body_id"]).get_effective_body(
                recompile_if_stale=False
            )
            binding = SimulationNavigationBinding.from_body(body)
            if (
                binding.body_snapshot_hash != config["body_snapshot_hash"]
                or binding.run_id != self.sources.run_id
                or binding.frame_id != self.sources.map_frame
                or binding.lease_service != self.sources.topic("lease")
            ):
                raise ValueError("bootstrap Body/source routes differ")
            self.binding = binding
        except (FileNotFoundError, ValueError, KeyError):
            # No admission or drive publisher exists during incomplete bootstrap.
            return

    def _try_admission(self):
        if self.admission is not None or not (self.control / "admission.json").exists():
            return
        if self.binding is None:
            raise ValueError("compiled Body unavailable for explicit admission")
        key_path = self.control / "fixture.key"
        info = key_path.stat(follow_symlinks=False)
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("private operator SDK key required")
        key = _read(key_path, 32)
        raw = _read(self.control / "admission.json", 16_000)
        envelope = json.loads(raw)
        signature = hmac.new(key, _canonical(envelope["payload"]), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(envelope["hmac_sha256"], signature):
            raise ValueError("explicit SDK operator admission authentication failed")
        facts = CurrentSimulationFacts(
            self.root,
            assets_directory=Path("/assets"),
            enrollment_sha256=envelope["payload"]["facts_enrollment_sha256"],
        )
        try:
            admission = authenticate_sink_admission(
                raw,
                key=key,
                binding=self.binding,
                facts=facts,
                p03_raw=_read(self.root / "P03-accepted-review.json", 128_000),
                p03_sha256=self.p03_sha256,
                now_ns=time.monotonic_ns(),
            )
            self.client = DaemonClient(
                socket_path=self.control / "daemon/run/rosclawd.sock",
                expected_daemon_uid=admission["daemon_uid"],
                timeout_sec=0.1,
            )
            self.admission, self.key, self.facts = admission, key, facts
            self.monitor = SdkCurrentFactsProcess(
                root=self.root,
                enrollment_sha256=envelope["payload"]["facts_enrollment_sha256"],
                admission=admission,
                binding=self.binding,
            )
            self.monitor.start()
        except BaseException:
            if self.monitor is not None:
                self.monitor.close()
            facts.close()
            raise

    def _current_local(self, *, require_active=True):
        if self.admission is None or time.monotonic_ns() >= self.admission["expires_ns"]:
            raise ValueError("explicit SDK operator context unavailable or expired")
        facts = self.monitor.get()
        status = facts["daemon_status"]
        a = self.admission
        review_active_sim_daemon(
            status,
            instance_id=a["daemon_instance_id"],
            pid=a["daemon_pid"],
            uid=a["daemon_uid"],
            body_id=self.binding.body_id,
            expected_running=1 if require_active else None,
        )
        self._event("CURRENT_LOCAL_FACTS", facts=facts, daemon_status=status)
        return facts

    def _event(self, event, **fields):
        self.events.write(
            json.dumps(
                {
                    "event": event,
                    "monotonic_ns": time.monotonic_ns(),
                    "run_id": self.sources.run_id,
                    **fields,
                },
                allow_nan=False,
            )
            + "\n"
        )

    def _physical_stopped(self):
        fd = os.open(self.root / "witness.jsonl", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            size = os.fstat(fd).st_size
            if not 0 < size <= 10_000_000:
                raise ValueError("bounded independent physics witness required")
            raw = os.pread(fd, min(size, 8192), max(0, size - 8192))
        finally:
            os.close(fd)
        complete = raw[: raw.rfind(b"\n") + 1]
        sample = json.loads(complete.splitlines()[-1])
        age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
        if not 0 <= age <= 0.3 or not self.binding.stopped(sample):
            raise ValueError("fresh independently measured stopped Body required before lease")

    def _zero(self):
        if self.velocity is not None:
            message = TwistStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            self.velocity.publish(message)
            self.zeroes += 1

    def _fatal(self, reason):
        self.failed_reason = self.failed_reason or str(reason)[:1000]
        self.deadline_ns = 0
        if self.gate is not None:
            self.gate.revoke()
        self._zero()
        self._write_status()

    def heartbeat(self, request, response):
        try:
            self._event("LEASE_REQUEST", enabled=request.data, admitted=self.admission is not None)
            if not request.data:
                if self.gate is not None:
                    self.gate.revoke()
                    self.gate = None
                self.deadline_ns = 0
                self._zero()
                self._write_status()
                response.success = True
                return response
            if self.failed_reason is not None:
                raise ValueError(self.failed_reason)
            self._bootstrap_binding()
            self._try_admission()
            if self.admission is None:
                self._event("LEASE_REJECTED_BEFORE_ADMISSION")
                response.success = False
                response.message = "bootstrap refuses lease without explicit SDK operator admission"
                return response
            raw = _read(self.control / "proof.json", 8192)
            action_id = json.loads(raw)["payload"]["action_id"]
            if self.gate is None:
                if (
                    action_id not in self.admission["allowed_action_ids"]
                    or action_id in self.used_actions
                ):
                    raise ValueError("operator action scope or terminal action reuse rejected")
                self._physical_stopped()
                keys = (
                    "body_id",
                    "body_snapshot_hash",
                    "manifest_hash",
                    "run_id",
                    "world_sha256",
                    "observer_epoch",
                    "daemon_instance_id",
                )
                pins = ExecutionProofPins(
                    **{k: self.admission[k] for k in keys}, action_id=action_id
                )
                self.gate = SimulationVelocityGate(
                    binding=self.binding, pins=pins, key=self.key, collect_local=self._current_local
                )
                self.action_id = action_id
            elif action_id != self.action_id:
                raise ValueError("active action identity changed without revoke")
            self.gate.refresh_lease(raw)
            self.deadline_ns = self.gate._deadline
            self.used_actions.add(action_id)
            self.lease_updates += 1
            if self.velocity is None:
                self.velocity = self.create_publisher(
                    TwistStamped, self.sources.topic("drive_command"), 1
                )
            self._write_status()
            self._event("LEASE_RENEWED", action_id=self.action_id, expires_ns=self.deadline_ns)
            response.success = True
        except Exception as exc:
            self._fatal(exc)
            response.success = False
            response.message = self.failed_reason
        return response

    def command(self, message):
        self.received += 1
        if self.gate is None or self.failed_reason is not None:
            return
        try:
            command = {
                "linear": {k: getattr(message.linear, k) for k in ("x", "y", "z")},
                "angular": {k: getattr(message.angular, k) for k in ("x", "y", "z")},
            }
            checked = self.gate.validate_command(
                command, raw=_read(self.control / "proof.json", 8192)
            )
            if checked is None:
                raise ValueError("command has no current independently guarded lease")
            stamped = TwistStamped()
            stamped.header.stamp = self.get_clock().now().to_msg()
            stamped.twist = message
            self.velocity.publish(stamped)
            self.published += 1
            self._event("VELOCITY_ACCEPTED", action_id=self.action_id, command=checked)
        except Exception as exc:
            self._fatal(exc)

    def tick(self):
        try:
            if self.failed_reason is None:
                self._bootstrap_binding()
                self._try_admission()
                if (
                    self.monitor is not None
                    and not self.monitor.ready
                    and not self.monitor.poll_ready()
                ):
                    self._write_status()
                    return
                if self.gate is not None:
                    self.gate.watchdog(_read(self.control / "proof.json", 8192))
                elif self.admission is not None:
                    self._current_local(require_active=False)
            if self.gate is None or self.failed_reason is not None:
                self._zero()
        except Exception as exc:
            self._fatal(exc)
        self._write_status()

    def _write_status(self):
        status = {
            "schema_version": "rosclaw.sim_sink_status.v1",
            "run_id": self.sources.run_id,
            "body_snapshot_hash": self.binding.body_snapshot_hash if self.binding else None,
            "monotonic_ns": time.monotonic_ns(),
            "lease_expires_ns": self.deadline_ns,
            "terminal": self.failed_reason is not None,
            "lease_updates": self.lease_updates,
            "received_commands": self.received,
            "published_velocity_messages": self.published,
            "stop_zero_messages": self.zeroes,
            "action_id": self.action_id,
            "daemon_instance_id": self.admission["daemon_instance_id"]
            if self.admission and self.monitor is not None and self.monitor.ready
            else None,
        }
        temporary = self.root / "governed-sink-status.tmp"
        temporary.write_text(json.dumps(status) + "\n")
        temporary.replace(self.root / "governed-sink-status.json")
        if self.failed_reason is not None:
            target = self.root / "governed-sink-terminal.json"
            if not target.exists():
                with target.open("x") as stream:
                    stream.write(
                        json.dumps({"reason": self.failed_reason, "status": status}) + "\n"
                    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources-path", type=Path, required=True)
    parser.add_argument("--sources-sha256", required=True)
    parser.add_argument("--p03-review-sha256", required=True)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = GovernedSimulationSink(
        source_path=args.sources_path,
        source_sha256=args.sources_sha256,
        p03_sha256=args.p03_review_sha256,
    )
    try:
        try:
            with suppress(ExternalShutdownException, KeyboardInterrupt):
                rclpy.spin(node)
        except RCLError:
            if rclpy.ok():
                raise

    finally:
        node.deadline_ns = 0
        if rclpy.ok():
            node._zero()
        if node.monitor is not None:
            node.monitor.close()
        if node.facts is not None:
            node.facts.close()
        node.events.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

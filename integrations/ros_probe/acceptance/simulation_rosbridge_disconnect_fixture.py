"""SDK operator fault closes only the existing daemon's four owned SIM sockets.

No Runtime, driver, publisher, goal, new ROS connection or authority is created.
The fixture server, Gazebo and independent DDS observations remain alive.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from simulation_current_facts import _read
from simulation_navigation_supervision import review_active_sim_daemon

from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport


class OwnedRosbridgeDisconnectFixture:
    def __init__(
        self, *, root, policy, policy_sha256, binding, transports, daemon_status, instance_id
    ):
        if (
            not isinstance(binding, SimulationNavigationBinding)
            or policy["case"] != "rosbridge_disconnect"
            or policy["body_snapshot_hash"] != binding.body_snapshot_hash
            or policy["run_id"] != binding.run_id
            or len(transports) != 4
            or any(not isinstance(t, RosbridgeTransport) or t._dry_run for t in transports)
            or not callable(daemon_status)
        ):
            raise ValueError("original owned SDK SIM disconnect context required")
        self.root = Path(root)
        self.policy = policy
        self.policy_sha256 = policy_sha256
        self.binding = binding
        self.transports = tuple(transports)
        self.daemon_status = daemon_status
        self.instance_id = instance_id
        self.fired = False

    def poll(self):
        trigger = self.root / "governed-rosbridge-disconnect-trigger-original.json"
        if self.fired or not trigger.exists():
            return
        value = json.loads(_read(trigger, 32_000))
        if (
            set(value) != {"policy_sha256", "injection"}
            or value["policy_sha256"] != self.policy_sha256
        ):
            raise ValueError("pinned actual-motion disconnect trigger required")
        injection = value["injection"]
        sample = injection["actual_prior_physics"]
        sink = injection["actual_prior_sink"]
        self.binding.validate_observation(sample)
        age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
        if (
            not 0 <= age <= 0.3
            or sample["lease_enabled"] is not True
            or not (
                abs(sample["linear_speed_mps"]) > self.binding.stop_speed_mps
                or abs(sample["angular_speed_rps"]) > self.binding.stop_yaw_rate_rps
            )
            or sink["daemon_instance_id"] != self.instance_id
            or sink["action_id"] != self.policy["action_id"]
        ):
            raise ValueError("fresh actual governed motion and original daemon action required")
        review_active_sim_daemon(
            self.daemon_status(),
            instance_id=self.instance_id,
            pid=os.getpid(),
            uid=os.getuid(),
            body_id=self.binding.body_id,
        )
        before = []
        for index, t in enumerate(self.transports):
            ws = t._ws
            sock = getattr(ws, "sock", None)
            if t._closed or ws is None or getattr(ws, "connected", False) is not True:
                raise ValueError(
                    "all original daemon navigation sockets must be connected before injection"
                )
            before.append(
                {
                    "index": index,
                    "endpoint": t.endpoint.url,
                    "connected": True,
                    "socket_fd": sock.fileno() if sock is not None else None,
                }
            )
        started = time.monotonic_ns()
        self.fired = True
        for t in self.transports:
            t.close()
        after = [
            {"index": i, "closed": t._closed, "websocket_released": t._ws is None}
            for i, t in enumerate(self.transports)
        ]
        if not all(x["closed"] and x["websocket_released"] for x in after):
            raise ValueError("actual original socket closure incomplete")
        result = {
            "schema_version": "rosclaw.owned_rosbridge_disconnect.v1",
            "case": "rosbridge_disconnect",
            "policy_sha256": self.policy_sha256,
            "run_id": self.binding.run_id,
            "body_snapshot_hash": self.binding.body_snapshot_hash,
            "daemon_instance_id": self.instance_id,
            "daemon_pid": os.getpid(),
            "before_monotonic_ns": started,
            "after_monotonic_ns": time.monotonic_ns(),
            "original_navigation_sockets_before": before,
            "original_navigation_sockets_after": after,
            "new_connections_opened": 0,
            "physics_server_signals_sent": 0,
        }
        pending = self.root / "governed-rosbridge-disconnect-effect.pending"
        with pending.open("x") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        pending.rename(self.root / "governed-rosbridge-disconnect-effect-original.json")

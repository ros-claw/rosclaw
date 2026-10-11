"""Read-only owned-fixture lifecycle replies, never an action or stop proof."""

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

REQUIRED_NODES = (
    "amcl",
    "map_server",
    "planner_server",
    "controller_server",
    "collision_monitor",
    "bt_navigator",
    "coverage_server",
)
SCHEMA = "rosclaw.fixture_lifecycle_readiness.v1"


def readiness(snapshot, *, now=None):
    now = time.monotonic() if now is None else now
    if (
        type(snapshot) is not dict
        or snapshot.get("schema_version") != SCHEMA
        or snapshot.get("source") != "actual_read_only_GetState_responses"
        or type(snapshot.get("responses")) is not dict
        or set(snapshot["responses"]) != set(REQUIRED_NODES)
    ):
        return False
    for name, row in snapshot["responses"].items():
        if type(row) is not dict:
            return False
        stamp = row.get("received_monotonic_sec")
        if (
            type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or not 0 <= now - stamp < 2.0
            or row.get("service") != "/" + name + "/get_state"
            or type(row.get("state_id")) is not int
            or row["state_id"] != 3
            or row.get("state_label") != "active"
        ):
            return False
    return True


class LifecycleProbe:
    """Bound pending requests; preserve direct replies and loss as UNKNOWN."""

    def __init__(self, node, service_type, output):
        self.service_type = service_type
        self.output = Path(output)
        self.clients = {
            name: node.create_client(service_type, "/" + name + "/get_state")
            for name in REQUIRED_NODES
        }
        self.pending = {}
        self.responses = dict.fromkeys(REQUIRED_NODES)
        self.trace = (self.output / "lifecycle-responses.jsonl").open("x", buffering=1)

    def poll(self):
        now = time.monotonic()
        for name, client in self.clients.items():
            pending = self.pending.get(name)
            if pending is not None:
                future, requested = pending
                if future.done():
                    del self.pending[name]
                    try:
                        reply = future.result().current_state
                        if now - requested >= 1.0:
                            raise ValueError("late lifecycle response")
                        row = {
                            "service": "/" + name + "/get_state",
                            "state_id": int(reply.id),
                            "state_label": reply.label,
                            "requested_monotonic_sec": requested,
                            "received_monotonic_sec": now,
                            "captured_at": datetime.now(UTC).isoformat(),
                        }
                        self.responses[name] = row
                        self.trace.write(json.dumps(row, allow_nan=False) + "\n")
                    except Exception as error:
                        self.responses[name] = None
                        self.trace.write(
                            json.dumps(
                                {"node": name, "error": str(error), "received_monotonic_sec": now}
                            )
                            + "\n"
                        )
                elif now - requested >= 1.0:
                    client.remove_pending_request(future)
                    future.cancel()
                    del self.pending[name]
                    self.responses[name] = None
                    self.trace.write(
                        json.dumps(
                            {
                                "node": name,
                                "error": "GetState response timeout",
                                "received_monotonic_sec": now,
                            }
                        )
                        + "\n"
                    )
            if name not in self.pending and client.service_is_ready():
                self.pending[name] = (client.call_async(self.service_type.Request()), now)
        value = {
            "schema_version": SCHEMA,
            "source": "actual_read_only_GetState_responses",
            "responses": self.responses,
            "sampled_monotonic_sec": now,
            "authorization": False,
            "physical_stop_proof": "NOT_MEASURED",
        }
        value["ready"] = readiness(value, now=now)
        temporary = self.output / "lifecycle-readiness.pending.json"
        temporary.write_text(json.dumps(value, allow_nan=False) + "\n")
        temporary.replace(self.output / "lifecycle-readiness.json")

    def close(self):
        self.trace.close()


def main():
    import rclpy
    from lifecycle_msgs.srv import GetState
    from rclpy.clock import Clock, ClockType
    from rclpy.node import Node

    rclpy.init()
    node = Node("rosclaw_read_only_lifecycle_probe")
    probe = LifecycleProbe(node, GetState, "/evidence")
    node.create_timer(0.25, probe.poll, clock=Clock(clock_type=ClockType.STEADY_TIME))
    try:
        rclpy.spin(node)
    finally:
        probe.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

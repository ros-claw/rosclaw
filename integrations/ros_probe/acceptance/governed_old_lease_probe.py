"""SDK-only negative lease probe, restricted to an already terminal SIM sink.

No publisher, goal, Runtime or daemon is created. Never probes a healthy sink.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from simulation_current_facts import _read
from simulation_witness_sources import SimulationWitnessSources
from std_srvs.srv import SetBool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("owned isolated SIM negative fixture required")
    root = Path("/evidence")
    raw = _read(root / "governed-fault.json", 4096)
    if hashlib.sha256(raw).hexdigest() != args.policy_sha256:
        raise ValueError("explicit original fault pin required")
    policy = json.loads(raw)
    actor = json.loads(_read(root / "governed-fault-actor-outcome-original.json", 256_000))
    if (
        actor["status"] != "PASS_ACTUAL_DAEMON_SIGKILL_INDEPENDENT_STABLE_STOP_LOCAL"
        or actor["policy_sha256"] != args.policy_sha256
    ):
        raise ValueError("actual injected fault and independent stop required first")
    source_raw = _read(Path("/assets/observer-sources.json"), 128_000)
    sources = SimulationWitnessSources.from_bytes(
        source_raw,
        expected_sha256=hashlib.sha256(source_raw).hexdigest(),
        urdf_bytes=_read(Path("/assets/robot.urdf"), 5_000_000),
        world_sdf_bytes=_read(Path("/assets/world.sdf"), 20_000_000),
    )
    if sources.run_id != policy["run_id"]:
        raise ValueError("original SIM run required")
    before = json.loads(_read(root / "governed-sink-status.json", 4096, allow_atomic_unlink=True))
    if (
        before["terminal"] is not True
        or before["lease_expires_ns"] != 0
        or before["run_id"] != policy["run_id"]
        or before["body_snapshot_hash"] != policy["body_snapshot_hash"]
    ):
        raise ValueError("never probe a nonterminal or differently bound sink")
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_fixture_old_lease_refusal")
    try:
        client = node.create_client(SetBool, sources.topic("lease"))
        if not client.wait_for_service(timeout_sec=1):
            raise ValueError("actual sink lease service required")
        request = SetBool.Request()
        request.data = True
        started = time.monotonic_ns()
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=1)
        if not future.done() or future.result().success is not False:
            raise ValueError("old final-sink lease was not actually rejected")
        after = json.loads(
            _read(root / "governed-sink-status.json", 4096, allow_atomic_unlink=True)
        )
        if (
            after["terminal"] is not True
            or after["lease_expires_ns"] != 0
            or after["published_velocity_messages"] != before["published_velocity_messages"]
        ):
            raise ValueError("terminal sink changed or forwarded a command during old-lease probe")
        result = {
            "status": "PASS_OLD_LEASE_ACTUALLY_REJECTED_BY_TERMINAL_SDK_SINK",
            "policy_sha256": args.policy_sha256,
            "run_id": policy["run_id"],
            "body_snapshot_hash": policy["body_snapshot_hash"],
            "service": sources.topic("lease"),
            "requested_enable": True,
            "response_success": future.result().success,
            "response_message": future.result().message,
            "before_monotonic_ns": started,
            "after_monotonic_ns": time.monotonic_ns(),
            "sink_before": before,
            "sink_after": after,
            "velocity_publishers_created": 0,
            "goals_sent": 0,
        }
        with (root / "governed-old-lease-refusal-original.json").open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({"status": result["status"]}), flush=True)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

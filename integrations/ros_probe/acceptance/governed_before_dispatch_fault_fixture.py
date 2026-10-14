"""SDK negative intent actor: actual MCP refusal and independent no-motion proof.

No ROS, Runtime, driver, publisher, goal client, lease operation or key access.
"""

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from governed_daemon_sigkill_fixture import fresh_physics
from simulation_completed_artifact import publish_completed_fault_outcome
from simulation_current_facts import _read
from simulation_execution_fault_policy import (
    BEFORE_FAULT_ACTOR_STATUS,
    parse_governed_before_dispatch_fault_policy,
)
from simulation_unadmitted_sink import require_unadmitted_sink

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding
from rosclaw.daemon.client import DaemonClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--daemon-pid", type=int, required=True)
    parser.add_argument("--daemon-start-time", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("owned isolated before-dispatch SDK fixture required")
    root = Path("/evidence")
    config = json.loads(_read(root / "generic-compiled/execution_config.json", 16384))
    binding = SimulationNavigationBinding.from_body(
        BodyResolver(
            workspace=root / "generic-compiled", body_id=config["body_id"]
        ).get_effective_body(recompile_if_stale=False)
    )
    goals = json.loads(_read(root / "governed-goals.json", 16000))
    policy = parse_governed_before_dispatch_fault_policy(
        _read(root / "governed-fault.json", 4096),
        expected_sha256=args.policy_sha256,
        run_id=binding.run_id,
        body_snapshot_hash=binding.body_snapshot_hash,
        world_sha256=hashlib.sha256(_read(Path("/assets/world.sdf"), 20_000_000)).hexdigest(),
        action_id=goals["goals"][0]["action_id"],
    )
    ready = json.loads(_read(root / "governed-daemon-ready-original.json", 16384))
    if (
        ready["pid"] != args.daemon_pid
        or Path(f"/proc/{args.daemon_pid}/stat").read_text().split(")", 1)[1].split()[19]
        != args.daemon_start_time
    ):
        raise ValueError("original owned daemon process required")
    before_sink = require_unadmitted_sink(
        root, run_id=binding.run_id, body_snapshot_hash=binding.body_snapshot_hash
    )
    before_physics = fresh_physics(root, binding)
    if not binding.stopped(before_physics):
        raise ValueError("before-dispatch fixture must start independently stopped")
    with (root / "governed-fault-actor-ready-original.json").open("x") as f:
        json.dump(
            {
                "case": policy["case"],
                "policy_sha256": args.policy_sha256,
                "ready_monotonic_ns": time.monotonic_ns(),
                "requests_sent": 0,
            },
            f,
        )
    deadline = time.monotonic_ns() + 15_000_000_000
    rejected = root / "governed-action-0-request-rejected-original.json"
    receipt_path = root / "governed-action-0-receipt-original.json"
    while not rejected.exists() and not receipt_path.exists():
        if time.monotonic_ns() >= deadline:
            raise ValueError("actual original canonical MCP refusal did not arrive")
        time.sleep(0.01)
    request = json.loads(_read(root / "governed-action-0-request-original.json", 32768))
    refusal = json.loads(_read(rejected if rejected.exists() else receipt_path, 256000))
    if policy["case"] == "wrong_body_hash":
        receipt = refusal.get("receipt", {})
        if (
            request["body_snapshot_hash"] == binding.body_snapshot_hash
            or receipt.get("body_snapshot_hash") != request["body_snapshot_hash"]
            or receipt.get("final_state") != "BLOCKED"
            or not any(e.get("code") == "BODY_SNAPSHOT_MISMATCH" for e in receipt.get("errors", []))
        ):
            raise ValueError(
                "actual wrong Body hash must have original snapshot-mismatch BLOCKED receipt"
            )
    else:
        receipt = refusal.get("receipt", {})
        if receipt.get("final_state") != "BLOCKED" or not any(
            e.get("code") == "NAVIGATION_SCOPE_REJECTED"
            and "bound small-motion distance" in e.get("message", "")
            for e in receipt.get("errors", [])
        ):
            raise ValueError("actual out-of-bounds goal must have original scope-rejected receipt")
    client = DaemonClient(
        socket_path=root / "execution-control/daemon/run/rosclawd.sock",
        expected_daemon_uid=os.getuid(),
        timeout_sec=0.5,
    )
    status = client.get_runtime_status()
    if (
        status["daemon_peer"]["pid"] != ready["pid"]
        or status["daemon_instance_id"] != ready["daemon_instance_id"]
        or status["queue"]["RUNNING"] != 0
        or status["queue"]["QUEUED"] != 0
    ):
        raise ValueError("same actual daemon must be idle after rejected intent")
    samples, stable, last = [], None, None
    deadline = time.monotonic_ns() + 3_000_000_000
    while time.monotonic_ns() < deadline:
        sample = fresh_physics(root, binding)
        require_unadmitted_sink(
            root, run_id=binding.run_id, body_snapshot_hash=binding.body_snapshot_hash
        )
        if not binding.stopped(sample):
            raise ValueError("invalid intent caused physical motion")
        if sample["captured_at"] != last:
            last = sample["captured_at"]
            samples.append(sample)
            stable = stable or datetime.fromisoformat(last)
            if (datetime.fromisoformat(last) - stable).total_seconds() >= 0.3:
                break
        time.sleep(0.01)
    else:
        raise ValueError("independent stable no-motion observation missing")
    outcome = {
        "status": BEFORE_FAULT_ACTOR_STATUS[policy["case"]],
        "case": policy["case"],
        "policy_sha256": args.policy_sha256,
        "injection": {
            **json.loads(_read(root / "governed-fault-injection-original.json", 32768)),
            "actual_request": request,
            "actual_refusal": refusal,
        },
        "before_sink": before_sink,
        "before_physics": before_physics,
        "daemon_status_after_refusal": status,
        "after_monotonic_ns": time.monotonic_ns(),
        "original_post_fault_physics": samples,
        "actual_unadmitted_sink": require_unadmitted_sink(
            root, run_id=binding.run_id, body_snapshot_hash=binding.body_snapshot_hash
        ),
        "G3_done": False,
        "v1_done": False,
    }
    publish_completed_fault_outcome(root, outcome)


if __name__ == "__main__":
    main()

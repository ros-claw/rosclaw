"""SDK-only negative lease probe through the existing fixture rosbridge.

No DDS participant, publisher, goal, Runtime or daemon is created. The socket
is prepared passively; the only lease RPC is AFTER independently confirmed
terminal stop. Never probes a healthy sink and never retries a lease request.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from simulation_current_facts import _read, read_current_native_pointer
from simulation_witness_sources import SimulationWitnessSources

from rosclaw.connectors.ros.transport.base import RosbridgeEndpoint
from rosclaw.connectors.ros.transport.rosbridge import RosbridgeTransport


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
    source_raw = _read(Path("/assets/observer-sources.json"), 128_000)
    sources = SimulationWitnessSources.from_bytes(
        source_raw,
        expected_sha256=hashlib.sha256(source_raw).hexdigest(),
        urdf_bytes=_read(Path("/assets/robot.urdf"), 5_000_000),
        world_sdf_bytes=_read(Path("/assets/world.sdf"), 20_000_000),
    )
    if sources.run_id != policy["run_id"]:
        raise ValueError("original SIM run required")
    pointer, _ = read_current_native_pointer(root)
    native_raw = _read(root / pointer["file"], 10_000_000)
    if hashlib.sha256(native_raw).hexdigest() != pointer["sha256"]:
        raise ValueError("original Native bytes required")
    native = json.loads(native_raw)
    if [x for x in native["graph"]["services"] if x.get("name") == sources.topic("lease")] != [
        {"name": sources.topic("lease"), "srv_type": "std_srvs/srv/SetBool"}
    ]:
        raise ValueError("actual original final-sink lease service declaration required")
    transport = RosbridgeTransport(RosbridgeEndpoint.from_url("ws://127.0.0.1:9090"), max_retries=0)
    try:
        setup_started_ns = time.monotonic_ns()
        connected = transport.connect()
        if not connected.ok:
            raise ValueError("owned existing SDK rosbridge unavailable: " + str(connected.error))
        resources = {}
        for directory in [Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/pids")]:
            for name in ["pids.current", "pids.max", "pids.peak", "pids.events"]:
                path = directory / name
                if path.is_file():
                    with path.open() as stream:
                        resources[str(path)] = stream.read(256)
        with (root / "governed-old-lease-probe-ready-original.json").open("x") as f:
            json.dump(
                {
                    "run_id": policy["run_id"],
                    "service": sources.topic("lease"),
                    "setup_started_ns": setup_started_ns,
                    "ready_monotonic_ns": time.monotonic_ns(),
                    "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID"),
                    "requests_sent": 0,
                    "velocity_publishers_created": 0,
                    "DDS_participants_created": 0,
                    "transport": "existing isolated SDK rosbridge",
                    "resource_observation": resources,
                },
                f,
            )
            f.write("\n")
        deadline = time.monotonic_ns() + 20_000_000_000
        actor_path = root / "governed-fault-actor-outcome-original.json"
        while not actor_path.exists():
            if time.monotonic_ns() >= deadline:
                raise ValueError("actual injected fault and independent stop did not arrive")
            time.sleep(0.01)
        actor = json.loads(_read(actor_path, 256_000))
        if (
            actor["status"] != "PASS_ACTUAL_DAEMON_SIGKILL_INDEPENDENT_STABLE_STOP_LOCAL"
            or actor["policy_sha256"] != args.policy_sha256
        ):
            raise ValueError("actual injected fault and independent stop required first")
        before = json.loads(
            _read(root / "governed-sink-status.json", 4096, allow_atomic_unlink=True)
        )
        if (
            before["terminal"] is not True
            or before["lease_expires_ns"] != 0
            or before["run_id"] != policy["run_id"]
            or before["body_snapshot_hash"] != policy["body_snapshot_hash"]
        ):
            raise ValueError("never probe a nonterminal or differently bound sink")
        started = time.monotonic_ns()
        response = transport.call_service(
            sources.topic("lease"),
            {"data": True},
            service_type="std_srvs/srv/SetBool",
            timeout_sec=1,
        )
        values = (response.data or {}).get("values", {})
        if response.ok is not True or values.get("success") is not False:
            raise ValueError(
                "old lease was not actually rejected by a successful service exchange: "
                + str(response.error)
            )
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
            "response_success": values["success"],
            "response_message": values.get("message"),
            "actual_original_service_exchange": response.data,
            "before_monotonic_ns": started,
            "after_monotonic_ns": time.monotonic_ns(),
            "sink_before": before,
            "sink_after": after,
            "velocity_publishers_created": 0,
            "DDS_participants_created": 0,
            "goals_sent": 0,
        }
        with (root / "governed-old-lease-refusal-original.json").open("x") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(json.dumps({"status": result["status"]}), flush=True)
    finally:
        transport.close()


if __name__ == "__main__":
    main()

"""SDK-owned passive ECM/current-ROS/daemon observer; no Runtime or actions.

Enrollment is written by the independent operator only after stopped G1/L1.
Every poll rebuilds the current model from original Native bytes, checks the
immutable compiled Body, and obtains actual kernel-authenticated rosclawd
status. Any loss latches permanently; this program never arms an executor.
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from simulation_source_stream import SimulationSourceStreamReader, SourceStreamPins

from rosclaw.connectors.ros.context.navigation_readiness import review_current_sim_navigation_health
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model
from rosclaw.daemon.client import DaemonClient


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def review_passive_daemon_status(status, pins, *, client_pid):
    """Pure rejection reasons; status must come from an actual DaemonClient."""
    if type(status) is not dict or any(
        type(status.get(key)) is not dict for key in ("client_peer", "recovery", "ledger")
    ):
        return ["DAEMON_MALFORMED_OR_MISSING_LEDGER_STATUS"]
    reasons = []
    for key in ("daemon_instance_id", "daemon_pid", "daemon_uid", "robot_id"):
        if status.get(key) != pins[key]:
            reasons.append("DAEMON_IDENTITY_CHANGED:" + key)
    if (
        status.get("running") is not True
        or status.get("process_separated") is not True
        or status.get("southbound_owner") != "rosclawd"
        or status.get("client_peer", {}).get("pid") != client_pid
        or status.get("runtime_state") != "RUNNING"
    ):
        reasons.append("DAEMON_NOT_INDEPENDENTLY_HEALTHY")
    if (
        status.get("emergency_stop_latched") is not False
        or status.get("recovery", {}).get("required") is not False
        or status.get("ledger", {}).get("integrity_verified") is not True
        or status.get("ledger", {}).get("write_failed") is not False
    ):
        reasons.append("DAEMON_LEDGER_RECOVERY_OR_ESTOP_UNHEALTHY")
    executors = status.get("registered_executors")
    builtin_sandbox = {"sandbox.reach:SIMULATION", "sandbox.reach:FIXTURE"}
    if (
        status.get("drivers") != []
        or type(executors) is not list
        or any(type(value) is not str or value not in builtin_sandbox for value in executors)
    ):
        reasons.append("PASSIVE_DAEMON_HAS_EXECUTION_PATH")
    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrollment-sha256", required=True)
    parser.add_argument("--duration", type=float, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.enrollment_sha256) or not 0 < args.duration <= 90:
        raise ValueError("independent enrollment SHA256 and bounded observation duration required")
    root = Path("/evidence")
    raw = (root / "dual-supervision-enrollment.json").read_bytes()
    if sha(raw) != args.enrollment_sha256:
        raise ValueError("independent enrollment hash mismatch")
    enrollment = json.loads(raw)
    keys = {
        "schema_version",
        "request",
        "policy",
        "policy_hash",
        "source_pins",
        "guard_inputs",
        "prefix_sha256",
        "initial_L1_sha256",
        "daemon_ready_sha256",
    }
    if (
        set(enrollment) != keys
        or enrollment["schema_version"] != "rosclaw.passive_dual_enrollment.v1"
    ):
        raise ValueError("closed passive SDK enrollment required")
    initial_raw = (root / "dual-initial-L1.json").read_bytes()
    initial = json.loads(initial_raw)
    request = enrollment["request"]
    if (
        sha(initial_raw) != enrollment["initial_L1_sha256"]
        or initial["status"] != "READY_FOR_SIM_TEST"
        or initial["source_binding"]["result"] != "MATCHED"
        or any(
            initial[key] != request[key]
            for key in ("manifest_hash", "body_snapshot_hash", "sim_run_id")
        )
    ):
        raise ValueError("independently accepted initial Body/run/G1/L1 required")
    ready_raw = (root / "dual-enrolled-daemon-ready.json").read_bytes()
    ready = json.loads(ready_raw)
    if (
        sha(ready_raw) != enrollment["daemon_ready_sha256"]
        or ready["body_snapshot_hash"] != request["body_snapshot_hash"]
        or ready["run_id"] != request["sim_run_id"]
        or ready["manifest_hash"] != request["manifest_hash"]
        or ready["source_only"] is not True
        or ready["L2"] != "NOT_RUN"
    ):
        raise ValueError("independent actual daemon/Body/run enrollment required")
    pins = {
        "daemon_instance_id": ready["daemon_instance_id"],
        "daemon_pid": ready["pid"],
        "daemon_uid": ready["uid"],
        "robot_id": ready["body_id"],
    }
    client = DaemonClient(
        socket_path=root / "supervision/run/rosclawd.sock",
        expected_daemon_uid=ready["uid"],
        timeout_sec=0.15,
    )
    reader = SimulationSourceStreamReader(
        root,
        "ecm-original.jsonl",
        pins=SourceStreamPins(**enrollment["source_pins"]),
        accepted_prefix=(root / "dual-enrolled-prefix.jsonl").read_bytes(),
        expected_prefix_sha256=enrollment["prefix_sha256"],
        guard_inputs=enrollment["guard_inputs"],
    )
    started = time.monotonic_ns()
    reason = None
    rows = 0
    try:
        with (root / "dual-supervision-original.jsonl").open("x") as trace:
            while True:
                health = status = source = None
                pointer = None
                body_sha = config_sha = None
                try:
                    pointer = json.loads((root / "live-native-pointer.json").read_bytes())
                    if not re.fullmatch(r"original-native-[0-9]{3,6}\.json", pointer["file"]):
                        raise ValueError("original bounded Native filename required")
                    native_raw = (root / pointer["file"]).read_bytes()
                    if sha(native_raw) != pointer["sha256"]:
                        raise ValueError("original Native hash mismatch")
                    native = json.loads(native_raw)
                    model = build_system_model(
                        RosGraphSnapshot.from_dict(
                            {
                                "ros_version": "ros2",
                                "distro": "jazzy",
                                "endpoint": "fixture://actual-gazebo-source",
                                "captured_at": native["captured_at"],
                                **native["graph"],
                            }
                        ),
                        native=native,
                    )
                    health = review_current_sim_navigation_health(
                        root / "generic-compiled",
                        request,
                        model=model,
                        policy=enrollment["policy"],
                        policy_hash=enrollment["policy_hash"],
                    )
                    health_monotonic_ns = time.monotonic_ns()
                    if health["status"] != "HEALTHY_SIM_OBSERVATION_ONLY":
                        raise ValueError("CURRENT_ROS_HEALTH:" + health["status"])
                    status = client.get_runtime_status()
                    daemon_reasons = review_passive_daemon_status(
                        status, pins, client_pid=os.getpid()
                    )
                    if daemon_reasons:
                        raise ValueError(";".join(daemon_reasons))
                    body_sha = sha(
                        (root / "supervision/home/body/refs/effective_body.json").read_bytes()
                    )
                    config_sha = sha((root / "supervision/execution_config.json").read_bytes())
                    if (
                        body_sha != ready["effective_body_file_sha256"]
                        or config_sha != ready["execution_config_sha256"]
                    ):
                        raise ValueError("DAEMON_IMMUTABLE_BODY_OR_RUN_CONFIG_CHANGED")
                    source = reader.poll()
                    now = time.monotonic_ns()
                    if not source["continuous"]:
                        raise ValueError("CURRENT_SOURCE:" + source["reason"])
                    if now - health_monotonic_ns > 500_000_000:
                        raise ValueError("CURRENT_HEALTH_REVIEW_EXPIRED")
                except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                    reason = str(exc)[:1000]
                now = time.monotonic_ns()
                row = {
                    "index": rows,
                    "monotonic_ns": now,
                    "native_pointer": pointer,
                    "health": health,
                    "daemon_status": status,
                    "source": source,
                    "daemon_body_file_sha256": body_sha,
                    "daemon_config_sha256": config_sha,
                    "terminal": reason is not None,
                    "reason": reason,
                    "execution_authority": "NONE",
                    "L2": "NOT_RUN",
                }
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                trace.flush()
                rows += 1
                if reason is not None or now - started >= args.duration * 1e9:
                    break
                time.sleep(0.05)
        outcome = {
            "status": "TERMINAL_LATCHED" if reason else "PASS_LOCAL_DUAL_READONLY_WINDOW",
            "reason": reason,
            "duration_s": (time.monotonic_ns() - started) / 1e9,
            "rows": rows,
            "source_only": True,
            "L2": "NOT_RUN",
            "G3_done": False,
            "v1_done": False,
            "enrollment_sha256": args.enrollment_sha256,
            "daemon_generation": pins["daemon_instance_id"],
            "execution_authority": "NONE",
        }
        with (root / "dual-supervision-outcome.json").open("x") as stream:
            stream.write(json.dumps(outcome, indent=2) + "\n")
        print(json.dumps(outcome), flush=True)
    finally:
        reader.close()


if __name__ == "__main__":
    main()

"""Owned SDK SIGKILL/MCP E-stop faults: actual motion trigger and independent stop.

This operator fixture imports no ROS, Runtime, driver, publisher or goal client.
SIGKILL targets only the pinned daemon child; E-stop is requested by canonical MCP.
"""

import argparse
import hashlib
import json
import os
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from simulation_completed_artifact import publish_completed_fault_outcome
from simulation_current_facts import _read, read_current_native_pointer
from simulation_execution_fault_policy import (
    MOTION_FAULT_ACTOR_STATUS,
    parse_governed_motion_fault_policy,
)
from simulation_owned_process_fault import pause_owned_process, resume_owned_process

from rosclaw.body.resolver import BodyResolver
from rosclaw.connectors.ros.mission.navigation_binding import SimulationNavigationBinding


def last_physics(root):
    fd = os.open(root / "witness.jsonl", os.O_RDONLY | os.O_NOFOLLOW)
    try:
        size = os.fstat(fd).st_size
        if not 0 < size <= 10_000_000:
            raise ValueError("bounded independent original physics required")
        raw = os.pread(fd, min(size, 16384), max(0, size - 16384))
    finally:
        os.close(fd)
    complete = raw[: raw.rfind(b"\n") + 1]
    return json.loads(complete.splitlines()[-1])


def fresh_physics(root, binding):
    sample = last_physics(root)
    age = (datetime.now(UTC) - datetime.fromisoformat(sample["captured_at"])).total_seconds()
    if (
        not 0 <= age <= 0.3
        or sample.get("run_id") != binding.run_id
        or sample.get("body_snapshot_hash") != binding.body_snapshot_hash
        or sample.get("body_id") != binding.body_id
        or sample.get("frame_id") != binding.frame_id
        or sample.get("observation_complete") is not True
        or sample.get("evidence_domain") != "GAZEBO_PHYSICS"
    ):
        raise ValueError("fresh independent physical Body/run required")
    return sample


def is_physical_motion(sample, binding):
    binding.validate_observation(sample)
    return sample["lease_enabled"] is True and (
        abs(sample["linear_speed_mps"]) > binding.stop_speed_mps
        or abs(sample["angular_speed_rps"]) > binding.stop_yaw_rate_rps
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-sha256", required=True)
    parser.add_argument("--daemon-pid", type=int, required=True)
    parser.add_argument("--daemon-start-time", required=True)
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("owned isolated disposable SIM fault required")
    root = Path("/evidence")
    raw = _read(root / "governed-fault.json", 4096)
    config = json.loads(_read(root / "generic-compiled/execution_config.json", 16384))
    binding = SimulationNavigationBinding.from_body(
        BodyResolver(
            workspace=root / "generic-compiled", body_id=config["body_id"]
        ).get_effective_body(recompile_if_stale=False)
    )
    goals = json.loads(_read(root / "governed-goals.json", 16_000))
    policy = parse_governed_motion_fault_policy(
        raw,
        expected_sha256=args.policy_sha256,
        run_id=binding.run_id,
        body_snapshot_hash=binding.body_snapshot_hash,
        world_sha256=hashlib.sha256(_read(Path("/assets/world.sdf"), 20_000_000)).hexdigest(),
        action_id=goals["goals"][0]["action_id"],
    )
    ready = json.loads(_read(root / "governed-daemon-ready-original.json", 16384))
    if ready["pid"] != args.daemon_pid:
        raise ValueError("actual owned daemon PID required")
    with (root / "governed-fault-actor-ready-original.json").open("x") as f:
        json.dump(
            {
                "policy_sha256": args.policy_sha256,
                "case": policy["case"],
                "run_id": binding.run_id,
                "body_snapshot_hash": binding.body_snapshot_hash,
                "ready_monotonic_ns": time.monotonic_ns(),
                "requests_sent": 0,
            },
            f,
        )
        f.write("\n")
    deadline = time.monotonic_ns() + 15_000_000_000
    while True:
        sample = fresh_physics(root, binding)
        sink = json.loads(_read(root / "governed-sink-status.json", 4096, allow_atomic_unlink=True))
        if (
            sink["terminal"] is False
            and sink["action_id"] == policy["action_id"]
            and sink["published_velocity_messages"] > 0
            and is_physical_motion(sample, binding)
        ):
            break
        if time.monotonic_ns() >= deadline:
            raise ValueError("original governed physical motion was not observed before injection")
        time.sleep(0.01)
    proc = Path(f"/proc/{args.daemon_pid}")
    if (
        proc.joinpath("stat").read_text().split(")", 1)[1].split()[19] != args.daemon_start_time
        or b"/repo/integrations/ros_probe/acceptance/daemon.py"
        not in proc.joinpath("cmdline").read_bytes()
        or b"/evidence/execution-control/daemon" not in proc.joinpath("cmdline").read_bytes()
    ):
        raise ValueError("original owned daemon process identity changed")
    before = time.monotonic_ns()
    utc = datetime.now(UTC).isoformat()
    injection = {
        "case": policy["case"],
        "phase": "during_motion",
        "run_id": binding.run_id,
        "body_snapshot_hash": binding.body_snapshot_hash,
        "daemon_pid": args.daemon_pid,
        "daemon_instance_id": ready["daemon_instance_id"],
        "daemon_start_time": args.daemon_start_time,
        "signal": "SIGKILL" if policy["case"] == "daemon_sigkill" else None,
        "canonical_tool": "emergency_stop" if policy["case"] == "emergency_stop" else None,
        "before_monotonic_ns": before,
        "captured_at": utc,
        "actual_prior_physics": sample,
        "actual_prior_sink": sink,
    }
    with (root / "governed-fault-injection-original.json").open("x") as f:
        json.dump(injection, f, indent=2)
        f.write("\n")
    paused = None
    fault_effect = {}
    if policy["case"] == "daemon_sigkill":
        os.kill(args.daemon_pid, signal.SIGKILL)
    elif policy["case"] == "emergency_stop":
        with (root / "governed-emergency-stop-trigger-original.json").open("x") as f:
            json.dump({"policy_sha256": args.policy_sha256, "injection": injection}, f)
            f.write("\n")
        deadline = time.monotonic_ns() + 3_000_000_000
        while not (root / "governed-emergency-stop-MCP-original.json").exists():
            if time.monotonic_ns() >= deadline:
                raise ValueError(
                    "canonical MCP E-stop was not observed after actual motion trigger"
                )
            time.sleep(0.01)
    elif policy["case"] == "lease_loss":
        with (root / "governed-lease-loss-trigger-original.json").open("x") as f:
            json.dump({"policy_sha256": args.policy_sha256, "injection": injection}, f)
            f.write("\n")
    elif policy["case"] == "lidar_stream_lost":
        paused = pause_owned_process(root, "scan_bridge")
        fault_effect["actual_owned_scan_bridge_pause"] = paused
        fault_effect["pause_completed_monotonic_ns"] = time.monotonic_ns()
    samples = []
    stable_since = None
    last_stamp = None
    deadline = time.monotonic_ns() + (
        13_000_000_000 if policy["case"] == "lease_loss" else 3_000_000_000
    )
    while time.monotonic_ns() < deadline:
        sample = fresh_physics(root, binding)
        if sample["captured_at"] != last_stamp and datetime.fromisoformat(
            sample["captured_at"]
        ) >= datetime.fromisoformat(utc):
            last_stamp = sample["captured_at"]
            samples.append(sample)
            if len(samples) > 400:
                raise ValueError("bounded original post-fault physical sample budget exhausted")
            if sample["lease_enabled"] is False and binding.stopped(sample):
                stable_since = stable_since or datetime.fromisoformat(sample["captured_at"])
                if (
                    datetime.fromisoformat(sample["captured_at"]) - stable_since
                ).total_seconds() >= 0.3:
                    break
            else:
                stable_since = None
        time.sleep(0.01)
    else:
        raise ValueError("independent lease-closed stable physical stop not observed")
    terminal = json.loads(_read(root / "governed-sink-terminal.json", 16384))
    if not terminal["status"]["terminal"] or terminal["status"]["lease_expires_ns"] != 0:
        raise ValueError("terminal final sink stop required")
    if paused is not None:
        pointer, _ = read_current_native_pointer(root)
        native_raw = _read(root / pointer["file"], 10_000_000)
        if hashlib.sha256(native_raw).hexdigest() != pointer["sha256"]:
            raise ValueError("original post-loss Native bytes changed")
        topic = json.loads(_read(Path("/assets/requirements.json"), 32_000))["interface_names"][
            "sensing.lidar"
        ]
        signals = [x for x in json.loads(native_raw)["signals"] if x["topic"] == topic]
        if len(signals) != 1 or signals[0]["last_message_age_ms"] <= 500:
            raise ValueError("actual original lidar stream loss was not observed")
        fault_effect["native_pointer_before_resume"] = pointer
        fault_effect["original_lidar_signal_before_resume"] = signals[0]
        fault_effect["resume_after_independent_terminal_stop"] = resume_owned_process(paused)
        fault_effect["resume_monotonic_ns"] = time.monotonic_ns()
    outcome = {
        "status": MOTION_FAULT_ACTOR_STATUS[policy["case"]],
        "case": policy["case"],
        "fault_effect": fault_effect,
        "policy_sha256": args.policy_sha256,
        "injection": injection,
        "after_monotonic_ns": time.monotonic_ns(),
        "original_post_fault_physics": samples,
        "actual_terminal_sink": terminal,
        "terminal_receipt_expectation": "MISSING_AFTER_SIGKILL_REPORTED_SEPARATELY"
        if policy["case"] == "daemon_sigkill"
        else "REQUIRE_ORIGINAL_FAILED_OR_CANCELLED_RECEIPT",
        "G3_done": False,
        "v1_done": False,
    }
    publish_completed_fault_outcome(root, outcome)
    print(json.dumps({"status": outcome["status"], "samples": len(samples)}), flush=True)


if __name__ == "__main__":
    main()

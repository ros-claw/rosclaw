"""SDK operator fault injection in one explicitly owned, source-only SIM.

Not an Agent tool. Never creates a velocity publisher or navigation goal.
"""

import argparse
from contextlib import suppress
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from gazebo_source_witness.prepare import check_prepared_assets
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

OUT = Path("/evidence")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "sensor_stop",
            "clock_freeze",
            "tf_loss",
            "nav2_inactive",
            "qos_mismatch",
            "namespace_mismatch",
        ],
    )
    args = parser.parse_args()
    if not Path("/.dockerenv").is_file() or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise ValueError("isolated disposable operator SIM required")
    check_prepared_assets("/assets", args.inventory_sha256, "/tmp/readiness-fault-assets")
    initialization = json.loads(Path("/assets/source-initialization.json").read_text())
    if (
        initialization["run_id"] != args.run_id
        or initialization["operator_owned_disposable_sim_only"] is not True
    ):
        raise ValueError("explicit owned SIM run required")
    processes = json.loads((OUT / "owned-stack-processes.json").read_text())
    paused, child, node = [], None, None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def pause(label):
        root = processes[label]["pid"]
        cmdline = Path(f"/proc/{root}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        if " ".join(processes[label]["argv"]) not in cmdline:
            raise ValueError("owned process argv changed")
        descendants = {root}
        while True:
            before = set(descendants)
            for path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    stat = path.read_text().rsplit(")", 1)[1].split()
                    if int(stat[1]) in descendants:
                        descendants.add(int(path.parent.name))
                except (FileNotFoundError, ProcessLookupError):
                    continue
            if before == descendants:
                break
        for pid in sorted(descendants, reverse=True):
            os.kill(pid, signal.SIGSTOP)
            paused.append(pid)
        return {"label": label, "root_pid": root, "paused_pids": paused}

    try:
        details = {
            "mode": args.mode,
            "run_id": args.run_id,
            "velocity_publishers_created": 0,
            "goals_sent": 0,
        }
        if args.mode in {"sensor_stop", "namespace_mismatch"}:
            details["owned_process_fault"] = pause("scan_bridge")
            if args.mode == "namespace_mismatch":
                argv = [
                    "ros2",
                    "run",
                    "ros_gz_bridge",
                    "parameter_bridge",
                    "--ros-args",
                    "-p",
                    "config_file:=/evidence/scan-bridge.yaml",
                    "-p",
                    "use_sim_time:=true",
                    "-r",
                    "/scan:=/misplaced/scan",
                ]
                details["replacement_argv"] = argv
                child = subprocess.Popen(argv)
        elif args.mode == "tf_loss":
            details["owned_process_fault"] = pause("nav2")
        elif args.mode in {"clock_freeze", "nav2_inactive"}:
            argv = (
                [
                    "gz",
                    "service",
                    "-s",
                    "/world/ros_expert/control",
                    "--reqtype",
                    "gz.msgs.WorldControl",
                    "--reptype",
                    "gz.msgs.Boolean",
                    "--timeout",
                    "5000",
                    "--req",
                    "pause: true",
                ]
                if args.mode == "clock_freeze"
                else ["ros2", "lifecycle", "set", "/collision_monitor", "deactivate"]
            )
            details["operator_fault_argv"] = argv
            result = subprocess.run(argv, capture_output=True, text=True, timeout=10)
            details.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
            if (
                result.returncode
                or (args.mode == "clock_freeze" and "true" not in result.stdout)
                or (args.mode == "nav2_inactive" and "successful" not in result.stdout.lower())
            ):
                raise RuntimeError("owned SIM fault did not acknowledge")
        else:
            rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
            node = Node("rosclaw_explicit_readiness_qos_fault")
            # Actual /scan publisher is VOLATILE; this incompatible subscriber
            # requests TRANSIENT_LOCAL and receives no fabricated scan.
            node.create_subscription(
                LaserScan,
                "/scan",
                lambda _: None,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            )
            details["subscription"] = {"topic": "/scan", "durability": "TRANSIENT_LOCAL"}
        (OUT / "readiness-fault-applied.json").write_text(json.dumps(details, indent=2) + "\n")
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if child and child.poll() is not None:
                raise RuntimeError("namespace fault bridge exited")
            if node:
                rclpy.spin_once(node, timeout_sec=0.1)
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for pid in paused:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGCONT)
        if child and child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        if node:
            node.destroy_node()
            rclpy.shutdown()
        (OUT / "readiness-fault-cleanup.json").write_text(
            json.dumps(
                {
                    "resumed_owned_pids": paused,
                    "replacement_returncode": child.returncode if child else None,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()

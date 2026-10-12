"""SDK-owned passive known-fixture capture and real-observation Body compilation.

Only the isolated operator fixture launches ROS processes. No action client,
velocity publisher, initial-pose publication, daemon or Runtime is created.
The external supervisor must separately inspect and pin the running instance.
"""

import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from rosclaw.connectors.ros.context.navigation_review import review_navigation_candidate
from rosclaw.connectors.ros.context.navigation_workspace import compile_navigation_body_workspace
from rosclaw.connectors.ros.context.navigation_workspace_review import (
    inspect_navigation_body_workspace,
)
from rosclaw.connectors.ros.discovery.graph import RosGraphSnapshot
from rosclaw.connectors.ros.intelligence import build_system_model

OUT = Path("/evidence")
ROOT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def process_fact(pid):
    path = Path("/proc") / str(pid)
    return {
        "pid": pid,
        "cmdline": path.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(),
        "stat": path.joinpath("stat").read_text(),
    }


def observer_facts():
    servers, loaded = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            maps = (path / "maps").read_text()
            if "libgz-sim8.so" in maps:
                servers.append(process_fact(int(path.name)))
            if "/observer/librosclaw_source_witness.so" in maps:
                loaded.append(process_fact(int(path.name)))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue
    return {
        "gz_servers": servers,
        "observer_loaded_servers": loaded,
        "library_sha256": hashlib.sha256(
            Path("/observer/librosclaw_source_witness.so").read_bytes()
        ).hexdigest(),
    }


def main():
    settings = json.loads(Path("/settings.json").read_text())
    requirements = {
        "interface_names": {
            "mapping.occupancy_map": "/map",
            "state.odometry": "/odom",
            "sensing.lidar": "/scan",
            "navigation.navigate_to_pose": "/navigate_to_pose",
        },
        "lifecycle_nodes": {
            "localization": "/amcl",
            "planner": "/planner_server",
            "controller": "/velocity_smoother",
            "collision_monitor": "/collision_monitor",
            "navigator": "/bt_navigator",
        },
        "costmap_topics": ["/local_costmap/costmap", "/global_costmap/costmap"],
        "localization_topic": "/amcl_pose",
        "velocity_input_topic": "/cmd_vel_smoothed",
        "velocity_output_topic": "/nav_cmd_vel",
        "velocity_consumer_node": "/rosclaw_sim_witness",
    }
    save("requirements.json", requirements)
    stack_log = (OUT / "stack-original.log").open("x")
    argv = [
        "python3",
        str(ROOT / "stack.py"),
        "--navigation-only",
        "--source-only-observer",
        "--external-source-witness",
        "--prepared-assets",
        "/assets",
        "--expected-prepared-assets-sha256",
        settings["inventory_sha256"],
        "--run-id",
        settings["run_id"],
        "--seed",
        "101201",
    ]
    save("stack-argv.json", argv)
    stack = subprocess.Popen(
        argv, stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True
    )
    witness = None
    witness_logs = []
    rclpy.init(args=["--ros-args", "-p", "use_sim_time:=true"])
    node = Node("rosclaw_source_capture")
    state = {"native": None, "model": None, "review": None, "physics": None}
    reviews = []

    def observed(message):
        native = json.loads(message.data)
        graph = RosGraphSnapshot.from_dict(
            {
                "ros_version": "ros2",
                "distro": "jazzy",
                "endpoint": "fixture://actual-gazebo-source",
                "captured_at": native["captured_at"],
                **native["graph"],
            }
        )
        model = build_system_model(graph, native=native)
        review = review_navigation_candidate(
            model, Path("/assets/robot.urdf").read_bytes(), requirements
        )
        index = len(reviews)
        save(f"original-native-{index:03}.json", native)
        save(f"sealed-system-{index:03}.json", model.to_dict())
        reviews.append({"index": index, "captured_at": native["captured_at"], "review": review})
        state.update(native=native, model=model, review=review)

    def physics(message):
        state["physics"] = json.loads(message.data)

    node.create_subscription(
        String,
        "/rosclaw_probe/snapshot",
        observed,
        QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
    )
    node.create_subscription(String, "/rosclaw_sim/observation", physics, 10)

    def launch_witness(name, workspace, config):
        log = (OUT / f"{name}-original.log").open("x")
        witness_logs.append(log)
        args = [
            "python3",
            str(ROOT / "witness.py"),
            "--ros-args",
            "-p",
            "use_sim_time:=true",
            "-p",
            "read_only_source_observer:=true",
            "-p",
            "body_workspace:=" + str(workspace),
            "-p",
            "execution_config_path:=" + str(config),
        ]
        save(f"{name}-argv.json", args)
        return subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def spin_until(condition, seconds):
        deadline = time.monotonic() + seconds
        while not condition():
            if time.monotonic() >= deadline:
                save("incomplete-original-reviews.json", reviews)
                raise RuntimeError("bounded passive capture condition not achieved")
            if stack.poll() is not None or (witness and witness.poll() is not None):
                raise RuntimeError("owned source stack or witness exited unexpectedly")
            rclpy.spin_once(node, timeout_sec=0.05)

    try:
        spin_until(lambda: (OUT / "execution_config.json").exists(), 20)
        witness = launch_witness(
            "initial-source-witness", OUT / "home", OUT / "execution_config.json"
        )
        spin_until(
            lambda: (
                state["review"]
                and state["review"]["status"] == "PROPOSED"
                and state["physics"]
                and state["physics"]["observation_complete"]
            ),
            65,
        )
        initial_physics = state["physics"]
        save("initial-physics-witness.json", initial_physics)
        if any(
            initial_physics[k] != 0
            for k in ("source_received_control_messages", "source_published_velocity_messages")
        ):
            raise RuntimeError("source-only capture received control messages")
        model, review = state["model"], state["review"]
        policy = {
            "schema_version": "rosclaw.navigation_sim_fixture_policy.v1",
            "source": "simulator_operator_fixture_policy",
            "approved": True,
            "policy_id": "G1-known-source-readonly-capture",
            "body_id": "g1-observed-known-base",
            "source_snapshot_hash": model.snapshot_hash,
            "source_urdf_sha256": review["candidate"]["source_urdf_sha256"],
            "navigation_requirements_hash": review["requirements_hash"],
            "navigation_limits": {
                "max_goal_distance_m": 0.5,
                "max_duration_sec": 30.0,
                "goal_position_tolerance_m": 0.05,
                "goal_yaw_tolerance_rad": 0.1,
                "max_speed_mps": 0.2,
                "max_yaw_rate_rps": 0.5,
                "stop_speed_mps": 0.005,
                "stop_yaw_rate_rps": 0.01,
            },
            "witness_declaration": {
                "observation_topic": "/rosclaw_sim/observation",
                "lease_service": "/rosclaw_sim/lease",
                "run_id": settings["run_id"],
            },
        }
        compiled = OUT / "generic-compiled"
        manifest = compile_navigation_body_workspace(
            compiled, model, Path("/assets/robot.urdf").read_bytes(), requirements, policy=policy
        )
        inspection = inspect_navigation_body_workspace(
            compiled,
            expected_manifest_hash=manifest["manifest_hash"],
            expected_body_hash=manifest["body_snapshot_hash"],
        )
        save("compiled-inspection.json", inspection)
        # The supervisor owns this single intentional witness replacement. The
        # stack never masks a crashed child, and the new process has no actuator.
        save("initial-witness-process.json", process_fact(witness.pid))
        stop(witness)
        save("initial-witness-exit.json", {"returncode": witness.returncode})
        state["physics"] = None
        witness = launch_witness(
            "compiled-source-witness", compiled, compiled / "execution_config.json"
        )
        spin_until(
            lambda: (
                state["physics"]
                and state["physics"].get("body_snapshot_hash") == manifest["body_snapshot_hash"]
                and state["physics"]["observation_complete"]
                and state["review"]["status"] == "PROPOSED"
            ),
            30,
        )
        save("latest-native.json", state["native"])
        save("latest-system.json", state["model"].to_dict())
        save("latest-navigation-review.json", state["review"])
        save("latest-physics-witness.json", state["physics"])
        save("all-navigation-reviews.json", reviews)
        save("observer-process-facts.json", observer_facts())
        save("compiled-witness-process.json", process_fact(witness.pid))
        save(
            "capture_ready.json",
            {
                "manifest_hash": manifest["manifest_hash"],
                "body_snapshot_hash": manifest["body_snapshot_hash"],
                "native_samples": len(reviews),
            },
        )
        spin_until(lambda: (OUT / "host_capture_ack.json").exists(), 20)
        print(
            json.dumps(
                {
                    "status": "ACTUAL_SOURCE_CAPTURE_COMPLETE_EXTERNAL_REVIEW_REQUIRED",
                    "native_samples": len(reviews),
                    "compiled_body": manifest["body_snapshot_hash"],
                }
            ),
            flush=True,
        )
    finally:
        if witness:
            stop(witness)
        stop(stack)
        stack_log.close()
        for log in witness_logs:
            log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
